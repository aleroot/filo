#include "SubagentOrchestrator.hpp"

#include "AgentDefinitionRegistry.hpp"
#include "AgentMode.hpp"
#include "DelegatedAgentRunner.hpp"
#include "DelegationBudget.hpp"
#include "../config/BuiltinSubagents.hpp"
#include "../config/ConfigManager.hpp"
#include "../llm/ProviderFactory.hpp"
#include "../llm/ProviderManager.hpp"
#include "../logging/Logger.hpp"
#include "../session/SessionStore.hpp"
#include "../tools/TaskTool.hpp"
#include "../tools/ToolManager.hpp"
#include "../tools/shell/ShellSessions.hpp"
#include "../tools/ToolNames.hpp"
#include "../tools/read/ReadTypes.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <format>
#include <functional>
#include <iterator>
#include <mutex>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace core::agent {

namespace {

[[nodiscard]] bool acquire_run_lock(std::unique_lock<std::timed_mutex>& lock,
                                    const std::function<bool()>& cancelled) {
    using namespace std::chrono_literals;
    do {
        if (cancelled && cancelled()) return false;
        if (lock.try_lock_for(25ms)) return true;
    } while (true);
}

class SubagentSlot final {
public:
    explicit SubagentSlot(
        std::shared_ptr<std::counting_semaphore<SubagentOrchestrator::kMaxParallelSubagents>> slots)
        : slots_(std::move(slots)) {}

    SubagentSlot(const SubagentSlot&) = delete;
    SubagentSlot& operator=(const SubagentSlot&) = delete;

    ~SubagentSlot() {
        if (acquired_) slots_->release();
    }

    [[nodiscard]] bool acquire(const std::function<bool()>& cancellation_requested) {
        using namespace std::chrono_literals;
        if (cancellation_requested && cancellation_requested()) return false;
        while (!slots_->try_acquire_for(25ms)) {
            if (cancellation_requested && cancellation_requested()) return false;
        }
        acquired_ = true;
        return true;
    }

private:
    std::shared_ptr<std::counting_semaphore<SubagentOrchestrator::kMaxParallelSubagents>> slots_;
    bool acquired_ = false;
};

[[nodiscard]] bool is_write_destructive_tool(std::string_view tool_name) {
    return core::tools::names::is_write_destructive_tool(tool_name);
}

[[nodiscard]] bool is_terminal_tool(const core::llm::Tool& tool) {
    return core::tools::names::is_terminal_tool(tool.function.name);
}

// A read-only profile keeps the shell, which then runs gated
// (SessionContext::read_only_shell).
[[nodiscard]] bool read_only_profile_keeps(const core::llm::Tool& tool) {
    return tool.function.annotations.read_only_hint || is_terminal_tool(tool);
}

class WorkerShell final {
public:
    WorkerShell(std::shared_ptr<core::tools::shell::ShellSessions> sessions, std::string id)
        : sessions_(std::move(sessions)), id_(std::move(id)) {}

    WorkerShell(const WorkerShell&) = delete;
    WorkerShell& operator=(const WorkerShell&) = delete;

    ~WorkerShell() { sessions_->clear(id_); }

    [[nodiscard]] const std::string& id() const noexcept { return id_; }

private:
    std::shared_ptr<core::tools::shell::ShellSessions> sessions_;
    std::string id_;
};

[[nodiscard]] int sanitize_max_steps(int value, int fallback) {
    return DelegationBudget::sanitize(value, fallback);
}

[[nodiscard]] bool is_filo_task_id(std::string_view value) {
    constexpr std::string_view kPrefix = "task_";
    constexpr std::size_t kHexDigits = 8;
    if (!value.starts_with(kPrefix) || value.size() != kPrefix.size() + kHexDigits) {
        return false;
    }

    return std::ranges::all_of(value.substr(kPrefix.size()), [](unsigned char ch) {
        return std::isxdigit(ch) != 0;
    });
}

[[nodiscard]] bool is_provider_generated_uuid(std::string_view value) {
    if (value.size() != 36) return false;
    for (std::size_t i = 0; i < value.size(); ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            if (value[i] != '-') return false;
        } else if (!std::isxdigit(static_cast<unsigned char>(value[i]))) return false;
    }
    return true;
}

[[nodiscard]] DelegatedAgentRunner::PermissionCheck adapt_permission_check(
    const SubagentOrchestrator::RunContext& context,
    bool read_only_shell) {
    if (!context.permission_check) return {};

    return [permission_check = context.permission_check, read_only_shell](
               std::string_view tool_name,
               std::string_view args) {
        // ShellTool refuses what a read-only shell may not run; asking is moot.
        if (read_only_shell && core::tools::names::is_terminal_tool(tool_name)) return true;
        return permission_check(
            std::string(tool_name),
            std::string(args));
    };
}

} // namespace

std::vector<std::string> SubagentOrchestrator::ExecutionPlan::allowed_tool_names() const {
    std::vector<std::string> names;
    names.reserve(tools.size() + (allow_task_tool ? 2 : 1));
    for (const auto& tool : tools) {
        names.push_back(tool.function.name);
    }
    if (allow_task_tool) {
        names.push_back(std::string(SubagentOrchestrator::kTaskToolName));
    }
    names.push_back(std::string(core::tools::names::kReadToolResult));
    for (const auto& name : supplemental_tools) {
        if (std::ranges::find(names, name) == names.end()) {
            names.push_back(name);
        }
    }
    return names;
}

SubagentOrchestrator::SubagentOrchestrator(
    core::tools::ToolManager& tool_manager,
    const core::config::AppConfig* app_config,
    std::shared_ptr<core::session::SessionStatsRegistry> session_stats_registry,
    std::shared_ptr<core::scm::WorkspaceLeaseRegistry> workspace_leases)
    : tool_manager_(tool_manager)
    , session_stats_registry_(std::move(session_stats_registry))
    , workspace_leases_(std::move(workspace_leases))
    , profiles_(compose_profiles({}, app_config))
    , app_config_(app_config) {}

std::vector<SubagentOrchestrator::Profile> SubagentOrchestrator::make_default_profiles() {
    std::vector<Profile> profiles;
    profiles.reserve(core::config::builtin_subagents::kAll.size());
    for (const auto& builtin : core::config::builtin_subagents::kAll) {
        profiles.push_back(Profile{
            .name = std::string(builtin.name),
            .description = std::string(builtin.description),
            .prompt = std::string(builtin.prompt),
            .provider_override = "",
            .model_override = "",
            .response_format = std::nullopt,
            .allowed_tools = {builtin.allowed_tools.begin(), builtin.allowed_tools.end()},
            .use_allow_list = !builtin.allowed_tools.empty(),
            .allow_task_tool = false,
            .max_steps = builtin.max_steps,
            .enabled = true,
            .read_only_tools = builtin.read_only,
        });
    }
    return profiles;
}

core::llm::Tool SubagentOrchestrator::task_tool_definition(
    const std::vector<std::filesystem::path>& workspace_roots) const {
    core::llm::Tool tool;
    tool.type = "function";
    tool.function.name = std::string(kTaskToolName);
    tool.function.title = std::string(core::tools::names::kTaskTitle);
    tool.function.description = build_task_description(workspace_roots);
    tool.function.parameters = {
        {
            .name = "description",
            .type = "string",
            .description = "A short 3-8 word label describing this delegated task.",
            .required = true,
        },
        {
            .name = "prompt",
            .type = "string",
            .description = "The full task instructions for the subagent.",
            .required = true,
        },
        {
            .name = "subagent_type",
            .type = "string",
            .description =
                "Configured execution profile to run (for example, general, explore, or a "
                "custom profile such as luna-explore).",
            .required = true,
        },
        {
            .name = "task_id",
            .type = "string",
            .description =
                "Optional Filo task_id returned by an earlier successful task call. "
                "Omit this field when starting a new task; never invent an ID.",
            .required = false,
        },
        {
            .name = "fork_turns",
            .type = "string",
            .description =
                "Parent history a new subagent inherits. `all` (default) gives it this "
                "conversation's user messages and your answers, so the prompt only needs the "
                "task itself; `none` starts it with just the prompt. Ignored on resume.",
            .required = false,
            .schema = R"({"type":"string","enum":["all","none"],"description":"Parent history a new subagent inherits. `all` (default) gives it this conversation's user messages and your answers, so the prompt only needs the task itself; `none` starts it with just the prompt. Ignored on resume."})",
        },
        {
            .name = "max_steps",
            .type = "integer",
            .description = "Optional model-step budget for this run (1-64, including the final summary). "
                           "Omit to use the profile default; increase when resuming partial work.",
            .required = false,
            .schema = R"({"type":"integer","minimum":1,"maximum":64,"description":"Model-step budget for this run, including the final summary. Omit to keep the profile or previous run's budget; increase when resuming partial work."})",
        },
        {
            .name = "command",
            .type = "string",
            .description = "Optional command or trigger label for traceability.",
            .required = false,
        },
    };
    return tool;
}

std::string SubagentOrchestrator::execute_task(
    std::string_view json_args,
    const std::shared_ptr<core::llm::LLMProvider>& provider,
    const RunContext& context) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded{std::string(json_args)};
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        return render_error_json("Invalid JSON arguments for task.");
    }

    simdjson::dom::object obj;
    if (doc.get(obj) != simdjson::SUCCESS) {
        return render_error_json("Task arguments must be a JSON object.");
    }

    std::string description;
    std::string prompt;
    std::string subagent_type;
    std::string task_id;

    std::string_view value;
    if (obj["description"].get(value) == simdjson::SUCCESS) description = std::string(value);
    if (obj["prompt"].get(value) == simdjson::SUCCESS) prompt = std::string(value);
    if (obj["subagent_type"].get(value) == simdjson::SUCCESS) subagent_type = std::string(value);
    if (obj["task_id"].get(value) == simdjson::SUCCESS) task_id = core::utils::str::trim_ascii_copy(value);
    bool fork_parent = true;
    if (obj["fork_turns"].get(value) == simdjson::SUCCESS) {
        const std::string fork_turns =
            core::utils::str::to_lower_ascii_copy(core::utils::str::trim_ascii_copy(value));
        if (fork_turns == "none") {
            fork_parent = false;
        } else if (!fork_turns.empty() && fork_turns != "all") {
            return render_error_json("fork_turns must be `all` or `none`.");
        }
    }

    const auto max_steps = DelegationBudget::parse(obj);
    if (!max_steps) return render_error_json(max_steps.error());

    if (description.empty()) {
        return render_error_json("Missing required 'description' for task.");
    }
    if (prompt.empty()) {
        return render_error_json("Missing required 'prompt' for task.");
    }
    if (subagent_type.empty()) {
        return render_error_json("Missing required 'subagent_type' for task.");
    }

    if (context.session_context.delegation_depth >= kMaxDelegationDepth) {
        return render_error_json(std::format(
            "Delegation depth limit reached ({} nested subagent levels). Complete this work "
            "directly instead of delegating it further.",
            kMaxDelegationDepth));
    }

    auto plan = build_execution_plan(ExecutionRequest{
        .worker = subagent_type,
        .worker_label = "subagent_type",
        .parent_mode = context.parent_mode,
        .inherited_provider = provider,
        .inherited_provider_name = context.active_provider_name,
        .inherited_model = context.active_model,
        .prefer_inherited_provider = true,
        .max_steps_override = *max_steps,
        .workspace_roots = context.session_context.workspace_view().ordered_roots(),
    });
    if (!plan) {
        return render_error_json(plan.error());
    }

    const Profile session_profile{
        .name = plan->worker_name,
        .description = plan->worker_description,
        .prompt = plan->worker_prompt,
        .provider_override = {},
        .model_override = {},
        .allowed_tools = {},
        .use_allow_list = false,
        .allow_task_tool = false,
        .max_steps = plan->max_steps,
        .enabled = true,
    };

    std::string session_error;
    auto session = get_or_create_session(
        description,
        session_profile,
        task_id,
        plan->model_name,
        session_error);
    if (!session) {
        return render_error_json(session_error.empty() ? "Failed to create task session." : session_error);
    }

    // A task_id identifies mutable conversational state. Only one invocation may
    // advance that state at a time, even when the parent emits duplicate resumes.
    std::unique_lock session_run_lock(session->run_mutex, std::defer_lock);
    if (!acquire_run_lock(session_run_lock, context.cancellation_requested)) {
        return render_error_json("Delegated task was cancelled while waiting to resume.");
    }
    if (session->pending_run.valid()
        && session->pending_run.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
        return render_error_json("This task's previous worker is still stopping. Retry the same task_id after it has stopped.");
    }
    // A resume keeps an explicitly chosen budget unless this call overrides it.
    if (!max_steps->has_value() && session->max_steps > 0) plan->max_steps = session->max_steps;
    auto subagent_slot = std::make_shared<SubagentSlot>(subagent_slots_);
    if (!subagent_slot->acquire(context.cancellation_requested)) {
        return render_error_json("Delegated task was cancelled while waiting to run.");
    }

    auto execution_provider = plan->provider;
    std::unique_lock serial_provider_lock(serial_provider_mutex_, std::defer_lock);
    if (execution_provider->capabilities().supports_parallel_requests) {
        if (auto isolated = execution_provider->fork_for_parallel_request()) {
            execution_provider = std::move(isolated);
        }
    } else {
        if (!acquire_run_lock(serial_provider_lock, context.cancellation_requested)) {
            return render_error_json("Delegated task was cancelled while waiting for its provider.");
        }
        if (pending_serial_run_.valid()
            && pending_serial_run_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
            return render_error_json("The provider's previous worker is still stopping. Retry after it has stopped.");
        }
    }

    std::optional<DelegatedAgentRunner::ResumeState> resume_state;
    bool forked_context = false;
    {
        std::lock_guard session_lock(session->mutex);
        if (session->model.empty()) {
            session->model = plan->model_name;
        }
        if (!session->history.empty() || !session->context_summary.empty()) {
            resume_state = DelegatedAgentRunner::ResumeState{
                .messages = session->history,
                .context_summary = session->context_summary,
                .mode = delegated_mode_name(context.parent_mode, plan->read_only),
            };
        } else if (fork_parent && !plan->isolated) {
            auto forked = context.parent_history
                ? fork_parent_history(*context.parent_history)
                : std::vector<core::llm::Message>{};
            if (!forked.empty() || !context.parent_context_summary.empty()) {
                forked_context = true;
                resume_state = DelegatedAgentRunner::ResumeState{
                    .messages = std::move(forked),
                    .context_summary = context.parent_context_summary,
                    .mode = delegated_mode_name(context.parent_mode, plan->read_only),
                };
            }
        }
    }

    // Parallel workers must not queue on, or change the cwd of, the parent's shell.
    const WorkerShell worker_shell(
        tool_manager_.shell_sessions(),
        core::context::worker_shell_session_id(context.session_context.shell_session(), session->id));
    core::context::SessionContext worker_context = context.session_context;
    worker_context.shell_session_id = worker_shell.id();
    worker_context.read_only_shell = plan->read_only_shell;

    const auto result = DelegatedAgentRunner::run({
        .provider = std::move(execution_provider),
        .provider_name = plan->provider_name,
        .tool_manager = tool_manager_,
        .session_context = worker_context,
        .mode = delegated_mode_name(context.parent_mode, plan->read_only),
        .model = plan->model_name,
        .response_format = plan->response_format,
        .allowed_tools = plan->allowed_tool_names(),
        .max_steps = plan->max_steps,
        .prompt = prompt,
        .worker_name = plan->worker_name,
        .worker_description = plan->worker_description,
        .worker_prompt = plan->worker_prompt,
        .parent_conversation_scope = context.parent_conversation_scope,
        .parent_turn_id = context.parent_turn_id,
        .root_turn_id = context.root_turn_id,
        .task_id = session->id,
        .task_description = description,
        .parent_tool_call_id = context.parent_tool_call_id,
        .resume_state = std::move(resume_state),
        .session_stats_registry = session_stats_registry_,
        .workspace_leases = workspace_leases_,
        .memory_system = context.memory_system,
        .change_tracker = context.change_tracker,
        .timeout = std::chrono::minutes(30),
        .cancellation_requested = context.cancellation_requested,
        .permission_check = adapt_permission_check(context, plan->read_only_shell),
        .on_event = context.on_subagent_event,
        .effort = context.effort,
        .advisor = context.advisor,
        .execution_journal = context.execution_journal,
        .execution_lease = subagent_slot,
        .isolated_context = plan->isolated,
        .forked_context = forked_context,
    });
    session->pending_run = result.settled;
    session->max_steps = plan->max_steps;
    if (serial_provider_lock.owns_lock()) pending_serial_run_ = result.settled;

    {
        std::lock_guard session_lock(session->mutex);
        session->model = plan->model_name;
        session->history = result.history;
        session->context_summary = result.context_summary;
    }

    return render_success_json(
        *session,
        *plan,
        result.final_text,
        result.status(),
        result.reason(),
        !result.settled.valid() || result.settled.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready,
        result.steps,
        result.tool_calls_total,
        result.tool_calls_total - result.tool_calls_success);
}

void SubagentOrchestrator::clear_sessions() {
    std::lock_guard lock(sessions_mutex_);
    sessions_.clear();
}

void SubagentOrchestrator::reload_profiles(const core::config::AppConfig& app_config) {
    app_config_ = &app_config;
    {
        std::lock_guard lock(profiles_mutex_);
        profiles_ = compose_profiles({}, &app_config);
    }
    clear_sessions();
}

SubagentOrchestrator::Profile SubagentOrchestrator::profile_from(const AgentDefinition& definition) {
    Profile profile{
        .name = definition.name,
        .description = definition.description,
        .prompt = definition.prompt,
        .provider_override = definition.provider,
        .model_override = definition.model,
        .response_format = std::nullopt,
        .allowed_tools = {},
        .use_allow_list = definition.tools.has_value(),
        .allow_task_tool = definition.allow_task_tool,
        .max_steps = sanitize_max_steps(definition.max_steps.value_or(16), 16),
        .enabled = true,
        .isolated = definition.isolated,
        .read_only_tools = definition.read_only,
    };
    if (definition.tools.has_value()) {
        profile.allowed_tools.insert(definition.tools->begin(), definition.tools->end());
    }
    return profile;
}

std::vector<SubagentOrchestrator::Profile> SubagentOrchestrator::compose_profiles(
    std::span<const AgentDefinition> definitions,
    const core::config::AppConfig* app_config) {
    auto profiles = make_default_profiles();
    // Agent files sit between the built-ins and the user's configuration, so
    // `subagents.<name>` can still pin a provider, model or budget for a
    // plugin's persona. They never replace a built-in: a third-party plugin
    // must not be able to redefine what `general` or `explore` means.
    std::vector<Profile> from_files;
    for (const auto& definition : definitions) {
        if (std::ranges::contains(profiles, definition.name, &Profile::name)
            || definition.name == core::tools::read::kReaderProfile) {
            core::logging::warn("Agent definition '{}' ({}) uses a reserved profile name; skipped",
                                definition.name, definition.source.string());
            continue;
        }
        from_files.push_back(profile_from(definition));
    }
    std::ranges::move(from_files, std::back_inserter(profiles));
    if (app_config != nullptr) {
        apply_config_overrides(profiles, *app_config);
    }
    return profiles;
}

std::vector<SubagentOrchestrator::Profile> SubagentOrchestrator::effective_profiles(
    const std::vector<std::filesystem::path>& workspace_roots) const {
    const auto definitions = AgentDefinitionRegistry::discover(workspace_roots);
    if (!definitions.empty()) return compose_profiles(definitions, app_config_);
    std::lock_guard lock(profiles_mutex_);
    return profiles_;
}

std::optional<SubagentOrchestrator::Profile> SubagentOrchestrator::find_profile(
    std::string_view name,
    const std::vector<std::filesystem::path>& workspace_roots) const {
    const std::string normalized = normalize_agent_name(name);
    for (auto& profile : effective_profiles(workspace_roots)) {
        if (profile.name == normalized) {
            return std::move(profile);
        }
    }
    return std::nullopt;
}

std::string SubagentOrchestrator::available_worker_names(
    const std::vector<std::filesystem::path>& workspace_roots) const {
    const auto profiles = effective_profiles(workspace_roots);
    if (profiles.empty()) return "none";

    std::string names;
    for (std::size_t i = 0; i < profiles.size(); ++i) {
        if (i > 0) names += ", ";
        names += profiles[i].name;
    }
    return names;
}

bool SubagentOrchestrator::task_is_read_only(
    std::string_view json_args,
    std::string_view parent_mode,
    const std::vector<std::filesystem::path>& workspace_roots) const {
    simdjson::dom::parser parser;
    simdjson::padded_string padded{std::string(json_args)};
    simdjson::dom::object object;
    if (parser.parse(padded).get(object) != simdjson::SUCCESS) return false;

    std::string_view worker;
    if (object["subagent_type"].get(worker) != simdjson::SUCCESS) return false;
    const auto profile = find_profile(worker, workspace_roots);
    if (!profile) return false;

    return runs_read_only(*profile, build_tools_for_profile(*profile, parent_mode));
}

std::expected<SubagentOrchestrator::ExecutionPlan, std::string>
SubagentOrchestrator::build_execution_plan(const ExecutionRequest& request) const {
    const std::optional<Profile> profile = find_profile(
        request.worker.empty() ? "general" : request.worker,
        request.workspace_roots);
    if (!profile) {
        return std::unexpected(std::format(
            "Unknown {} '{}'. Available workers: {}.",
            request.worker_label.empty() ? "worker" : request.worker_label,
            request.worker.empty() ? "general" : std::string(request.worker),
            available_worker_names(request.workspace_roots)));
    }

    const core::config::AppConfig& config = app_config_ != nullptr
        ? *app_config_
        : core::config::ConfigManager::get_instance().get_config();

    std::string provider_name = request.provider_override.empty()
        ? profile->provider_override
        : std::string(request.provider_override);
    if (provider_name.empty() && !request.prefer_inherited_provider) {
        provider_name = config.default_provider;
    }
    if (provider_name.empty() && !request.prefer_inherited_provider && !config.providers.empty()) {
        provider_name = config.providers.begin()->first;
    }
    if (provider_name.empty() && request.prefer_inherited_provider) {
        provider_name = std::string(request.inherited_provider_name);
    }

    std::shared_ptr<core::llm::LLMProvider> resolved_provider = request.inherited_provider;
    const bool reuses_inherited = resolved_provider && provider_name == request.inherited_provider_name;
    if (!provider_name.empty() && !reuses_inherited) {
        try {
            resolved_provider =
                core::llm::ProviderManager::get_instance().get_provider(provider_name);
        } catch (const std::exception&) {
            const auto it = config.providers.find(provider_name);
            if (it == config.providers.end()) {
                return std::unexpected(std::format("Unknown provider '{}'.", provider_name));
            }
            resolved_provider = core::llm::ProviderFactory::create_provider(provider_name, it->second);
            if (!resolved_provider) {
                return std::unexpected(std::format(
                    "Provider '{}' is configured but unavailable.",
                    provider_name));
            }
            core::llm::ProviderManager::get_instance().register_provider(provider_name, resolved_provider);
        }
    }

    if (!resolved_provider) {
        return std::unexpected("No provider is available for delegated task execution.");
    }

    std::string model_name = request.model_override.empty()
        ? profile->model_override
        : std::string(request.model_override);
    if (model_name.empty() && request.prefer_inherited_provider) {
        model_name = std::string(request.inherited_model);
    }
    if (model_name.empty()) {
        if (!provider_name.empty()) {
            if (const auto it = config.providers.find(provider_name); it != config.providers.end()) {
                model_name = it->second.model;
            }
        }
    }
    if (model_name.empty()) {
        model_name = std::string(request.inherited_model);
    }

    auto tools = build_tools_for_profile(*profile, request.parent_mode);
    const bool read_only = runs_read_only(*profile, tools);
    const bool read_only_shell = profile->read_only_tools
        && std::ranges::any_of(tools, is_terminal_tool);
    const int max_steps = sanitize_max_steps(
        request.max_steps_override.value_or(0),
        profile->max_steps);
    std::vector<std::string> supplemental;
    const std::string review_tool{core::tools::names::kReview};
    if (!profile->use_allow_list || profile->allowed_tools.contains(review_tool)) {
        supplemental.push_back(review_tool);
    }

    return ExecutionPlan{
        .worker_name = profile->name,
        .worker_description = profile->description,
        .worker_prompt = profile->prompt,
        .provider = std::move(resolved_provider),
        .provider_name = std::move(provider_name),
        .model_name = std::move(model_name),
        .response_format = profile->response_format,
        .tools = std::move(tools),
        .allow_task_tool = profile->allow_task_tool,
        .read_only = read_only,
        .read_only_shell = read_only_shell,
        .max_steps = max_steps,
        .supplemental_tools = std::move(supplemental),
        .isolated = profile->isolated,
    };
}

void SubagentOrchestrator::apply_config_overrides(
    std::vector<Profile>& profiles,
    const core::config::AppConfig& app_config) {
    for (const auto& [raw_name, config_profile] : app_config.subagents) {
        const std::string name = normalize_agent_name(raw_name);
        if (name.empty()) continue;
        // `reader` is the isolated evidence worker for `read.question`. It is not
        // a spawnable task profile; only provider/model/enabled are consumed.
        // Say so once, because a pre-existing profile of that name stops being
        // delegatable and silence would look like a lost configuration.
        if (name == core::tools::read::kReaderProfile) {
            static std::once_flag announced;
            std::call_once(announced, [] {
                core::logging::info(
                    "subagents.reader is reserved for the read tool's evidence "
                    "worker and is not spawnable with task/delegate_task");
            });
            continue;
        }

        auto it = std::find_if(
            profiles.begin(),
            profiles.end(),
            [&](const Profile& profile) { return profile.name == name; });

        if (it == profiles.end()) {
            profiles.push_back(Profile{
                .name = name,
                .description = std::format("Custom subagent profile '{}'.", name),
                .prompt = std::format(
                    "You are Filo's @{} worker subagent.\n"
                    "Complete delegated tasks autonomously.\n"
                    "Final summary: status line first, then observed evidence — files, commands, "
                    "file:line findings.",
                    name),
                .provider_override = "",
                .model_override = "",
                .response_format = std::nullopt,
                .allowed_tools = {},
                .use_allow_list = false,
                .allow_task_tool = false,
                .max_steps = 8,
                .enabled = true,
            });
            it = std::prev(profiles.end());
        }

        if (!config_profile.description.empty()) it->description = config_profile.description;
        if (!config_profile.prompt.empty()) it->prompt = config_profile.prompt;
        if (!config_profile.provider.empty()) it->provider_override = config_profile.provider;
        if (!config_profile.model.empty()) it->model_override = config_profile.model;
        if (config_profile.response_format.has_value()) {
            it->response_format = config_profile.response_format;
        }

        if (config_profile.allowed_tools.has_value()) {
            it->allowed_tools.clear();
            for (const auto& tool_name : *config_profile.allowed_tools) {
                if (!tool_name.empty()) it->allowed_tools.insert(tool_name);
            }
        }

        if (config_profile.use_allow_list.has_value()) {
            it->use_allow_list = *config_profile.use_allow_list;
        }
        if (config_profile.allow_task_tool.has_value()) {
            it->allow_task_tool = *config_profile.allow_task_tool;
        }
        if (config_profile.max_steps.has_value()) {
            it->max_steps = sanitize_max_steps(*config_profile.max_steps, it->max_steps);
        }
        if (config_profile.enabled.has_value()) {
            it->enabled = *config_profile.enabled;
        }
        if (config_profile.isolated.has_value()) {
            it->isolated = *config_profile.isolated;
        }
        if (config_profile.read_only.has_value()) {
            it->read_only_tools = *config_profile.read_only;
        }
    }

    std::erase_if(profiles, [](const Profile& profile) {
        return !profile.enabled;
    });
}

std::string SubagentOrchestrator::build_task_description(
    const std::vector<std::filesystem::path>& workspace_roots) const {
    const std::vector<Profile> profiles_snapshot = effective_profiles(workspace_roots);

    std::string description =
        "Delegates a well-scoped, self-contained subtask to a subagent. Unlike an async "
        "spawn, this call blocks until the subagent finishes and returns its result. To run "
        "subagents in parallel, put all of their task calls in the SAME response; task calls "
        "issued one per response run one after another. Delegation therefore saves time "
        "only for independent subtasks issued together, or keeps bulky side work out of "
        "your context.\n"
        "Do the critical path yourself: never delegate the step your next action "
        "depends on, work that is tightly coupled to what you are doing, or the core "
        "of what the user asked for. Do not redo a subagent's work; verify only "
        "surprising or load-bearing claims.\n\n"
        "Available subagent execution profiles:\n";

    if (profiles_snapshot.empty()) {
        description += "- none configured\n";
    } else {
        for (const auto& profile : profiles_snapshot) {
            const std::string_view provider = profile.provider_override.empty()
                ? "inherit parent"
                : std::string_view(profile.provider_override);
            const std::string_view model = profile.model_override.empty()
                ? "inherit parent"
                : std::string_view(profile.model_override);
            description += std::format(
                "- {}: {} [provider: {}; model: {}; model steps: {}{}{}{}]\n",
                profile.name,
                profile.description,
                provider,
                model,
                profile.max_steps,
                profile.read_only_tools ? "; read-only, shell limited to commands that change nothing" : "",
                profile.allow_task_tool ? "; may delegate to subagents" : "",
                profile.isolated ? "; fresh context without project instructions or memory" : "");
        }
    }

    description +=
        "\nWhen finished, the subagent returns a concise final result to this conversation. "
        "For a new task, omit task_id. To resume the same delegated thread later, pass "
        "the exact task_id returned by its earlier successful call; never invent one. "
        "Resume with this same task tool and the same subagent_type, prompt with follow-up "
        "instructions, and optionally a larger max_steps (up to 64). A partial status means "
        "the run used its budget; its findings and history are preserved. Do not pass task_id "
        "to delegate_task: that tool only resumes its own work_id values. "
        "A new subagent inherits this conversation (fork_turns defaults to `all`), so the "
        "prompt can focus on the task: state the concrete output you need next. "
        "Select the matching profile with subagent_type. A profile's provider and model are "
        "fixed by user configuration and cannot be overridden in a task prompt. "
        "Task calls in one response run concurrently, writers included; all subagents "
        "share this checkout, so give parallel writing tasks disjoint sets of files and "
        "have each list the files it changed.";

    return description;
}

std::vector<core::llm::Message> SubagentOrchestrator::fork_parent_history(
    const std::vector<core::llm::Message>& history) {
    std::vector<core::llm::Message> forked;
    forked.reserve(history.size());
    for (const auto& message : history) {
        if (message.role == "user") {
            core::llm::Message copy{.role = "user", .content = message.content};
            copy.content_parts = message.content_parts;
            copy.input_text = message.input_text;
            copy.synthetic = message.synthetic;
            forked.push_back(std::move(copy));
        } else if (message.role == "assistant" && message.tool_calls.empty()
                   && !core::utils::str::trim_ascii_copy(message.content).empty()) {
            forked.push_back(core::llm::Message{.role = "assistant", .content = message.content});
        }
    }
    return forked;
}

std::string SubagentOrchestrator::create_task_id() {
    // Called under sessions_mutex_. IDs must not alias an old conversation
    // after reload/recreation of an orchestrator.
    std::string id;
    do { id = "task_" + core::session::SessionStore::generate_id(); }
    while (sessions_.contains(id));
    return id;
}

std::shared_ptr<SubagentOrchestrator::TaskSession> SubagentOrchestrator::get_or_create_session(
    const std::string& description,
    const Profile& profile,
    std::string_view requested_task_id,
    std::string_view initial_model,
    std::string& error_out) {

    std::lock_guard lock(sessions_mutex_);

    // Some providers synthesize a UUID for optional identifier fields even on
    // the first call. Such a value cannot name a Filo session: our IDs live in
    // the `task_XXXXXXXX` namespace. Treat foreign identifiers like an omitted
    // optional field, while preserving a hard error for stale/mistyped Filo IDs.
    if (!requested_task_id.empty() && !is_filo_task_id(requested_task_id)
        && !is_provider_generated_uuid(requested_task_id)) {
        error_out = requested_task_id.starts_with("work_")
            ? "This is a persistent work_id. Resume it with delegate_task, action=resume and work_id."
            : "Invalid task_id. Use the exact task_id returned by task; omit it when starting new work.";
        return nullptr;
    }
    if (is_filo_task_id(requested_task_id)) {
        const std::string requested(requested_task_id);
        const auto it = sessions_.find(requested);
        if (it == sessions_.end()) {
            error_out = std::format("Task session '{}' was not found.", requested);
            return nullptr;
        }
        if (it->second->profile_name != profile.name) {
            error_out = std::format(
                "Task '{}' belongs to subagent_type '{}', not '{}'.",
                requested,
                it->second->profile_name,
                profile.name);
            return nullptr;
        }
        return it->second;
    }

    auto session = std::make_shared<TaskSession>();
    session->id = create_task_id();
    session->profile_name = profile.name;
    session->description = description;
    session->model = std::string(initial_model);

    sessions_.emplace(session->id, session);
    return session;
}

bool SubagentOrchestrator::runs_read_only(const Profile& profile,
                                          const std::vector<core::llm::Tool>& tools) {
    if (profile.allow_task_tool) return false;
    return profile.read_only_tools
        || std::ranges::all_of(tools, [](const core::llm::Tool& tool) {
               return tool.function.annotations.read_only_hint;
           });
}

std::vector<core::llm::Tool> SubagentOrchestrator::build_tools_for_profile(
    const Profile& profile,
    std::string_view parent_mode) const {

    std::vector<core::llm::Tool> tools = tool_manager_.get_all_tools();

    if (profile.use_allow_list) {
        std::erase_if(tools, [&](const core::llm::Tool& tool) {
            return !profile.allowed_tools.contains(tool.function.name);
        });
    }
    if (profile.read_only_tools) {
        std::erase_if(tools, std::not_fn(read_only_profile_keeps));
    }

    std::erase_if(tools, [&](const core::llm::Tool& tool) {
        return tool.function.name == core::tools::TaskTool::kToolName
            || (!profile.allow_task_tool && tool.function.name == kTaskToolName)
            // The pane belongs to the conversation the user reads; a worker
            // reports to its parent, which decides what to show.
            || tool.function.name == core::tools::names::kOpenDocument;
    });

    const std::string mode = normalize_mode(parent_mode);
    if (mode == "PLAN" || mode == "RESEARCH") {
        std::erase_if(tools, [](const core::llm::Tool& tool) {
            return is_write_destructive_tool(tool.function.name);
        });
    }

    return tools;
}

std::string SubagentOrchestrator::render_success_json(
    const TaskSession& session,
    const ExecutionPlan& plan,
    std::string_view result_text,
    std::string_view status,
    std::string_view stop_reason,
    bool resumable,
    int steps,
    int tool_calls,
    int failed_tool_calls) const {

    const std::string output = std::format(
        "task_id: {}\n\n<task_result>\n{}\n</task_result>",
        session.id,
        result_text);

    core::utils::JsonWriter writer(1024 + result_text.size() + output.size());
    {
        auto _obj = writer.object();
        writer.kv_str("task_id", session.id);
        writer.comma();
        writer.kv_str("status", status);
        writer.comma().kv_str("stop_reason", stop_reason);
        writer.comma().kv_bool("resumable", resumable);
        writer.comma();
        writer.kv_num("max_steps", plan.max_steps);
        if (status == "partial") {
            writer.comma().kv_num("suggested_max_steps", std::min(plan.max_steps * 2, DelegationBudget::kMaxSteps));
            writer.comma();
            writer.kv_str("resume_hint", "Call task with this task_id, the same subagent_type, "
                          "description and a follow-up prompt; optionally increase max_steps (1-64). "
                          "Do not use delegate_task or work_id for this task.");
        }
        writer.comma();
        writer.kv_str("subagent_type", plan.worker_name);
        writer.comma();
        writer.kv_str("description", session.description);
        writer.comma();
        writer.kv_str("result", result_text);
        writer.comma();
        writer.kv_str("output", output);
        writer.comma();
        writer.kv_num("steps", steps);
        writer.comma();
        writer.kv_num("tool_calls", tool_calls);
        writer.comma();
        writer.kv_num("failed_tool_calls", failed_tool_calls);
    }

    return std::move(writer).take();
}

std::string SubagentOrchestrator::render_error_json(std::string_view error) {
    core::utils::JsonWriter writer(256 + error.size());
    {
        auto _obj = writer.object();
        writer.kv_str("error", error);
    }
    return std::move(writer).take();
}

std::string SubagentOrchestrator::normalize_agent_name(std::string_view name) {
    std::string out;
    out.reserve(name.size());

    std::size_t start = 0;
    while (start < name.size() && std::isspace(static_cast<unsigned char>(name[start]))) {
        ++start;
    }
    if (start < name.size() && name[start] == '@') {
        ++start;
    }

    for (std::size_t i = start; i < name.size(); ++i) {
        const unsigned char ch = static_cast<unsigned char>(name[i]);
        if (std::isspace(ch)) break;
        out.push_back(static_cast<char>(std::tolower(ch)));
    }

    return out;
}

std::string SubagentOrchestrator::normalize_mode(std::string_view mode) {
    std::string out;
    out.reserve(mode.size());
    for (const unsigned char ch : mode) {
        if (std::isalpha(ch)) {
            out.push_back(static_cast<char>(std::toupper(ch)));
        }
    }
    if (out.empty()) out = "BUILD";
    return out;
}

} // namespace core::agent
