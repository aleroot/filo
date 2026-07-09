#include "SubagentOrchestrator.hpp"

#include "DelegatedAgentRunner.hpp"
#include "../config/ConfigManager.hpp"
#include "../llm/ProviderFactory.hpp"
#include "../llm/ProviderManager.hpp"
#include "../tools/TaskTool.hpp"
#include "../tools/ToolNames.hpp"
#include "../utils/JsonWriter.hpp"

#include <simdjson.h>

#include <algorithm>
#include <cctype>
#include <format>
#include <mutex>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace core::agent {

namespace {

constexpr int kMinSubagentSteps = 1;
constexpr int kMaxSubagentSteps = 64;

[[nodiscard]] bool is_write_destructive_tool(std::string_view tool_name) {
    return core::tools::names::is_write_destructive_tool(tool_name);
}

[[nodiscard]] int sanitize_max_steps(int value, int fallback) {
    if (value < kMinSubagentSteps) return fallback;
    if (value > kMaxSubagentSteps) return kMaxSubagentSteps;
    return value;
}

[[nodiscard]] DelegatedAgentRunner::PermissionCheck adapt_permission_check(
    const SubagentOrchestrator::RunContext& context) {
    if (!context.permission_check) return {};

    return [permission_check = context.permission_check](
               std::string_view tool_name,
               std::string_view args) {
        return permission_check(
            std::string(tool_name),
            std::string(args));
    };
}

} // namespace

std::vector<std::string> SubagentOrchestrator::ExecutionPlan::allowed_tool_names() const {
    std::vector<std::string> names;
    names.reserve(tools.size() + (allow_task_tool ? 1 : 0));
    for (const auto& tool : tools) {
        names.push_back(tool.function.name);
    }
    if (allow_task_tool) {
        names.push_back(std::string(SubagentOrchestrator::kTaskToolName));
    }
    return names;
}

SubagentOrchestrator::SubagentOrchestrator(core::tools::ToolManager& tool_manager,
                                           const core::config::AppConfig* app_config)
    : tool_manager_(tool_manager)
    , profiles_(make_default_profiles())
    , app_config_(app_config) {
    if (app_config != nullptr) {
        std::lock_guard lock(profiles_mutex_);
        apply_config_overrides_unlocked(*app_config);
    }
}

std::vector<SubagentOrchestrator::Profile> SubagentOrchestrator::make_default_profiles() {
    return {
        Profile{
            .name = "general",
            .description = "General-purpose subagent for complex multi-step tasks and broad research.",
            .prompt =
                "You are Filo's @general worker subagent.\\n"
                "Work independently on delegated tasks and use tools when needed.\\n"
                "Return a concise, actionable final summary for the parent agent.",
            .provider_override = "",
            .model_override = "",
            .allowed_tools = {},
            .use_allow_list = false,
            .allow_task_tool = false,
            .max_steps = 12,
            .enabled = true,
        },
        Profile{
            .name = "explore",
            .description = "Fast read-only codebase explorer for search-heavy investigations.",
            .prompt =
                "You are Filo's @explore worker subagent.\\n"
                "Focus on finding information quickly using read/search/list tools.\\n"
                "Do not edit files. Cite concrete files and findings in your final summary.",
            .provider_override = "",
            .model_override = "",
            .allowed_tools = [] {
                std::unordered_set<std::string> tool_names;
                for (const auto tool_name : core::tools::names::kExploreAllowedTools) {
                    tool_names.emplace(tool_name);
                }
                return tool_names;
            }(),
            .use_allow_list = true,
            .allow_task_tool = false,
            .max_steps = 10,
            .enabled = true,
        },
    };
}

core::llm::Tool SubagentOrchestrator::task_tool_definition() const {
    core::llm::Tool tool;
    tool.type = "function";
    tool.function.name = std::string(kTaskToolName);
    tool.function.description = build_task_description();
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
            .description = "Specialized subagent profile to run (e.g. general, explore).",
            .required = true,
        },
        {
            .name = "task_id",
            .type = "string",
            .description = "Optional prior task_id to resume an existing delegated subagent session.",
            .required = false,
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
    if (obj["task_id"].get(value) == simdjson::SUCCESS) task_id = std::string(value);

    if (description.empty()) {
        return render_error_json("Missing required 'description' for task.");
    }
    if (prompt.empty()) {
        return render_error_json("Missing required 'prompt' for task.");
    }
    if (subagent_type.empty()) {
        return render_error_json("Missing required 'subagent_type' for task.");
    }

    const auto plan = build_execution_plan(ExecutionRequest{
        .worker = subagent_type,
        .worker_label = "subagent_type",
        .parent_mode = context.parent_mode,
        .inherited_provider = provider,
        .inherited_provider_name = context.active_provider_name,
        .inherited_model = context.active_model,
        .prefer_inherited_provider = true,
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

    std::optional<DelegatedAgentRunner::ResumeState> resume_state;
    {
        std::lock_guard session_lock(session->mutex);
        if (session->model.empty()) {
            session->model = plan->model_name;
        }
        if (!session->history.empty() || !session->context_summary.empty()) {
            resume_state = DelegatedAgentRunner::ResumeState{
                .messages = session->history,
                .context_summary = session->context_summary,
                .mode = context.parent_mode.empty() ? "BUILD" : std::string(context.parent_mode),
            };
        }
    }

    const auto result = DelegatedAgentRunner::run({
        .provider = plan->provider,
        .provider_name = plan->provider_name,
        .tool_manager = tool_manager_,
        .session_context = context.session_context,
        .mode = context.parent_mode.empty() ? "BUILD" : std::string(context.parent_mode),
        .model = plan->model_name,
        .allowed_tools = plan->allowed_tool_names(),
        .max_steps = plan->max_steps,
        .prompt = prompt,
        .worker_name = plan->worker_name,
        .worker_description = plan->worker_description,
        .worker_prompt = plan->worker_prompt,
        .task_id = session->id,
        .task_description = description,
        .parent_tool_call_id = context.parent_tool_call_id,
        .resume_state = std::move(resume_state),
        .timeout = std::chrono::minutes(30),
        .permission_check = adapt_permission_check(context),
        .on_event = context.on_subagent_event,
    });
    if (result.timed_out) {
        return render_error_json("Delegated task timed out while waiting for the worker response.");
    }

    std::string final_text = result.streamed_text;
    if (!result.history.empty()) {
        for (auto it = result.history.rbegin(); it != result.history.rend(); ++it) {
            if (it->role == "assistant" && !it->content.empty()) {
                final_text = it->content;
                break;
            }
        }
    }

    {
        std::lock_guard session_lock(session->mutex);
        session->model = plan->model_name;
        session->history = result.history;
        session->context_summary = result.context_summary;
    }

    return render_success_json(
        *session,
        *plan,
        final_text,
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
        profiles_ = make_default_profiles();
        apply_config_overrides_unlocked(app_config);
    }
    clear_sessions();
}

std::optional<SubagentOrchestrator::Profile> SubagentOrchestrator::find_profile(
    std::string_view name) const {
    const std::string normalized = normalize_agent_name(name);
    std::lock_guard lock(profiles_mutex_);
    for (const auto& profile : profiles_) {
        if (profile.name == normalized) {
            return profile;
        }
    }
    return std::nullopt;
}

std::string SubagentOrchestrator::available_worker_names() const {
    std::lock_guard lock(profiles_mutex_);
    if (profiles_.empty()) return "none";

    std::string names;
    for (std::size_t i = 0; i < profiles_.size(); ++i) {
        if (i > 0) names += ", ";
        names += profiles_[i].name;
    }
    return names;
}

std::expected<SubagentOrchestrator::ExecutionPlan, std::string>
SubagentOrchestrator::build_execution_plan(const ExecutionRequest& request) const {
    const std::optional<Profile> profile = find_profile(request.worker.empty() ? "general" : request.worker);
    if (!profile) {
        return std::unexpected(std::format(
            "Unknown {} '{}'. Available workers: {}.",
            request.worker_label.empty() ? "worker" : request.worker_label,
            request.worker.empty() ? "general" : std::string(request.worker),
            available_worker_names()));
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
    if (!provider_name.empty()) {
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
    const int max_steps = sanitize_max_steps(
        request.max_steps_override.value_or(0),
        profile->max_steps);

    return ExecutionPlan{
        .worker_name = profile->name,
        .worker_description = profile->description,
        .worker_prompt = profile->prompt,
        .provider = std::move(resolved_provider),
        .provider_name = std::move(provider_name),
        .model_name = std::move(model_name),
        .tools = std::move(tools),
        .allow_task_tool = profile->allow_task_tool,
        .max_steps = max_steps,
    };
}

void SubagentOrchestrator::apply_config_overrides_unlocked(
    const core::config::AppConfig& app_config) {
    for (const auto& [raw_name, config_profile] : app_config.subagents) {
        const std::string name = normalize_agent_name(raw_name);
        if (name.empty()) continue;

        auto it = std::find_if(
            profiles_.begin(),
            profiles_.end(),
            [&](const Profile& profile) { return profile.name == name; });

        if (it == profiles_.end()) {
            profiles_.push_back(Profile{
                .name = name,
                .description = std::format("Custom subagent profile '{}'.", name),
                .prompt = std::format(
                    "You are Filo's @{} worker subagent.\\n"
                    "Complete delegated tasks autonomously and return a concise summary.",
                    name),
                .provider_override = "",
                .model_override = "",
                .allowed_tools = {},
                .use_allow_list = false,
                .allow_task_tool = false,
                .max_steps = 8,
                .enabled = true,
            });
            it = std::prev(profiles_.end());
        }

        if (!config_profile.description.empty()) it->description = config_profile.description;
        if (!config_profile.prompt.empty()) it->prompt = config_profile.prompt;
        if (!config_profile.provider.empty()) it->provider_override = config_profile.provider;
        if (!config_profile.model.empty()) it->model_override = config_profile.model;

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
    }

    std::erase_if(profiles_, [](const Profile& profile) {
        return !profile.enabled;
    });
}

std::string SubagentOrchestrator::build_task_description() const {
    std::vector<Profile> profiles_snapshot;
    {
        std::lock_guard lock(profiles_mutex_);
        profiles_snapshot = profiles_;
    }

    std::string description =
        "Delegates work to a specialized background subagent. "
        "Use this for complex multi-step research, broad file/code exploration, "
        "or splitting ambiguous work while you stay focused on the primary user goal.\\n\\n"
        "Available subagent_type values:\\n";

    if (profiles_snapshot.empty()) {
        description += "- none configured\\n";
    } else {
        for (const auto& profile : profiles_snapshot) {
            description += std::format("- {}: {}\\n", profile.name, profile.description);
        }
    }

    description +=
        "\\nWhen finished, the subagent returns a concise final result to this conversation. "
        "Use task_id to resume the same delegated thread later.";

    return description;
}

std::string SubagentOrchestrator::create_task_id() {
    const uint64_t id = next_task_id_.fetch_add(1, std::memory_order_relaxed);
    return std::format("task_{:08x}", id);
}

std::shared_ptr<SubagentOrchestrator::TaskSession> SubagentOrchestrator::get_or_create_session(
    const std::string& description,
    const Profile& profile,
    std::string_view requested_task_id,
    std::string_view initial_model,
    std::string& error_out) {

    std::lock_guard lock(sessions_mutex_);

    if (!requested_task_id.empty()) {
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

std::vector<core::llm::Tool> SubagentOrchestrator::build_tools_for_profile(
    const Profile& profile,
    std::string_view parent_mode) const {

    std::vector<core::llm::Tool> tools = tool_manager_.get_all_tools();

    if (profile.use_allow_list) {
        std::erase_if(tools, [&](const core::llm::Tool& tool) {
            return !profile.allowed_tools.contains(tool.function.name);
        });
    }

    if (!profile.allow_task_tool) {
        std::erase_if(tools, [](const core::llm::Tool& tool) {
            return tool.function.name == kTaskToolName
                || tool.function.name == core::tools::TaskTool::kToolName;
        });
    }

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
    int steps,
    int tool_calls,
    int failed_tool_calls) const {

    const std::string output = std::format(
        "task_id: {}\\n\\n<task_result>\\n{}\\n</task_result>",
        session.id,
        result_text);

    core::utils::JsonWriter writer(1024 + result_text.size() + output.size());
    {
        auto _obj = writer.object();
        writer.kv_str("task_id", session.id);
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
