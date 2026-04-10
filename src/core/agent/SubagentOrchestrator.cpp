#include "SubagentOrchestrator.hpp"

#include "PermissionGate.hpp"
#include "ToolOutputHistory.hpp"
#include "../budget/BudgetTracker.hpp"
#include "../config/ConfigManager.hpp"
#include "../llm/ProviderManager.hpp"
#include "../session/SessionStats.hpp"
#include "../utils/JsonWriter.hpp"

#include <simdjson.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <format>
#include <future>
#include <mutex>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

namespace core::agent {

namespace {

constexpr int kLoopBreakerThreshold = 3;
constexpr auto kModelStepTimeout = std::chrono::minutes(10);
constexpr int kMinSubagentSteps = 1;
constexpr int kMaxSubagentSteps = 64;

[[nodiscard]] bool is_write_destructive_tool(std::string_view tool_name) {
    return tool_name == "apply_patch"
        || tool_name == "write_file"
        || tool_name == "replace"
        || tool_name == "replace_in_file"
        || tool_name == "delete_file"
        || tool_name == "move_file";
}

[[nodiscard]] std::string normalize_ascii_lower(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char ch : value) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

[[nodiscard]] int sanitize_max_steps(int value, int fallback) {
    if (value < kMinSubagentSteps) return fallback;
    if (value > kMaxSubagentSteps) return kMaxSubagentSteps;
    return value;
}

} // namespace

SubagentOrchestrator::SubagentOrchestrator(core::tools::ToolManager& tool_manager,
                                           const core::config::AppConfig* app_config)
    : tool_manager_(tool_manager)
    , profiles_(make_default_profiles()) {
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
            .allowed_tools = {
                "read_file",
                "file_search",
                "grep_search",
                "list_directory",
                "get_current_time",
            },
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

    const std::optional<Profile> profile = find_profile(subagent_type);
    if (!profile) {
        return render_error_json(std::format(
            "Unknown subagent_type '{}'. Available: {}.",
            subagent_type,
            available_profile_names()));
    }

    std::shared_ptr<core::llm::LLMProvider> delegated_provider = provider;
    if (!profile->provider_override.empty()) {
        try {
            delegated_provider =
                core::llm::ProviderManager::get_instance().get_provider(profile->provider_override);
        } catch (const std::exception& e) {
            return render_error_json(std::format(
                "Subagent '{}' is configured with provider '{}' but it is unavailable: {}",
                profile->name,
                profile->provider_override,
                e.what()));
        }
        if (!delegated_provider) {
            return render_error_json(std::format(
                "Subagent '{}' could not resolve provider '{}'.",
                profile->name,
                profile->provider_override));
        }
    }
    if (!delegated_provider) {
        return render_error_json("No LLM provider is available for delegated task execution.");
    }

    const std::string initial_model = !profile->model_override.empty()
        ? profile->model_override
        : (profile->provider_override.empty() ? context.active_model : std::string{});

    std::string session_error;
    auto session = get_or_create_session(
        description,
        *profile,
        task_id,
        context.parent_mode,
        initial_model,
        session_error);
    if (!session) {
        return render_error_json(session_error.empty() ? "Failed to create task session." : session_error);
    }

    std::lock_guard session_lock(session->mutex);

    if (session->history.empty() || session->history.front().role != "system") {
        session->history.insert(
            session->history.begin(),
            core::llm::Message{
                .role = "system",
                .content = build_system_prompt(*profile, context.parent_mode),
                .name = "",
                .tool_call_id = "",
                .tool_calls = {},
            });
    }
    if (session->model.empty()) {
        session->model = initial_model;
    }

    session->history.push_back(
        core::llm::Message{
            .role = "user",
            .content = prompt,
            .name = "",
            .tool_call_id = "",
            .tool_calls = {},
        });

    LoopResult result = run_task_loop(*session, *profile, delegated_provider, context);
    if (!result.ok) {
        const std::string error = result.error.empty()
            ? "Delegated task failed for an unknown reason."
            : result.error;
        return render_error_json(error);
    }

    return render_success_json(*session, *profile, result);
}

void SubagentOrchestrator::clear_sessions() {
    std::lock_guard lock(sessions_mutex_);
    sessions_.clear();
}

void SubagentOrchestrator::reload_profiles(const core::config::AppConfig& app_config) {
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

std::string SubagentOrchestrator::available_profile_names() const {
    std::lock_guard lock(profiles_mutex_);
    if (profiles_.empty()) return "none";

    std::string names;
    for (std::size_t i = 0; i < profiles_.size(); ++i) {
        if (i > 0) names += ", ";
        names += profiles_[i].name;
    }
    return names;
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
    std::string_view parent_mode,
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
    session->history.push_back(core::llm::Message{
        .role = "system",
        .content = build_system_prompt(profile, parent_mode),
        .name = "",
        .tool_call_id = "",
        .tool_calls = {},
    });

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
            return tool.function.name == kTaskToolName;
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

SubagentOrchestrator::StepResult SubagentOrchestrator::run_model_step(
    TaskSession& session,
    const Profile& profile,
    const std::shared_ptr<core::llm::LLMProvider>& provider,
    std::string_view parent_mode) {
    StepResult step;

    core::llm::ChatRequest request;
    request.messages = session.history;
    request.model = session.model;
    if (provider->capabilities().supports_tool_calls) {
        request.tools = build_tools_for_profile(profile, parent_mode);
    }

    struct StreamState {
        std::mutex done_mutex;
        std::condition_variable done_cv;
        bool done = false;

        std::mutex accum_mutex;
        std::vector<core::llm::ToolCall> tool_calls_accum;
        std::string assistant_text;
        std::string reasoning_content_accum;
    };
    auto state = std::make_shared<StreamState>();

    try {
        provider->stream_response(
            request,
            [state](const core::llm::StreamChunk& chunk) {
                {
                    std::lock_guard lock(state->accum_mutex);

                    if (!chunk.content.empty()) {
                        state->assistant_text += chunk.content;
                    }
                    if (!chunk.reasoning_content.empty()) {
                        state->reasoning_content_accum += chunk.reasoning_content;
                    }

                    for (const auto& t : chunk.tools) {
                        bool found = false;
                        for (auto& acc : state->tool_calls_accum) {
                            const bool same = (t.index != -1 && acc.index == t.index)
                                           || (t.index == -1 && !t.id.empty() && acc.id == t.id)
                                           || (t.index == -1 && t.id.empty()
                                               && state->tool_calls_accum.size() == 1);
                            if (same) {
                                if (!t.id.empty()) acc.id = t.id;
                                if (!t.type.empty()) acc.type = t.type;
                                if (!t.function.name.empty()) acc.function.name = t.function.name;
                                acc.function.arguments += t.function.arguments;
                                found = true;
                                break;
                            }
                        }
                        if (!found) state->tool_calls_accum.push_back(t);
                    }
                }

                if (chunk.is_final) {
                    // Record API call outcome (is_error is true for HTTP 4XX/5XX or connection errors)
                    core::session::SessionStats::get_instance().record_api_call(!chunk.is_error);
                    {
                        std::lock_guard lock(state->done_mutex);
                        state->done = true;
                    }
                    state->done_cv.notify_one();
                }
            });
    } catch (const std::exception& e) {
        core::session::SessionStats::get_instance().record_api_call(false);
        step.error = std::format("Subagent model call failed: {}", e.what());
        return step;
    }

    {
        std::unique_lock lock(state->done_mutex);
        if (!state->done_cv.wait_for(lock, kModelStepTimeout, [&]() { return state->done; })) {
            step.error = "Subagent model call timed out waiting for completion.";
            return step;
        }
    }

    {
        std::lock_guard lock(state->accum_mutex);
        step.text = state->assistant_text;
        step.tool_calls = state->tool_calls_accum;
        step.reasoning_content = state->reasoning_content_accum;
    }

    {
        auto usage = provider->get_last_usage();
        std::string model = provider->get_last_model();
        const bool should_estimate_cost = provider->should_estimate_cost();
        if (model.empty()) model = request.model;
        core::budget::BudgetTracker::get_instance().record(usage, model, should_estimate_cost);
        core::session::SessionStats::get_instance().record_turn(
            model, usage, should_estimate_cost);
    }

    session.history.push_back(core::llm::Message{
        .role = "assistant",
        .content = step.text,
        .name = "",
        .tool_call_id = "",
        .tool_calls = step.tool_calls,
        .reasoning_content = step.reasoning_content,
    });

    step.ok = true;
    return step;
}

SubagentOrchestrator::LoopResult SubagentOrchestrator::run_task_loop(
    TaskSession& session,
    const Profile& profile,
    const std::shared_ptr<core::llm::LLMProvider>& provider,
    const RunContext& context) {
    LoopResult loop;

    for (int step_idx = 0; step_idx < profile.max_steps; ++step_idx) {
        StepResult model_step = run_model_step(session, profile, provider, context.parent_mode);
        if (!model_step.ok) {
            loop.error = model_step.error;
            return loop;
        }

        ++loop.steps;

        if (model_step.tool_calls.empty()) {
            loop.ok = true;
            loop.text = model_step.text;
            return loop;
        }

        std::vector<bool> approved(model_step.tool_calls.size(), true);
        std::vector<std::string> denied_payloads(model_step.tool_calls.size());
        bool denied_any = false;

        for (std::size_t i = 0; i < model_step.tool_calls.size(); ++i) {
            const auto& tc = model_step.tool_calls[i];

            if (denied_any) {
                approved[i] = false;
                denied_payloads[i] =
                    R"({"error":"Tool call skipped after a previous denial in this delegated task step."})";
                continue;
            }

            if (tc.function.name == kTaskToolName && !profile.allow_task_tool) {
                approved[i] = false;
                denied_any = true;
                denied_payloads[i] =
                    R"({"error":"Nested task delegation is disabled for this subagent profile."})";
                continue;
            }

            const bool requires_permission = needs_permission(tc.function.name);
            if (!requires_permission || !context.permission_check) {
                approved[i] = true;
                continue;
            }

            approved[i] = context.permission_check(tc.function.name, tc.function.arguments);
            if (!approved[i]) {
                denied_any = true;
                denied_payloads[i] = R"({"error":"Tool call denied by user."})";
            }
        }

        std::vector<std::future<core::llm::Message>> futures;
        futures.reserve(model_step.tool_calls.size());

        for (std::size_t i = 0; i < model_step.tool_calls.size(); ++i) {
            const auto tc = model_step.tool_calls[i];

            if (!approved[i]) {
                const std::string payload = denied_payloads[i].empty()
                    ? R"({"error":"Tool call denied."})"
                    : denied_payloads[i];
                futures.push_back(std::async(std::launch::deferred, [tc, payload]() {
                    return core::llm::Message{
                        .role = "tool",
                        .content = payload,
                        .name = tc.function.name,
                        .tool_call_id = tc.id,
                        .tool_calls = {},
                    };
                }));
                continue;
            }

            futures.push_back(std::async(
                std::launch::async,
                [this, tc, &context]() {
                    std::string output = tool_manager_.execute_tool(
                        tc.function.name,
                        tc.function.arguments,
                        context.session_context);
                    output = tool_output_history::clamp_for_history(tc.function.name, output);
                    return core::llm::Message{
                        .role = "tool",
                        .content = output,
                        .name = tc.function.name,
                        .tool_call_id = tc.id,
                        .tool_calls = {},
                    };
                }));
        }

        int failures_in_step = 0;
        for (auto& future : futures) {
            auto tool_message = future.get();
            const bool ok = !is_tool_error_payload(tool_message.content);
            if (!ok) ++failures_in_step;

            ++loop.tool_calls;
            if (!ok) ++loop.failed_tool_calls;
            core::session::SessionStats::get_instance().record_tool_call(ok);

            session.history.push_back(std::move(tool_message));
        }

        const bool all_failed = !futures.empty() && failures_in_step == static_cast<int>(futures.size());
        if (all_failed) {
            ++session.consecutive_failure_rounds;
        } else {
            session.consecutive_failure_rounds = 0;
        }

        if (session.consecutive_failure_rounds >= kLoopBreakerThreshold) {
            loop.error = std::format(
                "Subagent '{}' paused after {} consecutive fully-failed tool rounds.",
                profile.name,
                kLoopBreakerThreshold);
            session.consecutive_failure_rounds = 0;
            return loop;
        }

        if (denied_any) {
            loop.ok = true;
            if (!model_step.text.empty()) {
                loop.text = model_step.text + "\\n\\nPaused because a required tool call was denied.";
            } else {
                loop.text = "Paused because a required tool call was denied.";
            }
            return loop;
        }
    }

    loop.ok = true;
    loop.text = std::format(
        "Subagent '{}' reached its step limit ({}). Returning the best available intermediate result.",
        profile.name,
        profile.max_steps);
    return loop;
}

std::string SubagentOrchestrator::render_success_json(
    const TaskSession& session,
    const Profile& profile,
    const LoopResult& result) const {

    const std::string output = std::format(
        "task_id: {}\\n\\n<task_result>\\n{}\\n</task_result>",
        session.id,
        result.text);

    core::utils::JsonWriter writer(1024 + result.text.size() + output.size());
    {
        auto _obj = writer.object();
        writer.kv_str("task_id", session.id);
        writer.comma();
        writer.kv_str("subagent_type", profile.name);
        writer.comma();
        writer.kv_str("description", session.description);
        writer.comma();
        writer.kv_str("result", result.text);
        writer.comma();
        writer.kv_str("output", output);
        writer.comma();
        writer.kv_num("steps", result.steps);
        writer.comma();
        writer.kv_num("tool_calls", result.tool_calls);
        writer.comma();
        writer.kv_num("failed_tool_calls", result.failed_tool_calls);
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

bool SubagentOrchestrator::is_tool_error_payload(std::string_view payload) {
    return payload.find("\"error\"") != std::string_view::npos;
}

std::string SubagentOrchestrator::build_system_prompt(const Profile& profile,
                                                       std::string_view parent_mode) {
    const std::string mode = normalize_ascii_lower(normalize_mode(parent_mode));

    std::string prompt =
        "You are a specialized Filo subagent running in background worker mode.\\n"
        "Your intermediate reasoning and tool logs are not shown to the end user.\\n"
        "Solve the delegated task end-to-end, then provide a concise final result for the parent agent.\\n\\n";

    prompt += "Subagent profile: @" + profile.name + "\\n";
    prompt += profile.prompt;

    if (mode == "plan" || mode == "research") {
        prompt +=
            "\\n\\nParent mode is read-focused (plan/research). "
            "Do not perform file edits or destructive operations.";
    }

    return prompt;
}

} // namespace core::agent
