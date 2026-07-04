#include "TaskService.hpp"

#include "../agent/DelegatedAgentRunner.hpp"
#include "../agent/SubagentOrchestrator.hpp"
#include "../config/ConfigManager.hpp"
#include "../session/SessionHandoff.hpp"
#include "../session/SessionStore.hpp"
#include "../tools/ToolManager.hpp"
#include "../tools/ToolNames.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/StringUtils.hpp"
#include "WorkStore.hpp"

#include <simdjson.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <initializer_list>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace core::task {

namespace {

struct ParsedRequest {
    std::string action;
    std::string work_id;
    std::string title;
    std::string instructions;
    std::string worker;
    std::string mode;
    std::string provider;
    std::string model;
    std::string status_filter;
    std::string worker_filter;
    std::string cwd;
    int max_steps = 0;
};

struct RunOutcome {
    std::string final_text;
    std::string handoff_summary;
    std::vector<core::llm::Message> history;
    int steps = 0;
    int tool_calls = 0;
    int failed_tool_calls = 0;
    std::vector<std::string> files_touched;
    std::vector<std::string> commands_run;
    bool cancelled = false;
};

[[nodiscard]] std::string normalize_mode(std::string_view input) {
    std::string mode = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_copy(input));
    if (mode.empty()) return "BUILD";
    if (mode == "plan") return "PLAN";
    if (mode == "research") return "RESEARCH";
    if (mode == "debug") return "DEBUG";
    if (mode == "execute") return "EXECUTE";
    return "BUILD";
}

[[nodiscard]] std::string clamp_line(std::string_view input, std::size_t max_chars = 240) {
    std::string out = core::utils::str::trim_ascii_copy(input);
    if (out.size() <= max_chars) return out;
    out.resize(max_chars);
    out += "...";
    return out;
}

void append_unique(std::vector<std::string>& values,
                   std::string value,
                   std::size_t max_items = 16) {
    value = core::utils::str::trim_ascii_copy(value);
    if (value.empty()) return;
    if (std::ranges::find(values, value) != values.end()) return;
    if (values.size() >= max_items) return;
    values.push_back(std::move(value));
}

[[nodiscard]] std::string escape_json(std::string_view value) {
    return core::utils::escape_json_string(std::string(value));
}

[[nodiscard]] std::string error_json(std::string_view message) {
    return std::format(R"({{"error":"{}"}})", escape_json(message));
}

[[nodiscard]] std::optional<ParsedRequest> parse_request(std::string_view json_args,
                                                         std::string& error_out) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded{std::string(json_args)};
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        error_out = "Invalid JSON arguments for task.";
        return std::nullopt;
    }

    simdjson::dom::object obj;
    if (doc.get(obj) != simdjson::SUCCESS) {
        error_out = "Task arguments must be a JSON object.";
        return std::nullopt;
    }

    ParsedRequest parsed;
    std::string_view text;
    if (obj["action"].get(text) == simdjson::SUCCESS) parsed.action = core::utils::str::to_lower_ascii_copy(text);
    if (obj["work_id"].get(text) == simdjson::SUCCESS) parsed.work_id = std::string(text);
    if (obj["title"].get(text) == simdjson::SUCCESS) parsed.title = std::string(text);
    if (obj["instructions"].get(text) == simdjson::SUCCESS) parsed.instructions = std::string(text);
    if (obj["worker"].get(text) == simdjson::SUCCESS) parsed.worker = std::string(text);
    if (obj["mode"].get(text) == simdjson::SUCCESS) parsed.mode = std::string(text);
    if (obj["provider"].get(text) == simdjson::SUCCESS) parsed.provider = std::string(text);
    if (obj["model"].get(text) == simdjson::SUCCESS) parsed.model = std::string(text);
    if (obj["status"].get(text) == simdjson::SUCCESS) parsed.status_filter = core::utils::str::to_lower_ascii_copy(text);
    if (obj["cwd"].get(text) == simdjson::SUCCESS) parsed.cwd = std::string(text);

    int64_t max_steps = 0;
    if (obj["max_steps"].get(max_steps) == simdjson::SUCCESS) {
        parsed.max_steps = static_cast<int>(max_steps);
    }

    if (parsed.action.empty()) {
        error_out = "Missing required 'action' for task.";
        return std::nullopt;
    }
    if (parsed.action != "start" && parsed.action != "resume" && parsed.action != "list") {
        error_out = "Invalid task action. Supported values: start, resume, list.";
        return std::nullopt;
    }
    if (parsed.action == "start") {
        if (core::utils::str::trim_ascii_copy(parsed.title).empty()) {
            error_out = "Missing required 'title' for task start.";
            return std::nullopt;
        }
        if (core::utils::str::trim_ascii_copy(parsed.instructions).empty()) {
            error_out = "Missing required 'instructions' for task start.";
            return std::nullopt;
        }
    }
    if (parsed.action == "resume" && core::utils::str::trim_ascii_copy(parsed.work_id).empty()) {
        error_out = "Missing required 'work_id' for task resume.";
        return std::nullopt;
    }
    if (parsed.action == "resume" && !WorkStore::is_valid_work_id(parsed.work_id)) {
        error_out = "Invalid work_id for task resume.";
        return std::nullopt;
    }
    return parsed;
}

[[nodiscard]] std::string summary_from_text(std::string_view text) {
    const auto trimmed = core::utils::str::trim_ascii_copy(text);
    if (trimmed.empty()) return "No final summary was produced.";
    const auto newline = trimmed.find('\n');
    if (newline == std::string::npos) return clamp_line(trimmed, 220);
    return clamp_line(trimmed.substr(0, newline), 220);
}

[[nodiscard]] std::string resolved_path_string(const core::context::SessionContext& context,
                                               std::string_view raw_path) {
    if (raw_path.empty()) return {};
    const std::filesystem::path path(raw_path);
    if (path.is_absolute()) return path.lexically_normal().string();
    return context.resolve_path(path).string();
}

void capture_apply_patch_paths(std::vector<std::string>& files,
                               const core::context::SessionContext& context,
                               std::string_view patch_text) {
    std::size_t start = 0;
    while (start < patch_text.size()) {
        const auto end = patch_text.find('\n', start);
        std::string_view line = patch_text.substr(
            start,
            end == std::string_view::npos ? patch_text.size() - start : end - start);
        if (line.starts_with("+++ ") || line.starts_with("--- ")) {
            std::string candidate = core::utils::str::trim_ascii_copy(line.substr(4));
            if (candidate != "/dev/null" && !candidate.empty()) {
                if ((candidate.starts_with("a/") || candidate.starts_with("b/")) && candidate.size() > 2) {
                    candidate.erase(0, 2);
                }
                append_unique(files, resolved_path_string(context, candidate));
            }
        }
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
}

[[nodiscard]] RunOutcome summarize_run(const std::vector<core::llm::Message>& history,
                                       std::size_t history_before,
                                       const std::string& streamed_text,
                                       const core::context::SessionContext& context,
                                       bool cancelled) {
    RunOutcome outcome;
    outcome.cancelled = cancelled;
    outcome.history = history;

    std::unordered_map<std::string, core::llm::ToolCall> tool_calls_by_id;
    for (const auto& message : history) {
        if (message.role == "assistant") {
            for (const auto& tool_call : message.tool_calls) {
                if (!tool_call.id.empty()) {
                    tool_calls_by_id[tool_call.id] = tool_call;
                }
            }
        }
    }

    for (const auto& message : history) {
        if (message.role != "tool") continue;

        std::string tool_name = message.name;
        std::string tool_args;
        if (const auto it = tool_calls_by_id.find(message.tool_call_id); it != tool_calls_by_id.end()) {
            tool_name = it->second.function.name;
            tool_args = it->second.function.arguments;
        }

        if (tool_name == core::tools::names::kRunTerminalCommand) {
            append_unique(outcome.commands_run, core::utils::json::first_string_field_or_empty(tool_args, {"command"}), 12);
            continue;
        }

        if (tool_name == core::tools::names::kWriteFile
            || tool_name == core::tools::names::kReplace
            || tool_name == core::tools::names::kSearchReplace
            || tool_name == core::tools::names::kDeleteFile) {
            append_unique(
                outcome.files_touched,
                resolved_path_string(context, core::utils::json::first_string_field_or_empty(tool_args, {"file_path", "path"})),
                24);
            continue;
        }

        if (tool_name == core::tools::names::kMoveFile) {
            append_unique(
                outcome.files_touched,
                resolved_path_string(context, core::utils::json::first_string_field_or_empty(tool_args, {"source", "from"})),
                24);
            append_unique(
                outcome.files_touched,
                resolved_path_string(context, core::utils::json::first_string_field_or_empty(tool_args, {"destination", "to"})),
                24);
            continue;
        }

        if (tool_name == core::tools::names::kApplyPatch) {
            capture_apply_patch_paths(
                outcome.files_touched,
                context,
                core::utils::json::first_string_field_or_empty(tool_args, {"patch"}));
        }
    }

    if (history.size() >= history_before) {
        for (const auto& message : std::span<const core::llm::Message>(
                 history.data() + history_before,
                 history.size() - history_before)) {
            if (message.role == "assistant" && !message.content.empty()) {
                if (!outcome.final_text.empty()) outcome.final_text += "\n";
                outcome.final_text += message.content;
            }
        }
    }
    if (outcome.final_text.empty()) {
        outcome.final_text = streamed_text;
    }

    outcome.handoff_summary = core::session::build_handoff_summary(history, {}, "BUILD");
    return outcome;
}

[[nodiscard]] std::optional<core::context::SessionContext> build_worker_context(
    const core::context::SessionContext& parent_context,
    std::string_view session_id,
    std::string_view requested_cwd,
    std::string& error_out)
{
    auto snapshot = parent_context.effective_workspace();
    if (!requested_cwd.empty()) {
        const auto resolved = parent_context.resolve_path(std::filesystem::path(requested_cwd));
        if (!parent_context.is_path_allowed(resolved)) {
            error_out = std::format(
                "cwd '{}' is outside the allowed workspace.",
                std::string(requested_cwd));
            return std::nullopt;
        }
        std::error_code ec;
        if (!std::filesystem::is_directory(resolved, ec)) {
            error_out = std::format(
                "cwd '{}' does not exist or is not a directory.",
                std::string(requested_cwd));
            return std::nullopt;
        }
        snapshot.primary = resolved;
        snapshot.enforce = true;
    }
    return core::context::make_session_context(
        snapshot,
        parent_context.transport,
        std::string(session_id));
}

[[nodiscard]] std::string build_result_json(const ParsedRequest& request,
                                            const WorkItem& item) {
    core::utils::JsonWriter writer(2048 + item.result.size() + item.summary.size());
    {
        auto root = writer.object();
        writer.kv_str("action", request.action).comma();
        writer.kv_str("work_id", item.work_id).comma();
        writer.kv_str("status", item.status).comma();
        writer.kv_str("title", item.title).comma();
        writer.kv_str("worker", item.worker).comma();
        writer.kv_str("mode", item.mode).comma();
        writer.kv_str("provider", item.provider).comma();
        writer.kv_str("model", item.model).comma();
        writer.kv_str("summary", item.summary).comma();
        writer.kv_str("result", item.result).comma();
        writer.kv_num("steps", item.steps).comma();
        writer.kv_num("tool_calls", item.tool_calls).comma();
        writer.kv_num("failed_tool_calls", item.failed_tool_calls).comma();
        writer.key("files_touched");
        {
            auto files = writer.array();
            bool first = true;
            for (const auto& file : item.files_touched) {
                if (!first) writer.comma();
                first = false;
                writer.str(file);
            }
        }
        writer.comma().key("commands_run");
        {
            auto commands = writer.array();
            bool first = true;
            for (const auto& command : item.commands_run) {
                if (!first) writer.comma();
                first = false;
                writer.str(command);
            }
        }
        writer.comma().kv_str("handoff_summary", item.handoff_summary);
    }
    return std::move(writer).take();
}

[[nodiscard]] std::string build_list_json(const ParsedRequest& request,
                                          const std::vector<WorkInfo>& items) {
    core::utils::JsonWriter writer(2048 + items.size() * 256);
    {
        auto root = writer.object();
        writer.kv_str("action", request.action).comma().key("items");
        {
            auto array = writer.array();
            bool first_item = true;
            for (const auto& item : items) {
                if (!first_item) writer.comma();
                first_item = false;
                auto object = writer.object();
                writer.kv_str("work_id", item.work_id).comma();
                writer.kv_str("title", item.title).comma();
                writer.kv_str("status", item.status).comma();
                writer.kv_str("worker", item.worker).comma();
                writer.kv_str("provider", item.provider).comma();
                writer.kv_str("model", item.model).comma();
                writer.kv_str("working_dir", item.working_dir).comma();
                writer.kv_str("updated_at", item.updated_at);
            }
        }
    }
    return std::move(writer).take();
}

} // namespace

void TaskService::ExecutionControl::attach_agent(
    const std::shared_ptr<core::agent::Agent>& agent) {
    std::lock_guard lock(agent_mutex_);
    active_agent_ = agent;
    if (cancel_requested()) {
        if (auto shared = active_agent_.lock()) {
            shared->request_stop();
        }
    }
}

void TaskService::ExecutionControl::request_cancel() {
    cancel_requested_.store(true, std::memory_order_release);
    std::lock_guard lock(agent_mutex_);
    if (auto agent = active_agent_.lock()) {
        agent->request_stop();
    }
}

TaskService& TaskService::get_instance() {
    static TaskService instance;
    return instance;
}

std::string TaskService::execute(std::string_view json_args,
                                 const core::context::SessionContext& context,
                                 std::shared_ptr<ExecutionControl> control) {
    std::string parse_error;
    const auto request = parse_request(json_args, parse_error);
    if (!request.has_value()) {
        return error_json(parse_error);
    }

    const auto config = core::config::ConfigManager::get_instance().get_config();
    WorkStore work_store(WorkStore::default_work_dir(context));

    if (request->action == "list") {
        auto items = work_store.list();
        if (!request->status_filter.empty()) {
            std::erase_if(items, [&](const WorkInfo& item) {
                return core::utils::str::to_lower_ascii_copy(item.status) != request->status_filter;
            });
        }
        if (!request->worker.empty()) {
            std::erase_if(items, [&](const WorkInfo& item) {
                return core::utils::str::to_lower_ascii_copy(item.worker) != core::utils::str::to_lower_ascii_copy(request->worker);
            });
        }
        if (!request->cwd.empty()) {
            const auto resolved = context.resolve_path(std::filesystem::path(request->cwd)).string();
            std::erase_if(items, [&](const WorkInfo& item) {
                return item.working_dir != resolved;
            });
        }
        return build_list_json(*request, items);
    }

    WorkItem item;
    core::session::SessionStore session_store(core::session::SessionStore::default_sessions_dir());
    std::optional<core::session::SessionData> existing_session;

    if (request->action == "resume") {
        const auto loaded = work_store.load(request->work_id);
        if (!loaded.has_value()) {
            return error_json(std::format("Unknown work_id '{}'.", request->work_id));
        }
        item = *loaded;
        existing_session = session_store.load_by_id(item.session_id);
        if (!existing_session.has_value()) {
            return error_json(std::format(
                "Work item '{}' could not load its saved session '{}'.",
                item.work_id,
                item.session_id));
        }
    } else {
        item.work_id = WorkStore::generate_work_id();
        item.session_id = item.work_id;
        item.title = core::utils::str::trim_ascii_copy(request->title);
        item.created_at = core::session::SessionStore::now_iso8601();
    }

    item.mode = request->mode.empty()
        ? (item.mode.empty() ? (config.default_mode.empty() ? "BUILD" : normalize_mode(config.default_mode))
                             : normalize_mode(item.mode))
        : normalize_mode(request->mode);

    std::string requested_cwd = request->cwd;
    if (requested_cwd.empty()) requested_cwd = item.working_dir;

    std::string context_error;
    auto worker_context = build_worker_context(context, item.work_id, requested_cwd, context_error);
    if (!worker_context.has_value()) {
        return error_json(context_error);
    }
    item.working_dir = worker_context->effective_workspace().primary.string();

    if (control && control->cancel_requested()) {
        item.status = "paused";
        item.updated_at = core::session::SessionStore::now_iso8601();
        item.summary = "Execution cancelled before it started.";
        item.result.clear();
        item.handoff_summary = item.summary;
        std::string save_error;
        (void)work_store.save(item, &save_error);
        return build_result_json(*request, item);
    }

    auto& tool_manager = core::tools::ToolManager::get_instance();
    core::agent::SubagentOrchestrator orchestrator(tool_manager, &config);
    const auto plan = orchestrator.build_execution_plan(core::agent::SubagentOrchestrator::ExecutionRequest{
        .worker = request->worker.empty() ? std::string_view(item.worker) : std::string_view(request->worker),
        .worker_label = "worker",
        .parent_mode = item.mode,
        .inherited_model = item.model,
        .provider_override = request->provider.empty() ? std::string_view(item.provider) : std::string_view(request->provider),
        .model_override = request->model.empty() ? std::string_view(item.model) : std::string_view(request->model),
        .max_steps_override = request->max_steps > 0 ? std::optional<int>{request->max_steps} : std::nullopt,
    });
    if (!plan) {
        return error_json(plan.error());
    }

    item.worker = plan->worker_name;
    item.provider = plan->provider_name;
    item.model = plan->model_name;

    const std::string prompt = request->action == "resume"
        ? (core::utils::str::trim_ascii_copy(request->instructions).empty()
            ? "Continue from the saved state, complete the task if possible, and return a concise architect-facing result."
            : request->instructions)
        : request->instructions;

    const auto run_result = core::agent::DelegatedAgentRunner::run({
        .provider = plan->provider,
        .provider_name = plan->provider_name,
        .tool_manager = tool_manager,
        .session_context = *worker_context,
        .mode = item.mode,
        .model = plan->model_name,
        .allowed_tools = plan->allowed_tool_names(),
        .max_steps = plan->max_steps,
        .prompt = prompt,
        .worker_name = plan->worker_name,
        .worker_description = plan->worker_description,
        .worker_prompt = plan->worker_prompt,
        .resume_state = existing_session.has_value()
            ? std::optional<core::agent::DelegatedAgentRunner::ResumeState>{{
                  .messages = existing_session->messages,
                  .context_summary = existing_session->context_summary,
                  .mode = existing_session->mode,
              }}
            : std::nullopt,
        .timeout = std::chrono::minutes(30),
        .on_agent_ready = control
            ? std::function<void(const std::shared_ptr<core::agent::Agent>&)>{
                  [control](const std::shared_ptr<core::agent::Agent>& agent) {
                      control->attach_agent(agent);
                  }}
            : std::function<void(const std::shared_ptr<core::agent::Agent>&)>{},
        .cancellation_requested = control
            ? std::function<bool()>{[control]() { return control->cancel_requested(); }}
            : std::function<bool()>{},
    });
    if (run_result.timed_out) {
        if (control) {
            control->request_cancel();
        }
        item.status = "failed";
        item.updated_at = core::session::SessionStore::now_iso8601();
        item.summary = "Delegated task timed out while waiting for the worker response.";
        item.result = item.summary;
        item.handoff_summary = item.summary;
        std::string save_error;
        (void)work_store.save(item, &save_error);
        return build_result_json(*request, item);
    }

    const RunOutcome outcome = summarize_run(
        run_result.history,
        existing_session.has_value() ? existing_session->messages.size() : std::size_t{0},
        run_result.streamed_text,
        *worker_context,
        run_result.cancelled);

    core::session::SessionData session_data;
    session_data.session_id = item.session_id;
    session_data.created_at = existing_session.has_value()
        ? existing_session->created_at
        : item.created_at;
    session_data.last_active_at = core::session::SessionStore::now_iso8601();
    session_data.working_dir = item.working_dir;
    session_data.provider = plan->provider_name;
    session_data.model = plan->model_name;
    session_data.mode = item.mode;
    session_data.context_summary = run_result.context_summary;
    session_data.messages = run_result.history;
    session_data.stats.turn_count = run_result.steps;
    session_data.stats.tool_calls_total = run_result.tool_calls_total;
    session_data.stats.tool_calls_success = run_result.tool_calls_success;
    session_data.handoff_summary = core::session::build_handoff_summary(session_data);

    item.updated_at = session_data.last_active_at;
    item.summary = summary_from_text(outcome.final_text);
    item.result = outcome.final_text;
    item.handoff_summary = session_data.handoff_summary;
    item.steps = session_data.stats.turn_count;
    item.tool_calls = session_data.stats.tool_calls_total;
    item.failed_tool_calls =
        session_data.stats.tool_calls_total - session_data.stats.tool_calls_success;
    item.files_touched = outcome.files_touched;
    item.commands_run = outcome.commands_run;
    item.status = outcome.cancelled ? "paused" : "completed";

    if (item.result.empty()) {
        item.result = outcome.cancelled
            ? "Execution was cancelled before the worker produced a final result."
            : "The worker completed without producing final text.";
        item.summary = summary_from_text(item.result);
    }

    std::string save_error;
    if (!session_store.save(session_data, &save_error)) {
        return error_json(save_error.empty()
            ? "Failed to persist delegated task session."
            : save_error);
    }
    if (!work_store.save(item, &save_error)) {
        return error_json(save_error.empty()
            ? "Failed to persist delegated work item."
            : save_error);
    }

    return build_result_json(*request, item);
}

} // namespace core::task
