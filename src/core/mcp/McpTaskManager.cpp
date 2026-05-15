#include "McpTaskManager.hpp"

#include "ToolCallResult.hpp"
#include "../session/SessionStore.hpp"
#include "../task/TaskService.hpp"
#include "../tools/TaskTool.hpp"
#include "../tools/ToolManager.hpp"

#include <algorithm>
#include <chrono>
#include <exception>
#include <format>
#include <thread>

namespace core::mcp {

namespace {

constexpr int64_t kDefaultTtlMs = 60 * 60 * 1000;
constexpr int64_t kMinTtlMs = 1'000;
constexpr int64_t kMaxTtlMs = 24 * 60 * 60 * 1000;
constexpr int64_t kPollIntervalMs = 1'000;

[[nodiscard]] int64_t clamp_ttl(std::optional<int64_t> requested) {
    if (!requested.has_value()) return kDefaultTtlMs;
    return std::clamp(*requested, kMinTtlMs, kMaxTtlMs);
}

[[nodiscard]] bool is_terminal(std::string_view status) {
    return status == "completed" || status == "failed" || status == "cancelled";
}

struct TaskRunner {
    std::function<std::string()> execute;
    std::function<void()> request_cancel;
};

[[nodiscard]] TaskRunner build_runner(std::string_view tool_name,
                                      std::string_view arguments_json,
                                      const core::context::SessionContext& context) {
    const std::string tool_name_copy(tool_name);
    const std::string arguments(arguments_json);
    const auto copied_context = context;

    if (tool_name == core::tools::TaskTool::kToolName) {
        auto control = std::make_shared<core::task::TaskService::ExecutionControl>();
        return TaskRunner{
            .execute = [arguments, copied_context, control]() {
                return core::task::TaskService::get_instance().execute(
                    arguments,
                    copied_context,
                    control);
            },
            .request_cancel = [control]() {
                control->request_cancel();
            },
        };
    }

    return TaskRunner{
        .execute = [tool_name_copy, arguments, copied_context]() {
            return core::tools::ToolManager::get_instance().execute_tool(
                tool_name_copy,
                arguments,
                copied_context);
        },
        .request_cancel = []() {},
    };
}

} // namespace

McpTaskManager& McpTaskManager::get_instance() {
    static McpTaskManager instance;
    return instance;
}

std::string McpTaskManager::generate_task_id() {
    const auto sequence = next_task_id_.fetch_add(1, std::memory_order_relaxed);
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    const auto timestamp = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());

    return std::format("task_{:016x}_{:016x}", timestamp, sequence);
}

void McpTaskManager::purge_expired_locked() {
    const auto now = std::chrono::system_clock::now();
    std::erase_if(records_, [&](const auto& pair) {
        const auto& record = pair.second;
        std::lock_guard entry_lock(record->entry->mutex);
        if (!record->entry->state.ttl_ms.has_value()) return false;
        if (!is_terminal(record->entry->state.status)) return false;
        return now - record->entry->created_at_tp
            > std::chrono::milliseconds(*record->entry->state.ttl_ms);
    });
}

std::shared_ptr<McpTaskManager::TaskRecord> McpTaskManager::find_record_locked(
    std::string_view task_id,
    std::string_view session_scope) {
    const auto it = records_.find(std::string(task_id));
    if (it == records_.end()) return {};
    if (it->second->entry->session_scope != session_scope) return {};
    return it->second;
}

McpTaskManager::CreateResult McpTaskManager::create_task_tool_call(
    std::string_view tool_name,
    std::string_view arguments_json,
    const core::context::SessionContext& context,
    std::optional<int64_t> requested_ttl_ms) {
    auto entry = std::make_shared<Entry>();
    entry->state.task_id = generate_task_id();
    entry->state.status = "working";
    entry->state.status_message =
        "The delegated task has been accepted and is now in progress.";
    entry->state.created_at = core::session::SessionStore::now_iso8601();
    entry->state.last_updated_at = entry->state.created_at;
    entry->state.ttl_ms = clamp_ttl(requested_ttl_ms);
    entry->state.poll_interval_ms = kPollIntervalMs;
    entry->created_at_tp = std::chrono::system_clock::now();
    entry->session_scope = context.session_id;

    auto record = std::make_shared<TaskRecord>();
    record->entry = entry;
    record->stop_source = std::stop_source();

    TaskRunner runner = build_runner(tool_name, arguments_json, context);
    record->request_cancel = std::move(runner.request_cancel);
    std::thread worker(
        [entry,
         stop_token = record->stop_source.get_token(),
         execute = std::move(runner.execute)]() mutable {
            {
                std::lock_guard lock(entry->mutex);
                if (!entry->finished) {
                    if (stop_token.stop_requested()) {
                        entry->state.status = "cancelled";
                        entry->state.status_message = "The task was cancelled before it started.";
                        entry->payload.has_error = true;
                        entry->payload.error_code = -32000;
                        entry->payload.error_message = "Task cancelled before it started.";
                        entry->finished = true;
                    } else {
                        entry->state.status_message = "The delegated task is now in progress.";
                    }
                    entry->state.last_updated_at = core::session::SessionStore::now_iso8601();
                }
            }
            if (stop_token.stop_requested()) {
                entry->cv.notify_all();
                return;
            }

            try {
                const auto payload_json = execute();
                const auto classification = classify_tool_call_payload(payload_json);
                const auto final_result = build_call_tool_result_from_payload(
                    payload_json,
                    entry->state.task_id);

                {
                    std::lock_guard lock(entry->mutex);
                    if (!entry->finished) {
                        entry->payload.has_error = false;
                        entry->payload.result_json = final_result;
                        entry->state.status = classification.is_error ? "failed" : "completed";
                        entry->state.status_message = classification.is_error
                            ? "The delegated task finished with an execution error."
                            : "The delegated task completed successfully.";
                        entry->finished = true;
                    }
                    entry->state.last_updated_at = core::session::SessionStore::now_iso8601();
                }
            } catch (const std::exception& e) {
                std::lock_guard lock(entry->mutex);
                if (!entry->finished) {
                    entry->payload.has_error = true;
                    entry->payload.error_code = -32603;
                    entry->payload.error_message = std::format("Task execution failed: {}", e.what());
                    entry->state.status = "failed";
                    entry->state.status_message = "The delegated task crashed during execution.";
                    entry->finished = true;
                }
                entry->state.last_updated_at = core::session::SessionStore::now_iso8601();
            } catch (...) {
                std::lock_guard lock(entry->mutex);
                if (!entry->finished) {
                    entry->payload.has_error = true;
                    entry->payload.error_code = -32603;
                    entry->payload.error_message =
                        "Task execution failed with an unknown error.";
                    entry->state.status = "failed";
                    entry->state.status_message = "The delegated task crashed during execution.";
                    entry->finished = true;
                }
                entry->state.last_updated_at = core::session::SessionStore::now_iso8601();
            }
            entry->cv.notify_all();
        });
    worker.detach();

    {
        std::lock_guard lock(mutex_);
        purge_expired_locked();
        records_[entry->state.task_id] = record;
    }

    TaskState accepted_state;
    {
        std::lock_guard entry_lock(entry->mutex);
        accepted_state = entry->state;
    }

    return CreateResult{
        .task = std::move(accepted_state),
        .immediate_response =
            "Task accepted. Poll tasks/get for status updates or call tasks/result to await completion.",
    };
}

std::optional<McpTaskManager::TaskState> McpTaskManager::get(std::string_view task_id,
                                                             std::string_view session_scope) {
    std::lock_guard lock(mutex_);
    purge_expired_locked();
    const auto record = find_record_locked(task_id, session_scope);
    if (!record) return std::nullopt;
    std::lock_guard entry_lock(record->entry->mutex);
    return record->entry->state;
}

std::vector<McpTaskManager::TaskState> McpTaskManager::list(std::string_view session_scope) {
    std::vector<TaskState> out;
    std::lock_guard lock(mutex_);
    purge_expired_locked();
    out.reserve(records_.size());
    for (const auto& [task_id, record] : records_) {
        if (record->entry->session_scope != session_scope) continue;
        std::lock_guard entry_lock(record->entry->mutex);
        out.push_back(record->entry->state);
    }
    std::ranges::sort(out, [](const TaskState& lhs, const TaskState& rhs) {
        return lhs.last_updated_at > rhs.last_updated_at;
    });
    return out;
}

std::expected<McpTaskManager::TaskState, McpTaskManager::CancelError> McpTaskManager::cancel(
    std::string_view task_id,
    std::string_view session_scope) {
    std::shared_ptr<TaskRecord> record;
    {
        std::lock_guard lock(mutex_);
        purge_expired_locked();
        record = find_record_locked(task_id, session_scope);
    }
    if (!record) {
        return std::unexpected(CancelError::not_found);
    }

    {
        std::lock_guard lock(record->entry->mutex);
        if (is_terminal(record->entry->state.status)) {
            return std::unexpected(CancelError::already_terminal);
        }
        record->entry->state.status = "cancelled";
        record->entry->state.status_message = "The task was cancelled by request.";
        record->entry->state.last_updated_at = core::session::SessionStore::now_iso8601();
        record->entry->payload.has_error = true;
        record->entry->payload.error_code = -32000;
        record->entry->payload.error_message = "Task cancelled by request.";
        record->entry->finished = true;
    }
    if (record->request_cancel) {
        record->request_cancel();
    }
    record->stop_source.request_stop();
    record->entry->cv.notify_all();

    std::lock_guard lock(record->entry->mutex);
    return record->entry->state;
}

std::optional<McpTaskManager::ResultPayload> McpTaskManager::await_result(
    std::string_view task_id,
    std::string_view session_scope) {
    std::shared_ptr<TaskRecord> record;
    {
        std::lock_guard lock(mutex_);
        purge_expired_locked();
        record = find_record_locked(task_id, session_scope);
    }
    if (!record) return std::nullopt;

    std::unique_lock lock(record->entry->mutex);
    record->entry->cv.wait(lock, [&] { return record->entry->finished; });
    ResultPayload payload = record->entry->payload;
    lock.unlock();

    {
        std::lock_guard manager_lock(mutex_);
        const auto current = find_record_locked(task_id, session_scope);
        if (current == record) {
            records_.erase(std::string(task_id));
        }
    }

    return payload;
}

} // namespace core::mcp
