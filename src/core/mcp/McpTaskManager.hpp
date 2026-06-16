#pragma once

#include "../context/SessionContext.hpp"

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <expected>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::mcp {

class McpTaskManager {
public:
    struct TaskState {
        std::string task_id;
        std::string status;
        std::string status_message;
        std::string created_at;
        std::string last_updated_at;
        std::optional<int64_t> ttl_ms;
        std::optional<int64_t> poll_interval_ms;
    };

    struct ResultPayload {
        bool has_error = false;
        int error_code = -32603;
        std::string error_message;
        std::string result_json;
    };

    struct TaskSnapshot {
        TaskState task;
        std::optional<ResultPayload> result;
    };

    struct CreateResult {
        TaskState task;
        std::string immediate_response;
    };

    enum class CancelError {
        not_found,
        already_terminal,
    };

    static McpTaskManager& get_instance();

    [[nodiscard]] CreateResult create_task_tool_call(
        std::string_view tool_name,
        std::string_view arguments_json,
        const core::context::SessionContext& context,
        std::optional<int64_t> requested_ttl_ms);

    [[nodiscard]] std::optional<TaskSnapshot> get(std::string_view task_id,
                                                  std::string_view session_scope);
    [[nodiscard]] std::vector<TaskState> list(std::string_view session_scope);
    [[nodiscard]] std::expected<TaskState, CancelError> cancel(
        std::string_view task_id,
        std::string_view session_scope);
    [[nodiscard]] std::optional<ResultPayload> await_result(std::string_view task_id,
                                                            std::string_view session_scope);

private:
    struct Entry {
        TaskState state;
        std::chrono::system_clock::time_point created_at_tp{};
        std::string session_scope;
        std::mutex mutex;
        std::condition_variable cv;
        bool finished = false;
        ResultPayload payload;
    };

    struct TaskRecord {
        std::shared_ptr<Entry> entry;
        std::function<void()> request_cancel;
        std::stop_source stop_source;
    };

    McpTaskManager() = default;

    [[nodiscard]] std::string generate_task_id();
    void purge_expired_locked();
    [[nodiscard]] std::shared_ptr<TaskRecord> find_record_locked(std::string_view task_id,
                                                                 std::string_view session_scope);

    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<TaskRecord>> records_;
    std::atomic<uint64_t> next_task_id_{1};
};

} // namespace core::mcp
