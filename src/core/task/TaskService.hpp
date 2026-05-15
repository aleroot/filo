#pragma once

#include "../context/SessionContext.hpp"

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace core::agent {
class Agent;
}

namespace core::task {

class TaskService {
public:
    struct ExecutionControl {
        void attach_agent(const std::shared_ptr<core::agent::Agent>& agent);
        void request_cancel();
        [[nodiscard]] bool cancel_requested() const noexcept {
            return cancel_requested_.load(std::memory_order_acquire);
        }

    private:
        std::atomic<bool> cancel_requested_{false};
        mutable std::mutex agent_mutex_;
        std::weak_ptr<core::agent::Agent> active_agent_;
    };

    static TaskService& get_instance();

    [[nodiscard]] std::string execute(
        std::string_view json_args,
        const core::context::SessionContext& context,
        std::shared_ptr<ExecutionControl> control = {});

private:
    TaskService() = default;
};

} // namespace core::task
