#pragma once

#include "Agent.hpp"

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::tools {
class ToolManager;
}

namespace core::agent {

class DelegatedAgentRunner {
public:
    using PermissionCheck = std::function<bool(std::string_view, std::string_view)>;

    struct ResumeState {
        std::vector<core::llm::Message> messages;
        std::string context_summary;
        std::string mode;
    };

    struct Request {
        std::shared_ptr<core::llm::LLMProvider> provider;
        std::string provider_name;
        core::tools::ToolManager& tool_manager;
        core::context::SessionContext session_context;
        std::string mode = "BUILD";
        std::string model;
        std::vector<std::string> allowed_tools;
        int max_steps = 0;
        std::string prompt;
        std::string worker_name;
        std::string worker_description;
        std::string worker_prompt;
        std::optional<ResumeState> resume_state;
        std::chrono::milliseconds timeout = std::chrono::minutes(30);
        std::function<void(const std::shared_ptr<core::agent::Agent>&)> on_agent_ready = {};
        std::function<bool()> cancellation_requested = {};
        PermissionCheck permission_check = {};
    };

    struct Result {
        bool timed_out = false;
        bool cancelled = false;
        std::string streamed_text;
        std::string context_summary;
        std::vector<core::llm::Message> history;
        int steps = 0;
        int tool_calls_total = 0;
        int tool_calls_success = 0;
    };

    [[nodiscard]] static Result run(Request request);
};

} // namespace core::agent
