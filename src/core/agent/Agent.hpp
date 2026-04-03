#pragma once

#include "../llm/ProviderManager.hpp"
#include "../tools/ToolManager.hpp"
#include "SubagentOrchestrator.hpp"
#include "PermissionGate.hpp"
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <mutex>
#include <atomic>
#include <optional>

namespace core::agent {

class Agent : public std::enable_shared_from_this<Agent> {
public:
    struct TurnCallbacks {
        std::function<void()> on_step_begin;
        std::function<void(const core::llm::ToolCall&)> on_tool_start;
        std::function<void(const core::llm::ToolCall&, const core::llm::Message&)> on_tool_finish;
    };

    Agent(std::shared_ptr<core::llm::LLMProvider> provider,
          core::tools::ToolManager& skill_manager);

    // -----------------------------------------------------------------------
    // Cancellation support — stop the current LLM response generation.
    // -----------------------------------------------------------------------

    /// Request cancellation of the current streaming response.
    void request_stop();

    /// Check if a stop has been requested.
    [[nodiscard]] bool is_stop_requested() const;

    /// Clear the stop flag (call before starting a new turn).
    void clear_stop_request();

    void set_mode(const std::string& mode);

    // Set the permission profile (Interactive, Standard, Autonomous).
    void set_permission_profile(PermissionProfile profile) {
        std::lock_guard lock(history_mutex_);
        permission_profile_ = profile;
    }

    void send_message(const std::string& user_message,
                      std::function<void(const std::string&)> text_callback,
                      std::function<void(const std::string&, const std::string&)> tool_callback,
                      std::function<void()> done_callback,
                      TurnCallbacks turn_callbacks = {});

    void clear_history();
    void compact_history(std::string summary);
    void undo_last();
    std::string last_user_message();

    // -----------------------------------------------------------------------
    // Session persistence helpers.
    // -----------------------------------------------------------------------

    /// Return a snapshot of the conversation history (system message excluded).
    [[nodiscard]] std::vector<core::llm::Message> get_history() const;

    /// Restore history from a saved session.  Regenerates the system prompt.
    void load_history(std::vector<core::llm::Message> messages,
                      const std::string& context_summary,
                      const std::string& mode);

    [[nodiscard]] std::string get_mode() const;
    [[nodiscard]] std::string get_context_summary() const;
    [[nodiscard]] std::string get_active_model_name() const;

    // -----------------------------------------------------------------------
    // Permission gate — set by the TUI to approve/deny dangerous tool calls.
    // When null, all tool calls are executed without approval (headless mode).
    // -----------------------------------------------------------------------
    void set_permission_fn(std::function<bool(std::string_view, std::string_view)> fn) {
        std::lock_guard lock(history_mutex_);
        permission_fn_ = std::move(fn);
    }

    // Called by TUI when the loop breaker fires (agent stuck).
    void set_loop_break_fn(std::function<void(int)> fn) {
        std::lock_guard lock(history_mutex_);
        on_loop_break_ = std::move(fn);
    }

    void set_auto_compact_threshold(int threshold) {
        std::lock_guard lock(history_mutex_);
        auto_compact_threshold_ = threshold;
    }

    // The active model name used for cost estimation.
    void set_active_model(std::string model) {
        std::lock_guard lock(history_mutex_);
        active_model_ = std::move(model);
    }

    void set_provider(std::shared_ptr<core::llm::LLMProvider> provider) {
        std::lock_guard lock(history_mutex_);
        provider_ = std::move(provider);
    }

private:
    void step(std::function<void(const std::string&)> text_callback,
              std::function<void(const std::string&, const std::string&)> tool_callback,
              std::function<void()> done_callback,
              TurnCallbacks turn_callbacks);

    void check_auto_compact(std::function<void(const std::string&)> text_callback);

    void ensure_system_prompt();

    // Returns true if the tool call was approved (or doesn't need permission).
    [[nodiscard]] bool check_permission(const std::string& tool_name,
                                        const std::string& args);

    std::shared_ptr<core::llm::LLMProvider> provider_;
    core::tools::ToolManager& skill_manager_;
    SubagentOrchestrator orchestrator_;
    std::vector<core::llm::Message> history_;
    mutable std::mutex history_mutex_;
    std::string current_mode_ = "BUILD";
    PermissionProfile permission_profile_ = PermissionProfile::Interactive;

    // Loop-breaker state
    static constexpr int kLoopBreakerThreshold = 3;
    int consecutive_failure_rounds_ = 0;
    int auto_compact_threshold_ = 0;

    // Callbacks set by the TUI
    std::function<bool(std::string_view, std::string_view)> permission_fn_;
    std::function<void(int)> on_loop_break_;

    // Model name for budget tracking
    std::string active_model_;
    std::string context_summary_;

    // Cancellation support
    std::atomic<bool> stop_requested_{false};
};

} // namespace core::agent
