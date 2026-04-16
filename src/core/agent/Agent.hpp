#pragma once

#include "../context/SessionContext.hpp"
#include "../llm/ProviderManager.hpp"
#include "../session/SessionEfficiencyController.hpp"
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

namespace core::config {
struct AppConfig;
}

namespace core::agent {

class Agent : public std::enable_shared_from_this<Agent> {
public:
    struct LoopLimits {
        static constexpr int kDefaultMaxStepsPerTurn = 0;
        static constexpr int kMinMaxStepsPerTurn = 0;
        static constexpr int kMaxMaxStepsPerTurn = 4096;

        int max_steps_per_turn = kDefaultMaxStepsPerTurn;
    };

    struct TurnCallbacks {
        std::function<void()> on_step_begin = {};
        std::function<void(const core::llm::ToolCall&)> on_tool_start = {};
        std::function<void(const core::llm::ToolCall&, const core::llm::Message&)> on_tool_finish =
            {};
        std::shared_ptr<core::llm::LLMProvider> provider_override = {};
        std::string model_override;
        std::vector<std::string> allowed_tools;
        bool allow_efficiency_rotation = true;
        // Rotate only when current context usage reaches this fraction [0.0, 1.0].
        double min_context_utilization_for_rotation = 0.0;
    };

    Agent(std::shared_ptr<core::llm::LLMProvider> provider,
          core::tools::ToolManager& skill_manager,
          core::context::SessionContext session_context);

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
                      std::function<void()> done_callback);

    void send_message(const std::string& user_message,
                      std::function<void(const std::string&)> text_callback,
                      std::function<void(const std::string&, const std::string&)> tool_callback,
                      std::function<void()> done_callback,
                      TurnCallbacks turn_callbacks);

    void send_message(core::llm::Message user_message,
                      std::function<void(const std::string&)> text_callback,
                      std::function<void(const std::string&, const std::string&)> tool_callback,
                      std::function<void()> done_callback);

    void send_message(core::llm::Message user_message,
                      std::function<void(const std::string&)> text_callback,
                      std::function<void(const std::string&, const std::string&)> tool_callback,
                      std::function<void()> done_callback,
                      TurnCallbacks turn_callbacks);

    void clear_history();
    void compact_history(std::string summary);
    void undo_last();
    std::string last_user_message();
    [[nodiscard]] std::optional<core::llm::Message> last_user_turn() const;

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
    void set_effort_level(std::string effort);
    [[nodiscard]] std::string get_effort_level() const;

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

    void set_loop_limits(LoopLimits limits) {
        std::lock_guard lock(history_mutex_);
        loop_limits_.max_steps_per_turn = sanitize_max_steps_per_turn(limits.max_steps_per_turn);
    }

    [[nodiscard]] LoopLimits get_loop_limits() const {
        std::lock_guard lock(history_mutex_);
        return loop_limits_;
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

    void reload_subagent_profiles(const core::config::AppConfig& app_config);

    void set_efficiency_decision_fn(
        std::function<void(const core::session::SessionEfficiencyDecision&)> fn) {
        std::lock_guard lock(history_mutex_);
        efficiency_decision_fn_ = std::move(fn);
    }

private:
    struct TurnState {
        int steps_taken = 0;
        int max_steps = LoopLimits::kDefaultMaxStepsPerTurn;
    };

    void step(std::function<void(const std::string&)> text_callback,
              std::function<void(const std::string&, const std::string&)> tool_callback,
              std::function<void()> done_callback,
              TurnCallbacks turn_callbacks,
              std::shared_ptr<TurnState> turn_state);

    void check_auto_compact(std::function<void(const std::string&)> text_callback);

    void ensure_system_prompt();

    [[nodiscard]] std::string build_stable_prompt_prefix() const;
    [[nodiscard]] std::string build_dynamic_prompt_suffix() const;
    void refresh_stable_prompt_prefix_unlocked();
    void mark_stable_prompt_prefix_dirty() noexcept { stable_prompt_prefix_dirty_ = true; }

    // Returns true if the tool call was approved (or doesn't need permission).
    [[nodiscard]] bool check_permission(const std::string& tool_name,
                                        const std::string& args);
    [[nodiscard]] core::session::SessionEfficiencyDecision current_efficiency_decision_unlocked() const;
    void reset_efficiency_tracking_unlocked();
    void run_efficiency_rotation_if_needed(double min_context_utilization_for_rotation);

    [[nodiscard]] static int sanitize_max_steps_per_turn(int value) noexcept;
    [[nodiscard]] const core::context::SessionContext& session_context() const noexcept {
        return session_context_;
    }

    std::shared_ptr<core::llm::LLMProvider> provider_;
    core::tools::ToolManager& skill_manager_;
    core::context::SessionContext session_context_;
    SubagentOrchestrator orchestrator_;
    std::vector<core::llm::Message> history_;
    mutable std::mutex history_mutex_;
    std::string current_mode_ = "BUILD";
    PermissionProfile permission_profile_ = PermissionProfile::Interactive;

    // Loop-breaker state
    static constexpr int kLoopBreakerThreshold = 3;
    int consecutive_failure_rounds_ = 0;
    int auto_compact_threshold_ = 0;
    LoopLimits loop_limits_{};

    // Callbacks set by the TUI
    std::function<bool(std::string_view, std::string_view)> permission_fn_;
    std::function<void(int)> on_loop_break_;

    // Model name for budget tracking
    std::string active_model_;
    std::string effort_level_;
    std::string context_summary_;
    std::string stable_prompt_prefix_;
    bool stable_prompt_prefix_dirty_ = true;
    core::session::SessionEfficiencyController efficiency_controller_;
    std::function<void(const core::session::SessionEfficiencyDecision&)> efficiency_decision_fn_;

    // Cancellation support
    std::atomic<bool> stop_requested_{false};
};

} // namespace core::agent
