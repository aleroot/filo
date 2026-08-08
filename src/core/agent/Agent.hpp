#pragma once

#include "../context/SessionContext.hpp"
#include "../context/ContextBuilder.hpp"
#include "../context/ContextWindowTracker.hpp"
#include "../llm/ProviderManager.hpp"
#include "../memory/MemoryBackgroundService.hpp"
#include "../power/SleepInhibitor.hpp"
#include "../session/GoalManager.hpp"
#include "../session/TodoManager.hpp"
#include "../session/SessionData.hpp"
#include "../session/SessionEfficiencyController.hpp"
#include "../tools/ToolManager.hpp"
#include "../tools/ReadToolResultTool.hpp"
#include "../tools/TodoTool.hpp"
#include "HistoryCompactor.hpp"
#include "HistoryCompactionPlanner.hpp"
#include "SubagentOrchestrator.hpp"
#include "SubagentEvents.hpp"
#include "PermissionGate.hpp"
#include "ToolCallDeduplicator.hpp"
#include "ToolResultStore.hpp"

namespace core::budget {
class BudgetTracker;
}

namespace core::session {
class SessionStatsRegistry;
}

#include <filesystem>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <mutex>
#include <atomic>
#include <optional>
#include <cstdint>

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
        std::function<void(const SubagentEvent&)> on_subagent_event = {};
        // Live reasoning/thinking deltas, streamed separately from the assistant
        // answer body. Display-only: routing reasoning here keeps chain-of-thought
        // out of the visible response text while still allowing the UI to show it
        // in a disclosure. Upstream replay is governed independently by the
        // serializer (Message::reasoning_content / continuation_items).
        std::function<void(const std::string&)> on_reasoning = {};
        // Out-of-band status/log sink for lifecycle messages (e.g. auto-compaction
        // progress). Must NOT be the streaming assistant-message chunk callback:
        // routing status here keeps it out of the assistant response body and
        // avoids re-marking a finalized assistant message as pending.
        std::function<void(const std::string&)> on_status_log = {};
        // Raised only when an OAuth refresh credential was definitively
        // rejected and interactive sign-in can recover the provider.
        std::function<void(const core::llm::AuthenticationRecoveryRequest&)>
            on_authentication_required = {};
        std::shared_ptr<core::llm::LLMProvider> provider_override = {};
        std::string provider_name_override{};
        std::string model_override{};
        std::string effort_override{};
        std::optional<int> max_tokens_override{};
        std::optional<core::llm::ResponseFormat> response_format_override{};
        std::vector<std::string> allowed_tools{};
        std::string ledger_actor = "agent";
        bool allow_efficiency_rotation = true;
        // Rotate only when current context usage reaches this fraction [0.0, 1.0].
        double min_context_utilization_for_rotation = 0.0;
    };

    /// Dependency-injection constructor. Execution roots that run concurrent
    /// sessions (the threads TUI) inject a shared SessionStatsRegistry so
    /// every thread's accounting stays isolated; when no registry is supplied
    /// the process-shared compatibility registry is used. A budget tracker can
    /// likewise be injected for testability (defaults to the process tracker).
    Agent(std::shared_ptr<core::llm::LLMProvider> provider,
          core::tools::ToolManager& skill_manager,
          core::context::SessionContext session_context,
          std::filesystem::path tool_result_root = ToolResultStore::default_root(),
          std::shared_ptr<core::power::SleepInhibitor> sleep_inhibitor = {},
          std::shared_ptr<core::session::SessionStatsRegistry> session_stats_registry = {},
          core::budget::BudgetTracker* budget_tracker = nullptr);
    ~Agent();

    // -----------------------------------------------------------------------
    // Cancellation support — stop the current LLM response generation.
    // -----------------------------------------------------------------------

    /// Request cancellation of the current streaming response.
    void request_stop();

    /// Check if a stop has been requested.
    [[nodiscard]] bool is_stop_requested() const;

    /// Clear the stop flag (call before starting a new turn).
    void clear_stop_request();

    /// Check if the current/last turn ended with an error (provider, transport, or configuration failure). Reset when a new turn starts.
    [[nodiscard]] bool last_turn_failed() const;

    /// True from user-message admission until the turn's single completion
    /// callback wins. Session/history replacement must not run while this is
    /// true.
    [[nodiscard]] bool turn_in_progress() const noexcept;
    [[nodiscard]] std::string session_id() const;

    void set_mode(const std::string& mode);
    void set_session_id(std::string session_id);

    // Atomically extends the session workspace and refreshes prompt state that
    // describes or depends on its roots.
    void grant_workspace_paths(
        const std::vector<std::filesystem::path>& paths);
    [[nodiscard]] core::workspace::SessionWorkspace workspace_snapshot() const;
    void set_session_goal(std::optional<core::session::SessionGoal> goal);
    /// Supplies the live goal-graph context for the dynamic prompt suffix.
    /// Injected as a callable rather than a GoalEngine handle so the agent
    /// keeps no dependency on core::goal, and so ownership stays acyclic:
    /// the engine's hooks already hold a strong reference to this agent, so
    /// the callable must hold only a weak one.
    void set_goal_graph_context_fn(std::function<std::string()> fn);
    void restore_todos(std::vector<core::session::SessionTodoItem> todos);
    [[nodiscard]] std::vector<core::session::SessionTodoItem> get_todos() const;
    [[nodiscard]] std::expected<core::session::SessionTodoItem, std::string>
    add_todo(std::string_view text);
    [[nodiscard]] std::expected<core::session::SessionTodoItem, std::string>
    set_todo_status(std::string_view selector, core::session::TodoStatus status);
    [[nodiscard]] std::expected<core::session::SessionTodoItem, std::string>
    remove_todo(std::string_view selector);
    [[nodiscard]] std::size_t clear_completed_todos();
    void clear_todos();
    void set_memory_thread_policy(core::memory::MemoryThreadPolicy policy);
    [[nodiscard]] core::memory::MemoryThreadPolicy memory_thread_policy() const;
    void run_memory_review_async(std::function<void(std::string)> status_callback = {});
    void refresh_system_prompt();

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
    void compact_history_async(
        std::function<void(const std::string&)> text_callback,
        std::function<void()> done_callback = {},
        HistoryCompactionReason reason = HistoryCompactionReason::Manual);
    void undo_last();
    std::string last_user_message();
    [[nodiscard]] bool has_user_turn() const;
    [[nodiscard]] std::optional<core::llm::Message> last_user_turn() const;

    // -----------------------------------------------------------------------
    // Session persistence helpers.
    // -----------------------------------------------------------------------

    /// Return a snapshot of the conversation history (system message excluded).
    [[nodiscard]] std::vector<core::llm::Message> get_history() const;

    /// Append a message to conversation history without starting a model turn.
    void append_history_message(core::llm::Message message);

    /// Restore history from a saved session.  Regenerates the system prompt.
    void load_history(std::vector<core::llm::Message> messages,
                      const std::string& context_summary,
                      const std::string& mode);

    [[nodiscard]] std::string get_mode() const;
    [[nodiscard]] std::string get_context_summary() const;
    [[nodiscard]] core::context::ContextWindowSnapshot context_window_snapshot() const;
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
        set_auto_compact_threshold(
            threshold,
            threshold == core::context::CompactionTriggerPolicy::kDefaultFixedTokenThreshold);
    }

    void set_auto_compact_threshold(int threshold,
                                    bool use_model_aware_default_threshold) {
        std::lock_guard lock(history_mutex_);
        auto_compact_threshold_ = threshold;
        auto_compact_uses_model_window_default_ = use_model_aware_default_threshold;
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
        refresh_context_window_snapshot_unlocked();
    }

    void set_provider(std::shared_ptr<core::llm::LLMProvider> provider) {
        std::lock_guard lock(history_mutex_);
        provider_ = std::move(provider);
        refresh_context_window_snapshot_unlocked();
    }

    /// Active provider, for out-of-band one-shot completions that must NOT
    /// touch conversation history (goal planning, verification judging,
    /// reflection). Callers own the returned handle for the call's duration.
    [[nodiscard]] std::shared_ptr<core::llm::LLMProvider> get_provider() const {
        std::lock_guard lock(history_mutex_);
        return provider_;
    }

    void set_active_provider_name(std::string provider_name) {
        std::lock_guard lock(history_mutex_);
        active_provider_name_ = std::move(provider_name);
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
        int max_output_recovery_count = 0;
        std::string transport_turn_id;
        std::optional<core::context::PromptPlan> prompt_plan;
        ToolCallDeduplicator deduplicator;
        bool final_response_after_repeat_stop_requested = false;
        // Destructive history replacement invalidates callbacks from every
        // older turn. Ordinary appends and in-turn compaction keep this stable.
        std::uint64_t conversation_generation = 0;
        // Freeze provider selection for a complete agentic turn. A live model
        // switch applies to the next user turn, not halfway through a
        // tool-use/result exchange.
        std::shared_ptr<core::llm::LLMProvider> provider;
        std::string provider_name;
        std::string model;
    };

    void step(std::function<void(const std::string&)> text_callback,
              std::function<void(const std::string&, const std::string&)> tool_callback,
              std::function<void()> done_callback,
              TurnCallbacks turn_callbacks,
              std::shared_ptr<TurnState> turn_state);

    // True while the turn identified by `turn_state` still owns the current
    // conversation. Destructive history replacement bumps conversation_generation_
    // and makes every older turn stale.
    [[nodiscard]] bool is_turn_current(
        const std::shared_ptr<TurnState>& turn_state) const;

    // Freeze provider selection (and the conversation generation) into the turn
    // state so a live model switch applies to the next user turn, not halfway
    // through a tool-use/result exchange. Caller must hold history_mutex_.
    void capture_turn_provider_snapshot_unlocked(
        TurnState& turn_state, const TurnCallbacks& turn_callbacks);

    // Run `mutator` against the live history under history_mutex_, but only if
    // the turn's conversation generation is still current. Returns false (doing
    // nothing) when the conversation was replaced after the turn began — callers
    // must abort the turn in that case. Refreshes the context snapshot when
    // current. Used to make every in-turn history append generation-safe with a
    // single, uniform control-flow pattern.
    template <typename Mutator>
    bool apply_to_history_if_turn_current(
        const std::shared_ptr<TurnState>& turn_state,
        Mutator&& mutator) {
        std::lock_guard lock(history_mutex_);
        if (turn_state->conversation_generation != conversation_generation_) {
            return false;
        }
        std::forward<Mutator>(mutator)(history_);
        refresh_context_window_snapshot_unlocked();
        return true;
    }

    // status_log_callback receives out-of-band lifecycle status (compaction
    // progress). It must be distinct from the streaming assistant chunk
    // callback so status text never lands in the assistant response body.
    void check_auto_compact(std::function<void(const std::string&)> status_log_callback);
    void run_memory_background_review(
        core::llm::protocols::RateLimitInfo rate_limit,
        std::function<void(const std::string&)> status_log_callback);

    void ensure_system_prompt();
    void refresh_stable_prompt_state_unlocked();

    [[nodiscard]] std::string build_dynamic_prompt_suffix() const;
    void append_project_facts_update_unlocked(
        std::optional<core::context::ProjectFactsSnapshot> current);
    void refresh_stable_prompt_prefix_unlocked();
    void mark_stable_prompt_prefix_dirty() noexcept { stable_prompt_prefix_dirty_ = true; }
    void refresh_context_window_snapshot_unlocked() noexcept;
    void apply_compaction_unlocked(
        std::string summary,
        HistoryCompactionPlan plan,
        std::string active_skill_context);
    void launch_compaction_transaction(
        HistoryCompactionPlan plan,
        std::uint64_t base_revision,
        std::string active_skill_context,
        std::shared_ptr<core::llm::LLMProvider> provider,
        std::string model,
        HistoryCompactionReason reason,
        std::function<void(const std::string&)> status_log_callback,
        std::function<void()> done_callback = {});

    // Returns true if the tool call was approved (or doesn't need permission).
    [[nodiscard]] bool check_permission(const std::string& tool_name,
                                        const std::string& args);
    [[nodiscard]] core::session::SessionEfficiencyDecision current_efficiency_decision_unlocked() const;
    void reset_efficiency_tracking_unlocked();
    void run_efficiency_rotation_if_needed(double min_context_utilization_for_rotation);

    [[nodiscard]] static int sanitize_max_steps_per_turn(int value) noexcept;
    [[nodiscard]] core::context::SessionContext session_context_snapshot() const;

    std::shared_ptr<core::llm::LLMProvider> provider_;
    std::shared_ptr<core::power::SleepInhibitor> sleep_inhibitor_;
    core::tools::ToolManager& skill_manager_;
    core::context::SessionContext session_context_;
    /// Injected/shared session metrics registry; declared before the
    /// orchestrator so subagents can share it.
    std::shared_ptr<core::session::SessionStatsRegistry> session_stats_registry_;
    core::budget::BudgetTracker* budget_tracker_ = nullptr;
    SubagentOrchestrator orchestrator_;
    core::session::TodoManager todo_manager_;
    core::tools::TodoTool todo_tool_;
    ToolResultStore tool_result_store_;
    core::tools::ReadToolResultTool read_tool_result_tool_;
    std::vector<core::llm::Message> history_;
    mutable std::mutex history_mutex_;
    std::uint64_t history_revision_ = 0;
    std::uint64_t conversation_generation_ = 0;
    bool compaction_in_progress_ = false;
    std::string current_mode_ = "BUILD";
    PermissionProfile permission_profile_ = PermissionProfile::Interactive;

    // Loop-breaker state
    static constexpr int kLoopBreakerThreshold = 3;
    int consecutive_failure_rounds_ = 0;
    int auto_compact_threshold_ = 0;
    bool auto_compact_uses_model_window_default_ = false;
    LoopLimits loop_limits_{};

    // Callbacks set by the TUI
    std::mutex permission_check_mutex_;
    std::function<bool(std::string_view, std::string_view)> permission_fn_;
    std::function<void(int)> on_loop_break_;

    // Model name for budget tracking
    std::string active_provider_name_;
    std::string active_model_;
    std::string effort_level_;
    std::string context_summary_;
    std::optional<core::session::SessionGoal> session_goal_;
    std::function<std::string()> goal_graph_context_fn_;
    std::optional<core::context::ProjectFactsSnapshot> project_facts_snapshot_;
    core::context::PromptPlan stable_prompt_plan_;
    std::string stable_prompt_prefix_;
    std::size_t stable_prompt_prefix_tokens_ = 0;
    bool stable_prompt_prefix_dirty_ = true;
    core::context::ContextWindowSnapshot context_window_snapshot_;
    core::session::SessionEfficiencyController efficiency_controller_;
    HistoryCompactor history_compactor_;
    std::function<void(const core::session::SessionEfficiencyDecision&)> efficiency_decision_fn_;

    // Cancellation support
    std::atomic<bool> stop_requested_{false};
    std::atomic<bool> turn_in_progress_{false};
    // Turn outcome: set when a turn ends through an error path.
    std::atomic<bool> turn_failed_{false};
    std::atomic<std::uint64_t> next_transport_turn_id_{1};
};

} // namespace core::agent
