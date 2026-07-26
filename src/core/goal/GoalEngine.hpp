#pragma once

#include "GoalGraph.hpp"
#include "GoalPlanner.hpp"
#include "GoalReflector.hpp"
#include "GoalScheduler.hpp"
#include "GoalVerifier.hpp"

#include <atomic>
#include <expected>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace core::goal {

// ---------------------------------------------------------------------------
// GoalEngine — the façade the TUI/CLI talks to. Owns the plan graph, run
// state, and the composed verifier/reflector/planner; drives the scheduler
// cooperatively (one wave per call) or blocking.
//
// All host capabilities are injected once via EngineHooks (Dependency
// Inversion): the engine never touches the network, the shell, or the agent
// directly. Thread-safety: public methods are serialized by an internal
// mutex; pause/stop are atomic and safe from any thread.
// ---------------------------------------------------------------------------

struct GoalStatusSnapshot {
    bool has_graph = false;
    RunState run_state = RunState::Idle;
    int plan_version = 0;
    std::size_t nodes_total = 0;
    std::size_t nodes_succeeded = 0;
    std::size_t nodes_failed = 0;
    std::size_t nodes_open = 0;
    std::uint64_t waves_executed = 0;
    std::size_t lessons_total = 0;
    std::string latest_reason;
};

class GoalEngine {
public:
    struct Hooks {
        CompletionFn complete;
        CommandRunner run_command;
        std::function<WorkOutcome(const Node&, std::string_view)> run_work;
        std::function<bool(const Node&)> await_gate;
        std::function<void(const GoalEvent&)> on_event;      ///< must be thread-safe
        std::function<bool()> cancellation_requested;
    };

    explicit GoalEngine(Hooks hooks);

    /// Plan phase: decompose the objective into a validated DAG (fallback
    /// linear plan when the model is unavailable or malformed). Replaces any
    /// existing plan. Not allowed while a run is in progress.
    [[nodiscard]] std::expected<void, std::string> create_plan(
        std::string_view objective,
        std::string_view context = {});

    /// Cooperative execution: run exactly one wave. Safe to call from a UI
    /// pump or a background worker; returns the engine run state afterwards.
    [[nodiscard]] RunState step_wave();

    /// Blocking execution: waves until terminal, pause, or cancellation.
    [[nodiscard]] RunState run();

    /// Replan after exhaustion/block: rebuilds the DAG from the objective,
    /// seeded with the failure reason and accumulated lessons. Plan version
    /// is preserved and bumped — traces stay attributable.
    [[nodiscard]] std::expected<void, std::string> replan(std::string_view reason);

    void request_pause() noexcept { pause_requested_.store(true, std::memory_order_release); }
    void resume() noexcept { pause_requested_.store(false, std::memory_order_release); }
    [[nodiscard]] bool pause_requested() const noexcept {
        return pause_requested_.load(std::memory_order_acquire);
    }

    void clear();

    [[nodiscard]] GoalStatusSnapshot status() const;
    [[nodiscard]] std::string render_graph() const;

    /// Prompt suffix describing the live graph for the agent's system prompt.
    /// Follows GoalManager::prompt_context conventions (untrusted-content
    /// framing, empty when nothing is active).
    [[nodiscard]] std::string prompt_context() const;

    // -----------------------------------------------------------------------
    // Durable execution: full snapshot/restore so a session can resume a
    // goal mid-graph after a crash or reload.
    // -----------------------------------------------------------------------
    [[nodiscard]] std::string snapshot_json() const;
    [[nodiscard]] std::expected<void, std::string> restore_snapshot(std::string_view json);

private:
    [[nodiscard]] SchedulerDelegates make_delegates();
    void handle_event(const GoalEvent& event);

    Hooks hooks_;
    GoalVerifier verifier_;
    GoalReflector reflector_;
    GoalPlanner planner_;

    /// Serializes plan/replan/wave execution against each other. Held for the
    /// whole (potentially minutes-long) operation, so it must never be taken
    /// by observers — those use mutex_ only.
    mutable std::mutex run_mutex_;
    /// Guards the observable state below. Only ever held for short critical
    /// sections, so status()/render_graph() stay responsive while a wave runs.
    mutable std::mutex mutex_;
    std::optional<GoalGraph> graph_;
    RunState run_state_ = RunState::Idle;
    std::string latest_reason_;
    std::uint64_t waves_executed_ = 0;
    std::atomic<bool> pause_requested_{false};
};

} // namespace core::goal
