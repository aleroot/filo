#pragma once

#include "GoalGraph.hpp"
#include "GoalReflector.hpp"
#include "GoalVerifier.hpp"

#include <cstdint>
#include <functional>
#include <semaphore>
#include <string>
#include <string_view>

namespace core::goal {

// ---------------------------------------------------------------------------
// GoalScheduler — executes a GoalGraph wave by wave.
//
// Each wave computes the ready set (dependencies satisfied), then:
//   - parallelizable work nodes fan out concurrently, bounded by
//     kMaxWaveParallelism (mirrors the subagent orchestrator's read-only
//     semaphore discipline); delegates run off-thread, all graph mutations
//     are applied single-threaded after the join — the graph itself is
//     never touched concurrently;
//   - all other nodes execute serially in topological order.
//
// Retry semantics (Reflexion loop, kept edge-free so the graph stays a DAG):
//   work fails    -> reflect -> lessons appended -> requeued while budget lasts
//   verify fails  -> reflect on retry_target -> target requeued with lessons
//   budget out    -> OnExhausted edges fire; none -> goal blocks, never loops
//
// All side effects go through SchedulerDelegates, so the scheduler is fully
// deterministic under test (inject synchronous fakes) and host-agnostic.
// ---------------------------------------------------------------------------

struct WorkOutcome {
    bool ok = false;
    std::string output; ///< summary on success, error description on failure
};

struct GoalEvent {
    enum class Type : std::uint8_t {
        WaveBegin,
        NodeStarted,
        NodeFinished,
        VerdictIssued,
        ReflectionReady,
        GateAwaiting,
        ReplanRequested,
    };

    Type type = Type::WaveBegin;
    NodeId node = kInvalidNodeId;
    std::string node_name;
    std::string message;
};

struct SchedulerDelegates {
    std::function<WorkOutcome(const Node& node, std::string_view retry_context)> run_work;
    std::function<Verdict(const Node& node, const VerifyContext& context)> verify;
    std::function<Reflection(const Node& node, const Verdict& verdict,
                             const VerifyContext& context)> reflect;
    std::function<bool(const Node& node)> await_gate;            ///< null: gates block
    std::function<void(const GoalEvent& event)> on_event;        ///< must be thread-safe
    std::function<bool()> cancellation_requested;                ///< polled between nodes
};

enum class WaveOutcome : std::uint8_t {
    Progress,        ///< at least one node ran; call again
    Idle,            ///< no ready nodes; graph is terminal or stuck
    Cancelled,       ///< cancellation requested mid-wave
    ReplanRequested, ///< a replan node fired; engine should rebuild the plan
};

class GoalScheduler {
public:
    static constexpr std::ptrdiff_t kMaxWaveParallelism = 4;

    explicit GoalScheduler(SchedulerDelegates delegates);

    /// Execute one wave: every node ready right now. Single-threaded from the
    /// caller's perspective; parallel branches are joined before returning.
    [[nodiscard]] WaveOutcome step_wave(GoalGraph& graph) const;

    /// Blocking convenience: step waves until the graph is terminal, a replan
    /// is requested, or cancellation fires. Maps the result to a RunState.
    [[nodiscard]] RunState run(GoalGraph& graph) const;

private:
    struct ParallelResult {
        NodeId node = kInvalidNodeId;
        WorkOutcome outcome;
        Reflection reflection; ///< populated when the outcome failed
    };

    void emit(const GoalEvent& event) const;
    [[nodiscard]] bool cancelled() const;

    void execute_serial(GoalGraph& graph, NodeId id) const;
    void execute_parallel_batch(GoalGraph& graph,
                                const std::vector<NodeId>& batch) const;

    void apply_work_outcome(GoalGraph& graph, NodeId id, WorkOutcome outcome,
                            std::optional<Reflection> reflection) const;
    void execute_verify(GoalGraph& graph, NodeId id) const;

    /// Activate an edge target (Pending -> Ready). Returns true if activated.
    static bool activate(GoalGraph& graph, NodeId id);
    void route(GoalGraph& graph, NodeId from, EdgeCondition condition) const;

    /// Requeue a work node after a failed verification, or fail the verify
    /// node when the retry budget is exhausted. Returns true when requeued.
    bool requeue_or_fail(GoalGraph& graph, NodeId verify_id, const Verdict& verdict,
                         const VerifyContext& context) const;

    [[nodiscard]] VerifyContext make_context(const GoalGraph& graph, NodeId verify_id) const;

    SchedulerDelegates delegates_;
    // Mutable only to bound concurrent fan-out inside step_wave; logically const.
    mutable std::counting_semaphore<kMaxWaveParallelism> wave_slots_{kMaxWaveParallelism};
};

} // namespace core::goal
