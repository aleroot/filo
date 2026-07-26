#include "GoalScheduler.hpp"

#include <algorithm>
#include <format>
#include <future>
#include <optional>
#include <vector>

namespace core::goal {

GoalScheduler::GoalScheduler(SchedulerDelegates delegates)
    : delegates_(std::move(delegates)) {}

void GoalScheduler::emit(const GoalEvent& event) const {
    if (delegates_.on_event) {
        delegates_.on_event(event);
    }
}

bool GoalScheduler::cancelled() const {
    return delegates_.cancellation_requested && delegates_.cancellation_requested();
}

bool GoalScheduler::activate(GoalGraph& graph, NodeId id) {
    Node* node = graph.node(id);
    if (node == nullptr || node->state != NodeState::Pending) {
        return false;
    }
    return graph.transition(id, NodeState::Ready).has_value();
}

void GoalScheduler::route(GoalGraph& graph, NodeId from, EdgeCondition condition) const {
    for (const Edge& edge : graph.out_edges(from)) {
        if (edge.condition == condition || (condition == EdgeCondition::OnPass
                                            && edge.condition == EdgeCondition::Always)) {
            activate(graph, edge.to);
        }
    }
}

// ---------------------------------------------------------------------------
// Wave execution
// ---------------------------------------------------------------------------

WaveOutcome GoalScheduler::step_wave(GoalGraph& graph) const {
    const std::vector<NodeId> ready = graph.ready_nodes();
    if (ready.empty()) {
        return WaveOutcome::Idle;
    }
    if (cancelled()) {
        return WaveOutcome::Cancelled;
    }

    emit(GoalEvent{.type = GoalEvent::Type::WaveBegin,
                   .message = std::format("wave with {} ready node(s)", ready.size())});

    // Partition: parallelizable work fans out; everything else is serial.
    std::vector<NodeId> parallel_batch;
    std::vector<NodeId> serial;
    for (const NodeId id : ready) {
        const Node* node = graph.node(id);
        if (node != nullptr && node->kind == NodeKind::Work && node->parallelizable) {
            parallel_batch.push_back(id);
        } else {
            serial.push_back(id);
        }
    }

    if (!parallel_batch.empty()) {
        execute_parallel_batch(graph, parallel_batch);
    }
    for (const NodeId id : serial) {
        if (cancelled()) {
            return WaveOutcome::Cancelled;
        }
        const Node* node = graph.node(id);
        if (node == nullptr || node->kind == NodeKind::Replan) {
            emit(GoalEvent{.type = GoalEvent::Type::ReplanRequested,
                           .node = id,
                           .node_name = node != nullptr ? node->name : std::string{}});
            return WaveOutcome::ReplanRequested;
        }
        execute_serial(graph, id);
    }
    return WaveOutcome::Progress;
}

RunState GoalScheduler::run(GoalGraph& graph) const {
    for (;;) {
        if (cancelled()) {
            return RunState::Paused;
        }
        const WaveOutcome outcome = step_wave(graph);
        switch (outcome) {
        case WaveOutcome::Progress:
            continue;
        case WaveOutcome::Cancelled:
            return RunState::Paused;
        case WaveOutcome::ReplanRequested:
            return RunState::Blocked;
        case WaveOutcome::Idle:
            return graph.outcome().value_or(RunState::Blocked);
        }
    }
}

void GoalScheduler::execute_parallel_batch(GoalGraph& graph,
                                           const std::vector<NodeId>& batch) const {
    // Phase 1: run delegates concurrently against const node snapshots.
    // The graph is NOT touched from worker threads.
    std::vector<std::future<ParallelResult>> futures;
    futures.reserve(batch.size());
    for (const NodeId id : batch) {
        Node* node = graph.node(id);
        if (node == nullptr) {
            continue;
        }
        if (!graph.transition(id, NodeState::Running).has_value()) {
            continue;
        }
        ++node->attempts;
        emit(GoalEvent{.type = GoalEvent::Type::NodeStarted,
                       .node = id,
                       .node_name = node->name});

        const Node snapshot = *node;
        const std::string retry_context = GoalReflector::build_retry_context(snapshot);
        futures.push_back(std::async(std::launch::async, [this, snapshot, retry_context] {
            wave_slots_.acquire();
            ParallelResult result{.node = snapshot.id};
            if (delegates_.run_work) {
                result.outcome = delegates_.run_work(snapshot, retry_context);
            } else {
                result.outcome = WorkOutcome{.ok = false, .output = "no work delegate"};
            }
            if (!result.outcome.ok && delegates_.reflect) {
                Verdict verdict{.passed = false,
                                .deterministic = false,
                                .reason = result.outcome.output};
                result.reflection = delegates_.reflect(snapshot, verdict, VerifyContext{});
            }
            wave_slots_.release();
            return result;
        }));
    }

    // Phase 2: join, then apply all mutations single-threaded.
    for (std::future<ParallelResult>& future : futures) {
        ParallelResult result = future.get();
        apply_work_outcome(graph, result.node, std::move(result.outcome),
                           std::move(result.reflection));
    }
}

void GoalScheduler::execute_serial(GoalGraph& graph, NodeId id) const {
    Node* node = graph.node(id);
    if (node == nullptr) {
        return;
    }

    switch (node->kind) {
    case NodeKind::Plan:
        // Planning already happened at plan time; the node is a marker.
        node->result_summary = node->directive;
        (void)graph.transition(id, NodeState::Ready);
        (void)graph.transition(id, NodeState::Running);
        (void)graph.transition(id, NodeState::Succeeded);
        emit(GoalEvent{.type = GoalEvent::Type::NodeFinished,
                       .node = id,
                       .node_name = node->name,
                       .message = "plan acknowledged"});
        route(graph, id, EdgeCondition::OnPass);
        return;

    case NodeKind::Work: {
        if (!graph.transition(id, NodeState::Running).has_value()) {
            // Pending nodes need the Ready stepping stone first.
            if (!graph.transition(id, NodeState::Ready).has_value()
                || !graph.transition(id, NodeState::Running).has_value()) {
                return;
            }
        }
        ++node->attempts;
        emit(GoalEvent{.type = GoalEvent::Type::NodeStarted,
                       .node = id,
                       .node_name = node->name});
        const Node snapshot = *node;
        const std::string retry_context = GoalReflector::build_retry_context(snapshot);
        WorkOutcome outcome = delegates_.run_work
            ? delegates_.run_work(snapshot, retry_context)
            : WorkOutcome{.ok = false, .output = "no work delegate"};

        std::optional<Reflection> reflection;
        if (!outcome.ok && delegates_.reflect) {
            Verdict verdict{.passed = false,
                            .deterministic = false,
                            .reason = outcome.output};
            reflection = delegates_.reflect(snapshot, verdict, VerifyContext{});
        }
        apply_work_outcome(graph, id, std::move(outcome), std::move(reflection));
        return;
    }

    case NodeKind::Verify:
        execute_verify(graph, id);
        return;

    case NodeKind::Reflect: {
        // Explicit pipeline reflector: critique the retry target (or itself).
        (void)graph.transition(id, NodeState::Ready);
        (void)graph.transition(id, NodeState::Running);
        emit(GoalEvent{.type = GoalEvent::Type::NodeStarted,
                       .node = id,
                       .node_name = node->name});
        Node* target = graph.node(node->retry_target != kInvalidNodeId
                                      ? node->retry_target
                                      : id);
        if (target != nullptr && delegates_.reflect) {
            Verdict verdict{.passed = false,
                            .deterministic = false,
                            .reason = node->directive};
            Reflection reflection = delegates_.reflect(*target, verdict,
                                                       make_context(graph, target->id));
            for (std::string& lesson : reflection.lessons) {
                if (target->lessons.size() < Node::kMaxLessons) {
                    target->lessons.push_back(std::move(lesson));
                }
            }
            if (!reflection.critique.empty()) {
                target->result_summary = reflection.critique;
            }
            emit(GoalEvent{.type = GoalEvent::Type::ReflectionReady,
                           .node = id,
                           .node_name = node->name,
                           .message = reflection.critique});
        }
        (void)graph.transition(id, NodeState::Succeeded);
        route(graph, id, EdgeCondition::OnPass);
        return;
    }

    case NodeKind::FanIn: {
        (void)graph.transition(id, NodeState::Ready);
        (void)graph.transition(id, NodeState::Running);
        std::string merged;
        for (const Edge& edge : graph.in_edges(id)) {
            const Node* dependency = graph.node(edge.from);
            if (dependency == nullptr || dependency->result_summary.empty()) {
                continue;
            }
            if (!merged.empty()) {
                merged += "\n---\n";
            }
            merged += std::format("[{}] {}", dependency->name, dependency->result_summary);
            if (merged.size() > Node::kMaxResultChars) {
                merged.resize(Node::kMaxResultChars);
                break;
            }
        }
        node->result_summary = std::move(merged);
        (void)graph.transition(id, NodeState::Succeeded);
        emit(GoalEvent{.type = GoalEvent::Type::NodeFinished,
                       .node = id,
                       .node_name = node->name,
                       .message = "fan-in merged"});
        route(graph, id, EdgeCondition::OnPass);
        return;
    }

    case NodeKind::Gate: {
        (void)graph.transition(id, NodeState::Ready);
        (void)graph.transition(id, NodeState::Running);
        emit(GoalEvent{.type = GoalEvent::Type::GateAwaiting,
                       .node = id,
                       .node_name = node->name});
        const bool approved = delegates_.await_gate && delegates_.await_gate(*node);
        if (approved) {
            (void)graph.transition(id, NodeState::Succeeded);
            route(graph, id, EdgeCondition::OnPass);
        } else {
            (void)graph.transition(id, NodeState::Blocked);
            emit(GoalEvent{.type = GoalEvent::Type::NodeFinished,
                           .node = id,
                           .node_name = node->name,
                           .message = "gate rejected or no gate handler"});
            route(graph, id, EdgeCondition::OnFail);
        }
        return;
    }

    case NodeKind::Replan:
        // Handled in step_wave before dispatch.
        return;
    }
}

void GoalScheduler::apply_work_outcome(GoalGraph& graph, NodeId id,
                                       WorkOutcome outcome,
                                       std::optional<Reflection> reflection) const {
    Node* node = graph.node(id);
    if (node == nullptr) {
        return;
    }

    if (outcome.ok) {
        node->result_summary = outcome.output.size() > Node::kMaxResultChars
            ? outcome.output.substr(0, Node::kMaxResultChars)
            : outcome.output;
        (void)graph.transition(id, NodeState::Succeeded);
        emit(GoalEvent{.type = GoalEvent::Type::NodeFinished,
                       .node = id,
                       .node_name = node->name,
                       .message = "succeeded"});
        route(graph, id, EdgeCondition::OnPass);
        return;
    }

    if (reflection.has_value()) {
        for (std::string& lesson : reflection->lessons) {
            if (node->lessons.size() < Node::kMaxLessons) {
                node->lessons.push_back(std::move(lesson));
            }
        }
        emit(GoalEvent{.type = GoalEvent::Type::ReflectionReady,
                       .node = id,
                       .node_name = node->name,
                       .message = reflection->critique});
    }

    if (node->attempts < node->max_attempts) {
        (void)graph.transition(id, NodeState::Ready); // requeue with lessons
        emit(GoalEvent{.type = GoalEvent::Type::NodeFinished,
                       .node = id,
                       .node_name = node->name,
                       .message = std::format("failed, retrying ({}/{})",
                                              node->attempts, node->max_attempts)});
        route(graph, id, EdgeCondition::OnFail);
        return;
    }

    (void)graph.transition(id, NodeState::Failed);
    emit(GoalEvent{.type = GoalEvent::Type::NodeFinished,
                   .node = id,
                   .node_name = node->name,
                   .message = std::format("failed after {} attempt(s)", node->attempts)});
    route(graph, id, EdgeCondition::OnFail);
    route(graph, id, EdgeCondition::OnExhausted);
}

VerifyContext GoalScheduler::make_context(const GoalGraph& graph,
                                          NodeId verify_id) const {
    VerifyContext context;
    const Node* verify = graph.node(verify_id);
    if (verify == nullptr) {
        return context;
    }
    if (verify->retry_target != kInvalidNodeId) {
        if (const Node* target = graph.node(verify->retry_target)) {
            context.work_output = target->result_summary;
        }
    }
    if (context.work_output.empty()) {
        for (const Edge& edge : graph.in_edges(verify_id)) {
            const Node* dependency = graph.node(edge.from);
            if (dependency != nullptr && !dependency->result_summary.empty()) {
                context.work_output = dependency->result_summary;
                break;
            }
        }
    }
    return context;
}

bool GoalScheduler::requeue_or_fail(GoalGraph& graph, NodeId verify_id,
                                    const Verdict& verdict,
                                    const VerifyContext& context) const {
    const Node* verify = graph.node(verify_id);
    if (verify == nullptr) {
        return false;
    }

    Node* target = graph.node(verify->retry_target);
    if (target != nullptr && target->attempts < target->max_attempts) {
        if (delegates_.reflect) {
            Reflection reflection = delegates_.reflect(*target, verdict, context);
            for (std::string& lesson : reflection.lessons) {
                if (target->lessons.size() < Node::kMaxLessons) {
                    target->lessons.push_back(std::move(lesson));
                }
            }
            emit(GoalEvent{.type = GoalEvent::Type::ReflectionReady,
                           .node = verify_id,
                           .node_name = verify->name,
                           .message = reflection.critique});
        }
        // Requeue the target; the verify node goes back to Pending and becomes
        // ready again when the target next succeeds (hard dependency edge).
        // requeue() is required here because the target is Succeeded: this
        // failure retroactively invalidates that success.
        const NodeId target_id = target->id;
        (void)graph.transition(verify_id, NodeState::Ready);
        (void)graph.transition(verify_id, NodeState::Pending);
        if (!graph.requeue(target_id).has_value()) {
            return false;
        }
        return true;
    }
    return false;
}

void GoalScheduler::execute_verify(GoalGraph& graph, NodeId id) const {
    Node* node = graph.node(id);
    if (node == nullptr) {
        return;
    }
    if (!graph.transition(id, NodeState::Running).has_value()) {
        if (!graph.transition(id, NodeState::Ready).has_value()
            || !graph.transition(id, NodeState::Running).has_value()) {
            return;
        }
    }
    emit(GoalEvent{.type = GoalEvent::Type::NodeStarted,
                   .node = id,
                   .node_name = node->name});

    const VerifyContext context = make_context(graph, id);
    Verdict verdict = delegates_.verify
        ? delegates_.verify(*node, context)
        : Verdict{.passed = false, .reason = "no verify delegate"};

    emit(GoalEvent{.type = GoalEvent::Type::VerdictIssued,
                   .node = id,
                   .node_name = node->name,
                   .message = std::format("{}: {}", verdict.passed ? "PASS" : "FAIL",
                                          verdict.reason)});

    if (verdict.passed) {
        node = graph.node(id);
        node->result_summary = verdict.reason.size() > Node::kMaxResultChars
            ? verdict.reason.substr(0, Node::kMaxResultChars)
            : verdict.reason;
        (void)graph.transition(id, NodeState::Succeeded);
        route(graph, id, EdgeCondition::OnPass);
        return;
    }

    if (requeue_or_fail(graph, id, verdict, context)) {
        emit(GoalEvent{.type = GoalEvent::Type::NodeFinished,
                       .node = id,
                       .node_name = graph.node(id)->name,
                       .message = "verification failed; retry target requeued with lessons"});
        route(graph, id, EdgeCondition::OnFail);
        return;
    }

    (void)graph.transition(id, NodeState::Failed);
    emit(GoalEvent{.type = GoalEvent::Type::NodeFinished,
                   .node = id,
                   .node_name = graph.node(id)->name,
                   .message = "verification failed; retry budget exhausted"});
    route(graph, id, EdgeCondition::OnFail);
    route(graph, id, EdgeCondition::OnExhausted);
}

} // namespace core::goal
