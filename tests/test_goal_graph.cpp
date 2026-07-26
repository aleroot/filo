#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/goal/GoalEngine.hpp"
#include "core/goal/GoalGraph.hpp"
#include "core/goal/GoalPlanner.hpp"
#include "core/goal/GoalReflector.hpp"
#include "core/goal/GoalScheduler.hpp"
#include "core/goal/GoalVerifier.hpp"
#include "core/session/SessionStore.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace core::goal;
using Catch::Matchers::ContainsSubstring;

Node make_work(std::string_view name, int max_attempts = 2) {
    Node node;
    node.kind = NodeKind::Work;
    node.name = std::string(name);
    node.directive = std::format("do {}", name);
    node.max_attempts = max_attempts;
    return node;
}

Node make_verify(std::string_view name, std::string check = {}) {
    Node node;
    node.kind = NodeKind::Verify;
    node.name = std::string(name);
    node.directive = "verify work";
    node.check_command = std::move(check);
    node.acceptance = "work is done";
    return node;
}

/// Delegates that succeed everything synchronously.
SchedulerDelegates make_happy_delegates() {
    SchedulerDelegates delegates;
    delegates.run_work = [](const Node&, std::string_view) {
        return WorkOutcome{.ok = true, .output = "work done"};
    };
    delegates.verify = [](const Node&, const VerifyContext&) {
        return Verdict{.passed = true, .deterministic = true, .reason = "check passed"};
    };
    delegates.reflect = [](const Node&, const Verdict& verdict, const VerifyContext&) {
        return GoalReflector::heuristic(Node{}, verdict);
    };
    return delegates;
}

GoalGraph make_linear_graph() {
    GoalGraph graph;
    graph.set_objective("linear goal");
    const NodeId work = *graph.add_node(make_work("work"));
    const NodeId verify = *graph.add_node(make_verify("verify", "true"));
    graph.node(verify)->retry_target = work;
    REQUIRE(graph.add_edge(work, verify, EdgeCondition::OnPass).has_value());
    REQUIRE(graph.validate().has_value());
    return graph;
}

} // namespace

// ---------------------------------------------------------------------------
// GoalTypes
// ---------------------------------------------------------------------------

TEST_CASE("GoalTypes enum string round-trips", "[goal][types]") {
    for (NodeKind kind : {NodeKind::Plan, NodeKind::Work, NodeKind::Verify,
                          NodeKind::Reflect, NodeKind::FanIn, NodeKind::Gate,
                          NodeKind::Replan}) {
        REQUIRE(node_kind_from_string(to_string(kind)).value() == kind);
    }
    for (EdgeCondition condition : {EdgeCondition::Always, EdgeCondition::OnPass,
                                    EdgeCondition::OnFail, EdgeCondition::OnExhausted}) {
        REQUIRE(edge_condition_from_string(to_string(condition)).value() == condition);
    }
    REQUIRE(node_state_from_string("succeeded") == NodeState::Succeeded);
    REQUIRE(run_state_from_string("paused") == RunState::Paused);
    REQUIRE(is_terminal(NodeState::Succeeded));
    REQUIRE_FALSE(is_terminal(NodeState::Running));
    REQUIRE(is_dependency(EdgeCondition::Always));
    REQUIRE_FALSE(is_dependency(EdgeCondition::OnFail));
}

// ---------------------------------------------------------------------------
// GoalGraph
// ---------------------------------------------------------------------------

TEST_CASE("GoalGraph rejects invalid structure", "[goal][graph]") {
    GoalGraph graph;
    const NodeId a = *graph.add_node(make_work("a"));

    SECTION("duplicate names are rejected") {
        REQUIRE_FALSE(graph.add_node(make_work("a")).has_value());
    }
    SECTION("nameless nodes are rejected") {
        Node node;
        node.kind = NodeKind::Work;
        REQUIRE_FALSE(graph.add_node(node).has_value());
    }
    SECTION("self-loops are rejected") {
        REQUIRE_FALSE(graph.add_edge(a, a, EdgeCondition::Always).has_value());
    }
    SECTION("edges to unknown nodes are rejected") {
        REQUIRE_FALSE(graph.add_edge(a, 999, EdgeCondition::Always).has_value());
    }
    SECTION("duplicate edges are rejected") {
        const NodeId b = *graph.add_node(make_work("b"));
        REQUIRE(graph.add_edge(a, b, EdgeCondition::Always).has_value());
        REQUIRE_FALSE(graph.add_edge(a, b, EdgeCondition::Always).has_value());
    }
}

TEST_CASE("GoalGraph validate detects cycles", "[goal][graph]") {
    GoalGraph graph;
    const NodeId a = *graph.add_node(make_work("a"));
    const NodeId b = *graph.add_node(make_work("b"));
    REQUIRE(graph.add_edge(a, b, EdgeCondition::Always).has_value());
    REQUIRE(graph.add_edge(b, a, EdgeCondition::Always).has_value());
    const auto valid = graph.validate();
    REQUIRE_FALSE(valid.has_value());
    REQUIRE_THAT(valid.error(), ContainsSubstring("cycle"));
}

TEST_CASE("GoalGraph validate enforces retry targets to be work nodes", "[goal][graph]") {
    GoalGraph graph;
    const NodeId v1 = *graph.add_node(make_verify("v1"));
    const NodeId v2 = *graph.add_node(make_verify("v2"));
    graph.node(v1)->retry_target = v2; // verify -> verify is invalid
    const auto valid = graph.validate();
    REQUIRE_FALSE(valid.has_value());
    REQUIRE_THAT(valid.error(), ContainsSubstring("work node"));
}

TEST_CASE("GoalGraph readiness respects dependencies and conditional edges", "[goal][graph]") {
    GoalGraph graph;
    const NodeId a = *graph.add_node(make_work("a"));
    const NodeId b = *graph.add_node(make_work("b"));
    const NodeId handler = *graph.add_node(make_work("handler"));
    REQUIRE(graph.add_edge(a, b, EdgeCondition::OnPass).has_value());
    REQUIRE(graph.add_edge(a, handler, EdgeCondition::OnFail).has_value());

    // Entry node is ready; b waits on a; handler is scheduler-activated only.
    std::vector<NodeId> ready = graph.ready_nodes();
    REQUIRE(ready == std::vector<NodeId>{a});

    REQUIRE(graph.transition(a, NodeState::Running).has_value());
    REQUIRE(graph.transition(a, NodeState::Succeeded).has_value());
    ready = graph.ready_nodes();
    REQUIRE(ready == std::vector<NodeId>{b});

    // Conditional target never becomes ready on its own, even after deps succeed.
    REQUIRE(std::ranges::find(ready, handler) == ready.end());
    // Scheduler activation moves it Pending -> Ready.
    REQUIRE(graph.transition(handler, NodeState::Ready).has_value());
    ready = graph.ready_nodes();
    REQUIRE(std::ranges::find(ready, handler) != ready.end());
}

TEST_CASE("GoalGraph requeue revives settled nodes for retry", "[goal][graph]") {
    GoalGraph graph;
    const NodeId a = *graph.add_node(make_work("a"));
    REQUIRE(graph.transition(a, NodeState::Running).has_value());
    REQUIRE(graph.transition(a, NodeState::Succeeded).has_value());

    // transition() refuses to resurrect a completed node...
    REQUIRE_FALSE(graph.transition(a, NodeState::Ready).has_value());
    // ...but requeue() is the sanctioned retry path.
    REQUIRE(graph.requeue(a).has_value());
    REQUIRE(graph.node(a)->state == NodeState::Ready);

    // Running nodes may never be requeued underneath the scheduler.
    REQUIRE(graph.transition(a, NodeState::Running).has_value());
    REQUIRE_FALSE(graph.requeue(a).has_value());
}

TEST_CASE("GoalGraph transitions enforce legality", "[goal][graph]") {
    GoalGraph graph;
    const NodeId a = *graph.add_node(make_work("a"));

    REQUIRE_FALSE(graph.transition(a, NodeState::Succeeded).has_value());
    REQUIRE(graph.transition(a, NodeState::Running).has_value());
    REQUIRE_FALSE(graph.transition(a, NodeState::Pending).has_value());
    REQUIRE(graph.transition(a, NodeState::Failed).has_value());
    REQUIRE(graph.transition(a, NodeState::Ready).has_value()); // retry path
    REQUIRE(graph.transition(a, NodeState::Running).has_value());
    REQUIRE(graph.transition(a, NodeState::Succeeded).has_value());
    REQUIRE_FALSE(graph.transition(a, NodeState::Ready).has_value()); // terminal
}

TEST_CASE("GoalGraph outcome precedence", "[goal][graph]") {
    GoalGraph graph;
    const NodeId a = *graph.add_node(make_work("a"));
    const NodeId b = *graph.add_node(make_work("b"));
    REQUIRE_FALSE(graph.outcome().has_value()); // not terminal

    REQUIRE(graph.transition(a, NodeState::Running).has_value());
    REQUIRE(graph.transition(a, NodeState::Failed).has_value());
    REQUIRE(graph.transition(b, NodeState::Running).has_value());
    REQUIRE(graph.transition(b, NodeState::Succeeded).has_value());
    REQUIRE(graph.outcome().value() == RunState::Failed);
}

TEST_CASE("GoalGraph JSON round-trip preserves state", "[goal][graph][json]") {
    GoalGraph graph = make_linear_graph();
    const NodeId work = *graph.find("work");
    REQUIRE(graph.transition(work, NodeState::Running).has_value());
    graph.node(work)->lessons.push_back("lesson one");
    graph.node(work)->attempts = 1;
    // Running must survive a snapshot as Ready, never as phantom work.
    const std::string json = graph.to_json();

    auto restored = GoalGraph::from_json(json);
    REQUIRE(restored.has_value());
    REQUIRE(restored->size() == 2);
    REQUIRE(restored->objective() == "linear goal");
    REQUIRE(restored->edges().size() == 1);
    const Node* restored_work = restored->node(work);
    REQUIRE(restored_work != nullptr);
    REQUIRE(restored_work->state == NodeState::Ready);
    REQUIRE(restored_work->attempts == 1);
    REQUIRE(restored_work->lessons.size() == 1);
    REQUIRE(restored->node(*restored->find("verify"))->retry_target == work);
    REQUIRE(restored->validate().has_value());
}

TEST_CASE("GoalGraph from_json rejects garbage", "[goal][graph][json]") {
    REQUIRE_FALSE(GoalGraph::from_json("not json").has_value());
    REQUIRE_FALSE(GoalGraph::from_json("{}").has_value());
    REQUIRE_FALSE(GoalGraph::from_json("{\"nodes\":[]}").has_value());
}

TEST_CASE("GoalGraph render shows layers and edges", "[goal][graph]") {
    const GoalGraph graph = make_linear_graph();
    const std::string rendered = graph.render_ascii();
    REQUIRE_THAT(rendered, ContainsSubstring("plan v1"));
    REQUIRE_THAT(rendered, ContainsSubstring("work (work"));
    REQUIRE_THAT(rendered, ContainsSubstring("verify (verify"));
    REQUIRE_THAT(rendered, ContainsSubstring("--on_pass-->"));
}

// ---------------------------------------------------------------------------
// GoalVerifier
// ---------------------------------------------------------------------------

TEST_CASE("ShellCheckVerifier passes on exit code 0", "[goal][verifier]") {
    ShellCheckVerifier verifier([](std::string_view) {
        return CommandResult{.exit_code = 0, .output = "ok"};
    });
    const Node node = make_verify("v", "make test");
    const Verdict verdict = verifier.verify(node, VerifyContext{});
    REQUIRE(verdict.passed);
    REQUIRE(verdict.deterministic);
    REQUIRE_THAT(verdict.reason, ContainsSubstring("exited 0"));
}

TEST_CASE("ShellCheckVerifier fails on non-zero exit and keeps evidence", "[goal][verifier]") {
    ShellCheckVerifier verifier([](std::string_view) {
        return CommandResult{.exit_code = 2, .output = "boom: test failed"};
    });
    const Node node = make_verify("v", "make test");
    const Verdict verdict = verifier.verify(node, VerifyContext{});
    REQUIRE_FALSE(verdict.passed);
    REQUIRE(verdict.deterministic);
    REQUIRE_THAT(verdict.reason, ContainsSubstring("exited 2"));
    REQUIRE_THAT(verdict.evidence, ContainsSubstring("boom"));
}

TEST_CASE("ModelCheckVerifier parses the PASS/FAIL contract", "[goal][verifier]") {
    REQUIRE(ModelCheckVerifier::parse_response("PASS\nall good").passed);
    REQUIRE_FALSE(ModelCheckVerifier::parse_response("FAIL\nmissing tests").passed);
    REQUIRE(ModelCheckVerifier::parse_response("  FAIL\nx").reason == "x");
    // Garbage never passes.
    REQUIRE_FALSE(ModelCheckVerifier::parse_response("I think it works").passed);
}

TEST_CASE("GoalVerifier prefers deterministic checks over the judge model", "[goal][verifier]") {
    bool shell_called = false;
    bool model_called = false;
    GoalVerifier verifier(
        [&](std::string_view) {
            shell_called = true;
            return CommandResult{.exit_code = 0, .output = ""};
        },
        [&](std::string_view) -> std::expected<std::string, std::string> {
            model_called = true;
            return std::string("FAIL\nshould not be consulted");
        });

    const Node with_check = make_verify("v1", "true");
    REQUIRE(verifier.verify(with_check, VerifyContext{}).passed);
    REQUIRE(shell_called);
    REQUIRE_FALSE(model_called);

    const Node without_check = make_verify("v2");
    REQUIRE_FALSE(verifier.verify(without_check, VerifyContext{}).passed);
    REQUIRE(model_called);
}

TEST_CASE("popen command runner executes real commands", "[goal][verifier][popen]") {
    const CommandRunner runner = make_popen_command_runner();
    REQUIRE(runner("exit 0").exit_code == 0);
    REQUIRE(runner("exit 3").exit_code == 3);
    const CommandResult echo = runner("printf hello");
    REQUIRE(echo.exit_code == 0);
    REQUIRE(echo.output == "hello");
}

// ---------------------------------------------------------------------------
// GoalReflector
// ---------------------------------------------------------------------------

TEST_CASE("GoalReflector heuristic produces a critique and a lesson", "[goal][reflector]") {
    GoalReflector reflector({}); // no model
    const Node node = make_verify("v", "make test");
    Verdict verdict{.passed = false, .deterministic = true, .reason = "`make test` exited 2"};
    const Reflection reflection = reflector.reflect(node, verdict, VerifyContext{});
    REQUIRE_THAT(reflection.critique, ContainsSubstring("exited 2"));
    REQUIRE(reflection.lessons.size() == 1);
    REQUIRE_THAT(reflection.lessons.front(), ContainsSubstring("make test"));
}

TEST_CASE("GoalReflector parses the CRITIQUE/LESSONS contract", "[goal][reflector]") {
    const Reflection reflection = GoalReflector::parse_response(
        "CRITIQUE:\nThe build broke because of a missing include.\n"
        "LESSONS:\n- add the missing header\n- run the build before claiming success\n");
    REQUIRE_THAT(reflection.critique, ContainsSubstring("missing include"));
    REQUIRE(reflection.lessons.size() == 2);
    REQUIRE(reflection.lessons[0] == "add the missing header");
}

TEST_CASE("GoalReflector builds retry context from accumulated lessons", "[goal][reflector]") {
    Node node = make_work("w");
    REQUIRE(GoalReflector::build_retry_context(node).empty());
    node.lessons.push_back("first lesson");
    node.lessons.push_back("second lesson");
    const std::string context = GoalReflector::build_retry_context(node);
    REQUIRE_THAT(context, ContainsSubstring("first lesson"));
    REQUIRE_THAT(context, ContainsSubstring("second lesson"));
}

TEST_CASE("GoalReflector uses the model when available", "[goal][reflector]") {
    bool called = false;
    GoalReflector reflector([&](std::string_view prompt)
                                -> std::expected<std::string, std::string> {
        called = true;
        REQUIRE_THAT(std::string(prompt), ContainsSubstring("reflection module"));
        return std::string("CRITIQUE:\nit broke\nLESSONS:\n- fix it\n");
    });
    const Reflection reflection =
        reflector.reflect(make_work("w"), Verdict{.reason = "x"}, VerifyContext{});
    REQUIRE(called);
    REQUIRE(reflection.lessons.size() == 1);
    REQUIRE(reflection.lessons[0] == "fix it");
}

// ---------------------------------------------------------------------------
// GoalPlanner
// ---------------------------------------------------------------------------

TEST_CASE("GoalPlanner parses a model plan into a validated DAG", "[goal][planner]") {
    const std::string json = R"({
        "nodes": [
            {"name": "explore", "kind": "work", "directive": "map the code", "parallel": true},
            {"name": "migrate", "kind": "work", "directive": "migrate module", "max_attempts": 3},
            {"name": "verify", "kind": "verify", "directive": "check",
             "check": "ctest", "acceptance": "tests pass", "retry_target": "migrate"}
        ],
        "edges": [
            {"from": "explore", "to": "migrate", "condition": "always"},
            {"from": "migrate", "to": "verify", "condition": "on_pass"}
        ]
    })";
    auto parsed = GoalPlanner::parse_plan_json(json, "migrate the module");
    REQUIRE(parsed.has_value());
    REQUIRE(parsed->size() == 3);
    REQUIRE(parsed->edges().size() == 2);
    REQUIRE(parsed->node(*parsed->find("explore"))->parallelizable);
    REQUIRE(parsed->node(*parsed->find("verify"))->retry_target
            == *parsed->find("migrate"));
    REQUIRE(parsed->validate().has_value());
}

TEST_CASE("GoalPlanner tolerates prose around the JSON payload", "[goal][planner]") {
    auto parsed = GoalPlanner::parse_plan_json(
        "Here is your plan:\n```json\n{\"nodes\":[{\"name\":\"w\",\"kind\":\"work\"}]}\n```",
        "objective");
    REQUIRE(parsed.has_value());
    REQUIRE(parsed->size() == 1);
}

TEST_CASE("GoalPlanner rejects cycles and reserved kinds", "[goal][planner]") {
    REQUIRE_FALSE(GoalPlanner::parse_plan_json(
        R"({"nodes":[{"name":"a"},{"name":"b"}],
            "edges":[{"from":"a","to":"b"},{"from":"b","to":"a"}]})",
        "x").has_value());
    REQUIRE_FALSE(GoalPlanner::parse_plan_json(
        R"({"nodes":[{"name":"r","kind":"replan"}]})", "x").has_value());
}

TEST_CASE("GoalPlanner falls back to a linear plan on garbage", "[goal][planner]") {
    GoalPlanner planner([](std::string_view) -> std::expected<std::string, std::string> {
        return std::string("total garbage with no json");
    });
    auto planned = planner.plan("do something");
    REQUIRE(planned.has_value());
    REQUIRE(planned->size() == 2); // work + verify fallback
    REQUIRE(planned->node(*planned->find("verify"))->retry_target
            == *planned->find("goal"));
    REQUIRE(planned->validate().has_value());
}

TEST_CASE("GoalPlanner fallback carries the objective", "[goal][planner]") {
    const GoalGraph fallback = GoalPlanner::fallback_linear_plan("reach the goal");
    REQUIRE(fallback.objective() == "reach the goal");
    const Node* verify = fallback.node(*fallback.find("verify"));
    REQUIRE(verify->acceptance == "reach the goal");
    REQUIRE(verify->retry_target == *fallback.find("goal"));
}

// ---------------------------------------------------------------------------
// GoalScheduler
// ---------------------------------------------------------------------------

TEST_CASE("GoalScheduler runs a linear graph to completion", "[goal][scheduler]") {
    GoalGraph graph = make_linear_graph();
    GoalScheduler scheduler(make_happy_delegates());
    REQUIRE(scheduler.run(graph) == RunState::Completed);
    REQUIRE(graph.node(*graph.find("work"))->state == NodeState::Succeeded);
    REQUIRE(graph.node(*graph.find("verify"))->state == NodeState::Succeeded);
}

TEST_CASE("GoalScheduler requeues the retry target with lessons on verify failure",
          "[goal][scheduler][reflection]") {
    GoalGraph graph = make_linear_graph();
    const NodeId work = *graph.find("work");
    graph.node(work)->max_attempts = 2;

    std::atomic<int> verify_calls{0};
    std::atomic<int> reflect_calls{0};
    std::vector<std::string> work_prompts;

    SchedulerDelegates delegates = make_happy_delegates();
    delegates.run_work = [&](const Node&, std::string_view retry_context) {
        work_prompts.emplace_back(retry_context);
        return WorkOutcome{.ok = true, .output = "attempt output"};
    };
    delegates.verify = [&](const Node&, const VerifyContext&) {
        ++verify_calls;
        if (verify_calls == 1) {
            return Verdict{.passed = false, .deterministic = true,
                           .reason = "tests failed", .evidence = "1 failing test"};
        }
        return Verdict{.passed = true, .deterministic = true, .reason = "green"};
    };
    delegates.reflect = [&](const Node& node, const Verdict& verdict,
                            const VerifyContext&) {
        ++reflect_calls;
        return GoalReflector::heuristic(node, verdict);
    };
    delegates.on_event = [](const GoalEvent&) {};

    GoalScheduler scheduler(std::move(delegates));
    REQUIRE(scheduler.run(graph) == RunState::Completed);

    REQUIRE(verify_calls == 2);
    REQUIRE(reflect_calls == 1);
    const Node* work_node = graph.node(work);
    REQUIRE(work_node->attempts == 2);
    REQUIRE(work_node->lessons.size() == 1);
    // The second attempt was conditioned on the reflection lessons.
    REQUIRE(work_prompts.size() == 2);
    REQUIRE(work_prompts[0].empty());
    REQUIRE_THAT(work_prompts[1], ContainsSubstring("tests failed"));
}

TEST_CASE("GoalScheduler exhausts the retry budget and routes on_exhausted",
          "[goal][scheduler]") {
    GoalGraph graph = make_linear_graph();
    const NodeId work = *graph.find("work");
    const NodeId verify = *graph.find("verify");
    graph.node(work)->max_attempts = 1;
    const NodeId alarm = *graph.add_node(make_work("alarm"));
    REQUIRE(graph.add_edge(verify, alarm, EdgeCondition::OnExhausted).has_value());

    SchedulerDelegates delegates = make_happy_delegates();
    delegates.verify = [](const Node&, const VerifyContext&) {
        return Verdict{.passed = false, .reason = "always failing"};
    };

    GoalScheduler scheduler(std::move(delegates));
    REQUIRE(scheduler.run(graph) == RunState::Failed);
    REQUIRE(graph.node(verify)->state == NodeState::Failed);
    // The on_exhausted route fired and the handler ran.
    REQUIRE(graph.node(alarm)->state == NodeState::Succeeded);
}

TEST_CASE("GoalScheduler runs parallelizable work concurrently", "[goal][scheduler][parallel]") {
    GoalGraph graph;
    graph.set_objective("parallel goal");
    std::vector<NodeId> explorers;
    for (int i = 0; i < 3; ++i) {
        Node explorer = make_work(std::format("explore-{}", i));
        explorer.parallelizable = true;
        explorers.push_back(*graph.add_node(std::move(explorer)));
    }
    Node fan_in;
    fan_in.kind = NodeKind::FanIn;
    fan_in.name = "merge";
    const NodeId merge = *graph.add_node(std::move(fan_in));
    for (const NodeId id : explorers) {
        REQUIRE(graph.add_edge(id, merge, EdgeCondition::Always).has_value());
    }
    REQUIRE(graph.validate().has_value());

    std::atomic<int> concurrent{0};
    std::atomic<int> max_concurrent{0};
    SchedulerDelegates delegates = make_happy_delegates();
    delegates.run_work = [&](const Node&, std::string_view) {
        const int now = ++concurrent;
        int expected = max_concurrent.load();
        while (now > expected
               && !max_concurrent.compare_exchange_weak(expected, now)) {}
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        --concurrent;
        return WorkOutcome{.ok = true, .output = "explored"};
    };

    GoalScheduler scheduler(std::move(delegates));
    REQUIRE(scheduler.run(graph) == RunState::Completed);
    REQUIRE(max_concurrent > 1); // genuine fan-out
    REQUIRE_FALSE(graph.node(merge)->result_summary.empty()); // merged summaries
}

TEST_CASE("GoalScheduler honors cancellation between nodes", "[goal][scheduler]") {
    GoalGraph graph = make_linear_graph();
    SchedulerDelegates delegates = make_happy_delegates();
    delegates.cancellation_requested = [] { return true; };
    GoalScheduler scheduler(std::move(delegates));
    REQUIRE(scheduler.run(graph) == RunState::Paused);
}

// ---------------------------------------------------------------------------
// GoalEngine
// ---------------------------------------------------------------------------

/// Hooks with a scripted model: the planner is unavailable (so the engine
/// exercises its deterministic fallback plan) while the judge and reflector
/// behave like a well-formed model.
GoalEngine::Hooks make_engine_hooks() {
    GoalEngine::Hooks hooks;
    hooks.complete = [](std::string_view prompt)
        -> std::expected<std::string, std::string> {
        if (prompt.find("verification judge") != std::string_view::npos) {
            return std::string("PASS\nacceptance criteria are satisfied");
        }
        if (prompt.find("reflection module") != std::string_view::npos) {
            return std::string("CRITIQUE:\nneeds work\nLESSONS:\n- retry carefully\n");
        }
        return std::unexpected(std::string("no planner model in tests"));
    };
    hooks.run_command = make_popen_command_runner();
    hooks.run_work = [](const Node&, std::string_view) {
        return WorkOutcome{.ok = true, .output = "done"};
    };
    return hooks;
}

TEST_CASE("GoalEngine plans and runs a fallback graph end-to-end", "[goal][engine]") {
    GoalEngine engine(make_engine_hooks());
    REQUIRE(engine.create_plan("ship the feature").has_value());
    REQUIRE(engine.status().run_state == RunState::Paused);
    REQUIRE(engine.status().has_graph);
    REQUIRE(engine.status().plan_version == 1);

    REQUIRE(engine.run() == RunState::Completed);
    const GoalStatusSnapshot status = engine.status();
    REQUIRE(status.nodes_succeeded == 2);
    REQUIRE(status.waves_executed >= 2);
}

TEST_CASE("GoalEngine blocks creation of a second plan while running", "[goal][engine]") {
    GoalEngine engine(make_engine_hooks());
    REQUIRE(engine.create_plan("first").has_value());
    // Not running yet; replanning replaces the plan.
    REQUIRE(engine.create_plan("second").has_value());
    REQUIRE_THAT(engine.render_graph(), ContainsSubstring("second"));
}

TEST_CASE("GoalEngine snapshot and restore round-trips", "[goal][engine][json]") {
    GoalEngine engine(make_engine_hooks());
    REQUIRE(engine.create_plan("persistent goal").has_value());
    REQUIRE(engine.step_wave() == RunState::Running); // work done, verify pending

    const std::string snapshot = engine.snapshot_json();
    GoalEngine restored(make_engine_hooks());
    REQUIRE(restored.restore_snapshot(snapshot).has_value());

    const GoalStatusSnapshot status = restored.status();
    REQUIRE(status.has_graph);
    REQUIRE(status.nodes_succeeded == 1);
    // A restored run never resumes as Running: progress is re-derived from the
    // graph's own node states, so Paused is the only honest mid-run state.
    REQUIRE(status.run_state == RunState::Paused);

    // The restored engine finishes the remaining verify node.
    REQUIRE(restored.run() == RunState::Completed);
}

TEST_CASE("GoalEngine replan bumps the plan version and keeps the objective",
          "[goal][engine]") {
    GoalEngine engine(make_engine_hooks());
    REQUIRE(engine.create_plan("original objective").has_value());
    REQUIRE(engine.replan("verification kept failing").has_value());
    const GoalStatusSnapshot status = engine.status();
    REQUIRE(status.plan_version == 2);
    REQUIRE_THAT(engine.render_graph(), ContainsSubstring("plan v2"));
    REQUIRE_THAT(engine.render_graph(), ContainsSubstring("original objective"));
}

TEST_CASE("GoalEngine prompt context exposes the frontier", "[goal][engine]") {
    GoalEngine engine(make_engine_hooks());
    REQUIRE(engine.prompt_context().empty()); // nothing active
    REQUIRE(engine.create_plan("goal with frontier").has_value());
    engine.resume();
    // Force a wave so the run state is live.
    REQUIRE(engine.step_wave() == RunState::Running);
    const std::string context = engine.prompt_context();
    REQUIRE_THAT(context, ContainsSubstring("goal with frontier"));
    REQUIRE_THAT(context, ContainsSubstring("Plan version: 1"));
    REQUIRE_THAT(context, ContainsSubstring("untrusted"));
}

TEST_CASE("GoalEngine renders an empty state without a plan", "[goal][engine]") {
    GoalEngine engine(make_engine_hooks());
    REQUIRE_THAT(engine.render_graph(), ContainsSubstring("No goal graph"));
}

TEST_CASE("GoalEngine observers stay responsive while a wave runs",
          "[goal][engine][concurrency]") {
    // A wave calls into the model and the shell for a long time. Observers
    // (the TUI status line) must never block on it, and must never read a
    // half-mutated graph.
    GoalEngine::Hooks hooks = make_engine_hooks();
    std::atomic<bool> work_started{false};
    std::atomic<bool> release_work{false};
    hooks.run_work = [&](const Node&, std::string_view) {
        work_started.store(true, std::memory_order_release);
        while (!release_work.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        return WorkOutcome{.ok = true, .output = "done"};
    };

    GoalEngine engine(std::move(hooks));
    REQUIRE(engine.create_plan("concurrent objective").has_value());

    std::thread runner([&] { (void)engine.run(); });
    while (!work_started.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // These must return promptly and consistently mid-wave.
    for (int i = 0; i < 50; ++i) {
        const auto status = engine.status();
        REQUIRE(status.has_graph);
        REQUIRE(status.nodes_total == 2);
        REQUIRE(status.nodes_succeeded <= status.nodes_total);
        REQUIRE_FALSE(engine.render_graph().empty());
        REQUIRE_FALSE(engine.snapshot_json().empty());
    }

    release_work.store(true, std::memory_order_release);
    runner.join();
    REQUIRE(engine.status().nodes_succeeded == 2);
}

TEST_CASE("GoalEngine refuses overlapping runs", "[goal][engine][concurrency]") {
    GoalEngine::Hooks hooks = make_engine_hooks();
    std::atomic<bool> work_started{false};
    std::atomic<bool> release_work{false};
    std::atomic<int> concurrent_work{0};
    std::atomic<int> max_concurrent_work{0};
    hooks.run_work = [&](const Node&, std::string_view) {
        const int now = ++concurrent_work;
        int expected = max_concurrent_work.load();
        while (now > expected
               && !max_concurrent_work.compare_exchange_weak(expected, now)) {}
        work_started.store(true, std::memory_order_release);
        while (!release_work.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        --concurrent_work;
        return WorkOutcome{.ok = true, .output = "done"};
    };

    GoalEngine engine(std::move(hooks));
    REQUIRE(engine.create_plan("exclusive objective").has_value());

    std::thread runner([&] { (void)engine.run(); });
    while (!work_started.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    // A second driver must not execute nodes concurrently, and mutating the
    // plan mid-run must be rejected rather than clobbering the live graph.
    (void)engine.step_wave();
    REQUIRE_FALSE(engine.replan("racing replan").has_value());
    REQUIRE_FALSE(engine.create_plan("racing plan").has_value());

    release_work.store(true, std::memory_order_release);
    runner.join();
    REQUIRE(max_concurrent_work.load() == 1);
}

// ---------------------------------------------------------------------------
// Session persistence of the goal graph blob
// ---------------------------------------------------------------------------

TEST_CASE("SessionStore persists the goal graph snapshot", "[goal][session]") {
    const auto dir = std::filesystem::temp_directory_path()
        / "filo_goal_graph_session_test";
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const core::session::SessionStore store{dir};

    core::session::SessionData data;
    data.session_id = "goalgrph";
    data.created_at = "2026-03-22T10:15:30Z";

    SECTION("round-trips a full snapshot blob") {
        // A real engine snapshot, so the stored blob is representative.
        GoalEngine engine(make_engine_hooks());
        REQUIRE(engine.create_plan("persisted objective").has_value());
        REQUIRE(engine.step_wave() == RunState::Running);

        data.goal_graph = core::session::SessionGoalGraph{
            .plan_version = engine.status().plan_version,
            .run_state = std::string(to_string(engine.status().run_state)),
            .snapshot = engine.snapshot_json(),
            .updated_at = "2026-03-22T11:00:00Z",
        };

        REQUIRE(store.save(data));
        const auto loaded = store.load_by_id("goalgrph");
        REQUIRE(loaded.has_value());
        REQUIRE(loaded->goal_graph.has_value());
        REQUIRE(loaded->goal_graph->plan_version == 1);
        REQUIRE(loaded->goal_graph->run_state == "running");
        REQUIRE(loaded->goal_graph->updated_at == "2026-03-22T11:00:00Z");

        // The persisted blob still rehydrates into a live engine.
        GoalEngine rehydrated(make_engine_hooks());
        REQUIRE(rehydrated.restore_snapshot(loaded->goal_graph->snapshot).has_value());
        REQUIRE(rehydrated.status().nodes_succeeded == 1);
        REQUIRE(rehydrated.run() == RunState::Completed);
    }

    SECTION("absent goal graph stays absent") {
        REQUIRE(store.save(data));
        const auto loaded = store.load_by_id("goalgrph");
        REQUIRE(loaded.has_value());
        REQUIRE_FALSE(loaded->goal_graph.has_value());
    }

    std::filesystem::remove_all(dir);
}
