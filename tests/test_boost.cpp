#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestSessionContext.hpp"
#include "core/agent/Agent.hpp"
#include "core/agent/BoostPipeline.hpp"
#include "core/scm/EphemeralWorktree.hpp"
#include "core/commands/CommandExecutor.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/tools/ToolNames.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "core/tools/VerificationTool.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <vector>
#include <stdexcept>
#include <thread>

using namespace core::agent;
using core::session::TurnCompletionAction;
using Catch::Matchers::ContainsSubstring;

namespace {
// ConfigManager is a process-wide singleton and other suites leave subagent
// overrides in it (one disables the `explore` worker entirely). Boost depends
// on real subagents, so every Boost workspace pins configuration to a private,
// empty config home and restores it afterwards. Without this the suite passes
// or fails depending on test order.
class ScopedConfigHome {
public:
  explicit ScopedConfigHome(const std::filesystem::path &home) {
    if (const char *current = std::getenv("XDG_CONFIG_HOME")) {
      previous_ = current;
      had_previous_ = true;
    }
    std::filesystem::create_directories(home);
    ::setenv("XDG_CONFIG_HOME", home.string().c_str(), 1);
  }
  ~ScopedConfigHome() {
    if (had_previous_)
      ::setenv("XDG_CONFIG_HOME", previous_.c_str(), 1);
    else
      ::unsetenv("XDG_CONFIG_HOME");
    // Leave the singleton no dirtier than we found it.
    core::config::ConfigManager::get_instance().load();
  }
  ScopedConfigHome(const ScopedConfigHome &) = delete;
  ScopedConfigHome &operator=(const ScopedConfigHome &) = delete;

private:
  std::string previous_;
  bool had_previous_ = false;
};

struct BoostWorkspace {
  std::filesystem::path root = std::filesystem::temp_directory_path() /
      ("filo_boost_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  // Kept outside `root` so it never lands in the workspace's Git history.
  std::filesystem::path config_home{root.string() + "_xdg"};
  std::optional<ScopedConfigHome> config;

  BoostWorkspace() {
    std::filesystem::create_directories(root);
    config.emplace(config_home);
    core::config::ConfigManager::get_instance().load(root);
  }
  ~BoostWorkspace() {
    config.reset();
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::remove_all(config_home, ec);
  }
  void recipe() {
    std::filesystem::create_directories(root / ".filo");
    std::ofstream(root / ".filo/verification.json") <<
        R"({"version":1,"recipes":[{"id":"quality","kind":"test","executable":"test","arguments":["-f","marker"],"required":true}]})";
  }
  // Boost's isolation and baseline attribution both need a real checkout.
  [[nodiscard]] bool git() const {
    const std::string q = "'" + root.string() + "'";
    const std::string cfg = "-c user.email=t@filo.test -c user.name=Test -c commit.gpgsign=false";
    return std::system(("git init -q " + q + " >/dev/null 2>&1").c_str()) == 0 &&
           std::system(("cd " + q + " && echo seed > seed.txt && echo '.filo/' > .gitignore"
                        " && git add -A && git " + cfg +
                        " commit -q -m seed >/dev/null 2>&1").c_str()) == 0;
  }
};

AutoGraphOrchestrator::Hooks review_hooks(std::string verdict) {
  return {.explore = [verdict = std::move(verdict)](const auto &, auto) {
    return core::goal::WorkOutcome{.ok = true, .output = verdict};
  }};
}

void mutate(AutoTurnCoordinator &coordinator, AutoTurnState &turn) {
  coordinator.observe_tool(turn, {.name = core::tools::names::kApplyPatch,
      .result = R"({"ok":true})", .succeeded = true, .mutation_hint = true});
}

class BoostProvider final : public core::llm::LLMProvider {
public:
  std::atomic_int plans{0}, investigations{0}, reviews{0}, executions{0};
  bool reject_first = false;
  bool never_pass = false;
  bool write_correction = false;
  std::mutex mutex;
  std::vector<core::llm::ChatRequest> requests;

  void stream_response(const core::llm::ChatRequest &request,
                       std::function<void(const core::llm::StreamChunk &)> callback) override {
    {
      std::lock_guard lock(mutex);
      requests.push_back(request);
    }
    const auto has = [&](std::string_view text) {
      return std::ranges::any_of(request.messages, [&](const auto &m) { return m.content.contains(text); });
    };
    std::string response;
    if (has("You are the planning module")) {
      ++plans;
      response = "invalid plan"; // Exercise the mandatory Boost fallback perspectives.
    } else if (has("You are Filo's independent BOOST reviewer")) {
      const int round = ++reviews;
      response = never_pass || (reject_first && round == 1)
          ? R"({"verdict":"revise","evidence":"Missing boundary case at scheduler.cpp:42."})"
          : R"({"verdict":"pass","evidence":"scheduler.cpp:42 and tests/scheduler.cpp:9 cover the boundary case."})";
    } else if (has("[Delegated worker]")) {
      ++investigations;
      response = "Inspection evidence: scheduler.cpp:42 needs a boundary check.";
    } else {
      const int execution = ++executions;
      if (write_correction && (execution == 1 || execution == 3)) {
        core::llm::ToolCall call;
        call.id = "write-" + std::to_string(execution);
        call.function.name = "write_file";
        call.function.arguments = execution == 1
            ? R"({"file_path":"wrong-file","content":"not the required marker"})"
            : R"({"file_path":"marker","content":"corrected"})";
        core::llm::StreamChunk chunk;
        chunk.tools = {std::move(call)};
        chunk.is_final = true;
        callback(chunk);
        return;
      }
      response = "Investigation complete; boundary case documented.";
    }
    callback(core::llm::StreamChunk::make_content(response));
    callback(core::llm::StreamChunk::make_final());
  }
};

void send(const std::shared_ptr<Agent> &agent, const std::string &prompt,
          Agent::TurnCallbacks callbacks = {}) {
  std::mutex mutex;
  std::condition_variable cv;
  bool done = false;
  agent->send_message(prompt, [](const auto &) {}, [](const auto &, const auto &) {},
      [&] { std::lock_guard lock(mutex); done = true; cv.notify_one(); }, std::move(callbacks));
  std::unique_lock lock(mutex);
  REQUIRE(cv.wait_for(lock, std::chrono::seconds(10), [&] { return done; }));
}
} // namespace

TEST_CASE("Boost command is a precise one-turn prefix", "[boost][commands]") {
  CHECK(BoostPipeline::command_task(" /boost\tfix the race\n") == "fix the race");
  CHECK(BoostPipeline::command_task("/boost")->empty());
  CHECK_FALSE(BoostPipeline::command_task("/booster fix"));
  CHECK_FALSE(BoostPipeline::command_task("Explain /boost"));
  core::commands::CommandExecutor commands;
  std::string sent, history;
  core::commands::CommandContext ctx{
      .text = " /boost\tfix the race",
      .clear_input_fn = [] {},
      .append_history_fn = [&](const std::string &text) { history += text; },
      .send_user_message_fn = [&](const std::string &text) { sent = text; }};
  CHECK(commands.try_execute(ctx.text, ctx));
  CHECK(sent == "/boost fix the race");
  sent.clear();
  ctx.text = "/boost";
  CHECK(commands.try_execute(ctx.text, ctx));
  CHECK(sent.empty());
  CHECK_THAT(history, ContainsSubstring("Usage: /boost <task>"));
  CHECK(std::ranges::any_of(commands.describe_commands(), [](const auto &command) {
    return command.name == "/boost" && command.accepts_arguments;
  }));
}

TEST_CASE("AUTO routes difficult engineering to Boost without boosting conversation", "[boost][auto]") {
  AutoModePolicy policy;
  for (auto prompt : {"Fix the intermittent deadlock in the scheduler",
       "Investigate the root cause across subsystems", "Optimize this algorithm for adversarial input",
       "Refactor validation across multiple modules"}) {
    INFO(prompt);
    auto decision = policy.decide(prompt);
    CHECK(decision.boost);
    CHECK(decision.parallel_exploration);
    CHECK(decision.path == AutoExecutionPath::Orchestrated);
  }
  for (auto prompt : {"Thanks", "What is a race condition?", "Fix a typo in README", "Rename the local variable"}) {
    INFO(prompt);
    CHECK_FALSE(policy.decide(prompt).boost);
  }
  CHECK_FALSE(policy.decide("Fix a typo", {.history_tokens = 200000, .turn_count = 50, .has_tool_history = true}).boost);
  CHECK(policy.decide("Inspect this", {.boost_requested = true}).boost);
  CHECK(BoostPipeline::reasoning_effort("low") == "high");
  CHECK(BoostPipeline::reasoning_effort("max") == "max");
  CHECK(BoostPipeline::reasoning_effort("ultra") == "ultra");
}

TEST_CASE("Boost requires a strict independent verdict and bounds correction rounds", "[boost][review]") {
  BoostPipeline pipeline;
  auto hooks = review_hooks(R"({"verdict":"revise","evidence":"Missing stress test"})");
  CHECK(pipeline.review({.objective = "fix", .candidate = "done"}, hooks).action == TurnCompletionAction::Continue);
  CHECK(pipeline.review({.objective = "fix", .candidate = "done"}, hooks).action == TurnCompletionAction::Continue);
  auto exhausted = pipeline.review({.objective = "fix", .candidate = "done"}, hooks);
  CHECK(exhausted.action == TurnCompletionAction::Fail);
  CHECK_THAT(exhausted.message, ContainsSubstring("Missing stress test"));

  for (auto output : {"pass", R"({"verdict":"pass"})", R"({"verdict":"pass","evidence":" "})",
       R"({"verdict":"unknown","evidence":"test"})", R"({"verdict":true,"evidence":"test"})",
       R"({"verdict":"pass","verdict":"revise","evidence":"ambiguous"})"}) {
    BoostPipeline fresh;
    CHECK(fresh.review({.objective = "fix", .candidate = "done"}, review_hooks(output)).action == TurnCompletionAction::Continue);
  }
  BoostPipeline fresh;
  CHECK(fresh.review({.objective = "fix", .candidate = "done"}, review_hooks(
      R"({"verdict":"blocked","evidence":"Missing dependency"})")).action == TurnCompletionAction::Fail);
  // A mutating turn that cannot be independently reviewed must not report success.
  CHECK(fresh.review({.objective = "fix", .candidate = "done", .mutated = true}, {}).action ==
        TurnCompletionAction::Fail);
}

TEST_CASE("Boost cancellation never accepts a reviewer pass", "[boost][cancel]") {
  BoostPipeline pipeline;
  bool cancelled = false;
  AutoGraphOrchestrator::Hooks hooks{
      .explore = [&](const auto &, auto) {
        cancelled = true;
        return core::goal::WorkOutcome{.ok = true, .output = R"({"verdict":"pass","evidence":"looks good"})"};
      },
      .cancellation_requested = [&] { return cancelled; }};
  CHECK(pipeline.review({.objective = "fix", .candidate = "done"}, hooks).action == TurnCompletionAction::Fail);
}

TEST_CASE("Boost rechecks fresh mutations before independent review", "[boost][quality]") {
  BoostWorkspace workspace;
  workspace.recipe();
  AutoTurnCoordinator coordinator;
  auto turn = coordinator.start("fix", {.boost_requested = true}, workspace.root);
  int reviews = 0, checks = 0;
  std::string evidence;
  AutoGraphOrchestrator::Hooks hooks{
      .explore = [&](const auto &node, auto) {
        ++reviews;
        evidence = node.directive;
        return core::goal::WorkOutcome{.ok = true, .output = reviews == 1
            ? R"({"verdict":"revise","evidence":"Add the missing edge case"})"
            : R"({"verdict":"pass","evidence":"All cases covered by the fresh check"})"};
      }};
  static_cast<void>(coordinator.prepare(*turn, "fix", {}, workspace.root, hooks));
  auto verify = [&](std::string_view id, const std::filesystem::path &) -> std::expected<core::verification::Receipt, std::string> {
    ++checks;
    return core::verification::Receipt{.receipt_id = std::to_string(checks), .recipe_id = std::string(id),
        .kind = core::verification::Kind::Test, .source = core::verification::Source::ProjectConfig,
        .command = "test -f marker", .working_directory = workspace.root, .exit_code = 0,
        .evidence = "All regression tests pass"};
  };
  mutate(coordinator, *turn);
  CHECK(coordinator.evaluate_completion(*turn, "done", workspace.root, {}, verify).action == TurnCompletionAction::Continue);
  CHECK(checks == 1);
  mutate(coordinator, *turn);
  CHECK(coordinator.evaluate_completion(*turn, "done", workspace.root, {}, verify).action == TurnCompletionAction::Complete);
  CHECK(checks == 2);
  CHECK(reviews == 2);
  CHECK_THAT(evidence, ContainsSubstring("All regression tests pass"));
}

TEST_CASE("AUTO escalates verification failures with diagnostics then recovers", "[boost][auto][quality]") {
  BoostWorkspace workspace;
  workspace.recipe();
  AutoTurnCoordinator coordinator;
  auto turn = coordinator.start("Fix typo", {}, workspace.root);
  auto hooks = review_hooks(R"({"verdict":"pass","evidence":"The corrected file passes the required recipe"})");
  hooks.complete = [](auto) -> std::expected<std::string, std::string> { return "invalid plan"; };
  static_cast<void>(coordinator.prepare(*turn, "Fix typo", {}, workspace.root, hooks));
  CHECK_FALSE(coordinator.decision(*turn).boost);
  mutate(coordinator, *turn);
  const auto failed = coordinator.evaluate_completion(*turn, "done", workspace.root, {},
      [](auto, const std::filesystem::path &) -> std::expected<core::verification::Receipt, std::string> {
        return std::unexpected("Regression assertion failed: expected 4, got 3");
      });
  CHECK(coordinator.decision(*turn).boost);
  CHECK(failed.action == TurnCompletionAction::Continue);
  CHECK_THAT(failed.message, ContainsSubstring("expected 4, got 3"));
  CHECK_THAT(failed.message, ContainsSubstring("BOOST execution contract"));
  mutate(coordinator, *turn);
  const auto recovered = coordinator.evaluate_completion(*turn, "fixed", workspace.root,
      [] { return core::session::TurnCompletionResult{.quality_gate_satisfied = true}; }, {});
  CHECK(recovered.action == TurnCompletionAction::Complete);
  CHECK_THAT(recovered.status, ContainsSubstring("independent verification passed"));
}

TEST_CASE("Manual Boost drives the real agent loop and remains scoped to one turn", "[boost][loop]") {
  BoostWorkspace workspace;
  auto provider = std::make_shared<BoostProvider>();
  provider->reject_first = true;
  auto agent = std::make_shared<Agent>(provider, core::tools::ToolManager::get_instance(),
      test_support::make_session_context(core::workspace::WorkspaceSnapshot{
          .primary = workspace.root, .enforce = true, .version = 1}));
  agent->set_mode("BUILD");
  send(agent, "/boost inspect the scheduler");
  CHECK_FALSE(agent->last_turn_failed());
  // The first review asked for a revision, so Phase 3 fed the diagnostics back
  // into a fresh Phase 2 wave instead of only re-prompting the parent.
  CHECK(provider->plans == 2);
  CHECK(provider->investigations == 4);
  CHECK(provider->reviews == 2);
  CHECK(provider->executions == 2);
  CHECK(agent->get_mode() == "BUILD");
  for (const auto &request : provider->requests) {
    CHECK(request.effort == "high");
  }
  send(agent, "Thanks");
  CHECK(provider->plans == 2);
  CHECK(provider->reviews == 2);
  CHECK(provider->executions == 3);
  CHECK(provider->requests.back().effort != "high");
  CHECK_FALSE(provider->requests.back().prompt_plan.render().contains("BOOST execution contract"));
}

TEST_CASE("Boost respects read-only mode and explicit tool restrictions", "[boost][loop][permissions]") {
  BoostWorkspace workspace;
  auto provider = std::make_shared<BoostProvider>();
  auto agent = std::make_shared<Agent>(provider, core::tools::ToolManager::get_instance(),
      test_support::make_session_context(core::workspace::WorkspaceSnapshot{
          .primary = workspace.root, .enforce = true, .version = 1}));
  agent->set_mode("RESEARCH");
  send(agent, "/boost inspect the scheduler");
  CHECK_FALSE(agent->last_turn_failed());
  CHECK(agent->get_mode() == "RESEARCH");
  for (const auto &request : provider->requests)
    for (const auto &tool : request.tools)
      CHECK_FALSE(core::tools::names::is_write_destructive_tool(tool.function.name));
  // Excluding delegation removes the independent reviewer. A read-only turn
  // changed nothing, so it completes with a loud caveat rather than losing the
  // work; only unverified *writes* are refused (see the degradation test).
  std::string status;
  send(agent, "/boost inspect",
       {.on_status_log = [&](const std::string &text) { status += text; },
        .allowed_tools = {"read_file"}});
  CHECK_FALSE(agent->last_turn_failed());
  CHECK(provider->reviews == 1); // no second reviewer was ever consulted
  CHECK_THAT(status, ContainsSubstring("degraded"));
}

TEST_CASE("Boost exhaustion is a failed turn", "[boost][loop]") {
  BoostWorkspace workspace;
  auto provider = std::make_shared<BoostProvider>();
  provider->never_pass = true;
  auto agent = std::make_shared<Agent>(provider, core::tools::ToolManager::get_instance(),
      test_support::make_session_context(core::workspace::WorkspaceSnapshot{
          .primary = workspace.root, .enforce = true, .version = 1}));
  send(agent, "/boost inspect the scheduler");
  CHECK(agent->last_turn_failed());
  CHECK(provider->reviews == 3);
  CHECK(provider->executions == 3);
}

TEST_CASE("AUTO verification failure escalates the live agent and fixes the failing check", "[boost][loop][quality]") {
  BoostWorkspace workspace;
  workspace.recipe();
  auto provider = std::make_shared<BoostProvider>();
  provider->write_correction = true;
  auto &tools = core::tools::ToolManager::get_instance();
  tools.register_tool(std::make_shared<core::tools::WriteFileTool>());
  tools.register_tool(std::make_shared<core::tools::VerificationTool>());
  auto agent = std::make_shared<Agent>(provider, tools,
      test_support::make_session_context(core::workspace::WorkspaceSnapshot{
          .primary = workspace.root, .enforce = true, .version = 1}));
  agent->set_mode("AUTO");
  agent->set_permission_fn([](auto, auto) { return true; });
  send(agent, "Fix typo");
  CHECK_FALSE(agent->last_turn_failed());
  CHECK(provider->executions == 4);
  CHECK(provider->investigations == 2);
  CHECK(provider->reviews == 1);
  CHECK(std::filesystem::exists(workspace.root / "marker"));
  CHECK(std::ranges::any_of(provider->requests, [](const auto &request) {
    return request.effort == "high" && request.prompt_plan.render().contains("BOOST execution contract");
  }));
}

TEST_CASE("Boost honors a smaller configured parent step limit", "[boost][loop][budget]") {
  BoostWorkspace workspace;
  auto provider = std::make_shared<BoostProvider>();
  provider->reject_first = true;
  auto agent = std::make_shared<Agent>(provider, core::tools::ToolManager::get_instance(),
      test_support::make_session_context(core::workspace::WorkspaceSnapshot{
          .primary = workspace.root, .enforce = true, .version = 1}));
  agent->set_loop_limits({.max_steps_per_turn = 1});
  send(agent, "/boost inspect the scheduler");
  CHECK(agent->last_turn_failed());
  CHECK(provider->executions == 1);
  CHECK(provider->reviews == 1);
}

TEST_CASE("Missing repository checks do not escalate a small AUTO edit", "[boost][auto][quality]") {
  BoostWorkspace workspace;
  AutoTurnCoordinator coordinator;
  auto turn = coordinator.start("Fix typo", {}, workspace.root);
  auto hooks = review_hooks(R"({"verdict":"pass","evidence":"No code changed"})");
  static_cast<void>(coordinator.prepare(*turn, "Fix typo", {}, workspace.root, hooks));
  mutate(coordinator, *turn);
  const auto completion = coordinator.evaluate_completion(*turn, "done", workspace.root, {}, {});
  CHECK(completion.action == TurnCompletionAction::Continue);
  CHECK_FALSE(coordinator.decision(*turn).boost);
}


// ---------------------------------------------------------------------------
// Antigravity parity: the four documented Boost use cases must auto-activate.
// ---------------------------------------------------------------------------
TEST_CASE("AUTO activates Boost on Google's documented Boost use cases", "[boost][auto][parity]") {
  AutoModePolicy policy;
  // Verbatim from https://antigravity.google/docs/boost/ "Key use cases".
  for (auto prompt : {
           "Reproduce and fix the intermittent deadlock in the connection pool during high connection turnover.",
           "Implement a lock-free ring buffer for streaming telemetry events and write stress tests.",
           "Refactor the authentication middleware to use asynchronous token validation without breaking existing routes.",
           "Trace why HTTP request timeouts spike when batch payload size exceeds 2MB, without modifying code.",
           "Investigate the race condition in the session cache and implement a thread-safe fix with tests.",
           "Optimize the matrix transposition algorithm to use SIMD vectorization and benchmark throughput."}) {
    INFO(prompt);
    const auto decision = policy.decide(prompt);
    CHECK(decision.boost);
    CHECK(decision.parallel_exploration);
    CHECK_THAT(decision.reason, ContainsSubstring("BOOST automatically selected"));
  }
}

TEST_CASE("Boost activation needs lexical difficulty, not just derived complexity", "[boost][auto]") {
  AutoModePolicy policy;
  // Substring matches must respect word starts: "prefix"/"suffix" contain "fix".
  for (auto prompt : {"Explain the prefix handling in the parser",
                      "Document the suffix rules",
                      "Add a suffix to the generated filename",
                      "What is a deadlock and how do mutexes cause one?",
                      "Rename the local variable",
                      "Fix a typo in README"}) {
    INFO(prompt);
    CHECK_FALSE(policy.decide(prompt).boost);
  }
  // A long, tool-heavy conversation raises derived complexity; that alone must
  // never boost a trivial edit.
  CHECK_FALSE(policy.decide("Fix a typo",
                            {.history_tokens = 400000, .turn_count = 90, .has_tool_history = true})
                  .boost);
}

TEST_CASE("Boost degrades honestly when no independent reviewer exists", "[boost][review][degrade]") {
  // An investigation that changed nothing cannot have broken the workspace, so
  // the work survives with a loud caveat instead of being thrown away.
  BoostPipeline readonly_turn;
  const auto degraded = readonly_turn.review(
      {.objective = "trace the timeout", .candidate = "root cause is the retry loop", .mutated = false}, {});
  CHECK(degraded.action == TurnCompletionAction::Complete);
  CHECK_THAT(degraded.status, ContainsSubstring("degraded"));
  CHECK_THAT(degraded.status, ContainsSubstring("unverified"));

  // Unverified writes are the real risk and still fail.
  BoostPipeline writing_turn;
  const auto refused = writing_turn.review(
      {.objective = "fix it", .candidate = "done", .mutated = true}, {});
  CHECK(refused.action == TurnCompletionAction::Fail);
  CHECK_THAT(refused.message, ContainsSubstring("cannot certify workspace changes"));
}

TEST_CASE("Boost candidate rendering never implies a candidate was applied", "[boost][candidates]") {
  const std::vector<BoostCandidate> candidates{
      {.name = "BOOST candidate A", .ok = true, .verified = true,
       .patch = "--- a/x\n+++ b/x\n", .evidence = "cmake --build passed"},
      {.name = "BOOST candidate B", .ok = false, .verified = false, .evidence = "worker failed"}};
  const auto rendered = BoostPipeline::render_candidates(candidates);
  CHECK_THAT(rendered, ContainsSubstring("NOT been applied"));
  CHECK_THAT(rendered, ContainsSubstring("untrusted evidence"));
  CHECK_THAT(rendered, ContainsSubstring("checks passed"));
  CHECK_THAT(rendered, ContainsSubstring("checks not proven"));
  CHECK_THAT(rendered, ContainsSubstring("--- a/x"));
  CHECK_THAT(rendered, ContainsSubstring("no diff"));
  CHECK(BoostPipeline::render_candidates({}).empty());
}

TEST_CASE("Boost implementation workstreams run in parallel isolated worktrees", "[boost][candidates]") {
  std::atomic_int concurrent{0}, peak{0};
  AutoGraphOrchestrator::Hooks hooks;
  std::mutex names_mutex;
  std::vector<std::string> labels;
  hooks.implement = [&](const core::goal::Node &node, std::string_view label)
      -> std::optional<BoostCandidate> {
    const int now = ++concurrent;
    for (int seen = peak.load(); now > seen && !peak.compare_exchange_weak(seen, now);) {}
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    {
      std::lock_guard lock(names_mutex);
      labels.emplace_back(label);
    }
    --concurrent;
    CHECK(node.workspace_access == core::goal::WorkspaceAccess::ExclusiveWrite);
    CHECK_THAT(node.directive, ContainsSubstring("throwaway git worktree"));
    return BoostCandidate{.name = node.name, .ok = true, .patch = "diff", .evidence = "built"};
  };
  const auto candidates = BoostPipeline::implement("fix the deadlock", "findings", {}, hooks);
  REQUIRE(candidates.size() == BoostPipeline::kMaxCandidates);
  CHECK(peak == 2); // genuinely concurrent, not sequential
  CHECK(labels.size() == 2);
  // The alternative candidate must not just re-run the primary strategy.
  CHECK(candidates[0].name != candidates[1].name);

  // A worker that throws must not take the turn down with it.
  AutoGraphOrchestrator::Hooks throwing;
  throwing.implement = [](const auto &, auto) -> std::optional<BoostCandidate> {
    throw std::runtime_error("worktree unavailable");
  };
  CHECK(BoostPipeline::implement("fix", "", {}, throwing).empty());
  CHECK(BoostPipeline::implement("fix", "", {}, {}).empty());
}

TEST_CASE("A check that was already failing does not burn Boost correction rounds",
          "[boost][quality][baseline]") {
  BoostWorkspace workspace;
  if (!workspace.git())
    SKIP("git is unavailable");
  workspace.recipe();
  AutoTurnCoordinator coordinator;
  auto turn = coordinator.start("fix", {.boost_requested = true}, workspace.root);
  std::string review_evidence;
  AutoGraphOrchestrator::Hooks hooks{
      .explore = [&](const auto &node, auto) {
        review_evidence = node.directive;
        return core::goal::WorkOutcome{.ok = true,
            .output = R"({"verdict":"pass","evidence":"The objective is met; the red check predates the turn."})"};
      }};
  static_cast<void>(coordinator.prepare(*turn, "fix", {}, workspace.root, hooks));
  mutate(coordinator, *turn);

  int workspace_runs = 0, baseline_runs = 0;
  auto always_failing = [&](std::string_view id, const std::filesystem::path &root)
      -> std::expected<core::verification::Receipt, std::string> {
    (root.empty() ? workspace_runs : baseline_runs)++;
    return core::verification::Receipt{.receipt_id = "r", .recipe_id = std::string(id),
        .kind = core::verification::Kind::Test, .source = core::verification::Source::ProjectConfig,
        .command = "test -f marker", .working_directory = root.empty() ? workspace.root : root,
        .exit_code = 1, .evidence = "marker missing"};
  };
  const auto completion =
      coordinator.evaluate_completion(*turn, "done", workspace.root, {}, always_failing);
  CHECK(workspace_runs == 1);
  // The same recipe was re-run against the pristine turn-start copy, so the red
  // check is attributed to pre-existing state instead of consuming a round.
  CHECK(baseline_runs == 1);
  CHECK(completion.action == TurnCompletionAction::Complete);
  CHECK_THAT(review_evidence, ContainsSubstring("pre-existing"));
}

TEST_CASE("An ephemeral worktree isolates writes and returns only the candidate diff",
          "[boost][worktree]") {
  BoostWorkspace workspace;
  if (!workspace.git())
    SKIP("git is unavailable");
  workspace.recipe(); // gitignored project config
  // A dirty parent must be reproduced in the copy: candidates start from what
  // the user actually has, not from a stale HEAD.
  std::ofstream(workspace.root / "seed.txt") << "user edit\n";

  auto worktree = core::scm::EphemeralWorktree::create(workspace.root, "candidate A/../..");
  REQUIRE(worktree.has_value());
  REQUIRE(worktree->valid());
  const auto root = worktree->root();
  CHECK(root != workspace.root);
  CHECK(root.filename().string().starts_with("filo-boost-"));
  CHECK(std::filesystem::exists(root / "seed.txt"));
  {
    std::ifstream seeded(root / "seed.txt");
    std::string line;
    std::getline(seeded, line);
    CHECK(line == "user edit");
  }
  // Nothing has changed since seeding, so there is no candidate diff yet.
  CHECK(worktree->patch().value_or("nonempty").empty());

  // Filo's project config is conventionally gitignored, so a plain checkout
  // would omit it and the candidate could never discover a recipe to prove
  // itself against. It must be present, and must stay out of the diff.
  CHECK(std::filesystem::exists(root / ".filo/verification.json"));

  std::ofstream(root / "new_file.txt") << "candidate work\n";
  const auto patch = worktree->patch();
  REQUIRE(patch.has_value());
  CHECK_THAT(*patch, ContainsSubstring("new_file.txt"));
  CHECK_THAT(*patch, ContainsSubstring("candidate work"));
  // The user's own edit is the baseline, not part of the candidate's diff.
  CHECK_FALSE(patch->contains("user edit"));
  // Isolation: the candidate's file never appears in the real checkout.
  CHECK_FALSE(std::filesystem::exists(workspace.root / "new_file.txt"));

  worktree->release();
  CHECK_FALSE(worktree->valid());
  CHECK_FALSE(std::filesystem::exists(root));
  CHECK_FALSE(worktree->patch().has_value());
  CHECK_FALSE(core::scm::EphemeralWorktree::create(
                  std::filesystem::temp_directory_path() / "filo-not-a-repo-xyz", "x")
                  .has_value());
}


TEST_CASE("Boost candidate workstreams are opt-in per workspace", "[boost][candidates][settings]") {
  BoostWorkspace workspace;
  const auto settings = workspace.root / ".filo" / "settings.json";
  std::filesystem::create_directories(workspace.root / ".filo");

  // Absent config, a malformed file, an unrelated key and an explicit false all
  // mean "off": a large compiled project must never pay two cold gate runs by
  // surprise. Only an explicit opt-in enables them.
  CHECK_FALSE(BoostPipeline::candidates_enabled(workspace.root));
  CHECK_FALSE(BoostPipeline::candidates_enabled({}));
  CHECK_FALSE(BoostPipeline::candidates_enabled(workspace.root / "missing"));

  std::ofstream(settings) << "not json at all";
  CHECK_FALSE(BoostPipeline::candidates_enabled(workspace.root));

  std::ofstream(settings) << R"({"context_compression":"full"})";
  CHECK_FALSE(BoostPipeline::candidates_enabled(workspace.root));

  std::ofstream(settings) << R"({"boost_candidates":false})";
  CHECK_FALSE(BoostPipeline::candidates_enabled(workspace.root));

  // A non-boolean must not be coerced into an opt-in.
  std::ofstream(settings) << R"({"boost_candidates":"true"})";
  CHECK_FALSE(BoostPipeline::candidates_enabled(workspace.root));

  std::ofstream(settings) << R"({"context_compression":"full","boost_candidates":true})";
  CHECK(BoostPipeline::candidates_enabled(workspace.root));
}
