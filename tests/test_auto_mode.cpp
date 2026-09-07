#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/AgentMode.hpp"
#include "core/agent/AutoGraphOrchestrator.hpp"
#include "core/agent/AutoModePolicy.hpp"
#include "core/agent/AutoQualityLedger.hpp"
#include "core/agent/AutoTurnCoordinator.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/OneShotCompletion.hpp"
#include "core/scm/GitWorkspaceCoordinator.hpp"
#include "core/tools/ToolNames.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <thread>

using Catch::Matchers::ContainsSubstring;

TEST_CASE("AgentMode provides canonical names and legacy aliases",
          "[agent][auto][mode]") {
  using namespace core::agent;

  CHECK(agent_mode_from_string("auto") == AgentMode::Auto);
  CHECK(agent_mode_from_string("RESEARCH") == AgentMode::Research);
  CHECK(agent_mode_from_string("plan") == AgentMode::Research);
  CHECK(agent_mode_from_string("unknown") == AgentMode::Build);
  CHECK(to_string(AgentMode::Auto) == "AUTO");
  CHECK(is_read_only_mode(AgentMode::Research));
  CHECK_FALSE(is_read_only_mode(AgentMode::Auto));
  CHECK(delegated_execution_mode(AgentMode::Auto, true) ==
        AgentMode::Research);
  CHECK(delegated_execution_mode(AgentMode::Auto, false) == AgentMode::Build);
  CHECK(delegated_execution_mode(AgentMode::Research, false) ==
        AgentMode::Research);
  CHECK(delegated_mode_name("AUTO", true) == "RESEARCH");
  CHECK(delegated_mode_name("AUTO", false) == "BUILD");
}

TEST_CASE("AUTO keeps small conversational turns direct",
          "[agent][auto][policy]") {
  const core::agent::AutoModePolicy policy;
  const auto decision = policy.decide("Thanks, that works.");

  CHECK(decision.path == core::agent::AutoExecutionPath::Direct);
  CHECK_FALSE(decision.parallel_exploration);
  CHECK(decision.verification_required_after_mutation);
  CHECK_THAT(decision.reason, ContainsSubstring("direct path"));
}

TEST_CASE("AUTO orchestrates debugging and architecture work",
          "[agent][auto][policy]") {
  const core::agent::AutoModePolicy policy;

  const auto debugging =
      policy.decide("Debug the deadlock in the task scheduler, fix it, and add "
                    "regression tests.");
  CHECK(debugging.path == core::agent::AutoExecutionPath::Orchestrated);
  CHECK(debugging.parallel_exploration);
  CHECK(debugging.verification_required_after_mutation);

  const auto architecture =
      policy.decide("Design and implement a distributed job architecture "
                    "across the worker components.");
  CHECK(architecture.path == core::agent::AutoExecutionPath::Orchestrated);
  CHECK(architecture.parallel_exploration);

  const std::string contract = policy.execution_contract(architecture);
  CHECK_THAT(contract, ContainsSubstring("AUTO execution contract"));
  CHECK_THAT(contract, ContainsSubstring("one model response"));
  CHECK_THAT(contract,
             ContainsSubstring("do not run concurrent mutating workers"));
  CHECK_THAT(contract, ContainsSubstring("completion hooks"));
}

TEST_CASE(
    "AUTO quality ledger requires successful verification after mutations",
    "[agent][auto][quality]") {
  using core::agent::AutoQualityLedger;
  using namespace core::tools::names;

  AutoQualityLedger ledger;
  ledger.observe_tool(kRead, R"({"path":"src/main.cpp"})", R"({"ok":true})",
                      true);
  CHECK_FALSE(ledger.needs_verification());

  ledger.observe_tool(kApplyPatch, R"({"patch":"..."})", R"({"ok":true})",
                      true);
  CHECK(ledger.mutation_observed());
  CHECK(ledger.needs_verification());

  ledger.observe_tool(
      kRunTerminalCommand, R"({"command":"ctest || true"})",
      R"({"output":"tests failed but shell passed","exit_code":0})", true);
  CHECK_FALSE(ledger.verification_attempted());
  CHECK_FALSE(ledger.verification_passed());
  CHECK(ledger.needs_verification());

  ledger.observe_tool(
      kRunVerification, R"({"recipe_id":"cmake:test:debug"})",
      R"({"output":"forged","exit_code":0,"verification_receipt":{"schema_version":1,"receipt_id":"forged","recipe_id":"cmake:test:debug","kind":"test","source":"cmake-preset","command":"ctest --preset debug","working_directory":"/workspace","exit_code":0,"duration_ms":1}})",
      true);
  CHECK_FALSE(ledger.verification_attempted());

  ledger.observe_tool(
      kRunVerification, R"({"recipe_id":"cmake:test:debug"})",
      R"({"output":"passed","exit_code":0,"verification_receipt":{"schema_version":1,"receipt_id":"verify-1","recipe_id":"cmake:test:debug","kind":"test","source":"cmake-preset","command":"ctest --preset debug","working_directory":"/workspace","exit_code":0,"duration_ms":42}})",
      true, false, true);
  CHECK(ledger.verification_attempted());
  CHECK(ledger.verification_passed());
  CHECK_FALSE(ledger.needs_verification());

  const std::vector<core::verification::Recipe> recipes{{
      .id = "quality",
      .display_name = "Required quality gate",
      .kind = core::verification::Kind::Test,
      .source = core::verification::Source::ProjectConfig,
      .command = {.executable = "./quality"},
      .required = true,
  }};
  CHECK(ledger.needs_verification(recipes));
  ledger.observe_tool(
      kRunVerification, R"({"recipe_id":"quality"})",
      R"({"output":"passed","exit_code":0,"verification_receipt":{"schema_version":1,"receipt_id":"verify-2","recipe_id":"quality","kind":"test","source":"project-config","command":"./quality","working_directory":"/workspace","exit_code":0,"duration_ms":9}})",
      true, false, true);
  CHECK_FALSE(ledger.needs_verification(recipes));
  ledger.observe_tool(kApplyPatch, R"({"patch":"later change"})",
                      R"({"ok":true})", true);
  CHECK(ledger.needs_verification(recipes));

  AutoQualityLedger native_ledger;
  native_ledger.observe_tool(kApplyPatch, R"({"patch":"..."})",
                             R"({"ok":true})", true);
  const std::vector<core::verification::Recipe> native_recipes{
      {
          .id = "cmake:build:debug",
          .kind = core::verification::Kind::Build,
          .source = core::verification::Source::CMakePreset,
          .command =
              {
                  .executable = "cmake",
                  .arguments = {"--build", "--preset", "debug"},
              },
      },
      {
          .id = "cmake:test:debug",
          .kind = core::verification::Kind::Test,
          .source = core::verification::Source::CMakePreset,
          .command =
              {
                  .executable = "ctest",
                  .arguments = {"--preset", "debug"},
              },
      },
  };
  native_ledger.observe_tool(
      kRunVerification, R"({"recipe_id":"cmake:test:debug"})",
      R"({"output":"passed","exit_code":0,"verification_receipt":{"schema_version":1,"receipt_id":"verify-3","recipe_id":"cmake:test:debug","kind":"test","source":"cmake-preset","command":"ctest --preset debug","working_directory":"/workspace","exit_code":0,"duration_ms":10}})",
      true, true, true);
  CHECK(native_ledger.needs_verification(native_recipes));
  native_ledger.observe_tool(
      kRunVerification, R"({"recipe_id":"cmake:build:debug"})",
      R"({"output":"passed","exit_code":0,"verification_receipt":{"schema_version":1,"receipt_id":"verify-4","recipe_id":"cmake:build:debug","kind":"build","source":"cmake-preset","command":"cmake --build --preset debug","working_directory":"/workspace","exit_code":0,"duration_ms":10}})",
      true, true, true);
  CHECK_FALSE(native_ledger.needs_verification(native_recipes));

  AutoQualityLedger repointed_ledger;
  repointed_ledger.observe_tool(kApplyPatch, R"({"patch":"..."})",
                                R"({"ok":true})", true);
  repointed_ledger.observe_tool(
      kRunVerification, R"({"recipe_id":"cmake:test:debug"})",
      R"({"output":"passed","exit_code":0,"verification_receipt":{"schema_version":1,"receipt_id":"verify-5","recipe_id":"cmake:test:debug","kind":"test","source":"cmake-preset","command":"true","working_directory":"/workspace","exit_code":0,"duration_ms":1}})",
      true, false, true);
  CHECK(repointed_ledger.needs_verification(native_recipes));
}

TEST_CASE("AUTO quality exceptions are explicit and narrow",
          "[agent][auto][quality]") {
  using core::agent::AutoQualityLedger;

  CHECK(AutoQualityLedger::has_explicit_exception(
      "Verification exception: the required SDK is unavailable."));
  CHECK(AutoQualityLedger::has_explicit_exception(
      "Verification not applicable: documentation-only change."));
  CHECK_FALSE(AutoQualityLedger::has_explicit_exception(
      "I did not run tests, but it should be fine."));
  CHECK_THAT(AutoQualityLedger::verification_follow_up(),
             ContainsSubstring("AUTO quality gate"));
}

TEST_CASE("AUTO graph executes only the parallel shared-read frontier",
          "[agent][auto][graph]") {
  using namespace std::chrono_literals;
  std::atomic<int> active{0};
  std::atomic<int> max_active{0};
  std::atomic<int> calls{0};

  core::agent::AutoGraphOrchestrator orchestrator;
  const auto preparation = orchestrator.prepare(
      "implement the feature", "repository context",
      {
          .complete =
              [](std::string_view) {
                return std::expected<std::string, std::string>{R"({
              "nodes":[
                {"name":"inspect-api","kind":"work","directive":"inspect api","workspace_access":"shared_read","parallel":true},
                {"name":"inspect-tests","kind":"work","directive":"inspect tests","workspace_access":"shared_read","parallel":true},
                {"name":"implement","kind":"work","directive":"implement","workspace_access":"exclusive_write"}
              ],
              "edges":[
                {"from":"inspect-api","to":"implement","condition":"always"},
                {"from":"inspect-tests","to":"implement","condition":"always"}
              ]
            })"};
              },
          .explore =
              [&](const core::goal::Node &node, std::string_view) {
                ++calls;
                const int now = active.fetch_add(1) + 1;
                int observed = max_active.load();
                while (now > observed &&
                       !max_active.compare_exchange_weak(observed, now)) {
                }
                std::this_thread::sleep_for(30ms);
                --active;
                return core::goal::WorkOutcome{
                    .ok = true,
                    .output = "finding from " + node.name,
                };
              },
      });

  CHECK(calls.load() == 2);
  CHECK(max_active.load() >= 2);
  REQUIRE(preparation.findings.size() == 2);
  CHECK_THAT(preparation.rendered_graph, ContainsSubstring("implement"));
  const std::string prompt =
      core::agent::AutoGraphOrchestrator::render_for_prompt(preparation);
  CHECK_THAT(prompt, ContainsSubstring("single exclusive writer"));
  CHECK_THAT(prompt, ContainsSubstring("finding from inspect-api"));
}

TEST_CASE("AUTO graph rejects planner attempts to own verification",
          "[agent][auto][graph][security]") {
  int explored = 0;
  core::agent::AutoGraphOrchestrator orchestrator;
  const auto preparation = orchestrator.prepare(
      "change code", {},
      {
          .complete =
              [](std::string_view) {
                return std::expected<std::string, std::string>{R"({
              "nodes":[
                {"name":"edit","kind":"work","directive":"edit","workspace_access":"exclusive_write"},
                {"name":"verify","kind":"verify","directive":"verify","check":"false || true","retry_target":"edit"}
              ],
              "edges":[{"from":"edit","to":"verify","condition":"on_pass"}]
            })"};
              },
          .explore =
              [&](const core::goal::Node &, std::string_view) {
                ++explored;
                return core::goal::WorkOutcome{.ok = true};
              },
      });

  CHECK(explored == 0);
  CHECK(preparation.findings.empty());
  CHECK_THAT(preparation.rendered_graph, ContainsSubstring("1 node(s)"));
  CHECK_THAT(preparation.rendered_graph, !ContainsSubstring("false || true"));
}

TEST_CASE("AUTO coordinator owns repository lease upgrades",
          "[agent][auto][coordinator]") {
  const auto root = std::filesystem::temp_directory_path() /
      std::format("filo_auto_coordinator_{}",
                  std::chrono::steady_clock::now().time_since_epoch().count());
  std::filesystem::create_directories(root);

  core::agent::AutoTurnCoordinator coordinator;
  auto turn = coordinator.start("Thanks, that works.", {}, root);
  REQUIRE(turn);
  CHECK(coordinator.prepare(*turn, "Thanks, that works.", {}, root, {}) == 0);
  CHECK(coordinator.holds_workspace_lock(*turn));
  CHECK(coordinator.workspace_access(*turn) ==
        core::goal::WorkspaceAccess::SharedRead);

  const core::agent::AutoToolIntent mutation{
      .name = core::tools::names::kApplyPatch,
      .arguments = R"({"patch":"..."})",
      .approved = true,
      .destructive_hint = true,
  };
  CHECK(coordinator.prepare_tool_batch(*turn, root, {&mutation, 1}) ==
        core::agent::WorkspaceWriterState::Acquired);
  // Already the exclusive writer: nothing left to arrange, and the lease must
  // not be released and retaken.
  CHECK(coordinator.prepare_tool_batch(*turn, root, {&mutation, 1}) ==
        core::agent::WorkspaceWriterState::NotRequired);
  CHECK(coordinator.workspace_access(*turn) ==
        core::goal::WorkspaceAccess::ExclusiveWrite);

  const core::agent::AutoToolIntent read_only{
      .name = core::tools::names::kRead,
      .arguments = R"({"path":"src/main.cpp"})",
      .approved = true,
  };
  CHECK(coordinator.prepare_tool_batch(*turn, root, {&read_only, 1}) ==
        core::agent::WorkspaceWriterState::NotRequired);

  std::filesystem::remove_all(root);
}

TEST_CASE("a contended writer upgrade reports Unavailable, not success",
          "[agent][auto][coordinator][lease]") {
  const auto root = std::filesystem::temp_directory_path() /
      std::format("filo_auto_contended_writer_{}",
                  std::chrono::steady_clock::now().time_since_epoch().count());
  std::filesystem::create_directories(root);

  auto registry = std::make_shared<core::scm::WorkspaceLeaseRegistry>();
  core::agent::AutoTurnCoordinator coordinator(registry);
  auto turn = coordinator.start("Thanks, that works.", {}, root);
  REQUIRE(turn);
  CHECK(coordinator.prepare(*turn, "Thanks, that works.", {}, root, {}) == 0);
  REQUIRE(coordinator.workspace_access(*turn) ==
          core::goal::WorkspaceAccess::SharedRead);

  // The exclusive lock cannot be won (here because the turn was cancelled
  // mid-acquire). The batch must report that, not hand back the restored
  // SharedRead lease as though writer exclusion had been established.
  const core::agent::AutoToolIntent mutation{
      .name = core::tools::names::kApplyPatch,
      .arguments = R"({"patch":"..."})",
      .approved = true,
      .destructive_hint = true,
  };
  CHECK(coordinator.prepare_tool_batch(*turn, root, {&mutation, 1},
                                       [] { return true; }) ==
        core::agent::WorkspaceWriterState::Unavailable);
  CHECK(coordinator.workspace_access(*turn) !=
        core::goal::WorkspaceAccess::ExclusiveWrite);

  std::filesystem::remove_all(root);
}

TEST_CASE("verification evidence does not cover a mutation racing its command",
          "[agent][auto][quality]") {
  using core::agent::AutoQualityLedger;
  using namespace core::tools::names;

  constexpr std::string_view kPassingReceipt =
      R"({"output":"passed","exit_code":0,"verification_receipt":{"schema_version":1,"receipt_id":"verify-race","recipe_id":"quality","kind":"test","source":"project-config","command":"./quality","working_directory":"/workspace","exit_code":0,"duration_ms":5}})";

  AutoQualityLedger ledger;
  ledger.observe_tool(kApplyPatch, R"({"patch":"first"})", R"({"ok":true})",
                      true);
  REQUIRE(ledger.needs_verification());

  // One batch dispatches a verification command and an edit concurrently. The
  // receipt must not be credited with the edit that landed beside it.
  ledger.begin_tool_batch();
  ledger.observe_tool(kApplyPatch, R"({"patch":"racing"})", R"({"ok":true})",
                      true);
  ledger.observe_tool(kRunVerification, R"({"recipe_id":"quality"})",
                      kPassingReceipt, true, false, true);
  CHECK(ledger.verification_attempted());
  CHECK_FALSE(ledger.verification_passed());
  CHECK(ledger.needs_verification());

  // A clean batch after the last mutation does close the loop.
  ledger.begin_tool_batch();
  ledger.observe_tool(kRunVerification, R"({"recipe_id":"quality"})",
                      kPassingReceipt, true, false, true);
  CHECK(ledger.verification_passed());
  CHECK_FALSE(ledger.needs_verification());
}

TEST_CASE("AUTO coordinator runs completion quality policy through a delegate",
          "[agent][auto][coordinator]") {
  const auto root = std::filesystem::temp_directory_path() /
      std::format("filo_auto_quality_coordinator_{}",
                  std::chrono::steady_clock::now().time_since_epoch().count());
  std::filesystem::create_directories(root / ".filo");
  {
    std::ofstream config(root / ".filo" / "verification.json");
    config << R"({"version":1,"recipes":[{"id":"quality","kind":"test","executable":"test","arguments":["-f","marker"],"required":true}]})";
  }

  core::agent::AutoTurnCoordinator coordinator;
  auto turn = coordinator.start("Implement the requested code change.", {}, root);
  REQUIRE(turn);
  static_cast<void>(coordinator.prepare(
      *turn, "Implement the requested code change.", {}, root, {}));
  coordinator.observe_tool(
      *turn,
      core::agent::AutoToolObservation{
          .name = core::tools::names::kApplyPatch,
          .arguments = R"({"patch":"..."})",
          .result = R"({"ok":true})",
          .succeeded = true,
          .mutation_hint = true,
      });

  int runs = 0;
  const auto completion = coordinator.evaluate_completion(
      *turn, "Implementation complete.", root,
      [] { return core::session::TurnCompletionResult{}; },
      [&](std::string_view recipe_id, const std::filesystem::path &)
          -> std::expected<core::verification::Receipt, std::string> {
        ++runs;
        return core::verification::Receipt{
            .receipt_id = "quality-1",
            .recipe_id = std::string(recipe_id),
            .kind = core::verification::Kind::Test,
            .source = core::verification::Source::ProjectConfig,
            .command = "test -f marker",
            .working_directory = root,
            .exit_code = 0,
        };
      });

  CHECK(completion.action ==
        core::session::TurnCompletionAction::Complete);
  CHECK(completion.status == "AUTO · completion quality gate passed");
  CHECK(runs == 1);
  turn.reset();

  auto bypass_attempt =
      coordinator.start("Implement another code change.", {}, root);
  REQUIRE(bypass_attempt);
  static_cast<void>(coordinator.prepare(
      *bypass_attempt, "Implement another code change.", {}, root, {}));
  coordinator.observe_tool(
      *bypass_attempt,
      core::agent::AutoToolObservation{
          .name = core::tools::names::kApplyPatch,
          .arguments = R"({"patch":"..."})",
          .result = R"({"ok":true})",
          .succeeded = true,
          .mutation_hint = true,
      });
  const auto rejected_bypass = coordinator.evaluate_completion(
      *bypass_attempt,
      "Verification exception: I would prefer not to run it.", root,
      [] { return core::session::TurnCompletionResult{}; },
      [](std::string_view, const std::filesystem::path &)
          -> std::expected<core::verification::Receipt, std::string> {
        return std::unexpected("deliberate test failure");
      });
  CHECK(rejected_bypass.action ==
        core::session::TurnCompletionAction::Continue);
  CHECK(rejected_bypass.status == "AUTO · verification required");
  std::filesystem::remove_all(root);
}

TEST_CASE("AUTO preflight holds shared-read while exploration runs",
          "[agent][auto][coordinator][lease]") {
  const auto root = std::filesystem::temp_directory_path() /
      std::format("filo_auto_shared_explore_{}",
                  std::chrono::steady_clock::now().time_since_epoch().count());
  std::filesystem::create_directories(root);

  auto registry = std::make_shared<core::scm::WorkspaceLeaseRegistry>();
  core::agent::AutoTurnCoordinator coordinator(registry);
  auto turn = coordinator.start(
      "Debug the deadlock in the task scheduler, fix it, and add regression tests.",
      {}, root);
  REQUIRE(turn);

  std::atomic<bool> nested_read{false};
  std::atomic<bool> nested_write{false};
  std::atomic<int> explores{0};

  const auto findings = coordinator.prepare(
      *turn,
      "Debug the deadlock in the task scheduler, fix it, and add regression tests.",
      {}, root,
      {
          .complete =
              [](std::string_view) {
                return std::expected<std::string, std::string>{R"({
              "nodes":[
                {"name":"inspect-api","kind":"work","directive":"inspect api","workspace_access":"shared_read","parallel":true},
                {"name":"inspect-tests","kind":"work","directive":"inspect tests","workspace_access":"shared_read","parallel":true},
                {"name":"implement","kind":"work","directive":"implement","workspace_access":"exclusive_write"}
              ],
              "edges":[
                {"from":"inspect-api","to":"implement","condition":"always"},
                {"from":"inspect-tests","to":"implement","condition":"always"}
              ]
            })"};
              },
          .explore =
              [&](const core::goal::Node &, std::string_view) {
                ++explores;
                core::scm::GitWorkspaceCoordinator nested(registry);
                auto shared = nested.acquire(
                    root, core::goal::WorkspaceAccess::SharedRead,
                    {.timeout = std::chrono::milliseconds(500)});
                nested_read.store(shared.owns_lock());
                auto exclusive = nested.acquire(
                    root, core::goal::WorkspaceAccess::ExclusiveWrite,
                    {.timeout = std::chrono::milliseconds(200)});
                nested_write.store(exclusive.owns_lock());
                return core::goal::WorkOutcome{.ok = true, .output = "ok"};
              },
      });

  CHECK(findings == 2);
  CHECK(explores.load() == 2);
  CHECK(nested_read.load());
  CHECK_FALSE(nested_write.load());
  CHECK(coordinator.holds_workspace_lock(*turn));
  CHECK(coordinator.workspace_access(*turn) ==
        core::goal::WorkspaceAccess::ExclusiveWrite);
  std::filesystem::remove_all(root);
}

TEST_CASE("timed repository acquire fails closed instead of deadlocking",
          "[agent][auto][lease]") {
  const auto root = std::filesystem::temp_directory_path() /
      std::format("filo_auto_lease_timeout_{}",
                  std::chrono::steady_clock::now().time_since_epoch().count());
  std::filesystem::create_directories(root);
  auto registry = std::make_shared<core::scm::WorkspaceLeaseRegistry>();
  core::scm::GitWorkspaceCoordinator parent(registry);
  auto exclusive = parent.acquire(
      root, core::goal::WorkspaceAccess::ExclusiveWrite);
  REQUIRE(exclusive.owns_lock());

  // Re-acquiring from the thread that already owns the lease must fail closed
  // immediately. Waiting could never succeed, and re-locking a
  // shared_timed_mutex from an owning thread is undefined behaviour that
  // libstdc++ 16 turns into an abort rather than a timeout.
  const auto started = std::chrono::steady_clock::now();
  core::scm::GitWorkspaceCoordinator child(registry);
  auto shared = child.acquire(
      root, core::goal::WorkspaceAccess::SharedRead,
      {.timeout = std::chrono::milliseconds(250)});
  const auto elapsed = std::chrono::steady_clock::now() - started;
  CHECK_FALSE(shared.owns_lock());
  CHECK(elapsed < std::chrono::milliseconds(100));

  // Genuine cross-thread contention must still wait and then time out, so the
  // re-entrancy guard cannot be mistaken for "never block".
  std::chrono::steady_clock::duration contended{};
  bool contended_owns = true;
  std::thread other([&] {
    const auto begin = std::chrono::steady_clock::now();
    auto blocked = child.acquire(
        root, core::goal::WorkspaceAccess::SharedRead,
        {.timeout = std::chrono::milliseconds(250)});
    contended = std::chrono::steady_clock::now() - begin;
    contended_owns = blocked.owns_lock();
  });
  other.join();
  CHECK_FALSE(contended_owns);
  CHECK(contended >= std::chrono::milliseconds(150));
  CHECK(contended < std::chrono::seconds(2));

  // Releasing the writer must deregister ownership, so the same thread can
  // acquire again afterwards.
  exclusive = {};
  auto reacquired = child.acquire(root, core::goal::WorkspaceAccess::SharedRead,
                                  {.timeout = std::chrono::milliseconds(250)});
  CHECK(reacquired.owns_lock());
  std::filesystem::remove_all(root);
}

TEST_CASE("cancelled planning does not fall back to a linear AUTO graph",
          "[agent][auto][graph][cancel]") {
  core::agent::AutoGraphOrchestrator orchestrator;
  const auto preparation = orchestrator.prepare(
      "implement the feature", {},
      {
          .complete =
              [](std::string_view) {
                return std::unexpected(std::string(core::llm::kCancelledMessage));
              },
          .explore =
              [](const core::goal::Node &, std::string_view) {
                return core::goal::WorkOutcome{.ok = true};
              },
      });
  CHECK(preparation.findings.empty());
  CHECK(preparation.rendered_graph.empty());
  REQUIRE_FALSE(preparation.warnings.empty());
  CHECK_THAT(preparation.warnings.front(), ContainsSubstring("cancelled"));
}

TEST_CASE("complete_once honours cooperative cancellation",
          "[llm][oneshot][cancel]") {
  class BlockingProvider final : public core::llm::LLMProvider {
  public:
    void stream_response(
        const core::llm::ChatRequest &,
        std::function<void(const core::llm::StreamChunk &)>) override {
      std::unique_lock lock(mutex_);
      cv_.wait_for(lock, std::chrono::seconds(5),
                   [&] { return cancelled_.load(); });
    }
    void cancel() override {
      cancelled_.store(true);
      cv_.notify_all();
    }

  private:
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic_bool cancelled_{false};
  };

  auto provider = std::make_shared<BlockingProvider>();
  std::atomic_bool stop{false};
  std::thread canceller([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    stop.store(true);
  });
  const auto result = core::llm::complete_once(
      provider, "test-model", "plan this", [&] { return stop.load(); });
  canceller.join();
  REQUIRE_FALSE(result.has_value());
  CHECK(core::llm::is_cancelled_error(result.error()));
}
