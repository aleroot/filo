#include "GoalExecutor.hpp"

#include "core/goal/GoalVerifier.hpp"
#include "core/llm/OneShotCompletion.hpp"
#include "core/verification/Verification.hpp"

#include <atomic>
#include <format>
#include <memory>
#include <string>

namespace core::commands {

namespace {

using core::goal::GoalEngine;
using core::goal::GoalEvent;
using core::goal::Node;
using core::goal::RunState;
using core::goal::WorkOutcome;

constexpr std::string_view kStoppedMarker = "[Generation stopped by user]";

[[nodiscard]] std::string_view trim(std::string_view value) noexcept {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(start, end - start + 1);
}

[[nodiscard]] std::string_view describe(RunState state) noexcept {
    switch (state) {
    case RunState::Completed: return "goal completed";
    case RunState::Failed:    return "goal failed";
    case RunState::Blocked:   return "goal blocked";
    case RunState::Paused:    return "goal paused";
    case RunState::Running:   return "goal running";
    case RunState::Planning:  return "goal planning";
    case RunState::Idle:      return "goal idle";
    }
    return "goal idle";
}

/// Human-readable event line for the transcript. Only meaningful transitions
/// are surfaced; wave bookkeeping stays out of the user's way.
[[nodiscard]] std::string format_event(const GoalEvent& event) {
    switch (event.type) {
    case GoalEvent::Type::NodeStarted:
        return std::format("   ▸ {}\n", event.node_name);
    case GoalEvent::Type::VerdictIssued:
        return std::format("   ⚖ {}: {}\n", event.node_name, event.message);
    case GoalEvent::Type::ReflectionReady:
        return std::format("   ↻ reflection: {}\n", event.message);
    case GoalEvent::Type::GateAwaiting:
        return std::format("   ⏸ gate '{}' awaiting approval\n", event.node_name);
    case GoalEvent::Type::ReplanRequested:
        return "   ⟳ replan requested\n";
    case GoalEvent::Type::NodeFinished:
        return event.message.empty()
            ? std::string{}
            : std::format("   · {}: {}\n", event.node_name, event.message);
    case GoalEvent::Type::WaveBegin:
        return {};
    }
    return {};
}

} // namespace

// ---------------------------------------------------------------------------
// Hook construction
// ---------------------------------------------------------------------------

core::goal::CompletionFn
GoalExecutor::make_completion_fn(const CommandContext &ctx) {
  auto agent = ctx.agent;
  if (!agent) {
    return {};
  }

  return [agent](std::string_view prompt) {
    return core::llm::complete_once(
        agent->get_provider(), agent->get_active_model_name(), prompt,
        [agent] { return agent->is_stop_requested(); });
  };
}

std::function<WorkOutcome(const Node&, std::string_view)>
GoalExecutor::make_work_fn(const CommandContext& ctx) {
    auto agent = ctx.agent;
    if (!agent) {
        return {};
    }

    return [agent](const Node& node, std::string_view retry_context) -> WorkOutcome {
        std::string prompt = node.directive.empty() ? node.name : node.directive;
        if (!node.acceptance.empty()) {
            prompt += "\n\nAcceptance criteria:\n" + node.acceptance;
        }
        if (!retry_context.empty()) {
            prompt += std::string(retry_context);
        }

        if (node.workspace_access == core::goal::WorkspaceAccess::SharedRead) {
          const auto evidence =
              agent->run_read_only_goal_task(node.name, prompt);
          return evidence.has_value()
                     ? WorkOutcome{.ok = true, .output = *evidence}
                     : WorkOutcome{.ok = false, .output = evidence.error()};
        }

        auto output = std::make_shared<std::string>();
        auto interrupted = std::make_shared<std::atomic<bool>>(false);

        // Exclusive work is one full parent turn with tools: act, don't just
        // think. It is serialized by the graph scheduler.
        agent->clear_stop_request();
        agent->send_message(
            prompt,
            [output, interrupted](const std::string& chunk) {
                *output += chunk;
                if (chunk.find(kStoppedMarker) != std::string::npos) {
                    interrupted->store(true, std::memory_order_release);
                }
            },
            [](const std::string&, const std::string&) {},
            [] {});

        WorkOutcome outcome;
        const std::string text(trim(*output));
        if (interrupted->load(std::memory_order_acquire)
            || text.find(kStoppedMarker) != std::string::npos) {
            outcome.ok = false;
            outcome.output = "the turn was interrupted by the user";
            return outcome;
        }
        if (agent->last_turn_failed()) {
            outcome.ok = false;
            outcome.output = text.empty() ? "the agent turn failed" : text;
            return outcome;
        }
        outcome.ok = true;
        outcome.output = text;
        return outcome;
    };
}

GoalExecutor::Handle GoalExecutor::make_engine(const CommandContext& ctx) {
    GoalEngine::Hooks hooks;
    hooks.complete = make_completion_fn(ctx);
    auto verification_agent = ctx.agent;
    hooks.run_recipe = [verification_agent](std::string_view recipe_id) {
      core::goal::RecipeResult result;
      if (!verification_agent) {
        result.error = "goal agent is unavailable";
        return result;
      }
      const auto receipt =
          verification_agent->run_verification_recipe(recipe_id);
      if (!receipt.has_value()) {
        result.error = receipt.error();
        return result;
      }
      result.passed = receipt->passed();
      result.command = receipt->command;
      result.evidence = receipt->evidence;
      if (!result.passed) {
        result.error =
            std::format("{} exited {}", receipt->command, receipt->exit_code);
      }
      return result;
    };
    hooks.run_command = core::goal::make_popen_command_runner();
    hooks.run_work = make_work_fn(ctx);

    auto append = ctx.append_history_fn;
    hooks.on_event = [append](const GoalEvent& event) {
        if (!append) {
            return;
        }
        const std::string line = format_event(event);
        if (!line.empty()) {
            append(line);
        }
    };

    auto agent = ctx.agent;
    hooks.cancellation_requested = [agent] {
        return agent && agent->is_stop_requested();
    };

    return std::make_shared<GoalEngine>(std::move(hooks));
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

void GoalExecutor::plan(const CommandContext& ctx, Handle engine,
                        std::string_view objective) {
    if (!engine) {
        ctx.append_history_fn("\n✗  Goal graph is unavailable in this context.\n");
        return;
    }

    ctx.append_history_fn(std::format("\n»  Planning goal graph: {}…\n", objective));
    std::string repository_context;
    std::vector<std::string> verification_recipe_ids;
    if (ctx.agent) {
        const auto discovery = core::verification::Catalog{}.discover(
            ctx.agent->workspace_snapshot().primary());
      repository_context =
          core::verification::Catalog::render_for_prompt(discovery.recipes);
        repository_context +=
            core::verification::Catalog::render_warnings(discovery.warnings);
        verification_recipe_ids =
            core::verification::Catalog::default_quality_gate(discovery.recipes);
    }
    const auto planned = engine->create_plan(
        objective, repository_context, verification_recipe_ids);
    if (!planned.has_value()) {
        ctx.append_history_fn(std::format("\n✗  Planning failed: {}\n", planned.error()));
        return;
    }
    ctx.append_history_fn(std::format("\n✓  Plan ready.\n{}", engine->render_graph()));
    ctx.append_history_fn("   Run it with `/goal run`.\n");
}

void GoalExecutor::run(const CommandContext& ctx, Handle engine) {
    if (!engine) {
        ctx.append_history_fn("\n✗  Goal graph is unavailable in this context.\n");
        return;
    }
    if (!engine->status().has_graph) {
        ctx.append_history_fn(
            "\nℹ  No goal graph yet. Use `/goal plan <objective>` first.\n");
        return;
    }

    engine->resume();
    ctx.append_history_fn("\n»  Running goal graph…\n");
    const RunState state = engine->run();
    const auto status = engine->status();

    ctx.append_history_fn(std::format(
        "\n{}  {} — {}/{} node(s) succeeded, {} failed, {} wave(s), {} lesson(s).\n",
        state == RunState::Completed ? "✓" : (state == RunState::Paused ? "ℹ" : "✗"),
        describe(state),
        status.nodes_succeeded, status.nodes_total, status.nodes_failed,
        status.waves_executed, status.lessons_total));
    if (!status.latest_reason.empty()) {
        ctx.append_history_fn(std::format("   Last signal: {}\n", status.latest_reason));
    }
    ctx.append_history_fn(engine->render_graph());
    if (state == RunState::Failed || state == RunState::Blocked) {
        ctx.append_history_fn("   Try `/goal replan <reason>` to rebuild with lessons.\n");
    }
}

void GoalExecutor::replan(const CommandContext& ctx, Handle engine,
                          std::string_view reason) {
    if (!engine) {
        ctx.append_history_fn("\n✗  Goal graph is unavailable in this context.\n");
        return;
    }

    ctx.append_history_fn("\n»  Replanning goal graph…\n");
    const auto replanned = engine->replan(reason);
    if (!replanned.has_value()) {
        ctx.append_history_fn(std::format("\n✗  Replan failed: {}\n", replanned.error()));
        return;
    }
    ctx.append_history_fn(std::format("\n✓  New plan version ready.\n{}",
                                      engine->render_graph()));
}

std::string GoalExecutor::render(const Handle& engine) {
    if (!engine) {
        return "  Goal graph is unavailable in this context.\n";
    }
    return engine->render_graph();
}

std::string GoalExecutor::render_status(const Handle& engine) {
    if (!engine) {
        return {};
    }
    const auto status = engine->status();
    if (!status.has_graph) {
        return {};
    }
    std::string out = std::format(
        "  Graph: plan v{}, {} — {}/{} succeeded, {} failed, {} open, {} wave(s).\n",
        status.plan_version, describe(status.run_state),
        status.nodes_succeeded, status.nodes_total, status.nodes_failed,
        status.nodes_open, status.waves_executed);
    if (!status.latest_reason.empty()) {
        out += std::format("  Last signal: {}\n", status.latest_reason);
    }
    return out;
}

} // namespace core::commands
