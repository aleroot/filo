#include "GoalExecutor.hpp"

#include "core/goal/GoalVerifier.hpp"

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

core::goal::CompletionFn GoalExecutor::make_completion_fn(const CommandContext& ctx) {
    auto agent = ctx.agent;
    if (!agent) {
        return {};
    }

    return [agent](std::string_view prompt) -> std::expected<std::string, std::string> {
        auto provider = agent->get_provider();
        if (!provider) {
            return std::unexpected(std::string("no active provider"));
        }

        // One-shot, history-free request: planning, judging, and reflection
        // must never contaminate (or be contaminated by) the conversation.
        core::llm::ChatRequest request;
        request.model = agent->get_active_model_name();
        request.messages.push_back({"user", std::string(prompt), "", "", {}});

        auto answer = std::make_shared<std::string>();
        auto completed = std::make_shared<std::atomic<bool>>(false);
        auto failed = std::make_shared<std::atomic<bool>>(false);
        try {
            provider->stream_response(
                request,
                [answer, completed, failed](const core::llm::StreamChunk& chunk) {
                    if (!chunk.content.empty()) {
                        *answer += chunk.content;
                    }
                    if (!chunk.is_final) {
                        return;
                    }
                    if (chunk.is_error) {
                        failed->store(true, std::memory_order_release);
                    }
                    completed->store(true, std::memory_order_release);
                });
        } catch (const std::exception& e) {
            return std::unexpected(std::string(e.what()));
        } catch (...) {
            return std::unexpected(std::string("unknown provider error"));
        }

        if (failed->load(std::memory_order_acquire)) {
            return std::unexpected(std::string("provider returned an error"));
        }
        if (!completed->load(std::memory_order_acquire)) {
            return std::unexpected(std::string("stream ended without a final response"));
        }
        std::string text(trim(*answer));
        if (text.empty()) {
            return std::unexpected(std::string("empty model response"));
        }
        return text;
    };
}

std::function<WorkOutcome(const Node&, std::string_view)>
GoalExecutor::make_work_fn(const CommandContext& ctx) {
    auto agent = ctx.agent;
    if (!agent) {
        return {};
    }

    return [agent](const Node& node, std::string_view retry_context) -> WorkOutcome {
        // A work node is one full agent turn with tools: act, don't just think.
        std::string prompt = node.directive.empty() ? node.name : node.directive;
        if (!node.acceptance.empty()) {
            prompt += "\n\nAcceptance criteria:\n" + node.acceptance;
        }
        if (!retry_context.empty()) {
            prompt += std::string(retry_context);
        }

        auto output = std::make_shared<std::string>();
        auto interrupted = std::make_shared<std::atomic<bool>>(false);

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
    const auto planned = engine->create_plan(objective);
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
