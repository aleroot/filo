#pragma once

#include "CommandExecutor.hpp"
#include "core/goal/GoalEngine.hpp"

#include <memory>
#include <string>
#include <string_view>

namespace core::commands {

// ---------------------------------------------------------------------------
// GoalExecutor — binds core::goal::GoalEngine to a live Agent + TUI context.
//
// This is the composition root for the goal graph: it is the only place that
// knows how to turn engine hooks into real capabilities.
//   - complete()  -> one-shot provider call, isolated from conversation history
//                    (planner, verification judge, reflector)
//   - run_work()  -> a full agent turn with tools, blocking on the calling
//                    worker thread (same discipline as ReviewExecutor)
//   - run_command -> POSIX shell, for deterministic verification
//
// The engine instance is session-scoped and owned here, so a goal survives
// across /goal invocations within one session.
// ---------------------------------------------------------------------------
class GoalExecutor {
public:
    /// Session-scoped engine handle. Held by MainApp for the session lifetime.
    using Handle = std::shared_ptr<core::goal::GoalEngine>;

    /// Build an engine bound to @p ctx. The context is copied, so the returned
    /// engine must only be used while the agent it references is alive.
    [[nodiscard]] static Handle make_engine(const CommandContext& ctx);

    /// `/goal plan <objective>` — decompose into a DAG and show it.
    /// Blocking: call from a dispatch_async_fn worker.
    static void plan(const CommandContext& ctx, Handle engine, std::string_view objective);

    /// `/goal run` — execute waves until terminal, paused, or blocked.
    /// Blocking: call from a dispatch_async_fn worker.
    static void run(const CommandContext& ctx, Handle engine);

    /// `/goal replan [reason]` — rebuild the plan, seeded with lessons.
    /// Blocking: call from a dispatch_async_fn worker.
    static void replan(const CommandContext& ctx, Handle engine, std::string_view reason);

    /// Render `/goal graph` (non-blocking).
    [[nodiscard]] static std::string render(const Handle& engine);

    /// Render a compact one-line status for `/goal` (non-blocking).
    [[nodiscard]] static std::string render_status(const Handle& engine);

private:
    [[nodiscard]] static core::goal::CompletionFn make_completion_fn(
        const CommandContext& ctx);
    [[nodiscard]] static std::function<core::goal::WorkOutcome(
        const core::goal::Node&, std::string_view)> make_work_fn(const CommandContext& ctx);
};

} // namespace core::commands
