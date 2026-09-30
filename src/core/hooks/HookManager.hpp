#pragma once

#include "../context/SessionContext.hpp"
#include "../session/TurnCompletion.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace core::hooks {

enum class HookEvent {
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    PostToolBatch,
    Stop,
};

[[nodiscard]] std::string_view to_string(HookEvent event) noexcept;

struct HookDecision {
    bool allowed = true;
    bool approved = false;
    std::string reason = {};
};

// Synchronous stop hooks are completion gates rather than notifications. They
// can send the agent back for another turn and can optionally act as the
// repository's authoritative quality gate.
struct StopDecision {
    bool complete = true;
    bool quality_gate_configured = false;
    bool quality_gate_passed = false;
    std::string reason = {};
    std::string followup_message = {};
};

struct CompletionGateState {
    int blocked_attempts = 0;
};

/// Runs the hooks configured for `event` whose matcher accepts the payload,
/// each on its own detached thread, and returns how many were started. A
/// caller that must know whether anything ran behind its back — a change
/// tracker, whose summary a hook's shell command can invalidate — reads the
/// count; everyone else ignores it.
std::size_t dispatch(HookEvent event,
                     std::string payload_json,
                     const core::context::SessionContext& session_context,
                     std::vector<std::pair<std::string, std::string>> extra_env = {});

[[nodiscard]] HookDecision run_pre_tool_use(
    std::string payload_json,
    const core::context::SessionContext& session_context,
    std::vector<std::pair<std::string, std::string>> extra_env = {});

[[nodiscard]] StopDecision run_stop(
    std::string payload_json,
    const core::context::SessionContext& session_context,
    std::vector<std::pair<std::string, std::string>> extra_env = {});

// Applies the bounded retry policy for synchronous completion hooks. Agent
// transports consume the returned action but do not interpret hook semantics.
[[nodiscard]] core::session::TurnCompletionResult evaluate_completion(
    std::string payload_json,
    const core::context::SessionContext& session_context,
    CompletionGateState& state,
    std::vector<std::pair<std::string, std::string>> extra_env = {});

} // namespace core::hooks
