#include "HookManager.hpp"
#include "../landrun/LandrunPolicyCompiler.hpp"
#include "../landrun/LandrunSettings.hpp"

#include "../config/ConfigManager.hpp"
#include "../logging/Logger.hpp"
#include "../tools/shell/ShellExecutorFactory.hpp"
#include "../tools/shell/ShellUtils.hpp"
#include "../utils/Base64.hpp"

#include <simdjson.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <optional>
#include <regex>
#include <string>
#include <thread>

namespace core::hooks {

namespace {

using core::tools::detail::shell_single_quote;

[[nodiscard]] const std::vector<core::config::HookCommandConfig>& hooks_for_event(
    const core::config::HookConfig& hooks,
    HookEvent event) {
    switch (event) {
        case HookEvent::UserPromptSubmit:
            return hooks.user_prompt_submit;
        case HookEvent::PreToolUse:
            return hooks.pre_tool_use;
        case HookEvent::PostToolUse:
            return hooks.post_tool_use;
        case HookEvent::PostToolBatch:
            return hooks.post_tool_batch;
        case HookEvent::Stop:
            return hooks.stop;
    }
    return hooks.user_prompt_submit;
}

[[nodiscard]] std::string hook_name(
    const core::config::HookCommandConfig& hook,
    HookEvent event) {
    return hook.name.empty() ? std::string(to_string(event)) : hook.name;
}

[[nodiscard]] std::optional<bool> hook_matches(
    const core::config::HookCommandConfig& hook,
    std::string_view payload_json) noexcept {
    if (hook.matcher.empty()) {
        return true;
    }
    try {
        return std::regex_search(
            payload_json.begin(), payload_json.end(), std::regex(hook.matcher));
    } catch (const std::regex_error& error) {
        core::logging::warn(
            "[hooks] '{}' has an invalid matcher '{}': {}",
            hook.name.empty() ? "unnamed" : hook.name,
            hook.matcher,
            error.what());
        return std::nullopt;
    }
}

[[nodiscard]] std::string transport_name(core::context::SessionTransport transport) {
    switch (transport) {
        case core::context::SessionTransport::cli:
            return "cli";
        case core::context::SessionTransport::mcp_stdio:
            return "mcp_stdio";
        case core::context::SessionTransport::mcp_http:
            return "mcp_http";
        case core::context::SessionTransport::mcp_client:
            return "mcp_client";
        case core::context::SessionTransport::unspecified:
            break;
    }
    return "unspecified";
}

[[nodiscard]] std::string build_env_prefix(
    HookEvent event,
    std::string_view payload_json,
    const core::context::SessionContext& session_context,
    const core::config::HookCommandConfig& hook,
    const std::vector<std::pair<std::string, std::string>>& extra_env) {
    std::vector<std::pair<std::string, std::string>> values = {
        {"FILO_HOOK_EVENT", std::string(to_string(event))},
        {"FILO_HOOK_SESSION_ID", session_context.session_id},
        {"FILO_HOOK_WORKSPACE", session_context.workspace_view().primary().string()},
        {"FILO_HOOK_TRANSPORT", transport_name(session_context.transport)},
        {"FILO_HOOK_PAYLOAD_B64", core::utils::Base64::encode(payload_json)},
    };

    for (const auto& entry : hook.env) {
        const auto pos = entry.find('=');
        if (pos == std::string::npos || pos == 0) {
            continue;
        }
        values.emplace_back(entry.substr(0, pos), entry.substr(pos + 1));
    }
    values.insert(values.end(), extra_env.begin(), extra_env.end());

    std::string command_prefix;
    for (const auto& [key, value] : values) {
        if (key.empty()) {
            continue;
        }
        command_prefix += key;
        command_prefix += "='";
        command_prefix += shell_single_quote(value);
        command_prefix += "' ";
    }
    return command_prefix;
}

// Hooks receive the event payload on stdin, but plenty of legitimate hook
// commands never read it (`exit 0`, a linter invocation, a static JSON
// decision). Once the reader exits, the payload writer's pipe has no read end
// left and the write fails with EPIPE, at which point bash reports
// "printf: write error: Broken pipe" on *stderr*. ShellSession merges stderr
// into stdout, so that diagnostic would be appended to the hook's captured
// output -- corrupting plain-text reasons and, worse, making structured JSON
// decisions unparseable so an explicit "allow"/"deny" silently degrades to "no
// decision". Silence just the writer: the payload also always reaches hooks via
// FILO_HOOK_PAYLOAD_B64, and EPIPE is the only error this printf can hit.
[[nodiscard]] std::string build_stdin_prefix(std::string_view payload_json) {
    return "{ printf '%s' '" + shell_single_quote(payload_json)
        + "' 2>/dev/null; } | ";
}

// A PreToolUse hook that could not produce a decision has decided nothing.
// Advisory hooks stay permissive, but a fail-closed policy hook must deny:
// otherwise a missing interpreter, a timeout kill, or a crashed matcher script
// silently downgrades a security control into a no-op.
[[nodiscard]] HookDecision pre_tool_use_failure(
    const core::config::HookCommandConfig& hook,
    std::string reason) {
    if (!hook.fail_closed) {
        core::logging::warn("[hooks] {}", reason);
        return {};
    }
    return {.allowed = false, .approved = false, .reason = std::move(reason)};
}

[[nodiscard]] HookDecision interpret_pre_tool_use_result(
    const core::config::HookCommandConfig& hook,
    const core::tools::shell::IShellExecutor::Result& result) {
    if (result.exit_code == 2) {
        return {
            .allowed = false,
            .approved = false,
            .reason = result.output.empty()
                ? "PreToolUse hook blocked the tool call."
                : result.output,
        };
    }

    if (result.exit_code != 0) {
        return pre_tool_use_failure(
            hook,
            std::format(
                "PreToolUse hook '{}' exited with status {}{}{}",
                hook_name(hook, HookEvent::PreToolUse),
                result.exit_code,
                result.output.empty() ? "" : ": ",
                result.output));
    }

    simdjson::dom::parser parser;
    simdjson::padded_string padded(result.output);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        return {};
    }

    bool continue_session = true;
    if (doc["continue"].get(continue_session) == simdjson::SUCCESS
        && !continue_session) {
        std::string reason = "PreToolUse hook stopped the tool call.";
        std::string_view stop_reason;
        if (doc["stopReason"].get(stop_reason) == simdjson::SUCCESS
            && !stop_reason.empty()) {
            reason = std::string(stop_reason);
        }
        return {.allowed = false, .approved = false, .reason = std::move(reason)};
    }

    simdjson::dom::object hook_output;
    if (doc["hookSpecificOutput"].get(hook_output) != simdjson::SUCCESS) {
        return {};
    }

    std::string_view decision;
    if (hook_output["permissionDecision"].get(decision) != simdjson::SUCCESS) {
        return {};
    }

    if (decision == "allow") {
        return {.allowed = true, .approved = true};
    }
    if (decision == "deny") {
        std::string reason = "PreToolUse hook denied the tool call.";
        std::string_view decision_reason;
        if (hook_output["permissionDecisionReason"].get(decision_reason) == simdjson::SUCCESS
            && !decision_reason.empty()) {
            reason = std::string(decision_reason);
        }
        return {.allowed = false, .approved = false, .reason = std::move(reason)};
    }

    return {};
}

[[nodiscard]] std::string json_string(
    simdjson::dom::element document,
    std::string_view key) {
    std::string_view value;
    if (document[key].get(value) == simdjson::SUCCESS) {
        return std::string(value);
    }
    return {};
}

[[nodiscard]] StopDecision interpret_stop_result(
    const core::config::HookCommandConfig& hook,
    const core::tools::shell::IShellExecutor::Result& result) {
    StopDecision decision{
        .quality_gate_configured = hook.quality_gate,
        .quality_gate_passed = hook.quality_gate,
    };
    const auto block = [&](std::string reason) {
        decision.complete = false;
        decision.quality_gate_passed = false;
        decision.reason = std::move(reason);
        decision.followup_message = decision.reason;
    };

    if (result.exit_code == 2) {
        block(result.output.empty()
            ? std::format("Stop hook '{}' requested another agent turn.",
                          hook_name(hook, HookEvent::Stop))
            : result.output);
        return decision;
    }
    if (result.exit_code != 0) {
        const std::string failure = std::format(
            "Stop hook '{}' exited with status {}{}{}",
            hook_name(hook, HookEvent::Stop),
            result.exit_code,
            result.output.empty() ? "" : ": ",
            result.output);
        decision.quality_gate_passed = false;
        if (hook.fail_closed) {
            block(failure);
        } else {
            core::logging::warn("[hooks] {}", failure);
        }
        return decision;
    }

    simdjson::dom::parser parser;
    simdjson::padded_string padded(result.output);
    simdjson::dom::element document;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS) {
        // Empty/plain stdout is allowed: exit status remains the portable hook
        // contract. Structured output is only needed for control flow.
        return decision;
    }

    std::string followup = json_string(document, "followup_message");
    std::string reason = json_string(document, "reason");
    if (reason.empty()) {
        reason = json_string(document, "stopReason");
    }
    std::string action = json_string(document, "decision");
    bool continue_session = true;
    const bool has_continue =
        document["continue"].get(continue_session) == simdjson::SUCCESS;
    if (action == "block" || !followup.empty()
        || (has_continue && !continue_session)) {
        if (reason.empty()) {
            reason = followup.empty()
                ? std::format("Stop hook '{}' requested another agent turn.",
                              hook_name(hook, HookEvent::Stop))
                : followup;
        }
        block(std::move(reason));
        if (!followup.empty()) {
            decision.followup_message = std::move(followup);
        }
    }
    return decision;
}

[[nodiscard]] core::tools::shell::IShellExecutor::Result run_hook_command(
    const core::config::HookCommandConfig& hook,
    HookEvent event,
    std::string_view payload_json,
    const core::context::SessionContext& session_context,
    const std::vector<std::pair<std::string, std::string>>& extra_env) {
    const auto working_dir = hook.working_dir.empty()
        ? session_context.workspace_view().primary().string()
        : session_context.resolve_path(hook.working_dir).string();
    const std::string command = build_stdin_prefix(payload_json)
        + build_env_prefix(event, payload_json, session_context, hook, extra_env)
        + hook.command;
    const int timeout_seconds = std::max(1, hook.timeout_seconds);

    auto executor = core::tools::shell::make_shell_executor();
    executor->configure_landrun(core::landrun::LandrunPolicyCompiler::compile(
        session_context.workspace_view(),
        core::landrun::LandrunSettings::instance().mode()));
    return executor->run(
        command,
        working_dir,
        std::chrono::seconds(timeout_seconds));
}

} // namespace

std::string_view to_string(HookEvent event) noexcept {
    switch (event) {
        case HookEvent::UserPromptSubmit:
            return "user_prompt_submit";
        case HookEvent::PreToolUse:
            return "pre_tool_use";
        case HookEvent::PostToolUse:
            return "post_tool_use";
        case HookEvent::PostToolBatch:
            return "post_tool_batch";
        case HookEvent::Stop:
            return "stop";
    }
    return "unknown";
}

void dispatch(HookEvent event,
              std::string payload_json,
              const core::context::SessionContext& session_context,
              std::vector<std::pair<std::string, std::string>> extra_env) {
    const auto hooks = hooks_for_event(
        core::config::ConfigManager::get_instance().get_config().hooks,
        event);
    if (hooks.empty()) {
        return;
    }

    for (const auto& hook : hooks) {
        if (!hook.enabled || hook.command.empty()) {
            continue;
        }
        if (!hook_matches(hook, payload_json).value_or(false)) {
            continue;
        }

        std::thread([hook_name = hook.name.empty() ? std::string(to_string(event)) : hook.name,
                     hook,
                     event,
                     payload_json,
                     session_context,
                     extra_env]() {
            try {
                const auto result = run_hook_command(
                    hook,
                    event,
                    payload_json,
                    session_context,
                    extra_env);
                if (result.exit_code != 0) {
                    core::logging::warn(
                        "[hooks] '{}' exited with status {}: {}",
                        hook_name,
                        result.exit_code,
                        result.output);
                }
            } catch (const std::exception& e) {
                core::logging::warn("[hooks] '{}' failed: {}", hook_name, e.what());
            } catch (...) {
                core::logging::warn("[hooks] '{}' failed: unknown exception", hook_name);
            }
        }).detach();
    }
}

HookDecision run_pre_tool_use(
    std::string payload_json,
    const core::context::SessionContext& session_context,
    std::vector<std::pair<std::string, std::string>> extra_env) {
    const auto hooks = hooks_for_event(
        core::config::ConfigManager::get_instance().get_config().hooks,
        HookEvent::PreToolUse);
    HookDecision decision;
    for (const auto& hook : hooks) {
        if (!hook.enabled || hook.command.empty()) {
            continue;
        }
        const auto matches = hook_matches(hook, payload_json);
        if (!matches.has_value() && hook.fail_closed) {
            return {
                .allowed = false,
                .approved = false,
                .reason = std::format(
                    "PreToolUse hook '{}' has an invalid matcher.",
                    hook_name(hook, HookEvent::PreToolUse)),
            };
        }
        if (!matches.value_or(false)) {
            continue;
        }

        HookDecision current;
        try {
            current = interpret_pre_tool_use_result(
                hook,
                run_hook_command(
                    hook, HookEvent::PreToolUse, payload_json,
                    session_context, extra_env));
        } catch (const std::exception& error) {
            current = pre_tool_use_failure(
                hook,
                std::format(
                    "PreToolUse hook '{}' failed: {}",
                    hook_name(hook, HookEvent::PreToolUse), error.what()));
        } catch (...) {
            current = pre_tool_use_failure(
                hook,
                std::format(
                    "PreToolUse hook '{}' failed with an unknown error.",
                    hook_name(hook, HookEvent::PreToolUse)));
        }

        if (!current.allowed) {
            return current;
        }
        if (current.approved) {
            decision.approved = true;
        }
    }
    return decision;
}

StopDecision run_stop(
    std::string payload_json,
    const core::context::SessionContext& session_context,
    std::vector<std::pair<std::string, std::string>> extra_env) {
    const auto hooks = hooks_for_event(
        core::config::ConfigManager::get_instance().get_config().hooks,
        HookEvent::Stop);
    StopDecision aggregate;
    bool every_quality_gate_passed = true;

    for (const auto& hook : hooks) {
        if (!hook.enabled || hook.command.empty()) {
            continue;
        }
        const auto matches = hook_matches(hook, payload_json);
        if (!matches.has_value() && hook.fail_closed) {
            const std::string failure = std::format(
                "Stop hook '{}' has an invalid matcher.",
                hook_name(hook, HookEvent::Stop));
            return {
                .complete = false,
                .quality_gate_configured = hook.quality_gate,
                .quality_gate_passed = false,
                .reason = failure,
                .followup_message = failure,
            };
        }
        if (!matches.value_or(false)) {
            continue;
        }
        if (hook.quality_gate) {
            aggregate.quality_gate_configured = true;
        }

        StopDecision current;
        try {
            current = interpret_stop_result(
                hook,
                run_hook_command(
                    hook, HookEvent::Stop, payload_json,
                    session_context, extra_env));
        } catch (const std::exception& error) {
            const std::string failure = std::format(
                "Stop hook '{}' failed: {}",
                hook_name(hook, HookEvent::Stop), error.what());
            current = {
                .complete = !hook.fail_closed,
                .quality_gate_configured = hook.quality_gate,
                .quality_gate_passed = false,
                .reason = failure,
                .followup_message = hook.fail_closed ? failure : std::string{},
            };
            core::logging::warn("[hooks] {}", failure);
        } catch (...) {
            const std::string failure = std::format(
                "Stop hook '{}' failed with an unknown error.",
                hook_name(hook, HookEvent::Stop));
            current = {
                .complete = !hook.fail_closed,
                .quality_gate_configured = hook.quality_gate,
                .quality_gate_passed = false,
                .reason = failure,
                .followup_message = hook.fail_closed ? failure : std::string{},
            };
            core::logging::warn("[hooks] {}", failure);
        }

        if (hook.quality_gate) {
            every_quality_gate_passed =
                every_quality_gate_passed && current.quality_gate_passed;
        }
        if (!current.complete) {
            aggregate.complete = false;
            aggregate.reason = std::move(current.reason);
            aggregate.followup_message = std::move(current.followup_message);
            aggregate.quality_gate_passed = false;
            return aggregate;
        }
    }

    aggregate.quality_gate_passed =
        aggregate.quality_gate_configured && every_quality_gate_passed;
    return aggregate;
}

core::session::TurnCompletionResult evaluate_completion(
    std::string payload_json,
    const core::context::SessionContext& session_context,
    CompletionGateState& state,
    std::vector<std::pair<std::string, std::string>> extra_env) {
    const auto decision = run_stop(
        std::move(payload_json), session_context, std::move(extra_env));
    if (decision.complete) {
        return {
            .quality_gate_satisfied =
                decision.quality_gate_configured && decision.quality_gate_passed,
        };
    }

    constexpr int kMaxBlockedAttempts = 3;
    if (state.blocked_attempts++ < kMaxBlockedAttempts) {
        std::string followup = decision.followup_message.empty()
            ? decision.reason
            : decision.followup_message;
        if (followup.empty()) {
            followup = "A completion hook requested another turn.";
        }
        return {
            .action = core::session::TurnCompletionAction::Continue,
            .status = "hook · completion blocked",
            .message = std::move(followup),
        };
    }
    return {
        .action = core::session::TurnCompletionAction::Fail,
        .message = std::format(
            "Completion hook loop limit reached: {}",
            decision.reason.empty()
                ? "the hook continued to block completion"
                : decision.reason),
    };
}

} // namespace core::hooks
