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
    }
    return hooks.user_prompt_submit;
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

[[nodiscard]] std::string build_stdin_prefix(std::string_view payload_json) {
    return "printf '%s' '" + shell_single_quote(payload_json) + "' | ";
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
        core::logging::warn(
            "[hooks] '{}' exited with status {}: {}",
            hook.name.empty() ? "pre_tool_use" : hook.name,
            result.exit_code,
            result.output);
        return {};
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

[[nodiscard]] core::tools::shell::IShellExecutor::Result run_hook_command(
    const core::config::HookCommandConfig& hook,
    HookEvent event,
    std::string_view payload_json,
    const core::context::SessionContext& session_context,
    const std::vector<std::pair<std::string, std::string>>& extra_env) {
    const auto working_dir = hook.working_dir.empty()
        ? std::string{}
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

        try {
            const auto result = run_hook_command(
                hook,
                HookEvent::PreToolUse,
                payload_json,
                session_context,
                extra_env);
            const auto current = interpret_pre_tool_use_result(hook, result);
            if (!current.allowed) {
                return current;
            }
            if (current.approved) {
                decision.approved = true;
            }
        } catch (const std::exception& e) {
            core::logging::warn(
                "[hooks] '{}' failed: {}",
                hook.name.empty() ? "pre_tool_use" : hook.name,
                e.what());
        } catch (...) {
            core::logging::warn(
                "[hooks] '{}' failed: unknown exception",
                hook.name.empty() ? "pre_tool_use" : hook.name);
        }
    }
    return decision;
}

} // namespace core::hooks
