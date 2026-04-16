#include "HookManager.hpp"

#include "../config/ConfigManager.hpp"
#include "../logging/Logger.hpp"
#include "../tools/shell/ShellExecutorFactory.hpp"
#include "../tools/shell/ShellUtils.hpp"
#include "../utils/Base64.hpp"

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

        const auto working_dir = hook.working_dir.empty()
            ? std::string{}
            : session_context.resolve_path(hook.working_dir).string();
        const std::string command = build_env_prefix(
            event,
            payload_json,
            session_context,
            hook,
            extra_env) + hook.command;
        const int timeout_seconds = std::max(1, hook.timeout_seconds);

        std::thread([hook_name = hook.name.empty() ? std::string(to_string(event)) : hook.name,
                     command,
                     working_dir,
                     timeout_seconds]() {
            try {
                auto executor = core::tools::shell::make_shell_executor();
                const auto result = executor->run(
                    command,
                    working_dir,
                    std::chrono::seconds(timeout_seconds));
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

} // namespace core::hooks
