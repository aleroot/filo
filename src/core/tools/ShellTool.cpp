#include "ShellTool.hpp"
#include "SteeringEnforcement.hpp"
#include "ToolArgumentUtils.hpp"
#include "ToolNames.hpp"
#include "ToolPolicy.hpp"
#include "../context/SessionContext.hpp"
#include "../landrun/LandrunPolicyCompiler.hpp"
#include "../landrun/LandrunSettings.hpp"
#include "../utils/JsonUtils.hpp"
#include "../changes/TurnChangeTracker.hpp"
#include <simdjson.h>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <format>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace core::tools {

ShellTool::ShellTool() : ShellTool(std::make_shared<shell::ShellSessions>()) {}
ShellTool::ShellTool(std::unique_ptr<shell::IShellExecutor> executor)
    : ShellTool(std::make_shared<shell::ShellSessions>(std::move(executor))) {}
ShellTool::ShellTool(std::shared_ptr<shell::ShellSessions> sessions) : sessions_(std::move(sessions)) {
    if (!sessions_) throw std::invalid_argument("ShellTool requires a session owner");
}
void ShellTool::clear_session_state(std::string_view session_id) { sessions_->clear(session_id); }

ToolDefinition ShellTool::get_definition() const {
    return {
        .name  = std::string(names::kRunTerminalCommand),
        .title = "Run",
        .description =
            "Run a local command in persistent bash. stdout and stderr are merged; check exit_code. "
            "working_dir uses a per-call subshell. Timeout defaults to 600 seconds. ",
        .parameters = {
            {"command",          "string", "Bash command.", true},
            {"working_dir",      "string",
             "Per-call directory; defaults to the persistent session directory.", false},
            {"timeout_seconds",  "integer",
             "Timeout in seconds, capped at 3600.", false},
        },
        .output_schema =
            R"({"type":"object","properties":{"output":{"type":"string","description":"Combined stdout and stderr from the command."},"exit_code":{"type":"integer","description":"The command's process exit status."}},"required":["output","exit_code"],"additionalProperties":false})",
        .annotations = {
            .destructive_hint = true,  // can modify filesystem, kill processes, etc.
            .open_world_hint  = true,  // can make network calls, spawn arbitrary processes
        },
    };
}

std::string ShellTool::execute(
    const std::string& json_args,
    const core::context::SessionContext& context)
{
    return execute_impl(json_args, context, {});
}

std::string ShellTool::execute(
    const std::string& json_args,
    const ToolInvocationContext& invocation)
{
    return execute_impl(
        json_args,
        invocation.session_context,
        invocation.tool_call_id,
        invocation.change_tracker);
}

std::string ShellTool::execute_impl(
    const std::string& json_args,
    const core::context::SessionContext& context,
    std::string_view tool_call_id,
    std::shared_ptr<core::changes::TurnChangeTracker> tracker)
{
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return R"({"error":"Invalid JSON arguments provided to shell skill."})";
    }

    std::string_view command_view;
    if (doc["command"].get(command_view) != simdjson::SUCCESS) {
        return R"({"error":"Missing 'command' argument."})";
    }

    std::string working_dir;
    std::string_view wd_view;
    if (doc["working_dir"].get(wd_view) == simdjson::SUCCESS && !wd_view.empty()) {
        std::filesystem::path resolved_path;
        if (const auto access_error = detail::check_workspace_access(
                std::filesystem::path(wd_view),
                std::string(wd_view),
                context,
                &resolved_path,
                names::kRunTerminalCommand)) {
            return *access_error;
        }
        std::error_code ec;
        if (!std::filesystem::is_directory(resolved_path, ec)) {
            return std::format(
                R"({{"error":"working_dir does not exist or is not a directory: '{}'"}})",
                core::utils::escape_json_string(std::string(wd_view)));
        }
        working_dir = resolved_path.string();
    }

    // Optional per-command timeout.
    std::chrono::milliseconds timeout = shell::IShellExecutor::kDefaultTimeout;
    int64_t timeout_secs = 0;
    if (doc["timeout_seconds"].get(timeout_secs) == simdjson::SUCCESS
            && timeout_secs > 0) {
        // Cap at 1 hour to prevent accidental infinite waits.
        timeout_secs = std::min(timeout_secs, static_cast<int64_t>(3600));
        timeout = std::chrono::milliseconds{timeout_secs * 1000};
    }

    if (const auto policy_error = core::tools::policy::enforce_command_policy(
            names::kRunTerminalCommand,
            command_view)) {
        return std::format(
            R"({{"error":"Tool policy blocked command: {}"}})",
            core::utils::escape_json_string(*policy_error));
    }
    if (const auto policy_error = core::tools::policy::enforce_url_policy(
            names::kRunTerminalCommand,
            command_view)) {
        return std::format(
            R"({{"error":"Tool policy blocked URL: {}"}})",
            core::utils::escape_json_string(*policy_error));
    }

    // The shell is the escape hatch every path gate has: `cat AGENTS.md` reads
    // a file no path-taking tool would hand over. How that is prevented depends
    // on the host — the kernel subtracts the paths when it can, and only then is
    // the command text left alone — so the decision lives in one place.
    // Relative references resolve against the directory the command runs in,
    // which is where the persistent session shell starts.
    const auto steering = SteeringEnforcement::for_context(context);
    if (const auto steering_error = steering.command_error(
            command_view,
            working_dir.empty() ? context.workspace_view().primary()
                                : std::filesystem::path(working_dir))) {
        return *steering_error;
    }

    // Delegate to the platform executor.
    // Working-directory subshell logic is encapsulated inside the executor so
    // that ShellTool stays platform-agnostic.
    const auto landrun_policy = steering.child_policy(
        core::landrun::LandrunPolicyCompiler::compile(
            context.workspace_view(),
            core::landrun::LandrunSettings::instance().mode()));
    std::function<void(int)> before_dispatch;
    if (tracker) before_dispatch = [tracker](int group) { tracker->note_process_group(group); };
    const auto result = sessions_->run({
        .session_id = context.session_id,
        .command = std::string(command_view),
        .working_dir = working_dir,
        .tool_call_id = std::string(tool_call_id),
    }, context.workspace_view().primary(), landrun_policy, timeout, before_dispatch);

    // Field is named "output" because both stdout and stderr are captured.
    // Escaped straight into the reply: output can be megabytes, and an
    // intermediate escaped copy would double the peak for nothing.
    std::string json;
    json.reserve(result.output.size() + result.output.size() / 8 + 40);
    json += R"({"output":")";
    core::utils::append_escaped_utf8_safe(json, result.output);
    std::format_to(std::back_inserter(json), R"(","exit_code":{}}})", result.exit_code);
    return json;
}

} // namespace core::tools
