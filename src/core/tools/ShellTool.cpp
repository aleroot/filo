#include "ShellTool.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <filesystem>
#include <format>
#include <string>

namespace core::tools {

ToolDefinition ShellTool::get_definition() const {
    return {
        .name  = "run_terminal_command",
        .title = "Run Terminal Command",
        .description =
            "Executes a shell command in a persistent bash session on the user's local machine. "
            "State persists between calls: 'cd' changes the working directory for all subsequent "
            "commands, 'export' sets environment variables, and 'source' activates environments "
            "(virtualenv, nvm, etc.). "
            "stdout and stderr are merged into a single 'output' field. "
            "Provide 'working_dir' to run this specific command in a given directory without "
            "affecting the session's current directory (runs in a subshell); "
            "omit it to run in the session's current directory. "
            "Use 'timeout_seconds' for long-running commands such as compilation, test suites, "
            "or package installation; the default is 600 seconds (10 minutes). "
            "On timeout the command is killed and the session is reset automatically.",
        .parameters = {
            {"command",          "string",
             "The bash command to execute.", true},
            {"working_dir",      "string",
             "Absolute path to run this command in. Runs in a subshell so the "
             "session's working directory is unaffected. "
             "Optional; defaults to the session's current directory.", false},
            {"timeout_seconds",  "integer",
             "Maximum seconds to wait for the command to finish. "
             "Defaults to 600 (10 minutes). Increase for very long builds. "
             "On expiry the process group is killed and the session restarted.", false},
        },
        .annotations = {
            .destructive_hint = true,  // can modify filesystem, kill processes, etc.
            .open_world_hint  = true,  // can make network calls, spawn arbitrary processes
        },
    };
}

std::string ShellTool::execute(const std::string& json_args) {
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
        std::error_code ec;
        if (!std::filesystem::is_directory(std::string(wd_view), ec)) {
            return std::format(
                R"({{"error":"working_dir does not exist or is not a directory: '{}'"}})",
                core::utils::escape_json_string(std::string(wd_view)));
        }
        working_dir = std::string(wd_view);
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

    // Delegate to the platform executor.
    // Working-directory subshell logic is encapsulated inside the executor so
    // that ShellTool stays platform-agnostic.
    shell::IShellExecutor::Result result;
    {
        std::lock_guard<std::mutex> lock(executor_mutex_);
        result = executor_->run(command_view, working_dir, timeout);
    }

    const std::string escaped = core::utils::escape_json_string(result.output);
    // Field is named "output" because both stdout and stderr are captured.
    return std::format(R"({{"output":"{}","exit_code":{}}})", escaped, result.exit_code);
}

} // namespace core::tools
