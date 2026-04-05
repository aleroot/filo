#include "ShellTool.hpp"
#include "../utils/JsonUtils.hpp"
#include "../workspace/Workspace.hpp"
#include <simdjson.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <unordered_map>

namespace core::tools {

namespace {

thread_local std::string g_mcp_session_id;

struct SessionShellState {
    std::unique_ptr<shell::IShellExecutor> executor{shell::make_shell_executor()};
    std::mutex mutex;
    std::chrono::steady_clock::time_point last_used{std::chrono::steady_clock::now()};

    ~SessionShellState() {
        if (!executor || !executor->is_alive()) return;
        [[maybe_unused]] const auto ignored =
            executor->run("exit 0", {}, std::chrono::seconds{2});
    }
};

std::mutex g_session_shells_mutex;
std::unordered_map<std::string, std::shared_ptr<SessionShellState>> g_session_shells;
constexpr std::size_t kMaxSessionShells = 64;

void mark_session_shell_used(const std::shared_ptr<SessionShellState>& state) {
    std::lock_guard<std::mutex> lock(g_session_shells_mutex);
    state->last_used = std::chrono::steady_clock::now();
}

[[nodiscard]] std::shared_ptr<SessionShellState>
get_or_create_session_shell(const std::string& session_id) {
    std::shared_ptr<SessionShellState> evicted;
    std::lock_guard<std::mutex> lock(g_session_shells_mutex);
    if (auto it = g_session_shells.find(session_id); it != g_session_shells.end()) {
        it->second->last_used = std::chrono::steady_clock::now();
        return it->second;
    }

    auto state = std::make_shared<SessionShellState>();
    state->last_used = std::chrono::steady_clock::now();
    g_session_shells[session_id] = state;
    if (g_session_shells.size() > kMaxSessionShells) {
        auto oldest_it = g_session_shells.begin();
        for (auto it = std::next(g_session_shells.begin()); it != g_session_shells.end(); ++it) {
            if (it->second->last_used < oldest_it->second->last_used) oldest_it = it;
        }
        evicted = oldest_it->second;
        g_session_shells.erase(oldest_it);
    }
    (void)evicted; // keep until after erase; destruction may block but happens outside map use
    return state;
}

} // namespace

ShellTool::ScopedMcpSessionContext::ScopedMcpSessionContext(std::string session_id) {
    previous_ = std::move(g_mcp_session_id);
    g_mcp_session_id = std::move(session_id);
    active_ = true;
}

ShellTool::ScopedMcpSessionContext::~ScopedMcpSessionContext() {
    if (!active_) return;
    g_mcp_session_id = std::move(previous_);
}

ShellTool::ScopedMcpSessionContext::ScopedMcpSessionContext(
    ScopedMcpSessionContext&& other) noexcept
    : active_(other.active_),
      previous_(std::move(other.previous_)) {
    other.active_ = false;
}

ShellTool::ScopedMcpSessionContext& ShellTool::ScopedMcpSessionContext::operator=(
    ScopedMcpSessionContext&& other) noexcept {
    if (this == &other) return *this;
    if (active_) g_mcp_session_id = std::move(previous_);
    active_ = other.active_;
    previous_ = std::move(other.previous_);
    other.active_ = false;
    return *this;
}

ShellTool::ScopedMcpSessionContext ShellTool::scoped_mcp_session(std::string session_id) {
    return ScopedMcpSessionContext(std::move(session_id));
}

void ShellTool::clear_mcp_session(std::string_view session_id) {
    if (session_id.empty()) return;

    {
        std::lock_guard<std::mutex> lock(g_session_shells_mutex);
        auto it = g_session_shells.find(std::string(session_id));
        if (it == g_session_shells.end()) return;
        g_session_shells.erase(it);
    }
}

std::string_view ShellTool::current_mcp_session_id() noexcept {
    return g_mcp_session_id;
}

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
        .output_schema =
            R"({"type":"object","properties":{"output":{"type":"string","description":"Combined stdout and stderr from the command."},"exit_code":{"type":"integer","description":"The command's process exit status."}},"required":["output","exit_code"],"additionalProperties":false})",
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
        if (!core::workspace::Workspace::get_instance().is_path_allowed(std::filesystem::path(wd_view))) {
            return std::format(
                R"({{"error":"Access denied: Path '{}' is outside the allowed workspace scope."}})",
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
    const std::string session_id(current_mcp_session_id());
    if (session_id.empty()) {
        std::lock_guard<std::mutex> lock(executor_mutex_);
        result = executor_->run(command_view, working_dir, timeout);
    } else {
        const auto state = get_or_create_session_shell(session_id);
        mark_session_shell_used(state);
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            result = state->executor->run(command_view, working_dir, timeout);
        }
        mark_session_shell_used(state);
    }

    const std::string escaped = core::utils::escape_json_string(result.output);
    // Field is named "output" because both stdout and stderr are captured.
    return std::format(R"({{"output":"{}","exit_code":{}}})", escaped, result.exit_code);
}

} // namespace core::tools
