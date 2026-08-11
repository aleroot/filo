#include "ShellTool.hpp"
#include "ToolArgumentUtils.hpp"
#include "ToolNames.hpp"
#include "ToolPolicy.hpp"
#include "shell/ShellUtils.hpp"
#include "../context/SessionContext.hpp"
#include "../landrun/LandrunPolicyCompiler.hpp"
#include "../landrun/LandrunSettings.hpp"
#include "../utils/JsonUtils.hpp"
#include "../workspace/Workspace.hpp"
#include <simdjson.h>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <format>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace core::tools {

namespace {

using detail::shell_single_quote;

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

std::mutex g_active_commands_mutex;
std::unordered_map<std::string, ShellTool::ActiveCommand> g_active_commands;

void mark_session_shell_used(const std::shared_ptr<SessionShellState>& state) {
    std::lock_guard<std::mutex> lock(g_session_shells_mutex);
    state->last_used = std::chrono::steady_clock::now();
}

[[nodiscard]] std::shared_ptr<SessionShellState>
get_or_create_session_shell(
    const std::string& session_id,
    const std::filesystem::path& initial_working_dir,
    const core::landrun::LandrunPolicy& landrun_policy) {
    std::shared_ptr<SessionShellState> evicted;
    std::lock_guard<std::mutex> lock(g_session_shells_mutex);
    if (auto it = g_session_shells.find(session_id); it != g_session_shells.end()) {
        it->second->last_used = std::chrono::steady_clock::now();
        return it->second;
    }

    auto state = std::make_shared<SessionShellState>();
    state->executor->configure_landrun(landrun_policy);
    if (!initial_working_dir.empty()) {
        [[maybe_unused]] const auto ignored = state->executor->run(
            std::format("cd '{}'", shell_single_quote(initial_working_dir.string())),
            {},
            std::chrono::seconds{5});
    }
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

class ActiveCommandRegistration {
public:
    explicit ActiveCommandRegistration(ShellTool::ActiveCommand command)
        : session_id_(command.session_id)
    {
        if (session_id_.empty()) return;
        std::lock_guard<std::mutex> lock(g_active_commands_mutex);
        g_active_commands[session_id_] = std::move(command);
    }

    ~ActiveCommandRegistration() {
        if (session_id_.empty()) return;
        std::lock_guard<std::mutex> lock(g_active_commands_mutex);
        g_active_commands.erase(session_id_);
    }

    ActiveCommandRegistration(const ActiveCommandRegistration&) = delete;
    ActiveCommandRegistration& operator=(const ActiveCommandRegistration&) = delete;

private:
    std::string session_id_;
};

} // namespace

ShellTool::ShellTool()
    : ShellTool(shell::make_shell_executor()) {}

ShellTool::ShellTool(std::unique_ptr<shell::IShellExecutor> executor)
    : executor_(std::move(executor)) {
    if (!executor_) {
        executor_ = shell::make_shell_executor();
    }
}

void ShellTool::clear_mcp_session(std::string_view session_id) {
    if (session_id.empty()) return;

    {
        std::lock_guard<std::mutex> lock(g_session_shells_mutex);
        auto it = g_session_shells.find(std::string(session_id));
        if (it != g_session_shells.end()) {
            g_session_shells.erase(it);
        }
    }
    {
        std::lock_guard<std::mutex> lock(g_active_commands_mutex);
        g_active_commands.erase(std::string(session_id));
    }
}

void ShellTool::clear_session_state(std::string_view session_id) {
    clear_mcp_session(session_id);
}

bool ShellTool::interrupt_mcp_session(std::string_view session_id) {
    if (session_id.empty()) return false;

    std::shared_ptr<SessionShellState> state;
    {
        std::lock_guard<std::mutex> lock(g_session_shells_mutex);
        auto it = g_session_shells.find(std::string(session_id));
        if (it == g_session_shells.end()) return false;
        state = it->second;
    }
    if (!state || !state->executor) return false;
    return state->executor->interrupt();
}

std::vector<ShellTool::ActiveCommand> ShellTool::active_commands() {
    std::lock_guard<std::mutex> lock(g_active_commands_mutex);
    std::vector<ActiveCommand> commands;
    commands.reserve(g_active_commands.size());
    for (const auto& [_, command] : g_active_commands) {
        commands.push_back(command);
    }
    std::ranges::sort(commands, [](const ActiveCommand& lhs, const ActiveCommand& rhs) {
        return lhs.started_at < rhs.started_at;
    });
    return commands;
}

ToolDefinition ShellTool::get_definition() const {
    return {
        .name  = std::string(names::kRunTerminalCommand),
        .title = "Run Terminal Command",
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
        invocation.tool_call_id);
}

std::string ShellTool::execute_impl(
    const std::string& json_args,
    const core::context::SessionContext& context,
    std::string_view tool_call_id)
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

    // Delegate to the platform executor.
    // Working-directory subshell logic is encapsulated inside the executor so
    // that ShellTool stays platform-agnostic.
    shell::IShellExecutor::Result result;
    const auto landrun_policy = core::landrun::LandrunPolicyCompiler::compile(
        context.workspace_view(),
        core::landrun::LandrunSettings::instance().mode());
    const std::string_view session_id = context.session_id;
    if (session_id.empty()) {
        std::lock_guard<std::mutex> lock(executor_mutex_);
        executor_->configure_landrun(landrun_policy);
        result = executor_->run(command_view, working_dir, timeout);
    } else {
        ActiveCommandRegistration active_command({
            .session_id = std::string(session_id),
            .command = std::string(command_view),
            .working_dir = working_dir,
            .tool_call_id = std::string(tool_call_id),
            .started_at = std::chrono::steady_clock::now(),
        });
        const auto state = get_or_create_session_shell(
            std::string(session_id),
            context.workspace_view().primary(),
            landrun_policy);
        mark_session_shell_used(state);
        {
            std::lock_guard<std::mutex> lock(state->mutex);
            state->executor->configure_landrun(landrun_policy);
            result = state->executor->run(command_view, working_dir, timeout);
        }
        mark_session_shell_used(state);
    }

    const std::string escaped = core::utils::escape_json_string_utf8_safe(result.output);
    // Field is named "output" because both stdout and stderr are captured.
    return std::format(R"({{"output":"{}","exit_code":{}}})", escaped, result.exit_code);
}

} // namespace core::tools
