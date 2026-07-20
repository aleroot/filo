#include "ShellTool.hpp"
#include "TempFileAccessRegistry.hpp"
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
#include <set>
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

struct TempCandidateSnapshot {
    std::filesystem::path path;
    bool existed{false};
};

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

[[nodiscard]] bool is_shell_token_separator(char c) {
    return std::isspace(static_cast<unsigned char>(c))
        || c == '<'
        || c == '>'
        || c == '|'
        || c == ';'
        || c == '&'
        || c == '('
        || c == ')'
        || c == '{'
        || c == '}';
}

void append_temp_candidate_from_token(
    const std::string& token,
    std::set<std::string>& seen,
    std::vector<std::filesystem::path>& candidates)
{
    if (token.empty()) return;

    auto try_append = [&](std::string value) {
        if (value.empty() || value.front() != '/') return;
        std::filesystem::path path(value);
        const auto normalized = path.lexically_normal();
        if (!TempFileAccessRegistry::is_lexically_temp_path(normalized)
            && !TempFileAccessRegistry::is_temp_path(normalized)) {
            return;
        }
        if (seen.insert(normalized.string()).second) {
            candidates.push_back(normalized);
        }
    };

    try_append(token);

    for (const auto& prefix : TempFileAccessRegistry::temp_path_prefixes()) {
        std::size_t offset = 0;
        while ((offset = token.find(prefix, offset)) != std::string::npos) {
            std::size_t end = offset;
            while (end < token.size() && !is_shell_token_separator(token[end])) {
                ++end;
            }
            try_append(token.substr(offset, end - offset));
            offset = end;
        }
    }
}

[[nodiscard]] std::vector<std::filesystem::path> extract_temp_candidates(std::string_view command) {
    std::vector<std::filesystem::path> candidates;
    std::set<std::string> seen;
    std::string token;
    char quote = '\0';
    bool escaped = false;

    auto flush = [&] {
        append_temp_candidate_from_token(token, seen, candidates);
        token.clear();
    };

    for (char c : command) {
        if (escaped) {
            token += c;
            escaped = false;
            continue;
        }

        if (c == '\\' && quote != '\'') {
            escaped = true;
            continue;
        }

        if (quote != '\0') {
            if (c == quote) {
                quote = '\0';
            } else {
                token += c;
            }
            continue;
        }

        if (c == '\'' || c == '"') {
            quote = c;
            continue;
        }

        if (is_shell_token_separator(c)) {
            flush();
            continue;
        }

        token += c;
    }

    flush();
    return candidates;
}

[[nodiscard]] TempCandidateSnapshot snapshot_temp_candidate(const std::filesystem::path& path) {
    TempCandidateSnapshot snapshot{.path = path};

    std::error_code symlink_ec;
    const auto symlink_status = std::filesystem::symlink_status(path, symlink_ec);
    if (symlink_ec || symlink_status.type() == std::filesystem::file_type::not_found) {
        return snapshot;
    }

    snapshot.existed = true;
    return snapshot;
}

[[nodiscard]] bool temp_candidate_was_created(
    const TempCandidateSnapshot& before)
{
    if (before.existed) {
        return false;
    }

    std::error_code symlink_ec;
    const auto symlink_status = std::filesystem::symlink_status(before.path, symlink_ec);
    if (symlink_ec || symlink_status.type() == std::filesystem::file_type::symlink) {
        return false;
    }

    std::error_code status_ec;
    const auto status = std::filesystem::status(before.path, status_ec);
    if (status_ec || status.type() != std::filesystem::file_type::regular) {
        return false;
    }

    if (!TempFileAccessRegistry::is_temp_path(before.path)) {
        return false;
    }
    return true;
}

void grant_created_temp_candidates(
    TempFileAccessRegistry* temp_file_access_registry,
    std::string_view session_id,
    const std::vector<TempCandidateSnapshot>& before)
{
    if (temp_file_access_registry == nullptr) return;
    for (const auto& candidate : before) {
        if (!temp_candidate_was_created(candidate)) continue;
        temp_file_access_registry->grant_read(session_id, candidate.path);
    }
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
    : ShellTool(shell::make_shell_executor(), std::make_shared<TempFileAccessRegistry>()) {}

ShellTool::ShellTool(std::shared_ptr<TempFileAccessRegistry> temp_file_access_registry)
    : ShellTool(shell::make_shell_executor(), std::move(temp_file_access_registry)) {}

ShellTool::ShellTool(std::unique_ptr<shell::IShellExecutor> executor)
    : ShellTool(std::move(executor), std::make_shared<TempFileAccessRegistry>()) {}

ShellTool::ShellTool(
    std::unique_ptr<shell::IShellExecutor> executor,
    std::shared_ptr<TempFileAccessRegistry> temp_file_access_registry)
    : executor_(std::move(executor))
    , temp_file_access_registry_(
          temp_file_access_registry
              ? std::move(temp_file_access_registry)
              : std::make_shared<TempFileAccessRegistry>()) {
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
    if (temp_file_access_registry_) {
        temp_file_access_registry_->clear_session(session_id);
    }
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
             "Absolute or relative path to run this command in. Relative paths resolve "
             "against the effective workspace root for this MCP session. Runs in a "
             "subshell so the session's working directory is unaffected. "
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

    const auto temp_candidates = extract_temp_candidates(command_view);
    std::vector<TempCandidateSnapshot> temp_snapshots;
    temp_snapshots.reserve(temp_candidates.size());
    for (const auto& candidate : temp_candidates) {
        temp_snapshots.push_back(snapshot_temp_candidate(candidate));
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

    if (result.exit_code == 0) {
        grant_created_temp_candidates(
            temp_file_access_registry_.get(),
            context.session_id,
            temp_snapshots);
    }

    const std::string escaped = core::utils::escape_json_string(result.output);
    // Field is named "output" because both stdout and stderr are captured.
    return std::format(R"({{"output":"{}","exit_code":{}}})", escaped, result.exit_code);
}

} // namespace core::tools
