#include "CommandExecutor.hpp"
#include <format>
#include <thread>
#include <array>
#include <memory>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <optional>
#include <string>
#include <fstream>
#include <string_view>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <simdjson.h>
#if !defined(_WIN32) && !defined(__APPLE__)
#include <unistd.h>
#endif
#include "core/auth/AuthLogin.hpp"
#include "core/agent/RepositoryContextMessage.hpp"
#include "core/budget/BudgetTracker.hpp"
#include "core/budget/TokenUsageFormatters.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/permissions/PermissionSystem.hpp"
#include "core/session/SessionStats.hpp"
#include "core/utils/Base64.hpp"
#include "ReviewExecutor.hpp"

namespace core::commands {

namespace {

// Strip leading/trailing ASCII whitespace from a command token.
std::string_view trim(std::string_view s) {
    auto start = s.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) return {};
    auto end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

std::string join(std::string_view separator, const std::vector<std::string>& values) {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out += separator;
        out += values[i];
    }
    return out;
}

std::string to_lower_ascii(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (const unsigned char ch : input) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

std::optional<std::string_view> first_argument(std::string_view input) {
    const std::string_view trimmed = trim(input);
    const std::size_t first_space = trimmed.find_first_of(" \t\r\n");
    if (first_space == std::string_view::npos) {
        return std::nullopt;
    }

    const std::string_view remainder = trim(trimmed.substr(first_space + 1));
    if (remainder.empty()) {
        return std::nullopt;
    }

    const std::size_t end = remainder.find_first_of(" \t\r\n");
    return end == std::string_view::npos
        ? std::optional<std::string_view>{remainder}
        : std::optional<std::string_view>{remainder.substr(0, end)};
}

std::string trailing_arguments(std::string_view input) {
    const std::string_view trimmed = trim(input);
    const std::size_t first_space = trimmed.find_first_of(" \t\r\n");
    if (first_space == std::string_view::npos) {
        return {};
    }
    return std::string(trim(trimmed.substr(first_space + 1)));
}

std::vector<std::string_view> split_ascii_whitespace(std::string_view input) {
    std::vector<std::string_view> tokens;
    std::size_t cursor = 0;
    while (cursor < input.size()) {
        const std::size_t start = input.find_first_not_of(" \t\r\n", cursor);
        if (start == std::string_view::npos) {
            break;
        }
        const std::size_t end = input.find_first_of(" \t\r\n", start);
        if (end == std::string_view::npos) {
            tokens.push_back(input.substr(start));
            break;
        }
        tokens.push_back(input.substr(start, end - start));
        cursor = end + 1;
    }
    return tokens;
}

struct ShellLikeTokenizeResult {
    std::vector<std::string> tokens;
    std::string error;
};

ShellLikeTokenizeResult split_shell_like_tokens(std::string_view input) {
    ShellLikeTokenizeResult result;
    std::string current;
    bool in_single_quotes = false;
    bool in_double_quotes = false;

    auto flush_token = [&]() {
        if (!current.empty()) {
            result.tokens.push_back(std::move(current));
            current.clear();
        }
    };

    for (std::size_t i = 0; i < input.size(); ++i) {
        const char ch = input[i];

        if (in_single_quotes) {
            if (ch == '\'') {
                in_single_quotes = false;
            } else {
                current.push_back(ch);
            }
            continue;
        }

        if (in_double_quotes) {
            if (ch == '"') {
                in_double_quotes = false;
                continue;
            }
            if (ch == '\\' && i + 1 < input.size() && input[i + 1] == '"') {
                current.push_back('"');
                ++i;
                continue;
            }
            current.push_back(ch);
            continue;
        }

        if (std::isspace(static_cast<unsigned char>(ch))) {
            flush_token();
            continue;
        }
        if (ch == '\'') {
            in_single_quotes = true;
            continue;
        }
        if (ch == '"') {
            in_double_quotes = true;
            continue;
        }
        if (ch == '\\') {
            if (i + 1 < input.size()) {
                const char next = input[i + 1];
                if (std::isspace(static_cast<unsigned char>(next))
                    || next == '\''
                    || next == '"') {
                    current.push_back(next);
                    ++i;
                    continue;
                }
            }
            current.push_back('\\');
            continue;
        }

        current.push_back(ch);
    }

    if (in_single_quotes || in_double_quotes) {
        result.error = "Unterminated quote in command arguments.";
        return result;
    }

    flush_token();
    return result;
}

std::optional<std::string> parse_tools_rule_spec(std::string_view raw_spec,
                                                 std::string& error) {
    const std::string_view trimmed_spec = trim(raw_spec);
    if (trimmed_spec.empty()) {
        error = "Missing trust rule. Use `/tools allow shell <program|*>` or `/tools help`.";
        return std::nullopt;
    }

    const auto tokens = split_ascii_whitespace(trimmed_spec);
    if (tokens.empty()) {
        error = "Missing trust rule.";
        return std::nullopt;
    }

    const std::string command = to_lower_ascii(tokens.front());
    auto require_single_value = [&](std::string_view usage) -> std::optional<std::string_view> {
        if (tokens.size() != 2) {
            error = std::format("Invalid rule syntax. Usage: {}", usage);
            return std::nullopt;
        }
        return tokens[1];
    };

    auto normalize_scope = [](std::string_view raw_scope) -> std::optional<std::string> {
        const std::string scope = to_lower_ascii(raw_scope);
        if (scope == "*" || scope == "all" || scope == "any") {
            return std::string("*");
        }
        if (scope == "write" || scope == "modify" || scope == "modification"
            || scope == "modifications" || scope == "edit" || scope == "edits") {
            return std::string("write");
        }
        if (scope == "delete" || scope == "deletion" || scope == "deletions"
            || scope == "remove" || scope == "removal" || scope == "removals") {
            return std::string("delete");
        }
        if (scope == "move" || scope == "moves" || scope == "rename" || scope == "renames") {
            return std::string("move");
        }
        return std::nullopt;
    };

    if (command == "shell" || command == "terminal") {
        const auto value = require_single_value("`/tools allow shell <program|*>`");
        if (!value.has_value()) {
            return std::nullopt;
        }
        const std::string program = to_lower_ascii(*value);
        if (program.empty()) {
            error = "Shell trust rule requires a program name.";
            return std::nullopt;
        }
        return program == "*" || program == "all" || program == "any"
            ? std::optional<std::string>{"shell:*"}
            : std::optional<std::string>{std::format("shell:{}", program)};
    }

    if (command == "file" || command == "files") {
        const auto value = require_single_value("`/tools allow files <write|delete|move|*>`");
        if (!value.has_value()) {
            return std::nullopt;
        }
        const auto normalized = normalize_scope(*value);
        if (!normalized.has_value()) {
            error = "Unknown files scope. Use write, delete, move, or *.";
            return std::nullopt;
        }
        return *normalized == "*" ? std::optional<std::string>{"files:*"}
                                   : std::optional<std::string>{std::format("files:{}", *normalized)};
    }

    if (command == "tool") {
        const auto value = require_single_value("`/tools allow tool <tool_name>`");
        if (!value.has_value()) {
            return std::nullopt;
        }
        const std::string tool = to_lower_ascii(*value);
        if (tool.empty()) {
            error = "Tool trust rule requires a tool name.";
            return std::nullopt;
        }
        return std::format("tool:{}", tool);
    }

    if (command == "key") {
        const auto value = require_single_value("`/tools allow key <allow_key>`");
        if (!value.has_value()) {
            return std::nullopt;
        }
        const std::string canonical =
            core::permissions::normalize_session_allow_rule(*value);
        if (canonical.empty()) {
            error = "Allow key cannot be empty.";
            return std::nullopt;
        }
        return canonical;
    }

    if (tokens.size() == 1) {
        const std::string single_token = to_lower_ascii(tokens.front());
        if (single_token.starts_with("shell:")) {
            const std::string value = single_token.substr(6);
            if (value.empty()) {
                error = "Shell trust rule requires a program name or `*`.";
                return std::nullopt;
            }
            return value == "*"
                ? std::optional<std::string>{"shell:*"}
                : std::optional<std::string>{std::format("shell:{}", value)};
        }
        if (single_token.starts_with("files:")) {
            const auto normalized = normalize_scope(single_token.substr(6));
            if (!normalized.has_value()) {
                error = "Unknown files scope. Use write, delete, move, or *.";
                return std::nullopt;
            }
            return *normalized == "*" ? std::optional<std::string>{"files:*"}
                                       : std::optional<std::string>{std::format("files:{}", *normalized)};
        }
        if (single_token.starts_with("tool:")) {
            const std::string value = single_token.substr(5);
            if (value.empty()) {
                error = "Tool trust rule requires a tool name.";
                return std::nullopt;
            }
            return std::format("tool:{}", value);
        }

        const std::string canonical =
            core::permissions::normalize_session_allow_rule(tokens.front());
        if (canonical.empty()) {
            error = "Invalid trust rule.";
            return std::nullopt;
        }
        return canonical;
    }

    error = "Invalid trust rule syntax. Use `/tools help` for examples.";
    return std::nullopt;
}

std::vector<std::string> sorted_rule_snapshot(const ToolRuleCallbacks& tool_rules) {
    std::vector<std::string> rules;
    if (tool_rules.list) {
        rules = tool_rules.list();
    }
    std::sort(rules.begin(), rules.end());
    return rules;
}

std::string format_tool_rules(const std::vector<std::string>& rules) {
    if (rules.empty()) {
        return "\n[Tool Trust]\n  No session trust rules. Sensitive tools will prompt.\n";
    }

    std::ostringstream out;
    out << "\n[Tool Trust]\n";
    for (std::size_t i = 0; i < rules.size(); ++i) {
        out << std::format("  {:>2}. {:<20}  {}\n",
                           i + 1,
                           rules[i],
                           core::permissions::describe_session_allow_rule(rules[i]));
    }
    return out.str();
}

bool is_valid_provider_token(std::string_view value) {
    if (value.empty()) return false;
    for (const unsigned char ch : value) {
        if (std::isalnum(ch) || ch == '-' || ch == '_') continue;
        return false;
    }
    return true;
}

FILE* open_pipe_write(const char* command) {
#if defined(_WIN32)
    return _popen(command, "w");
#else
    return popen(command, "w");
#endif
}

int close_pipe(FILE* pipe) {
#if defined(_WIN32)
    return _pclose(pipe);
#else
    return pclose(pipe);
#endif
}

bool write_text_to_pipe(std::string_view command,
                        std::string_view text,
                        std::string& error) {
    const std::string command_str(command);
    FILE* pipe = open_pipe_write(command_str.c_str());
    if (!pipe) {
        error = std::format("could not execute `{}`", command_str);
        return false;
    }

    const std::size_t written = std::fwrite(text.data(), sizeof(char), text.size(), pipe);
    const int status = close_pipe(pipe);
    if (written != text.size()) {
        error = std::format("failed writing data to `{}`", command_str);
        return false;
    }
    if (status != 0) {
        error = std::format("`{}` exited with status {}", command_str, status);
        return false;
    }
    return true;
}

std::string default_export_filename() {
    const auto now = std::chrono::system_clock::now();
    const auto sec = std::chrono::floor<std::chrono::seconds>(now);
    return std::format("filo-session-{:%Y%m%d-%H%M%S}.md", sec);
}

std::string markdown_fence(std::string_view text, std::string_view language = {}) {
    std::string out;
    out.reserve(text.size() + language.size() + 16);
    out += "~~~";
    out += language;
    out += "\n";
    out += text;
    if (!text.empty() && text.back() != '\n') {
        out.push_back('\n');
    }
    out += "~~~\n";
    return out;
}

std::string trim_copy(std::string_view input) {
    return std::string(trim(input));
}

std::filesystem::path expand_user_path(std::string_view input) {
    std::string text = trim_copy(input);
    if (text == "~" || text.starts_with("~/")) {
        if (const char* home = std::getenv("HOME"); home && home[0] != '\0') {
            return std::filesystem::path(home) / text.substr(text == "~" ? 1 : 2);
        }
    }
    return std::filesystem::path(text);
}

std::string format_token_count(int32_t n) {
    return core::budget::formatters::CompactTokenCountFormatter{}.format(n);
}

std::string format_usage_cost(double usd) {
    return core::budget::formatters::UsageCostFormatter{}.format(usd);
}

std::string summarize_tool_payload(std::string_view payload) {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(payload).get(document) != simdjson::SUCCESS) {
        return trim_copy(payload);
    }

    simdjson::dom::object object;
    if (document.get(object) != simdjson::SUCCESS) {
        return trim_copy(payload);
    }

    std::string_view error;
    if (object["error"].get(error) == simdjson::SUCCESS) {
        return std::format("Error: {}", std::string(error));
    }

    std::string_view output;
    if (object["output"].get(output) == simdjson::SUCCESS) {
        return std::string(output);
    }

    return trim_copy(payload);
}

std::string role_title(std::string_view role) {
    if (role == "user") return "User";
    if (role == "assistant") return "Assistant";
    if (role == "tool") return "Tool";
    if (role == "system") return "System";
    return std::string(role);
}

[[nodiscard]] bool is_visible_transcript_message(
    const core::llm::Message& message) noexcept {
    return !core::agent::is_repository_context_message(message);
}

[[nodiscard]] std::size_t visible_transcript_message_count(
    const std::vector<core::llm::Message>& messages) {
    return static_cast<std::size_t>(
        std::ranges::count_if(messages, is_visible_transcript_message));
}

std::string export_history_markdown(const std::vector<core::llm::Message>& messages) {
    const auto now = std::chrono::system_clock::now();
    const auto sec = std::chrono::floor<std::chrono::seconds>(now);
    const std::size_t visible_message_count =
        visible_transcript_message_count(messages);

    std::string out;
    out.reserve(2048 + visible_message_count * 256);
    out += "# Filo Conversation Export\n\n";
    out += std::format("- Exported: {:%Y-%m-%d %H:%M:%S}\n", sec);
    out += std::format("- Message count: {}\n\n", visible_message_count);

    std::size_t visible_index = 0;
    for (const auto& msg : messages) {
        if (!is_visible_transcript_message(msg)) continue;
        out += std::format("## {}. {}\n\n", ++visible_index, role_title(msg.role));

        if (msg.role == "tool") {
            const std::string tool_name = msg.name.empty() ? std::string("tool") : msg.name;
            out += std::format("Tool: `{}`", tool_name);
            if (!msg.tool_call_id.empty()) {
                out += std::format("  (`{}`)", msg.tool_call_id);
            }
            out += "\n\n";
            const std::string summary = summarize_tool_payload(msg.content);
            out += markdown_fence(summary);
            out += "\n";
            continue;
        }

        if (!msg.content.empty()) {
            out += msg.content;
            out += "\n\n";
        } else {
            out += "_No text content._\n\n";
        }

        if (!msg.tool_calls.empty()) {
            out += "Tool calls:\n";
            for (const auto& call : msg.tool_calls) {
                out += std::format("- `{}`", call.function.name.empty() ? "tool" : call.function.name);
                if (!call.id.empty()) {
                    out += std::format(" (`{}`)", call.id);
                }
                out += "\n";
                if (!trim(call.function.arguments).empty()) {
                    out += markdown_fence(call.function.arguments, "json");
                }
            }
            out += "\n";
        }
    }

    return out;
}

std::string format_todos(const std::vector<core::session::SessionTodoItem>& todos) {
    std::ostringstream out;
    out << "\n[Session Todos]\n";
    if (todos.empty()) {
        out << "  No todos yet. Use `/todo add <text>` to capture the next step.\n";
        return out.str();
    }

    std::size_t index = 1;
    for (const auto& todo : todos) {
        const std::string_view marker = todo.status == core::session::TodoStatus::Completed
            ? "[x] "
            : todo.status == core::session::TodoStatus::InProgress ? "[~] " : "[ ] ";
        out << "  " << index++ << ". "
            << marker
            << todo.text;
        if (!todo.id.empty()) {
            out << "  {" << todo.id << "}";
        }
        out << '\n';
    }
    return out.str();
}

std::string format_memory_state(const core::memory::MemoryState& state) {
    std::ostringstream out;
    out << "\n[Filo Memory]\n";
    out << "  Context: " << (state.settings.enabled ? "on" : "off") << '\n';
    out << "  Auto capture: " << (state.settings.auto_capture ? "on" : "off") << '\n';
    out << "  Background review: " << (state.settings.background_review ? "on" : "off") << '\n';
    out << "  Consolidation: " << (state.settings.consolidation ? "on" : "off") << '\n';
    out << "  Skill curation: " << (state.settings.skill_curation ? "on" : "off") << '\n';
    out << "  Rate-limit reserve: " << state.settings.min_rate_limit_remaining_percent << "%\n";

    std::size_t active_count = 0;
    for (const auto& entry : state.entries) {
        if (!entry.archived) ++active_count;
    }
    if (active_count == 0) {
        out << "  No active memories. Use `/memory add <text>` to store one.\n";
        return out.str();
    }

    out << "  Active memories:\n";
    std::size_t index = 1;
    for (const auto& entry : state.entries) {
        if (entry.archived) continue;
        out << "  " << index++ << ". " << entry.content;
        if (!entry.scope.empty() && entry.scope != "global") {
            out << "  [" << entry.scope << "]";
        }
        if (!entry.id.empty()) {
            out << "  {" << entry.id << "}";
        }
        out << '\n';
    }
    return out.str();
}

std::string format_goal(const std::optional<core::session::SessionGoal>& goal) {
    std::ostringstream out;
    out << "\n[Session Goal]\n";
    if (!goal.has_value() || goal->objective.empty()) {
        out << "  No active goal. Use `/goal <objective>` to set one.\n";
        return out.str();
    }

    out << "  Objective: " << goal->objective << '\n';
    out << "  Status: " << core::session::to_string(goal->status) << '\n';
    if (!goal->note.empty()) {
        out << "  Note: " << goal->note << '\n';
    }
    if (!goal->created_at.empty()) {
        out << "  Created: " << goal->created_at << '\n';
    }
    if (!goal->updated_at.empty()) {
        out << "  Updated: " << goal->updated_at << '\n';
    }
    if (!goal->completed_at.empty()) {
        out << "  Completed: " << goal->completed_at << '\n';
    }
    return out.str();
}

std::string format_mcp_servers(const std::vector<core::config::McpServerConfig>& servers) {
    std::ostringstream out;
    out << "\n[MCP Servers]\n";
    if (servers.empty()) {
        out << "  No MCP servers configured.\n";
        return out.str();
    }

    for (const auto& server : servers) {
        out << "  - " << server.name << " [" << server.transport << "]";
        if (server.transport == "stdio") {
            out << " " << server.command;
            if (!server.args.empty()) {
                out << " " << join(" ", server.args);
            }
        } else if (!server.url.empty()) {
            out << " " << server.url;
        }
        out << '\n';
    }
    return out.str();
}

#if !defined(_WIN32) && !defined(__APPLE__)
bool is_executable_in_path(const std::filesystem::path& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path, ec) || ec) {
        return false;
    }
    if (std::filesystem::is_directory(path, ec) || ec) {
        return false;
    }
    return ::access(path.c_str(), X_OK) == 0;
}

bool command_exists(std::string_view command) {
    const std::string cmd = trim_copy(command);
    if (cmd.empty()) {
        return false;
    }

    if (cmd.find('/') != std::string::npos) {
        return is_executable_in_path(std::filesystem::path(cmd));
    }

    const char* raw_path = std::getenv("PATH");
    if (raw_path == nullptr || raw_path[0] == '\0') {
        return false;
    }

    const std::string_view path_view(raw_path);
    std::size_t start = 0;

    while (start <= path_view.size()) {
        const std::size_t end = path_view.find(':', start);
        const std::string_view segment = end == std::string_view::npos
            ? path_view.substr(start)
            : path_view.substr(start, end - start);

        const std::filesystem::path dir = segment.empty()
            ? std::filesystem::path(".")
            : std::filesystem::path(segment);
        if (is_executable_in_path(dir / cmd)) {
            return true;
        }

        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }

    return false;
}

std::optional<std::string> copy_via_osc52(std::string_view text) {
    std::ofstream tty("/dev/tty", std::ios::binary | std::ios::out);
    if (!tty) {
        return std::string("failed to open /dev/tty for OSC52 copy");
    }

    const std::string payload = core::utils::Base64::encode(text);
    const bool inside_tmux = std::getenv("TMUX") != nullptr;
    if (inside_tmux) {
        tty << "\x1bPtmux;\x1b\x1b]52;c;" << payload << "\x07\x1b\\";
    } else {
        tty << "\x1b]52;c;" << payload << "\x07";
    }
    tty.flush();
    if (!tty.good()) {
        return std::string("failed to write OSC52 escape sequence");
    }
    return std::nullopt;
}
#endif


void wait_for_enter() {
    std::cout << "\nPress Enter to return to Filo...";
    std::cout.flush();
    std::string line;
    std::getline(std::cin, line);
}

std::optional<std::string> choose_review_menu_request() {
    while (true) {
        std::cout
            << "\nReview Targets\n"
            << "  1. Uncommitted changes\n"
            << "  2. Base branch\n"
            << "  3. Customised\n"
            << "Select an option [1-3] or press Enter to cancel: ";
        std::cout.flush();

        std::string line;
        if (!std::getline(std::cin, line)) {
            return std::nullopt;
        }

        const std::string_view input = trim(line);
        if (input.empty()) {
            return std::nullopt;
        }

        if (input == "1") {
            return std::string{};
        }
        if (input == "2") {
            std::cout << "Base branch: ";
            std::cout.flush();

            std::string branch;
            if (!std::getline(std::cin, branch)) {
                return std::nullopt;
            }
            const std::string_view trimmed_branch = trim(branch);
            if (trimmed_branch.empty()) {
                std::cout << "Base branch cannot be empty.\n";
                continue;
            }
            return std::format("--base {}", std::string(trimmed_branch));
        }
        if (input == "3") {
            std::cout << "Custom review prompt: ";
            std::cout.flush();

            std::string prompt;
            if (!std::getline(std::cin, prompt)) {
                return std::nullopt;
            }
            const std::string_view trimmed_prompt = trim(prompt);
            if (trimmed_prompt.empty()) {
                std::cout << "Custom review prompt cannot be empty.\n";
                continue;
            }
            return std::string(trimmed_prompt);
        }

        std::cout << "Invalid selection. Choose 1, 2, or 3.\n";
    }
}

void run_provider_auth(const std::string& provider,
                       const std::string& config_dir,
                       const std::function<void(std::function<void()>)>& suspend_fn,
                       const std::function<void(const std::string&)>& append_fn,
                       const std::function<std::string(std::string_view)>& switch_model_fn = {}) {
    struct AuthState {
        bool success = false;
        bool completed = false;
        std::string provider;
        std::vector<std::string> hints;
        bool profile_persisted = false;
        std::string selected_provider;
        std::string profile_error;
        std::string model_switch_message;
        std::string error;
    };

    auto state = std::make_shared<AuthState>();
    suspend_fn([provider, config_dir, state, switch_model_fn]() {
        try {
            const auto outcome = core::auth::login_and_persist(
                provider, config_dir);

            std::cout << "\nAuthentication successful!\n";
            std::cout << "Provider: " << outcome.result.provider << "\n";
            std::cout << "Credentials saved in: " << config_dir << "\n";

            if (outcome.profile_persisted) {
                state->profile_persisted = true;
                state->selected_provider = outcome.selected_provider;
                std::cout << "Default provider switched to: "
                          << state->selected_provider << "\n";
            } else {
                state->profile_error = outcome.profile_error;
                std::cout << "Warning: authenticated, but couldn't update default profile: "
                          << state->profile_error << "\n";
            }

            if (state->profile_persisted && switch_model_fn) {
                state->model_switch_message = switch_model_fn(state->selected_provider);
            }

            for (const auto& hint : outcome.result.hints) {
                std::cout << "Hint: " << hint << "\n";
            }
            wait_for_enter();

            state->success = true;
            state->provider = outcome.result.provider;
            state->hints = outcome.result.hints;
            state->completed = true;
        } catch (const std::exception& e) {
            std::cerr << "\nAuthentication failed: " << e.what() << "\n";
            wait_for_enter();

            state->success = false;
            state->error = e.what();
            state->completed = true;
        }
    });

    if (!state->completed) {
        append_fn(
            "\n\xe2\x9a\xa0  Authentication started. Check terminal output for progress.\n");
        return;
    }

    if (!state->success) {
        append_fn(std::format(
            "\n\xe2\x9c\x97  Authentication failed: {}\n",
            state->error.empty() ? std::string("Unknown error.") : state->error));
        return;
    }

    append_fn(std::format(
        "\n\xe2\x9c\x93  Authenticated with {}. Credentials are now saved for reuse.\n",
        state->provider.empty() ? provider : state->provider));
    if (state->profile_persisted) {
        append_fn(std::format(
            "\xe2\x9c\x93  Selected provider is now {}.\n",
            state->selected_provider));
    } else if (!state->profile_error.empty()) {
        append_fn(std::format(
            "\xe2\x9a\xa0  Auth succeeded, but profile update failed: {}\n",
            state->profile_error));
    }
    for (const auto& hint : state->hints) {
        append_fn(std::format("\xe2\x84\xb9  {}\n", hint));
    }
    if (!state->model_switch_message.empty()) {
        append_fn(std::format("\xe2\x84\xb9  {}\n", state->model_switch_message));
    }
}

} // namespace

std::optional<std::string> copy_text_to_clipboard(std::string_view text) {
    if (text.empty()) {
        return std::string("nothing to copy");
    }

    std::string error;

#if defined(_WIN32)
    if (write_text_to_pipe("clip", text, error)) {
        return std::nullopt;
    }
    return error;
#elif defined(__APPLE__)
    if (write_text_to_pipe("pbcopy", text, error)) {
        return std::nullopt;
    }
    return error;
#else
    const bool in_wayland = std::getenv("WAYLAND_DISPLAY") != nullptr;
    if (in_wayland && command_exists("wl-copy") && write_text_to_pipe("wl-copy", text, error)) {
        return std::nullopt;
    }
    if (command_exists("xclip")
        && write_text_to_pipe("xclip -selection clipboard", text, error)) {
        return std::nullopt;
    }
    if (command_exists("xsel")
        && write_text_to_pipe("xsel --clipboard --input", text, error)) {
        return std::nullopt;
    }

    if (const auto osc52_error = copy_via_osc52(text); !osc52_error.has_value()) {
        return std::nullopt;
    } else {
        if (error.empty()) {
            return *osc52_error;
        }
        return std::format("{}; {}", error, *osc52_error);
    }
#endif
}

std::optional<ActiveCommandToken> find_active_command(std::string_view input,
                                                      std::size_t cursor) {
    cursor = std::min(cursor, input.size());

    const std::size_t start = input.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos || input[start] != '/') {
        return std::nullopt;
    }

    std::size_t end = start + 1;
    while (end < input.size() && !std::isspace(static_cast<unsigned char>(input[end]))) {
        ++end;
    }

    if (cursor < start || cursor > end) {
        return std::nullopt;
    }

    return ActiveCommandToken{
        .replace_begin = start,
        .replace_end = end,
        .token = std::string(input.substr(start, end - start)),
    };
}

CommandCompletion apply_command_completion(std::string_view input,
                                           const ActiveCommandToken& active,
                                           std::string_view replacement) {
    CommandCompletion out;
    out.text.reserve(input.size() + replacement.size() + 4);
    out.text.append(input.substr(0, active.replace_begin));
    out.text.append(replacement);
    out.text.append(input.substr(std::min(active.replace_end, input.size())));
    out.cursor = active.replace_begin + replacement.size();
    return out;
}

// ---------------------------------------------------------
// Built-in Commands
// ---------------------------------------------------------

class AuthCommand : public Command {
public:
    std::string get_name() const override { return "/auth"; }
    std::vector<std::string> get_aliases() const override { return {"/login"}; }
    std::string get_description() const override { return "Interactive authentication wizard (use /auth or /auth <provider>)"; }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        const std::string config_dir = core::config::ConfigManager::get_instance().get_config_dir();
        const auto available = core::auth::available_login_providers(config_dir);

        const bool interactive_mode = static_cast<bool>(ctx.suspend_tui_fn);
        const auto arg = first_argument(ctx.text);
        if (!arg.has_value()) {
            if (!interactive_mode) {
                ctx.append_history_fn(std::format(
                    "\n\xe2\x84\xb9  Usage: /auth [provider] (e.g., /auth claude, /auth google)\n"
                    "\xe2\x84\xb9  Available providers: {}\n",
                    available.empty() ? std::string("<none>") : join(", ", available)));
                return;
            }
            if (available.empty()) {
                ctx.append_history_fn("\n\xe2\x9c\x97  No authentication providers are currently available.\n");
                return;
            }

            if (ctx.open_provider_picker_fn) {
                auto append_fn  = ctx.append_history_fn;
                auto suspend_fn = ctx.suspend_tui_fn;
                auto switch_model_fn = ctx.switch_model_fn;
                ctx.open_provider_picker_fn(available, [append_fn, suspend_fn, config_dir, switch_model_fn](std::optional<std::string> chosen) {
                    if (!chosen.has_value()) {
                        append_fn("\n\xe2\x84\xb9  Authentication cancelled.\n");
                        return;
                    }
                    run_provider_auth(chosen.value(),
                                      config_dir,
                                      suspend_fn,
                                      append_fn,
                                      switch_model_fn);
                });
                return;
            }

            // Fallback: text menu (no TUI picker available)
            auto chosen_provider = std::make_shared<std::optional<std::string>>();
            ctx.suspend_tui_fn([available, chosen_provider]() {
                *chosen_provider = core::auth::choose_login_provider_console(
                    available,
                    "Authentication Providers",
                    std::cin,
                    std::cout);
            });
            if (!chosen_provider->has_value()) {
                ctx.append_history_fn("\n\xe2\x84\xb9  Authentication cancelled.\n");
                return;
            }
            run_provider_auth(chosen_provider->value(),
                              config_dir,
                              ctx.suspend_tui_fn,
                              ctx.append_history_fn,
                              ctx.switch_model_fn);
            return;
        }

        if (!interactive_mode) {
            ctx.append_history_fn(
                "\n\xe2\x9a\xa0  Interactive authentication not supported in this mode.\n"
                "\xe2\x84\xb9  Use `filo --login <provider>` in a terminal.\n");
            return;
        }

        run_provider_auth(std::string(*arg),
                          config_dir,
                          ctx.suspend_tui_fn,
                          ctx.append_history_fn,
                          ctx.switch_model_fn);
    }
};

class LogoutCommand : public Command {
public:
    std::string get_name() const override { return "/logout"; }
    std::string get_description() const override {
        return "Sign out of a provider OAuth session (e.g. /logout grok, /logout openai)";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        const auto provider = first_argument(ctx.text);
        if (!provider.has_value()) {
            ctx.append_history_fn(
                "\nℹ  Usage: /logout <provider> "
                "(one of: claude, google, grok, kimi, openai, qwen)\n");
            return;
        }
        const std::string provider_name(*provider);
        std::string display_name;
        std::string error_message;
        const auto run_logout = [&provider_name, &display_name, &error_message]() {
            try {
                const std::string config_dir =
                    core::config::ConfigManager::get_instance().get_config_dir();
                auto manager =
                    core::auth::AuthenticationManager::create_with_defaults(config_dir);
                // Best-effort server-side revocation + local credential clearing.
                display_name = manager.logout(provider_name);
            } catch (const std::exception& error) {
                error_message = error.what();
            }
        };

        if (ctx.suspend_tui_fn) {
            // Revocation performs blocking network requests; suspend the
            // TUI so the interface does not appear frozen meanwhile.
            ctx.suspend_tui_fn(run_logout);
        } else {
            run_logout();
        }

        if (error_message.empty()) {
            ctx.append_history_fn("\n✓  Signed out of " + display_name + ".\n");
        } else {
            ctx.append_history_fn("\n✗  Logout failed: " + error_message + "\n");
        }
    }
};

class QuitCommand : public Command {
public:
    std::string get_name() const override { return "/quit"; }
    std::vector<std::string> get_aliases() const override { return {"/exit", "/q"}; }
    std::string get_description() const override { return "Exit Filo"; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        if (ctx.quit_fn) ctx.quit_fn();
        else exit(0);
    }
};

class ClearCommand : public Command {
public:
    std::string get_name() const override { return "/clear"; }
    std::vector<std::string> get_aliases() const override { return {"/cls"}; }
    std::string get_description() const override { return "Clear the screen and conversation history"; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        ctx.clear_screen_fn();
    }
};

class SessionsCommand : public Command {
public:
    std::string get_name() const override { return "/sessions"; }
    std::string get_description() const override { return "List and manage conversation sessions"; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        if (ctx.open_sessions_picker_fn && ctx.open_sessions_picker_fn()) {
            return;
        }
        ctx.append_history_fn("\n\xe2\x9a\xa0  Session management is only available in the interactive TUI.\n");
    }
};

class ResumeCommand : public Command {
public:
    std::string get_name() const override { return "/resume"; }
    std::string get_description() const override { return "Restore a saved session by ID, index, or name"; }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        if (!ctx.resume_session_fn) {
            ctx.append_history_fn("\n\xe2\x9a\xa0  Session resumption is only available in the interactive TUI.\n");
            return;
        }

        const auto arg = first_argument(ctx.text);
        ctx.resume_session_fn(arg.value_or(""));
    }
};

class ContinueCommand : public Command {
public:
    std::string get_name() const override { return "/continue"; }
    std::string get_description() const override {
        return "Resume the last session when empty, or push the current one forward";
    }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        const bool conversation_empty =
            !ctx.agent || ctx.agent->get_history().empty();

        if (conversation_empty) {
            // Nothing in flight — behave like `filo --continue`: reload the
            // most recent saved session and keep working inside it.
            if (!ctx.resume_session_fn) {
                ctx.append_history_fn(
                    "\n\xe2\x9a\xa0  Session resumption is only available in the interactive TUI.\n");
                return;
            }
            ctx.append_history_fn("\n\xc2\xbb  Resuming the most recent session\xe2\x80\xa6\n");
            ctx.resume_session_fn("");
            return;
        }

        // A conversation is already active (e.g. the previous turn ended in
        // an error or was interrupted) — nudge the agent to carry on from
        // exactly where it stopped instead of replacing the session.
        if (ctx.send_user_message_fn) {
            ctx.append_history_fn("\n\xc2\xbb  Continuing the current session\xe2\x80\xa6\n");
            ctx.send_user_message_fn(
                "Continue from where you left off. If the previous step failed "
                "or was interrupted, diagnose the error, retry it, and carry "
                "the task forward to completion.");
            return;
        }
        ctx.append_history_fn("\n\xe2\x84\xb9  Nothing to continue.\n");
    }
};

class RenameCommand : public Command {
public:
    std::string get_name() const override { return "/rename"; }
    std::string get_description() const override {
        return "Name the current session (no arg shows it, 'clear' removes it)";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        if (!ctx.rename_session_fn) {
            ctx.append_history_fn("\n\xe2\x9a\xa0  Session renaming is only available in the interactive TUI.\n");
            return;
        }

        const std::string arg = trailing_arguments(ctx.text);
        if (arg.empty()) {
            const std::string current = ctx.session_name_fn ? ctx.session_name_fn() : "";
            if (current.empty()) {
                ctx.append_history_fn(
                    "\n\xe2\x84\xb9  This session has no name. Use /rename <name> to set one "
                    "(then resume it later with /resume <name>).\n");
            } else {
                ctx.append_history_fn(std::format(
                    "\n\xe2\x84\xb9  Current session name: {}\n", current));
            }
            return;
        }

        const bool clearing = (arg == "clear" || arg == "none");
        const auto result = ctx.rename_session_fn(clearing ? std::string_view{} : std::string_view{arg});
        ctx.append_history_fn(std::format(
            "\n{}  {}\n",
            result.ok ? "\xc2\xbb" : "\xe2\x9c\x97",
            result.message));
    }
};

class PsCommand : public Command {
public:
    std::string get_name() const override { return "/ps"; }
    std::string get_description() const override { return "Show active terminal commands"; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        if (!ctx.list_active_terminals_fn) {
            ctx.append_history_fn("\nℹ  Terminal process tracking is unavailable.\n");
            return;
        }

        const auto terminals = ctx.list_active_terminals_fn();
        if (terminals.empty()) {
            ctx.append_history_fn("\nℹ  No terminal commands are currently running.\n");
            return;
        }

        std::ostringstream out;
        out << "\n[Active terminal commands]\n";
        for (std::size_t i = 0; i < terminals.size(); ++i) {
            const auto& terminal = terminals[i];
            out << "  " << (i + 1) << ". ";
            if (!terminal.command.empty()) {
                out << terminal.command;
            } else {
                out << "(unknown command)";
            }
            out << "  [" << terminal.elapsed.count() << "s]";
            if (!terminal.working_dir.empty()) {
                out << "\n      cwd: " << terminal.working_dir;
            }
            if (!terminal.tool_call_id.empty()) {
                out << "\n      tool: " << terminal.tool_call_id;
            }
            out << "\n";
        }
        ctx.append_history_fn(out.str());
    }
};

class StopCommand : public Command {
public:
    std::string get_name() const override { return "/stop"; }
    std::string get_description() const override { return "Stop the active terminal command or generation"; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        if (!ctx.stop_active_terminal_fn) {
            if (ctx.agent) {
                ctx.agent->request_stop();
                ctx.append_history_fn("\n»  Stop requested.\n");
            } else {
                ctx.append_history_fn("\nℹ  Nothing is currently running.\n");
            }
            return;
        }

        const auto result = ctx.stop_active_terminal_fn();
        ctx.append_history_fn(std::format(
            "\n{}  {}\n",
            result.ok ? "»" : "ℹ",
            result.message.empty()
                ? (result.ok ? "Stop requested." : "Nothing is currently running.")
                : result.message));
    }
};

class HelpCommand : public Command {
public:
    std::string get_name() const override { return "/help"; }
    std::vector<std::string> get_aliases() const override { return {"/?", "/h"}; }
    std::string get_description() const override { return "Show this help message"; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        ctx.append_history_fn(
            std::format("\n»  {}\n", ctx.text) +
            "\n[Filo Commands]\n"
            "  /auth [provider]     Authenticate with Google or Claude\n"
            "  /help, /?, /h       Show this help message\n"
            "  /clear, /cls        Clear the screen and conversation history\n"
            "  /quit, /exit, /q    Exit the application\n"
            "  /sessions           List and manage conversation sessions\n"
            "  /resume [id|name]   Restore a saved session by ID, index, or name\n"
            "  /continue           Resume last session when empty, else push the current one on\n"
            "  /rename [name]      Name the current session for /resume <name>\n"
            "  /ps                 Show active terminal commands\n"
            "  /stop               Stop active generation or terminal command\n"
            "  /compact            Summarise and compress the conversation history\n"
            "  /compress [mode]    Open/set tool output compression (off|light|full|ultra)\n"
            "  /undo               Remove the last user message from history\n"
            "  /retry              Re-send the last user message\n"
            "  /model [selector]   Open model picker or switch manual/router/provider/model target\n"
            "  /profile [name]     Show/list/switch named configuration profiles\n"
            "  /effort [level]     Open/show/set model effort (auto|low|medium|high|max)\n"
            "  /settings           Open the settings panel for user/workspace preferences\n"
            "  /yolo [on|off]      Toggle or set auto-approval for sensitive tools\n"
            "  /tools [action]     Manage session trust rules for sensitive tools\n"
            "  /usage              Show token usage, cost, and tool payload breakdown\n"
            "  /copy [target]      Copy response, prompt, or full transcript to clipboard\n"
            "  /history            Show or manage input history (use 'clear' to erase)\n"
            "  /review [target]    Open review menu or run Codex-style AI code review\n"
            "  /export [file]      Export the current conversation to Markdown\n"
            "  /fork               Branch the current conversation into a new session\n"
            "  /rewind             Restore and branch from an earlier prompt\n"
            "  /init [provider] [options]  Scaffold .filo/config.json (and optional FILO.md)\n"
            "  /goal [action]      Set or manage the session goal\n"
            "  /todo [action]      Manage session-backed todo items\n"
            "  /memory [action]    Manage opt-in durable memory and auto capture\n"
            "  /mcp [action]       List or manage workspace/global MCP server overlays\n"
            "  !<command>          Execute a shell command  (e.g., !ls -la)\n"
            "\n[Keyboard Shortcuts]\n"
            "  ↑/↓      Navigate input history (previous/next prompt)\n"
            "  Ctrl+P/N Navigate input history (alternative to arrows)\n"
            "  Ctrl+F   Search the conversation history\n"
            "  F2       Cycle agent mode (BUILD → DEBUG → RESEARCH → EXECUTE)\n"
            "  Ctrl+Y   Toggle YOLO auto-approval mode\n"
            "  Ctrl+O   Toggle verbose output view (compact/full)\n"
            "  Ctrl+X   Open current input in external editor ($VISUAL/$EDITOR)\n"
            "  Ctrl+D   Delete char right; when input is empty press twice to quit\n"
            "  Ctrl+L   Clear the screen\n"
            "  Esc      Stop active generation or terminal command\n"
            "  Esc Esc  Clear draft to history, or open rewind menu on empty input\n"
            "  Ctrl+C   Stop active generation or terminal command; quits when idle\n"
        );
    }
};

// Summarise the conversation to save context window space.
class CompactCommand : public Command {
public:
    std::string get_name() const override { return "/compact"; }
    std::string get_description() const override { return "Compress conversation history into a summary"; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        if (!ctx.agent) {
            ctx.append_history_fn("\n✗  No agent available for compaction.\n");
            return;
        }

        ctx.agent->compact_history_async(ctx.append_history_fn);
    }
};

class CompressionCommand : public Command {
public:
    std::string get_name() const override { return "/compression"; }
    std::vector<std::string> get_aliases() const override {
        return {"/compress"};
    }
    std::string get_description() const override {
        return "Open or set tool output compression mode";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        const std::string arg_text = trailing_arguments(ctx.text);
        std::string_view arg = trim(arg_text);

        if (arg.empty()) {
            if (ctx.open_command_option_picker_fn
                && ctx.open_command_option_picker_fn(get_name())) {
                return;
            }
            const std::string body = ctx.compression_status_fn
                ? ctx.compression_status_fn()
                : "Use /compression off, /compression light, /compression full, or /compression ultra.";
            ctx.append_history_fn(std::format("\nℹ  {}\n", body));
            return;
        }

        const std::string mode = to_lower_ascii(arg);
        const std::string body = ctx.switch_compression_fn
            ? ctx.switch_compression_fn(mode)
            : "Compression switching is not available in this session.";
        const bool success = body.starts_with("Set");
        ctx.append_history_fn(std::format(
            "\n{}\n",
            success
                ? "✓  " + body
                : "✗  " + body));
    }
};

class UndoCommand : public Command {
public:
    std::string get_name() const override { return "/undo"; }
    std::string get_description() const override { return "Remove the last user message from history"; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        ctx.agent->undo_last();
        ctx.append_history_fn("\n»  Last message removed from history.\n");
    }
};

class RetryCommand : public Command {
public:
    std::string get_name() const override { return "/retry"; }
    std::string get_description() const override { return "Re-send the last user message"; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        const auto last = ctx.agent->last_user_turn();
        if (!last.has_value()) {
            ctx.append_history_fn("\n✗  No previous message to retry.\n");
            return;
        }
        ctx.append_history_fn(std::format(
            "\n»  Retrying: {}\n",
            core::llm::message_text_for_display(*last)));

        auto append_fn = ctx.append_history_fn;
        auto run_retry = [agent = ctx.agent, last = *last, append_fn]() {
            agent->send_message(last,
                [append_fn](const std::string& chunk) {
                    append_fn(chunk);
                },
                [append_fn](const std::string& name, const std::string& args) {
                    std::string msg = "\n[Tool] " + name + " " + args + "\n";
                    append_fn(msg);
                },
                [append_fn]() {
                    append_fn("\n");
                }
            );
        };
        if (ctx.dispatch_async_fn) {
            ctx.dispatch_async_fn(std::move(run_retry));
            return;
        }
        run_retry();
    }
};

class ModelCommand : public Command {
public:
    std::string get_name() const override { return "/model"; }
    std::string get_description() const override {
        return "Open model picker or switch manual/router/provider/model target";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        std::string_view arg = trim(std::string_view{ctx.text}.substr(std::min(ctx.text.size(), std::string::size_type{6})));

        if (arg.empty()) {
            if (ctx.open_model_picker_fn && ctx.open_model_picker_fn()) {
                return;
            }
            const std::string body = ctx.model_status_fn
                ? ctx.model_status_fn()
                : "Use /model manual, /model router, /model <provider-name>, or /model <provider-name> <model>.";
            // body may be multi-line ("Active: ...\n        Available: ...")
            // emit each non-empty segment as a separate info notification
            std::string remaining = body;
            bool first = true;
            while (!remaining.empty()) {
                const auto nl = remaining.find('\n');
                std::string line = nl == std::string::npos
                    ? remaining
                    : remaining.substr(0, nl);
                remaining = nl == std::string::npos ? "" : remaining.substr(nl + 1);
                const auto start = line.find_first_not_of(" \t");
                if (start == std::string::npos) continue;
                ctx.append_history_fn(std::format(
                    "{}\xe2\x84\xb9  {}\n", first ? "\n" : "", line.substr(start)));
                first = false;
            }
            if (first) ctx.append_history_fn("\n\xe2\x84\xb9  No model info available.\n");
            return;
        }

        const std::string body = ctx.switch_model_fn
            ? ctx.switch_model_fn(arg)
            : "Switching provider is not available in this session.";
        const bool success = body.starts_with("Switched");
        ctx.append_history_fn(std::format(
            "\n{}\n", success
                ? "\xe2\x9c\x93  " + body   // ✓
                : "\xe2\x9c\x97  " + body)); // ✗
    }
};

class ProfileCommand : public Command {
public:
    std::string get_name() const override { return "/profile"; }
    std::string get_description() const override {
        return "Show/list/switch named configuration profiles";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        std::string_view arg = trim(std::string_view{ctx.text}.substr(
            std::min(ctx.text.size(), std::string::size_type{8})));

        auto emit_info_body = [&](const std::string& body) {
            std::string remaining = body;
            bool first = true;
            while (!remaining.empty()) {
                const auto nl = remaining.find('\n');
                std::string line = nl == std::string::npos
                    ? remaining
                    : remaining.substr(0, nl);
                remaining = nl == std::string::npos ? "" : remaining.substr(nl + 1);
                const auto start = line.find_first_not_of(" \t");
                if (start == std::string::npos) continue;
                ctx.append_history_fn(std::format(
                    "{}ℹ  {}\n", first ? "\n" : "", line.substr(start)));
                first = false;
            }
            if (first) ctx.append_history_fn("\nℹ  No profile information is available.\n");
        };

        const std::string lowered = to_lower_ascii(arg);
        if (arg.empty() || lowered == "status"
            || lowered == "list" || lowered == "ls") {
            const std::string body = ctx.profile_status_fn
                ? ctx.profile_status_fn()
                : "Use /profile <name> to switch profiles (not available in this session).";
            emit_info_body(body);
            return;
        }

        const std::string body = ctx.switch_profile_fn
            ? ctx.switch_profile_fn(arg)
            : "Profile switching is not available in this session.";
        const bool success = body.starts_with("Switched")
            || body.starts_with("Cleared");
        ctx.append_history_fn(std::format(
            "\n{}\n", success
                ? "✓  " + body
                : "✗  " + body));
    }
};

class EffortCommand : public Command {
public:
    std::string get_name() const override { return "/effort"; }
    std::string get_description() const override {
        return "Open, show, or set model effort level (auto|low|medium|high|max)";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        std::string_view arg = trim(std::string_view{ctx.text}.substr(
            std::min(ctx.text.size(), std::string::size_type{7})));

        auto emit_info_body = [&](const std::string& body) {
            std::string remaining = body;
            bool first = true;
            while (!remaining.empty()) {
                const auto nl = remaining.find('\n');
                std::string line = nl == std::string::npos
                    ? remaining
                    : remaining.substr(0, nl);
                remaining = nl == std::string::npos ? "" : remaining.substr(nl + 1);
                const auto start = line.find_first_not_of(" \t");
                if (start == std::string::npos) continue;
                ctx.append_history_fn(std::format(
                    "{}ℹ  {}\n", first ? "\n" : "", line.substr(start)));
                first = false;
            }
            if (first) ctx.append_history_fn("\nℹ  No effort information is available.\n");
        };

        if (arg.empty()) {
            if (ctx.open_command_option_picker_fn
                && ctx.open_command_option_picker_fn(get_name())) {
                return;
            }
            const std::string body = ctx.effort_status_fn
                ? ctx.effort_status_fn()
                : "Use /effort auto|low|medium|high|max, or /effort status.";
            emit_info_body(body);
            return;
        }

        const std::string lowered = to_lower_ascii(arg);
        if (lowered == "status" || lowered == "current") {
            const std::string body = ctx.effort_status_fn
                ? ctx.effort_status_fn()
                : "Use /effort auto|low|medium|high|max.";
            emit_info_body(body);
            return;
        }

        const std::string body = ctx.switch_effort_fn
            ? ctx.switch_effort_fn(arg)
            : "Effort switching is not available in this session.";
        const bool success = body.starts_with("Set");
        ctx.append_history_fn(std::format(
            "\n{}\n", success
                ? "✓  " + body
                : "✗  " + body));
    }
};

class SettingsCommand : public Command {
public:
    std::string get_name() const override { return "/settings"; }
    std::string get_description() const override {
        return "Open the settings panel and edit user/workspace preferences";
    }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        if (const auto arg = first_argument(ctx.text); arg.has_value()) {
            ctx.append_history_fn(
                "\n✗  `/settings` does not take arguments. Open the panel with `/settings`.\n");
            return;
        }

        if (ctx.open_settings_picker_fn && ctx.open_settings_picker_fn()) {
            return;
        }

        std::string body;
        if (ctx.settings_status_fn) {
            body = ctx.settings_status_fn();
        } else {
            auto& config_manager = core::config::ConfigManager::get_instance();
            const auto& config = config_manager.get_config();
            const auto user_path = config_manager.get_settings_path(core::config::SettingsScope::User);
            const auto workspace_path = config_manager.get_settings_path(
                core::config::SettingsScope::Workspace);

            body = std::format(
                "Effective settings\n"
                "Start mode: {}\n"
                "Approval mode: {}\n"
                "Default router policy: {}\n"
                "Startup banner: {}\n"
                "Footer: {}\n"
                "Footer model badge: {}\n"
                "Footer context meter: {}\n"
                "Message timestamps: {}\n"
                "Activity spinner: {}\n"
                "User settings file: {}\n"
                "Workspace settings file: {}\n"
                "Model defaults are managed separately via `/model`.",
                config.default_mode,
                config.default_approval_mode.empty() ? std::string("prompt")
                                                     : config.default_approval_mode,
                config.router.default_policy.empty() ? std::string("<unset>")
                                                     : config.router.default_policy,
                config.ui_banner,
                config.ui_footer,
                config.ui_model_info,
                config.ui_context_usage,
                config.ui_timestamps,
                config.ui_spinner,
                user_path.string(),
                workspace_path.string());
        }

        std::string remaining = std::move(body);
        bool first = true;
        while (!remaining.empty()) {
            const auto nl = remaining.find('\n');
            std::string line = nl == std::string::npos
                ? remaining
                : remaining.substr(0, nl);
            remaining = nl == std::string::npos ? "" : remaining.substr(nl + 1);
            const auto start = line.find_first_not_of(" \t");
            if (start == std::string::npos) {
                continue;
            }
            ctx.append_history_fn(std::format(
                "{}ℹ  {}\n",
                first ? "\n" : "",
                line.substr(start)));
            first = false;
        }

        if (first) {
            ctx.append_history_fn("\nℹ  No settings information is available.\n");
        }
    }
};

class YoloCommand : public Command {
public:
    std::string get_name() const override { return "/yolo"; }
    std::string get_description() const override {
        return "Toggle or set approval mode (YOLO auto-approve on/off)";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        if (!ctx.yolo_mode_enabled_fn || !ctx.set_yolo_mode_enabled_fn) {
            ctx.append_history_fn("\n✗  Approval mode controls are unavailable in this session.\n");
            return;
        }

        std::string_view arg = trim(std::string_view{ctx.text}.substr(
            std::min(ctx.text.size(), std::string::size_type{5})));
        const std::string normalized = to_lower_ascii(arg);

        bool current = ctx.yolo_mode_enabled_fn();
        bool next = current;
        bool show_status_only = false;

        if (normalized.empty() || normalized == "toggle") {
            next = !current;
        } else if (normalized == "on" || normalized == "enable"
                   || normalized == "enabled" || normalized == "yes"
                   || normalized == "true" || normalized == "1") {
            next = true;
        } else if (normalized == "off" || normalized == "disable"
                   || normalized == "disabled" || normalized == "no"
                   || normalized == "false" || normalized == "0") {
            next = false;
        } else if (normalized == "status") {
            show_status_only = true;
        } else {
            ctx.append_history_fn(
                "\n✗  Unknown /yolo option. Use /yolo, /yolo on, /yolo off, or /yolo status.\n");
            return;
        }

        if (show_status_only) {
            ctx.append_history_fn(current
                ? "\n⚠  Approval mode: YOLO (auto-approve sensitive tools).\n"
                : "\nℹ  Approval mode: PROMPT (ask before sensitive tools).\n");
            return;
        }

        ctx.set_yolo_mode_enabled_fn(next);
        ctx.append_history_fn(next
            ? "\n⚠  Approval mode set to YOLO: sensitive tools now run without confirmation.\n"
            : "\nℹ  Approval mode set to PROMPT: sensitive tools require confirmation.\n");
    }
};

class ToolsCommand : public Command {
public:
    std::string get_name() const override { return "/tools"; }
    std::string get_description() const override {
        return "Manage session trust rules for sensitive tools";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        if (!ctx.tool_rules.available()) {
            ctx.append_history_fn(
                "\n✗  Tool trust controls are unavailable in this session.\n");
            return;
        }

        const std::string args = trailing_arguments(ctx.text);
        const auto tokens = split_ascii_whitespace(args);
        const std::string action = tokens.empty()
            ? std::string("status")
            : to_lower_ascii(tokens.front());

        auto usage = [&]() {
            ctx.append_history_fn(
                "\nℹ  Usage:\n"
                "   /tools                     Show trust status and active rules\n"
                "   /tools list               List session trust rules\n"
                "   /tools allow shell git    Trust one terminal program\n"
                "   /tools allow shell *      Trust all terminal programs\n"
                "   /tools allow files write  Trust file modifications\n"
                "   /tools allow files delete Trust file deletions\n"
                "   /tools allow files move   Trust file moves\n"
                "   /tools allow tool write_file\n"
                "   /tools remove <index|rule>\n"
                "   /tools clear              Remove all session trust rules\n");
        };

        auto remainder_after_action = [&]() -> std::string_view {
            if (tokens.empty()) {
                return {};
            }
            const std::size_t offset =
                static_cast<std::size_t>(tokens.front().data() - args.data())
                + tokens.front().size();
            if (offset >= args.size()) {
                return {};
            }
            return trim(std::string_view(args).substr(offset));
        };

        auto append_result = [&](const CommandOperationResult& result,
                                 std::string_view success_message = "Rule updated.") {
            const std::string message = result.message.empty()
                ? (result.ok ? std::string(success_message) : std::string("Operation failed."))
                : result.message;
            ctx.append_history_fn(std::format(
                "\n{}  {}\n",
                result.ok ? "✓" : "✗",
                message));
        };

        if (action == "help") {
            usage();
            return;
        }

        if (action == "status" || action == "show") {
            const auto rules = sorted_rule_snapshot(ctx.tool_rules);
            const bool yolo_enabled = ctx.yolo_mode_enabled_fn
                ? ctx.yolo_mode_enabled_fn()
                : false;
            ctx.append_history_fn(std::format(
                "{}\n{}\n",
                yolo_enabled
                    ? "\n⚠  Approval mode: YOLO (all sensitive tools auto-approved)."
                    : "\nℹ  Approval mode: PROMPT (sensitive tools require approval unless trusted).",
                format_tool_rules(rules)));
            return;
        }

        if (action == "list" || action == "ls") {
            ctx.append_history_fn(format_tool_rules(sorted_rule_snapshot(ctx.tool_rules)));
            return;
        }

        if (action == "allow" || action == "add" || action == "trust") {
            std::string error;
            const auto rule = parse_tools_rule_spec(remainder_after_action(), error);
            if (!rule.has_value()) {
                ctx.append_history_fn(std::format("\n✗  {}\n", error));
                return;
            }

            const auto result = ctx.tool_rules.add(*rule);
            append_result(result, "Rule added.");
            if (result.ok) {
                ctx.append_history_fn(format_tool_rules(sorted_rule_snapshot(ctx.tool_rules)));
            }
            return;
        }

        if (action == "remove" || action == "rm" || action == "revoke"
            || action == "delete" || action == "untrust") {
            const std::string_view target = remainder_after_action();
            if (target.empty()) {
                ctx.append_history_fn("\n✗  Usage: /tools remove <index|rule>\n");
                return;
            }

            std::string target_rule;
            const bool numeric_target = std::all_of(
                target.begin(),
                target.end(),
                [](unsigned char ch) { return std::isdigit(ch); });
            if (numeric_target) {
                std::size_t index = 0;
                try {
                    index = static_cast<std::size_t>(std::stoul(std::string(target)));
                } catch (...) {
                    ctx.append_history_fn("\n✗  Invalid rule index.\n");
                    return;
                }
                const auto rules = sorted_rule_snapshot(ctx.tool_rules);
                if (index == 0 || index > rules.size()) {
                    if (rules.empty()) {
                        ctx.append_history_fn("\n✗  There are no trust rules to remove.\n");
                        return;
                    }
                    ctx.append_history_fn(std::format(
                        "\n✗  Rule index {} is out of range (1-{}).\n",
                        index,
                        rules.size()));
                    return;
                }
                target_rule = rules[index - 1];
            } else {
                std::string error;
                const auto parsed = parse_tools_rule_spec(target, error);
                if (!parsed.has_value()) {
                    ctx.append_history_fn(std::format("\n✗  {}\n", error));
                    return;
                }
                target_rule = *parsed;
            }

            const auto result = ctx.tool_rules.remove(target_rule);
            append_result(result, "Rule removed.");
            if (result.ok) {
                ctx.append_history_fn(format_tool_rules(sorted_rule_snapshot(ctx.tool_rules)));
            }
            return;
        }

        if (action == "clear" || action == "reset") {
            const auto result = ctx.tool_rules.clear();
            append_result(result, "All tool trust rules cleared.");
            if (result.ok) {
                ctx.append_history_fn(format_tool_rules(sorted_rule_snapshot(ctx.tool_rules)));
            }
            return;
        }

        usage();
    }
};

class UsageCommand : public Command {
public:
    std::string get_name() const override { return "/usage"; }
    std::vector<std::string> get_aliases() const override {
        return {"/tokens", "/cost", "/toolcost"};
    }
    std::string get_description() const override {
        return "Show token usage, estimated cost, and tool payload breakdown";
    }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        const auto snapshot = core::session::SessionStats::get_instance().snapshot();
        const auto total = core::budget::BudgetTracker::get_instance().session_total();
        const double cost = core::budget::BudgetTracker::get_instance().session_cost_usd();

        if (snapshot.turn_count == 0 && !total.has_data()) {
            ctx.append_history_fn(
                "\n[Usage]\n"
                "  No token usage recorded yet in this session.\n");
            return;
        }

        std::ostringstream out;
        out << "\n[Usage]\n";
        out << "  Turns: " << snapshot.turn_count << "\n";

        if (snapshot.api_calls_total > 0) {
            out << "  API calls: " << snapshot.api_calls_total
                << " (" << snapshot.api_calls_success << " ok, "
                << (snapshot.api_calls_total - snapshot.api_calls_success) << " failed)\n";
        }

        if (snapshot.tool_calls_total > 0) {
            out << "  Tool calls: " << snapshot.tool_calls_total
                << " (" << snapshot.tool_calls_success << " ok, "
                << (snapshot.tool_calls_total - snapshot.tool_calls_success) << " failed)\n";
        }

        out << "  Tokens: input " << format_token_count(total.prompt_tokens)
            << ", output " << format_token_count(total.completion_tokens)
            << ", total " << format_token_count(total.total_tokens) << "\n";
        out << "  Estimated cost: " << format_usage_cost(cost) << "\n";

        if (!snapshot.per_model.empty()) {
            out << "\n[By Model]\n";
            for (const auto& model : snapshot.per_model) {
                out << std::format(
                    "  {:<24} {:>3} calls  in {:>7}  out {:>7}  total {:>7}  {}\n",
                    model.model,
                    model.call_count,
                    format_token_count(model.prompt_tokens),
                    format_token_count(model.completion_tokens),
                    format_token_count(model.prompt_tokens + model.completion_tokens),
                    model.cost_usd > 0.0 ? format_usage_cost(model.cost_usd) : std::string("-"));
            }
        }

        if (!snapshot.per_tool.empty()) {
            out << "\n[By Tool]\n";
            out << "  Cost is attributed from model completion tokens; args/results are estimated payload tokens.\n";
            for (const auto& tool : snapshot.per_tool) {
                out << std::format(
                    "  {:<24} {:>3} calls  {:>3} ok  {:>3} failed  out {:>7}  cost {:>8}  args {:>7}  results {:>7}\n",
                    tool.tool,
                    tool.call_count,
                    tool.success_count,
                    tool.call_count - tool.success_count,
                    format_token_count(tool.attributed_completion_tokens),
                    tool.attributed_cost_usd > 0.0
                        ? format_usage_cost(tool.attributed_cost_usd)
                        : std::string("-"),
                    format_token_count(tool.argument_tokens),
                    format_token_count(tool.result_tokens));
            }
        }

        ctx.append_history_fn(out.str());
    }
};

class CopyCommand : public Command {
public:
    std::string get_name() const override { return "/copy"; }
    std::string get_description() const override {
        return "Copy response, prompt, or full transcript to clipboard";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        const auto target = parse_target(trailing_arguments(ctx.text));
        if (!target.has_value()) {
            ctx.append_history_fn(
                "\nℹ  Usage: /copy [response|prompt|full]\n");
            return;
        }

        const auto source = resolve_source(ctx, *target);
        if (!source.has_value()) {
            return;
        }

        auto copy_fn = ctx.copy_to_clipboard_fn;
        if (!copy_fn) {
            copy_fn = [](std::string_view text) {
                return copy_text_to_clipboard(text);
            };
        }

        if (const auto error = copy_fn(source->text); error.has_value()) {
            ctx.append_history_fn(std::format(
                "\n✗  Could not copy {}: {}\n",
                source->label,
                *error));
            return;
        }

        ctx.append_history_fn(std::format(
            "\n✓  Copied {} to clipboard.\n",
            source->success_label));
    }

private:
    enum class Target {
        Response,
        Prompt,
        Full,
    };

    struct CopySource {
        std::string text;
        std::string_view label;
        std::string_view success_label;
    };

    static std::optional<Target> parse_target(std::string_view args) {
        const std::string normalized = to_lower_ascii(trim(args));
        if (normalized.empty() || normalized == "response") {
            return Target::Response;
        }
        if (normalized == "prompt") {
            return Target::Prompt;
        }
        if (normalized == "full") {
            return Target::Full;
        }
        return std::nullopt;
    }

    static std::optional<CopySource> resolve_source(const CommandContext& ctx,
                                                    Target target) {
        switch (target) {
            case Target::Response:
                return response_source(ctx);
            case Target::Prompt:
                return prompt_source(ctx);
            case Target::Full:
                return full_source(ctx);
        }
        return std::nullopt;
    }

    static std::optional<CopySource> response_source(const CommandContext& ctx) {
        if (!ctx.latest_assistant_output_fn) {
            ctx.append_history_fn("\n✗  Copy is unavailable in this session.\n");
            return std::nullopt;
        }

        std::string text = ctx.latest_assistant_output_fn();
        if (trim(text).empty()) {
            ctx.append_history_fn(
                "\nℹ  Nothing to copy yet. Wait for the first completed assistant response.\n");
            return std::nullopt;
        }

        return CopySource{
            .text = std::move(text),
            .label = "response",
            .success_label = "the latest assistant response",
        };
    }

    static std::optional<CopySource> prompt_source(const CommandContext& ctx) {
        if (!ctx.agent) {
            ctx.append_history_fn("\n✗  Prompt copy is unavailable without an active agent.\n");
            return std::nullopt;
        }

        std::string text = ctx.agent->last_user_message();
        if (trim(text).empty()) {
            ctx.append_history_fn("\nℹ  Nothing to copy yet. No user prompt found.\n");
            return std::nullopt;
        }

        return CopySource{
            .text = std::move(text),
            .label = "prompt",
            .success_label = "the latest user prompt",
        };
    }

    static std::optional<CopySource> full_source(const CommandContext& ctx) {
        if (!ctx.agent) {
            ctx.append_history_fn("\n✗  Transcript copy is unavailable without an active agent.\n");
            return std::nullopt;
        }

        const auto messages = ctx.agent->get_history();
        if (visible_transcript_message_count(messages) == 0) {
            ctx.append_history_fn("\nℹ  Nothing to copy yet. No conversation transcript found.\n");
            return std::nullopt;
        }

        return CopySource{
            .text = export_history_markdown(messages),
            .label = "transcript",
            .success_label = "the full conversation transcript",
        };
    }
};

class HistoryCommand : public Command {
public:
    std::string get_name() const override { return "/history"; }
    std::vector<std::string> get_aliases() const override { return {"/hist"}; }
    std::string get_description() const override {
        return "Manage input history: /history, /history clear, /history show [n]";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        if (!ctx.history_store_fn) {
            ctx.append_history_fn("\n✗  History is not available in this session.\n");
            return;
        }

        // Parse arguments
        std::string_view arg = trim(std::string_view{ctx.text}.substr(
            std::min(ctx.text.size(), std::string::size_type{8})));

        // Default: show recent history
        if (arg.empty() || arg == "show" || arg == "list") {
            const auto& entries = ctx.history_store_fn->entries();
            const std::size_t total = entries.size();

            if (total == 0) {
                ctx.append_history_fn("\nℹ  No history entries yet.\n");
                return;
            }

            // Parse limit if provided (e.g., "show 20" or just "20")
            std::size_t limit = 20;
            std::string_view limit_str = arg;
            if (limit_str.starts_with("show ")) {
                limit_str = trim(limit_str.substr(5));
            }
            if (!limit_str.empty()) {
                try {
                    limit = static_cast<std::size_t>(std::stoul(std::string(limit_str)));
                } catch (...) {
                    // Use default
                }
            }

            std::ostringstream oss;
            oss << "\n📜  Input History (" << total << " total, showing last " << std::min(limit, total) << ")\n";
            oss << "   Use ↑/↓ or Ctrl+P/Ctrl+N to navigate in the prompt.\n\n";

            const std::size_t start = total > limit ? total - limit : 0;
            for (std::size_t i = total; i > start; --i) {
                const std::size_t idx = i - 1;
                const std::string& entry = entries[idx];
                // Show entry number and truncated text
                oss << std::format("  {:4}  ", idx + 1);
                if (entry.size() > 70) {
                    oss << entry.substr(0, 67) << "...\n";
                } else {
                    oss << entry << "\n";
                }
            }
            ctx.append_history_fn(oss.str());
            return;
        }

        // Clear history
        if (arg == "clear" || arg == "erase" || arg == "delete") {
            if (ctx.clear_history_fn) {
                ctx.clear_history_fn();
                ctx.append_history_fn("\n🗑️  Input history cleared.\n");
            } else {
                ctx.append_history_fn("\n✗  Cannot clear history in this context.\n");
            }
            return;
        }

        // Show stats
        if (arg == "stats" || arg == "info") {
            const auto& entries = ctx.history_store_fn->entries();
            const auto path = ctx.history_store_fn->path();
            ctx.append_history_fn(std::format(
                "\n📊  History Stats\n"
                "   Entries: {}\n"
                "   Max entries: {}\n"
                "   Storage: {}\n",
                entries.size(),
                ctx.history_store_fn->max_entries(),
                path.string()));
            return;
        }

        // Unknown subcommand
        ctx.append_history_fn(
            "\nℹ  Usage: /history [show [n]|clear|stats]\n"
            "   show [n]  — Show last n entries (default: 20)\n"
            "   clear     — Clear all history\n"
            "   stats     — Show history statistics\n");
    }
};

class ReviewCommand : public Command {
public:
    std::string get_name() const override { return "/review"; }
    std::string get_description() const override {
        return "Open the review menu or run Codex-style review (/review [--uncommitted|--base|--commit|custom])";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        const std::string args = trailing_arguments(ctx.text);
        auto run_review = [ctx](std::string selection) {
            if (ctx.dispatch_async_fn) {
                ctx.dispatch_async_fn([ctx, selection = std::move(selection)]() {
                    ReviewExecutor::execute(ctx, selection);
                });
                return;
            }
            ReviewExecutor::execute(ctx, selection);
        };

        if (!args.empty()) {
            run_review(args);
            return;
        }

        if (!ctx.agent) {
            run_review(args);
            return;
        }

        if (ctx.open_review_picker_fn) {
            const CommandContext picker_ctx = ctx;
            ctx.open_review_picker_fn([picker_ctx](std::optional<std::string> selection) {
                if (!selection.has_value()) {
                    return;
                }
                if (picker_ctx.dispatch_async_fn) {
                    picker_ctx.dispatch_async_fn([picker_ctx, selection = std::move(*selection)]() {
                        ReviewExecutor::execute(picker_ctx, selection);
                    });
                    return;
                }
                ReviewExecutor::execute(picker_ctx, *selection);
            });
            return;
        }

        if (ctx.suspend_tui_fn) {
            std::optional<std::string> selection;
            ctx.suspend_tui_fn([&selection]() {
                selection = choose_review_menu_request();
            });
            if (selection.has_value()) {
                if (ctx.dispatch_async_fn) {
                    ctx.dispatch_async_fn([ctx, selection = std::move(*selection)]() {
                        ReviewExecutor::execute(ctx, selection);
                    });
                    return;
                }
                ReviewExecutor::execute(ctx, *selection);
            }
            return;
        }

        ReviewExecutor::execute(ctx, args);
    }
};

class ExportCommand : public Command {
public:
    std::string get_name() const override { return "/export"; }
    std::string get_description() const override {
        return "Export conversation history to Markdown (/export [filename])";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        if (!ctx.agent) {
            ctx.append_history_fn("\n✗  Export is unavailable without an active agent.\n");
            return;
        }

        const std::string args = trailing_arguments(ctx.text);
        const std::filesystem::path output_path = args.empty()
            ? std::filesystem::path(default_export_filename())
            : std::filesystem::path(args);

        std::error_code ec;
        const auto parent = output_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent, ec);
            if (ec) {
                ctx.append_history_fn(std::format(
                    "\n✗  Could not create export directory '{}': {}\n",
                    parent.string(),
                    ec.message()));
                return;
            }
        }

        const auto messages = ctx.agent->get_history();
        const std::size_t visible_message_count =
            visible_transcript_message_count(messages);
        if (visible_message_count == 0) {
            ctx.append_history_fn(
                "\nℹ  Nothing to export yet. No conversation transcript found.\n");
            return;
        }
        const std::string markdown = export_history_markdown(messages);

        std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
        if (!out) {
            ctx.append_history_fn(std::format(
                "\n✗  Could not open '{}' for writing.\n",
                output_path.string()));
            return;
        }

        out.write(markdown.data(), static_cast<std::streamsize>(markdown.size()));
        if (!out) {
            ctx.append_history_fn(std::format(
                "\n✗  Failed while writing '{}'.\n",
                output_path.string()));
            return;
        }

        ctx.append_history_fn(std::format(
            "\n✓  Exported {} messages to {}.\n",
            visible_message_count,
            output_path.string()));
    }
};

class ForkCommand : public Command {
public:
    std::string get_name() const override { return "/fork"; }
    std::string get_description() const override {
        return "Fork the current conversation into a new session";
    }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        if (!ctx.fork_session_fn) {
            ctx.append_history_fn("\n✗  Session forking is unavailable in this context.\n");
            return;
        }

        const std::string message = ctx.fork_session_fn();
        const bool success = !message.starts_with("Failed")
            && !message.starts_with("Error")
            && !message.starts_with("✗");
        ctx.append_history_fn(std::format(
            "\n{}  {}\n",
            success ? "✓" : "✗",
            message.empty() ? (success ? "Forked current session." : "Session fork failed.")
                            : message));
    }
};

class RewindCommand : public Command {
public:
    std::string get_name() const override { return "/rewind"; }
    std::vector<std::string> get_aliases() const override { return {"/checkpoint"}; }
    std::string get_description() const override {
        return "Restore and branch the conversation from an earlier prompt";
    }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        if (!ctx.open_rewind_picker_fn) {
            ctx.append_history_fn("\n✗  Rewind is only available in the interactive TUI.\n");
            return;
        }
        if (!ctx.open_rewind_picker_fn()) {
            ctx.append_history_fn("\nℹ  Nothing to rewind yet.\n");
        }
    }
};

class InitCommand : public Command {
public:
    std::string get_name() const override { return "/init"; }
    std::string get_description() const override {
        return "Scaffold .filo project config (/init [provider] [--with-prompt] [--force])";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        bool with_prompt = false;
        bool force = false;
        bool provider_set = false;
        std::string default_provider = "grok";
        for (const auto token : split_ascii_whitespace(trailing_arguments(ctx.text))) {
            const std::string option = to_lower_ascii(token);
            if (option == "--with-prompt" || option == "--prompt") {
                with_prompt = true;
                continue;
            }
            if (option == "--force" || option == "-f") {
                force = true;
                continue;
            }
            if (option == "-h" || option == "--help") {
                ctx.append_history_fn(
                    "\nℹ  Usage: /init [provider] [--with-prompt] [--force]\n"
                    "   provider       optional default provider (for example: grok, openai).\n"
                    "   --with-prompt  also create FILO.md in the project root.\n"
                    "   --force        overwrite existing scaffold files.\n");
                return;
            }
            if (option.starts_with('-')) {
                ctx.append_history_fn(std::format(
                    "\n✗  Unknown /init option '{}'. Use /init [provider] [--with-prompt] [--force].\n",
                    std::string(token)));
                return;
            }
            if (!is_valid_provider_token(option)) {
                ctx.append_history_fn(std::format(
                    "\n✗  Invalid provider '{}'. Use letters, numbers, '-' or '_'.\n",
                    std::string(token)));
                return;
            }
            if (provider_set) {
                ctx.append_history_fn(std::format(
                    "\n✗  Multiple providers given ('{}' and '{}'). Use only one provider.\n",
                    default_provider,
                    option));
                return;
            }
            default_provider = option;
            provider_set = true;
            continue;

        }

        const std::filesystem::path root = std::filesystem::current_path();
        const std::filesystem::path filo_dir = root / ".filo";
        const std::filesystem::path config_path = filo_dir / "config.json";
        const std::filesystem::path prompt_path = root / "FILO.md";

        std::error_code ec;
        std::filesystem::create_directories(filo_dir, ec);
        if (ec) {
            ctx.append_history_fn(std::format(
                "\n✗  Could not create '{}': {}\n",
                filo_dir.string(),
                ec.message()));
            return;
        }

        const std::string config_template = std::format(R"({{
    "default_provider": "{}",
    "default_model_selection": "manual",
    "default_mode": "BUILD",
    "default_approval_mode": "prompt",
    "auto_compact_threshold": 25000,
    "context_compression": "off"
}}
)", default_provider);

        constexpr std::string_view kPromptTemplate = R"(# FILO.md

Project-specific instructions for Filo.

- Coding conventions:
- Testing requirements:
- Deployment or runtime constraints:
- Preferred architecture patterns:
)";

        auto write_file = [&](const std::filesystem::path& path,
                              std::string_view content,
                              std::string& outcome) -> bool {
            if (std::filesystem::exists(path) && !force) {
                outcome = std::format("Skipped existing {}", path.string());
                return true;
            }

            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out) {
                outcome = std::format("Failed to write {}", path.string());
                return false;
            }
            out.write(content.data(), static_cast<std::streamsize>(content.size()));
            if (!out) {
                outcome = std::format("Failed while writing {}", path.string());
                return false;
            }

            outcome = std::format("Wrote {}", path.string());
            return true;
        };

        std::vector<std::string> outcomes;
        std::string outcome;
        if (!write_file(config_path, config_template, outcome)) {
            ctx.append_history_fn(std::format("\n✗  {}\n", outcome));
            return;
        }
        outcomes.push_back(std::move(outcome));

        if (with_prompt) {
            if (!write_file(prompt_path, kPromptTemplate, outcome)) {
                ctx.append_history_fn(std::format("\n✗  {}\n", outcome));
                return;
            }
            outcomes.push_back(std::move(outcome));
        }

        std::ostringstream summary;
        summary << "\n✓  Project scaffold ready.\n";
        summary << "   - default_provider: " << default_provider << "\n";
        for (const auto& line : outcomes) {
            summary << "   - " << line << "\n";
        }
        if (!with_prompt) {
            summary << "   Tip: run `/init --with-prompt` to add FILO.md.\n";
        }

        ctx.append_history_fn(summary.str());
    }
};

class GoalCommand : public Command {
public:
    std::string get_name() const override { return "/goal"; }
    std::vector<std::string> get_aliases() const override { return {"/g"}; }
    std::string get_description() const override {
        return "Set or manage the session goal";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        if (!ctx.current_goal_fn) {
            ctx.append_history_fn("\n✗  Goal management is unavailable in this context.\n");
            return;
        }

        const std::string args = trailing_arguments(ctx.text);
        const auto tokens = split_ascii_whitespace(args);
        auto show_goal = [&]() {
            ctx.append_history_fn(format_goal(ctx.current_goal_fn()));
        };

        if (tokens.empty()) {
            show_goal();
            return;
        }

        const std::string action = to_lower_ascii(tokens.front());
        const std::size_t remainder_offset =
            static_cast<std::size_t>(tokens.front().data() - args.data()) + tokens.front().size();
        const std::string remainder = trim_copy(
            remainder_offset >= args.size() ? std::string_view{} : std::string_view(args).substr(remainder_offset));

        CommandOperationResult result;
        std::string objective_to_submit;
        if (action == "show" || action == "status" || action == "list") {
            show_goal();
            return;
        }

        if (action == "clear" || action == "reset" || action == "remove" || action == "rm") {
            if (!ctx.clear_goal_fn) {
                ctx.append_history_fn("\n✗  Goal clearing is unavailable in this context.\n");
                return;
            }
            result = ctx.clear_goal_fn();
        } else if (action == "done" || action == "complete" || action == "completed") {
            if (!ctx.set_goal_status_fn) {
                ctx.append_history_fn("\n✗  Goal status updates are unavailable in this context.\n");
                return;
            }
            result = ctx.set_goal_status_fn("complete", remainder);
        } else if (action == "block" || action == "blocked") {
            if (!ctx.set_goal_status_fn) {
                ctx.append_history_fn("\n✗  Goal status updates are unavailable in this context.\n");
                return;
            }
            result = ctx.set_goal_status_fn("blocked", remainder);
        } else if (action == "resume" || action == "active" || action == "reopen") {
            if (!ctx.set_goal_status_fn) {
                ctx.append_history_fn("\n✗  Goal status updates are unavailable in this context.\n");
                return;
            }
            result = ctx.set_goal_status_fn("active", remainder);
        } else {
            if (!ctx.set_goal_fn) {
                ctx.append_history_fn("\n✗  Goal creation is unavailable in this context.\n");
                return;
            }
            const std::string objective =
                (action == "set" || action == "update") ? remainder : args;
            if (trim(objective).empty()) {
                ctx.append_history_fn(
                    "\nℹ  Usage: /goal <objective>\n"
                    "   Other actions: show, set <objective>, done [note], blocked [note], resume [note], clear.\n");
                return;
            }
            objective_to_submit = objective;
            result = ctx.set_goal_fn(objective);
        }

        ctx.append_history_fn(std::format(
            "\n{}  {}\n",
            result.ok ? "✓" : "✗",
            result.message.empty()
                ? (result.ok ? "Goal updated." : "Goal update failed.")
                : result.message));
        if (result.ok) {
            show_goal();
            if (!objective_to_submit.empty() && ctx.send_user_message_fn) {
                ctx.send_user_message_fn(objective_to_submit);
            }
        }
    }
};

class TodoCommand : public Command {
public:
    std::string get_name() const override { return "/todo"; }
    std::string get_description() const override {
        return "Manage session-backed todo items";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        if (!ctx.list_todos_fn) {
            ctx.append_history_fn("\n✗  Todo management is unavailable in this context.\n");
            return;
        }

        const std::string args = trailing_arguments(ctx.text);
        const auto tokens = split_ascii_whitespace(args);
        auto show_list = [&]() {
            ctx.append_history_fn(format_todos(ctx.list_todos_fn()));
        };

        if (tokens.empty() || to_lower_ascii(tokens.front()) == "list") {
            show_list();
            return;
        }

        const std::string action = to_lower_ascii(tokens.front());
        const std::size_t remainder_offset =
            static_cast<std::size_t>(tokens.front().data() - args.data()) + tokens.front().size();
        const std::string remainder = trim_copy(
            remainder_offset >= args.size() ? std::string_view{} : std::string_view(args).substr(remainder_offset));

        CommandOperationResult result;
        if (action == "add") {
            if (!ctx.add_todo_fn) {
                ctx.append_history_fn("\n✗  Todo creation is unavailable in this context.\n");
                return;
            }
            if (remainder.empty()) {
                ctx.append_history_fn(
                    "\nℹ  Usage: /todo add <text>\n"
                    "   Other actions: list, done <id|index>, undo <id|index>, remove <id|index>, clear-done.\n");
                return;
            }
            result = ctx.add_todo_fn(remainder);
        } else if (action == "done" || action == "complete") {
            if (!ctx.set_todo_completed_fn) {
                ctx.append_history_fn("\n✗  Todo completion is unavailable in this context.\n");
                return;
            }
            if (remainder.empty()) {
                ctx.append_history_fn("\n✗  Usage: /todo done <id|index>\n");
                return;
            }
            result = ctx.set_todo_completed_fn(remainder, true);
        } else if (action == "undo" || action == "open" || action == "reopen") {
            if (!ctx.set_todo_completed_fn) {
                ctx.append_history_fn("\n✗  Todo reopening is unavailable in this context.\n");
                return;
            }
            if (remainder.empty()) {
                ctx.append_history_fn("\n✗  Usage: /todo undo <id|index>\n");
                return;
            }
            result = ctx.set_todo_completed_fn(remainder, false);
        } else if (action == "remove" || action == "rm" || action == "delete") {
            if (!ctx.remove_todo_fn) {
                ctx.append_history_fn("\n✗  Todo removal is unavailable in this context.\n");
                return;
            }
            if (remainder.empty()) {
                ctx.append_history_fn("\n✗  Usage: /todo remove <id|index>\n");
                return;
            }
            result = ctx.remove_todo_fn(remainder);
        } else if (action == "clear-done" || action == "clear_done") {
            if (!ctx.clear_completed_todos_fn) {
                ctx.append_history_fn("\n✗  Clearing completed todos is unavailable in this context.\n");
                return;
            }
            result = ctx.clear_completed_todos_fn();
        } else {
            ctx.append_history_fn(
                "\nℹ  Usage: /todo [list|add <text>|done <id|index>|undo <id|index>|remove <id|index>|clear-done]\n");
            return;
        }

        ctx.append_history_fn(std::format(
            "\n{}  {}\n",
            result.ok ? "✓" : "✗",
            result.message.empty()
                ? (result.ok ? "Todo updated." : "Todo update failed.")
                : result.message));
        if (result.ok) {
            show_list();
        }
    }
};

class MemoryCommand : public Command {
public:
    std::string get_name() const override { return "/memory"; }
    std::string get_description() const override {
        return "Manage opt-in durable memory and auto capture";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        if (!ctx.memory_state_fn) {
            ctx.append_history_fn("\n✗  Memory management is unavailable in this context.\n");
            return;
        }

        const std::string args = trailing_arguments(ctx.text);
        const auto tokens = split_ascii_whitespace(args);
        auto show_memory = [&]() {
            ctx.append_history_fn(format_memory_state(ctx.memory_state_fn()));
        };
        auto refresh_agent_prompt = [&]() {
            if (ctx.agent) {
                ctx.agent->refresh_system_prompt();
            }
        };
        auto usage = [&]() {
            ctx.append_history_fn(
                "\nℹ  Usage: /memory [status|on|off|auto on|auto off|background on|background off|consolidate on|consolidate off|skills on|skills off|thread status|thread use on|thread generate off|review|add <text>|forget <id>|clean|clear|save [file.md]|load <file.md>]\n");
        };
        auto parse_switch = [&](std::string_view text, std::optional<bool>& out) {
            const auto parts = split_ascii_whitespace(text);
            if (parts.empty()) return false;
            const std::string value = to_lower_ascii(parts.front());
            if (value == "on" || value == "enable") {
                out = true;
                return true;
            }
            if (value == "off" || value == "disable") {
                out = false;
                return true;
            }
            return false;
        };

        if (tokens.empty() || to_lower_ascii(tokens.front()) == "status"
            || to_lower_ascii(tokens.front()) == "list") {
            show_memory();
            return;
        }

        const std::string action = to_lower_ascii(tokens.front());
        const std::size_t remainder_offset =
            static_cast<std::size_t>(tokens.front().data() - args.data()) + tokens.front().size();
        const std::string remainder = trim_copy(
            remainder_offset >= args.size() ? std::string_view{} : std::string_view(args).substr(remainder_offset));

        CommandOperationResult result;
        bool changed_prompt = false;
        if (action == "on" || action == "enable") {
            if (!ctx.set_memory_settings_fn) {
                ctx.append_history_fn("\n✗  Memory settings are unavailable.\n");
                return;
            }
            auto state = ctx.memory_state_fn();
            state.settings.enabled = true;
            result = ctx.set_memory_settings_fn(state.settings);
            changed_prompt = result.ok;
        } else if (action == "off" || action == "disable") {
            if (!ctx.set_memory_settings_fn) {
                ctx.append_history_fn("\n✗  Memory settings are unavailable.\n");
                return;
            }
            auto state = ctx.memory_state_fn();
            state.settings.enabled = false;
            state.settings.auto_capture = false;
            state.settings.background_review = false;
            state.settings.consolidation = false;
            state.settings.skill_curation = false;
            result = ctx.set_memory_settings_fn(state.settings);
            changed_prompt = result.ok;
        } else if (action == "auto") {
            if (!ctx.set_memory_settings_fn) {
                ctx.append_history_fn("\n✗  Memory settings are unavailable.\n");
                return;
            }
            const auto auto_tokens = split_ascii_whitespace(remainder);
            if (auto_tokens.empty()) {
                usage();
                return;
            }
            const std::string value = to_lower_ascii(auto_tokens.front());
            if (value != "on" && value != "off" && value != "enable" && value != "disable") {
                usage();
                return;
            }
            auto state = ctx.memory_state_fn();
            state.settings.auto_capture = value == "on" || value == "enable";
            if (state.settings.auto_capture) {
                state.settings.enabled = true;
                state.settings.background_review = true;
            } else {
                state.settings.background_review = false;
            }
            result = ctx.set_memory_settings_fn(state.settings);
            changed_prompt = result.ok;
        } else if (action == "background") {
            if (!ctx.set_memory_settings_fn) {
                ctx.append_history_fn("\n✗  Memory settings are unavailable.\n");
                return;
            }
            std::optional<bool> value;
            if (!parse_switch(remainder, value)) {
                usage();
                return;
            }
            auto state = ctx.memory_state_fn();
            state.settings.background_review = *value;
            if (*value) state.settings.enabled = true;
            result = ctx.set_memory_settings_fn(state.settings);
            changed_prompt = result.ok;
        } else if (action == "consolidate" || action == "consolidation") {
            if (!ctx.set_memory_settings_fn) {
                ctx.append_history_fn("\n✗  Memory settings are unavailable.\n");
                return;
            }
            std::optional<bool> value;
            if (!parse_switch(remainder, value)) {
                usage();
                return;
            }
            auto state = ctx.memory_state_fn();
            state.settings.consolidation = *value;
            if (*value) state.settings.enabled = true;
            result = ctx.set_memory_settings_fn(state.settings);
            changed_prompt = result.ok;
        } else if (action == "skills" || action == "skill") {
            if (!ctx.set_memory_settings_fn) {
                ctx.append_history_fn("\n✗  Memory settings are unavailable.\n");
                return;
            }
            std::optional<bool> value;
            if (!parse_switch(remainder, value)) {
                usage();
                return;
            }
            auto state = ctx.memory_state_fn();
            state.settings.skill_curation = *value;
            if (*value) state.settings.enabled = true;
            result = ctx.set_memory_settings_fn(state.settings);
            changed_prompt = result.ok;
        } else if (action == "thread") {
            if (!ctx.memory_thread_policy_fn || !ctx.set_memory_thread_policy_fn) {
                ctx.append_history_fn("\n✗  Thread memory policy is unavailable.\n");
                return;
            }
            const auto thread_tokens = split_ascii_whitespace(remainder);
            if (thread_tokens.empty() || to_lower_ascii(thread_tokens.front()) == "status") {
                const auto policy = ctx.memory_thread_policy_fn();
                ctx.append_history_fn(std::format(
                    "\n[Filo Memory · Thread]\n  Use memories: {}\n  Generate memories: {}\n  Curate skills: {}\n",
                    policy.use_memories ? "on" : "off",
                    policy.generate_memories ? "on" : "off",
                    policy.curate_skills ? "on" : "off"));
                return;
            }
            if (thread_tokens.size() < 2) {
                usage();
                return;
            }
            const std::string field = to_lower_ascii(thread_tokens[0]);
            std::optional<bool> value;
            if (!parse_switch(thread_tokens[1], value)) {
                usage();
                return;
            }
            auto policy = ctx.memory_thread_policy_fn();
            if (field == "use" || field == "context") {
                policy.use_memories = *value;
                changed_prompt = true;
            } else if (field == "generate" || field == "capture") {
                policy.generate_memories = *value;
            } else if (field == "skills" || field == "curate") {
                policy.curate_skills = *value;
            } else {
                usage();
                return;
            }
            result = ctx.set_memory_thread_policy_fn(policy);
            changed_prompt = result.ok && changed_prompt;
        } else if (action == "review") {
            if (!ctx.run_memory_review_fn) {
                ctx.append_history_fn("\n✗  Memory background review is unavailable.\n");
                return;
            }
            result = ctx.run_memory_review_fn();
        } else if (action == "add" || action == "remember") {
            if (!ctx.add_memory_fn) {
                ctx.append_history_fn("\n✗  Memory storage is unavailable.\n");
                return;
            }
            if (remainder.empty()) {
                ctx.append_history_fn("\n✗  Usage: /memory add <text>\n");
                return;
            }
            result = ctx.add_memory_fn(remainder);
            changed_prompt = result.ok;
        } else if (action == "forget" || action == "remove" || action == "rm" || action == "delete") {
            if (!ctx.forget_memory_fn) {
                ctx.append_history_fn("\n✗  Memory removal is unavailable.\n");
                return;
            }
            if (remainder.empty()) {
                ctx.append_history_fn("\n✗  Usage: /memory forget <id>\n");
                return;
            }
            result = ctx.forget_memory_fn(remainder);
            changed_prompt = result.ok;
        } else if (action == "save" || action == "export") {
            if (!ctx.save_memory_markdown_fn) {
                ctx.append_history_fn("\n✗  Memory markdown export is unavailable.\n");
                return;
            }
            const auto path = remainder.empty()
                ? std::filesystem::path("filo-memory.md")
                : expand_user_path(remainder);
            result = ctx.save_memory_markdown_fn(path);
        } else if (action == "load" || action == "import") {
            if (!ctx.load_memory_markdown_fn) {
                ctx.append_history_fn("\n✗  Memory markdown import is unavailable.\n");
                return;
            }
            if (remainder.empty()) {
                ctx.append_history_fn("\n✗  Usage: /memory load <file.md>\n");
                return;
            }
            result = ctx.load_memory_markdown_fn(expand_user_path(remainder));
            changed_prompt = result.ok;
        } else if (action == "clean") {
            if (!ctx.clean_memory_fn) {
                ctx.append_history_fn("\n✗  Memory cleaning is unavailable.\n");
                return;
            }
            result = ctx.clean_memory_fn();
            changed_prompt = result.ok;
        } else if (action == "clear") {
            if (!ctx.clear_memory_fn) {
                ctx.append_history_fn("\n✗  Memory clearing is unavailable.\n");
                return;
            }
            result = ctx.clear_memory_fn();
            changed_prompt = result.ok;
        } else {
            usage();
            return;
        }

        ctx.append_history_fn(std::format(
            "\n{}  {}\n",
            result.ok ? "✓" : "✗",
            result.message.empty()
                ? (result.ok ? "Memory updated." : "Memory update failed.")
                : result.message));
        if (changed_prompt) {
            refresh_agent_prompt();
        }
        if (result.ok) {
            show_memory();
        }
    }
};

class McpCommand : public Command {
public:
    std::string get_name() const override { return "/mcp"; }
    std::string get_description() const override {
        return "List or manage MCP server overlays";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        if (!ctx.list_mcp_servers_fn) {
            ctx.append_history_fn("\n✗  MCP management is unavailable in this context.\n");
            return;
        }

        const std::string args = trailing_arguments(ctx.text);
        const auto parsed_tokens = split_shell_like_tokens(args);
        if (!parsed_tokens.error.empty()) {
            ctx.append_history_fn(std::format("\n✗  {}\n", parsed_tokens.error));
            return;
        }
        const auto& tokens = parsed_tokens.tokens;
        auto usage = [&]() {
            ctx.append_history_fn(
                "\nℹ  Usage:\n"
                "   /mcp list\n"
                "   /mcp add [--global|--workspace] [--env KEY=VALUE ...] <name> stdio <command> [args...]\n"
                "   /mcp add [--global|--workspace] <name> http <url>\n"
                "   /mcp remove [--global|--workspace] <name>\n");
        };

        if (tokens.empty() || to_lower_ascii(tokens.front()) == "list") {
            ctx.append_history_fn(format_mcp_servers(ctx.list_mcp_servers_fn()));
            return;
        }

        const std::string action = to_lower_ascii(tokens.front());
        std::size_t index = 1;
        core::config::SettingsScope scope = core::config::SettingsScope::Workspace;

        auto consume_scope_options = [&](std::vector<std::string>& env_values) -> bool {
            while (index < tokens.size()) {
                const std::string option = to_lower_ascii(tokens[index]);
                if (option == "--global") {
                    scope = core::config::SettingsScope::User;
                    ++index;
                    continue;
                }
                if (option == "--workspace") {
                    scope = core::config::SettingsScope::Workspace;
                    ++index;
                    continue;
                }
                if (option == "--env") {
                    if (index + 1 >= tokens.size()) {
                        ctx.append_history_fn("\n✗  /mcp --env requires KEY=VALUE.\n");
                        return false;
                    }
                    env_values.push_back(std::string(tokens[index + 1]));
                    index += 2;
                    continue;
                }
                break;
            }
            return true;
        };

        if (action == "add") {
            if (!ctx.add_mcp_server_fn) {
                ctx.append_history_fn("\n✗  MCP persistence is unavailable in this context.\n");
                return;
            }

            std::vector<std::string> env_values;
            if (!consume_scope_options(env_values)) {
                return;
            }
            if (index + 2 >= tokens.size()) {
                usage();
                return;
            }

            core::config::McpServerConfig server;
            server.name = std::string(tokens[index++]);
            server.transport = to_lower_ascii(tokens[index++]);
            server.env = std::move(env_values);

            if (server.transport == "stdio") {
                if (index >= tokens.size()) {
                    usage();
                    return;
                }
                server.command = std::string(tokens[index++]);
                while (index < tokens.size()) {
                    server.args.push_back(std::string(tokens[index++]));
                }
            } else if (server.transport == "http") {
                if (index >= tokens.size()) {
                    usage();
                    return;
                }
                server.url = std::string(tokens[index++]);
                if (index != tokens.size()) {
                    usage();
                    return;
                }
            } else {
                ctx.append_history_fn("\n✗  MCP transport must be `stdio` or `http`.\n");
                return;
            }

            const auto result = ctx.add_mcp_server_fn(server, scope);
            ctx.append_history_fn(std::format(
                "\n{}  {}\n",
                result.ok ? "✓" : "✗",
                result.message.empty()
                    ? (result.ok ? "MCP server saved." : "Failed to save MCP server.")
                    : result.message));
            if (result.ok) {
                ctx.append_history_fn(format_mcp_servers(ctx.list_mcp_servers_fn()));
            }
            return;
        }

        if (action == "remove" || action == "rm" || action == "delete") {
            if (!ctx.remove_mcp_server_fn) {
                ctx.append_history_fn("\n✗  MCP persistence is unavailable in this context.\n");
                return;
            }
            std::vector<std::string> ignored_env;
            if (!consume_scope_options(ignored_env)) {
                return;
            }
            if (index >= tokens.size()) {
                usage();
                return;
            }

            const auto result = ctx.remove_mcp_server_fn(tokens[index], scope);
            ctx.append_history_fn(std::format(
                "\n{}  {}\n",
                result.ok ? "✓" : "✗",
                result.message.empty()
                    ? (result.ok ? "MCP server removed." : "Failed to remove MCP server.")
                    : result.message));
            if (result.ok) {
                ctx.append_history_fn(format_mcp_servers(ctx.list_mcp_servers_fn()));
            }
            return;
        }

        usage();
    }
};

class ShellCommand : public Command {
public:
    std::string get_name() const override { return "!"; }
    std::string get_description() const override { return "Execute a shell command directly"; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        std::string cmd = ctx.text.substr(1);
        auto first = cmd.find_first_not_of(" \t");
        if (first != std::string::npos) cmd = cmd.substr(first);
        if (first == std::string::npos) cmd.clear();
        if (cmd.empty()) {
            ctx.append_history_fn("\n✗  Shell command is empty.\n");
            return;
        }

        if (!ctx.direct_shell_command_fn) {
            ctx.append_history_fn("\n✗  Direct shell execution is unavailable in this context.\n");
            return;
        }

        ctx.direct_shell_command_fn(std::move(cmd));
    }
};

// ---------------------------------------------------------
// Executor Implementation
// ---------------------------------------------------------

CommandExecutor::CommandExecutor() {
    register_command(std::make_unique<AuthCommand>());
    register_command(std::make_unique<LogoutCommand>());
    register_command(std::make_unique<QuitCommand>());
    register_command(std::make_unique<ClearCommand>());
    register_command(std::make_unique<HelpCommand>());
    register_command(std::make_unique<SessionsCommand>());
    register_command(std::make_unique<ResumeCommand>());
    register_command(std::make_unique<ContinueCommand>());
    register_command(std::make_unique<RenameCommand>());
    register_command(std::make_unique<PsCommand>());
    register_command(std::make_unique<StopCommand>());
    register_command(std::make_unique<CompactCommand>());
    register_command(std::make_unique<CompressionCommand>());
    register_command(std::make_unique<UndoCommand>());
    register_command(std::make_unique<RetryCommand>());
    register_command(std::make_unique<ModelCommand>());
    register_command(std::make_unique<ProfileCommand>());
    register_command(std::make_unique<EffortCommand>());
    register_command(std::make_unique<SettingsCommand>());
    register_command(std::make_unique<YoloCommand>());
    register_command(std::make_unique<ToolsCommand>());
    register_command(std::make_unique<UsageCommand>());
    register_command(std::make_unique<CopyCommand>());
    register_command(std::make_unique<HistoryCommand>());
    register_command(std::make_unique<ReviewCommand>());
    register_command(std::make_unique<ExportCommand>());
    register_command(std::make_unique<ForkCommand>());
    register_command(std::make_unique<RewindCommand>());
    register_command(std::make_unique<InitCommand>());
    register_command(std::make_unique<GoalCommand>());
    register_command(std::make_unique<TodoCommand>());
    register_command(std::make_unique<MemoryCommand>());
    register_command(std::make_unique<McpCommand>());
    register_command(std::make_unique<ShellCommand>());
}

void CommandExecutor::register_command(std::unique_ptr<Command> cmd) {
    const auto name = cmd->get_name();
    const auto aliases = cmd->get_aliases();
    std::erase_if(commands_, [&](const std::unique_ptr<Command>& existing) {
        if (existing->get_name() == name) return true;
        const auto existing_aliases = existing->get_aliases();
        const auto overlaps = [&](const std::string& alias) {
            return alias == name
                || std::ranges::find(existing_aliases, alias) != existing_aliases.end();
        };
        if (std::ranges::find(aliases, existing->get_name()) != aliases.end()) {
            return true;
        }
        return std::ranges::any_of(aliases, overlaps);
    });
    commands_.push_back(std::move(cmd));
}

std::vector<CommandDescriptor> CommandExecutor::describe_commands() const {
    std::vector<CommandDescriptor> descriptors;
    descriptors.reserve(commands_.size());

    for (const auto& cmd : commands_) {
        if (!cmd->get_name().starts_with('/')) {
            continue;
        }

        descriptors.push_back(CommandDescriptor{
            .name = cmd->get_name(),
            .aliases = cmd->get_aliases(),
            .description = cmd->get_description(),
            .accepts_arguments = cmd->accepts_arguments(),
        });
    }

    return descriptors;
}

bool CommandExecutor::try_execute(const std::string& raw_input, const CommandContext& ctx) {
    if (raw_input.empty()) return false;

    // Shell passthrough: starts with '!' (possibly after spaces, but typically first char).
    if (raw_input.starts_with("!")) {
        for (auto& cmd : commands_) {
            if (cmd->get_name() == "!") {
                cmd->execute(ctx);
                return true;
            }
        }
        return false;
    }

    // For slash commands, extract only the command token (first word) and trim.
    // This allows "/clear extra-args" to still match "/clear".
    std::string_view sv{raw_input};
    sv = trim(sv);
    if (sv.empty()) return false;

    // Extract the command verb (up to the first space).
    auto space = sv.find(' ');
    std::string_view verb = (space == std::string_view::npos) ? sv : sv.substr(0, space);

    for (auto& cmd : commands_) {
        if (verb == cmd->get_name()) {
            cmd->execute(ctx);
            return true;
        }
        for (const auto& alias : cmd->get_aliases()) {
            if (verb == alias) {
                cmd->execute(ctx);
                return true;
            }
        }
    }

    return false; // Not a recognised command
}

} // namespace core::commands
