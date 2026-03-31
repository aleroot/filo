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
#if !defined(_WIN32)
#include <sys/wait.h>
#endif
#if !defined(_WIN32) && !defined(__APPLE__)
#include <unistd.h>
#endif
#include "core/auth/AuthenticationManager.hpp"
#include "core/config/ConfigManager.hpp"

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

FILE* open_pipe_read(const char* command) {
#if defined(_WIN32)
    return _popen(command, "r");
#else
    return popen(command, "r");
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

int normalize_pipe_exit_status(int status) {
    if (status == -1) {
        return -1;
    }
#if defined(_WIN32)
    return status;
#else
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return status;
#endif
}

bool run_command_capture(std::string_view command,
                         std::string& output,
                         std::string& error,
                         int* exit_code = nullptr) {
    output.clear();
    error.clear();

    const std::string command_str(command);
    FILE* pipe = open_pipe_read(command_str.c_str());
    if (!pipe) {
        error = std::format("could not execute `{}`", command_str);
        if (exit_code) *exit_code = -1;
        return false;
    }

    std::array<char, 4096> buffer{};
    while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    const int status = close_pipe(pipe);
    const int normalized = normalize_pipe_exit_status(status);
    if (exit_code) {
        *exit_code = normalized;
    }
    if (normalized != 0) {
        error = std::format("`{}` exited with status {}", command_str, normalized);
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

std::string export_history_markdown(const std::vector<core::llm::Message>& messages) {
    const auto now = std::chrono::system_clock::now();
    const auto sec = std::chrono::floor<std::chrono::seconds>(now);

    std::string out;
    out.reserve(2048 + messages.size() * 256);
    out += "# Filo Conversation Export\n\n";
    out += std::format("- Exported: {:%Y-%m-%d %H:%M:%S}\n", sec);
    out += std::format("- Message count: {}\n\n", messages.size());

    for (std::size_t i = 0; i < messages.size(); ++i) {
        const auto& msg = messages[i];
        out += std::format("## {}. {}\n\n", i + 1, role_title(msg.role));

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

std::string base64_encode(std::string_view input) {
    static constexpr std::string_view kAlphabet =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string out;
    out.reserve(((input.size() + 2) / 3) * 4);

    std::size_t i = 0;
    while (i + 2 < input.size()) {
        const auto b0 = static_cast<unsigned char>(input[i++]);
        const auto b1 = static_cast<unsigned char>(input[i++]);
        const auto b2 = static_cast<unsigned char>(input[i++]);

        out.push_back(kAlphabet[(b0 >> 2) & 0x3F]);
        out.push_back(kAlphabet[((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F)]);
        out.push_back(kAlphabet[((b1 & 0x0F) << 2) | ((b2 >> 6) & 0x03)]);
        out.push_back(kAlphabet[b2 & 0x3F]);
    }

    const std::size_t remaining = input.size() - i;
    if (remaining == 1) {
        const auto b0 = static_cast<unsigned char>(input[i]);
        out.push_back(kAlphabet[(b0 >> 2) & 0x3F]);
        out.push_back(kAlphabet[(b0 & 0x03) << 4]);
        out.push_back('=');
        out.push_back('=');
    } else if (remaining == 2) {
        const auto b0 = static_cast<unsigned char>(input[i]);
        const auto b1 = static_cast<unsigned char>(input[i + 1]);
        out.push_back(kAlphabet[(b0 >> 2) & 0x3F]);
        out.push_back(kAlphabet[((b0 & 0x03) << 4) | ((b1 >> 4) & 0x0F)]);
        out.push_back(kAlphabet[(b1 & 0x0F) << 2]);
        out.push_back('=');
    }

    return out;
}

std::optional<std::string> copy_via_osc52(std::string_view text) {
    std::ofstream tty("/dev/tty", std::ios::binary | std::ios::out);
    if (!tty) {
        return std::string("failed to open /dev/tty for OSC52 copy");
    }

    const std::string payload = base64_encode(text);
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

std::optional<std::string> choose_auth_provider_menu(const std::vector<std::string>& providers) {
    if (providers.empty()) {
        return std::nullopt;
    }

    while (true) {
        std::cout << "\nAuthentication Providers\n";
        for (std::size_t i = 0; i < providers.size(); ++i) {
            std::cout << "  " << (i + 1) << ". " << providers[i] << "\n";
        }
        std::cout << "Select a provider [1-" << providers.size() << "] or press Enter to cancel: ";
        std::cout.flush();

        std::string line;
        if (!std::getline(std::cin, line)) {
            return std::nullopt;
        }

        const std::string_view input = trim(line);
        if (input.empty()) {
            return std::nullopt;
        }

        bool digits_only = true;
        for (const char ch : input) {
            if (!std::isdigit(static_cast<unsigned char>(ch))) {
                digits_only = false;
                break;
            }
        }
        if (!digits_only) {
            std::cout << "Invalid selection. Please enter a number.\n";
            continue;
        }

        try {
            const std::size_t selected = static_cast<std::size_t>(std::stoul(std::string(input)));
            if (selected >= 1 && selected <= providers.size()) {
                return providers[selected - 1];
            }
        } catch (...) {
            // Fall through to generic invalid message below.
        }

        std::cout << "Invalid selection. Choose a number between 1 and "
                  << providers.size() << ".\n";
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
            auto auth_manager = core::auth::AuthenticationManager::create_with_defaults(config_dir);
            const auto result = auth_manager.login(provider);

            std::cout << "\nAuthentication successful!\n";
            std::cout << "Provider: " << result.provider << "\n";
            std::cout << "Credentials saved in: " << config_dir << "\n";

            std::string persist_error;
            if (core::config::ConfigManager::get_instance().persist_login_profile(
                    provider, &persist_error)) {
                const auto& config =
                    core::config::ConfigManager::get_instance().get_config();
                state->profile_persisted = true;
                state->selected_provider = config.default_provider;
                std::cout << "Default provider switched to: "
                          << state->selected_provider << " (OAuth enabled)\n";
            } else {
                state->profile_error = std::move(persist_error);
                std::cout << "Warning: authenticated, but couldn't update default profile: "
                          << state->profile_error << "\n";
            }

            if (state->profile_persisted && switch_model_fn) {
                state->model_switch_message = switch_model_fn(state->selected_provider);
            }

            for (const auto& hint : result.hints) {
                std::cout << "Hint: " << hint << "\n";
            }
            wait_for_enter();

            state->success = true;
            state->provider = result.provider;
            state->hints = result.hints;
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
            "\xe2\x9c\x93  Selected provider is now {} with OAuth enabled.\n",
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
    std::string get_description() const override { return "Interactive authentication wizard (use /auth or /auth <provider>)"; }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        const std::string config_dir = core::config::ConfigManager::get_instance().get_config_dir();
        auto auth_manager = core::auth::AuthenticationManager::create_with_defaults(config_dir);
        const auto available = auth_manager.available_login_providers();

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
                *chosen_provider = choose_auth_provider_menu(available);
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
    std::string get_description() const override { return "Restore a saved session by ID or index"; }
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
            "  /resume [id]        Restore a saved session by ID or index\n"
            "  /compact            Summarise and compress the conversation history\n"
            "  /undo               Remove the last user message from history\n"
            "  /retry              Re-send the last user message\n"
            "  /model [selector]   Open model picker or switch manual/router/provider/model target\n"
            "  /settings           Open the settings panel for user/workspace preferences\n"
            "  /yolo [on|off]      Toggle or set auto-approval for sensitive tools\n"
            "  /copy               Copy the latest assistant response to clipboard\n"
            "  /history            Show or manage input history (use 'clear' to erase)\n"
            "  /review [--staged]  Run AI code review on git diff (HEAD or staged)\n"
            "  /export [file]      Export the current conversation to Markdown\n"
            "  /fork               Branch the current conversation into a new session\n"
            "  /init [provider] [options]  Scaffold .filo/config.json (and optional FILO.md)\n"
            "  !<command>          Execute a shell command  (e.g., !ls -la)\n"
            "\n[Keyboard Shortcuts]\n"
            "  ↑/↓      Navigate input history (previous/next prompt)\n"
            "  Ctrl+P/N Navigate input history (alternative to arrows)\n"
            "  Ctrl+F   Search the conversation history\n"
            "  F2       Cycle agent mode (BUILD → DEBUG → RESEARCH → EXECUTE)\n"
            "  Ctrl+Y   Toggle YOLO auto-approval mode\n"
            "  Ctrl+X   Open current input in external editor ($VISUAL/$EDITOR)\n"
            "  Ctrl+D   Delete char right; when input is empty press twice to quit\n"
            "  Ctrl+L   Clear the screen\n"
            "  Ctrl+C   Interrupt / quit\n"
        );
    }
};

// Summarise the conversation to save context window space.
class CompactCommand : public Command {
public:
    std::string get_name() const override { return "/compact"; }
    std::vector<std::string> get_aliases() const override { return {"/compress"}; }
    std::string get_description() const override { return "Compress conversation history into a summary"; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        if (!ctx.agent) {
            ctx.append_history_fn("\n✗  No agent available for compaction.\n");
            return;
        }

        std::string summary_prompt =
            "Please summarise our conversation so far in a concise paragraph, "
            "preserving all key decisions, code changes, and open tasks. "
            "Do NOT call any tools. Reply ONLY with the summary text.";

        ctx.append_history_fn("\n»  Compacting conversation\xe2\x80\xa6\n");

        auto append_fn = ctx.append_history_fn;
        auto summary = std::make_shared<std::string>();
        ctx.agent->send_message(
            summary_prompt,
            [summary](const std::string& chunk) {
                *summary += chunk;
            },
            [](const std::string&, const std::string&) {},
            [append_fn, agent = ctx.agent, summary]() {
                std::string compacted = *summary;
                std::string_view trimmed = trim(compacted);
                if (trimmed.empty()
                    || trimmed.find("[Error") != std::string_view::npos
                    || trimmed.find("[HTTP") != std::string_view::npos
                    || trimmed.find("[Anthropic") != std::string_view::npos
                    || trimmed.find("[Mistral") != std::string_view::npos) {
                    append_fn("\n✗  History compaction failed.\n");
                    return;
                }
                agent->compact_history(std::string(trimmed));
                append_fn("\n✓  History compacted and summary preserved.\n");
            }
        );
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
        std::string last = ctx.agent->last_user_message();
        if (last.empty()) {
            ctx.append_history_fn("\n✗  No previous message to retry.\n");
            return;
        }
        ctx.append_history_fn(std::format("\n»  Retrying: {}\n", last));

        auto append_fn = ctx.append_history_fn;
        ctx.agent->send_message(last,
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

class CopyCommand : public Command {
public:
    std::string get_name() const override { return "/copy"; }
    std::string get_description() const override {
        return "Copy the latest completed assistant response to clipboard";
    }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        if (!ctx.latest_assistant_output_fn) {
            ctx.append_history_fn("\n✗  Copy is unavailable in this session.\n");
            return;
        }

        const std::string latest_output = ctx.latest_assistant_output_fn();
        if (trim(latest_output).empty()) {
            ctx.append_history_fn(
                "\nℹ  Nothing to copy yet. Wait for the first completed assistant response.\n");
            return;
        }

        auto copy_fn = ctx.copy_to_clipboard_fn;
        if (!copy_fn) {
            copy_fn = [](std::string_view text) {
                return copy_text_to_clipboard(text);
            };
        }

        if (const auto error = copy_fn(latest_output); error.has_value()) {
            ctx.append_history_fn(std::format(
                "\n✗  Could not copy response: {}\n",
                *error));
            return;
        }

        ctx.append_history_fn("\n✓  Copied the latest assistant response to clipboard.\n");
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
        return "Review git diff with AI (/review [--staged|--cached])";
    }
    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        if (!ctx.agent) {
            ctx.append_history_fn("\n✗  Review requires an active agent session.\n");
            return;
        }

        bool staged = false;
        for (const auto token : split_ascii_whitespace(trailing_arguments(ctx.text))) {
            const std::string option = to_lower_ascii(token);
            if (option == "--staged" || option == "--cached") {
                staged = true;
                continue;
            }
            if (option == "-h" || option == "--help") {
                ctx.append_history_fn(
                    "\nℹ  Usage: /review [--staged|--cached]\n"
                    "   /review           Review working tree changes against HEAD.\n"
                    "   /review --staged  Review only staged changes.\n");
                return;
            }

            ctx.append_history_fn(std::format(
                "\n✗  Unknown /review option '{}'. Use /review [--staged|--cached].\n",
                std::string(token)));
            return;
        }

        const std::string command = staged
            ? "git diff --staged 2>&1"
            : "git diff HEAD 2>&1";

        std::string diff_output;
        std::string error;
        int exit_code = 0;
        if (!run_command_capture(command, diff_output, error, &exit_code)) {
            std::string detail = trim_copy(diff_output);
            if (detail.empty()) {
                detail = error.empty() ? "unknown git error" : error;
            }
            ctx.append_history_fn(std::format(
                "\n✗  Could not collect a git diff ({}).\n",
                detail));
            return;
        }

        if (trim(diff_output).empty()) {
            ctx.append_history_fn(staged
                ? "\nℹ  No staged changes to review.\n"
                : "\nℹ  No local changes to review.\n");
            return;
        }

        constexpr std::size_t kMaxDiffChars = 120000;
        bool truncated = false;
        if (diff_output.size() > kMaxDiffChars) {
            diff_output.resize(kMaxDiffChars);
            truncated = true;
        }

        std::string prompt;
        prompt.reserve(diff_output.size() + 1024);
        prompt += "You are doing a code review.\n";
        prompt += staged
            ? "Review the staged git diff below.\n"
            : "Review the working-tree git diff below (vs HEAD).\n";
        prompt += "Focus on bugs, regressions, security issues, and missing tests.\n";
        prompt += "Output findings first, sorted by severity, with file/line pointers when possible.\n";
        if (truncated) {
            prompt += "Note: the diff is truncated; call out uncertainty for unseen parts.\n";
        }
        prompt += "\n";
        prompt += markdown_fence(diff_output, "diff");

        ctx.append_history_fn(std::format(
            "\n»  Reviewing {} diff…\n",
            staged ? "staged" : "working-tree"));

        auto append_fn = ctx.append_history_fn;
        ctx.agent->send_message(
            prompt,
            [append_fn](const std::string& chunk) {
                append_fn(chunk);
            },
            [append_fn](const std::string& name, const std::string& args) {
                append_fn(std::format("\n[Tool] {} {}\n", name, args));
            },
            [append_fn]() {
                append_fn("\n");
            });
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
            messages.size(),
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
    "auto_compact_threshold": 50000
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

class ShellCommand : public Command {
public:
    std::string get_name() const override { return "!"; }
    std::string get_description() const override { return "Execute a shell command directly"; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();
        ctx.append_history_fn(std::format("\n»  {}\n", ctx.text));

        std::string cmd = ctx.text.substr(1);
        auto first = cmd.find_first_not_of(" \t");
        if (first != std::string::npos) cmd = cmd.substr(first);
        if (cmd.empty()) return;

        auto append_fn = ctx.append_history_fn;
        std::thread([append_fn, cmd]() {
            std::array<char, 1024> buffer;
            std::string result;
            auto pclose_wrapper = [](FILE* f) { pclose(f); };
            std::unique_ptr<FILE, decltype(pclose_wrapper)> pipe(
                popen((cmd + " 2>&1").c_str(), "r"), pclose_wrapper);
            if (pipe) {
                while (fgets(buffer.data(), buffer.size(), pipe.get()) != nullptr) {
                    result += buffer.data();
                }
            } else {
                result = "Error: could not execute command.\n";
            }

            if (result.empty() || result.back() != '\n') result += '\n';
            append_fn(result);
        }).detach();
    }
};

// ---------------------------------------------------------
// Executor Implementation
// ---------------------------------------------------------

CommandExecutor::CommandExecutor() {
    register_command(std::make_unique<AuthCommand>());
    register_command(std::make_unique<QuitCommand>());
    register_command(std::make_unique<ClearCommand>());
    register_command(std::make_unique<HelpCommand>());
    register_command(std::make_unique<SessionsCommand>());
    register_command(std::make_unique<ResumeCommand>());
    register_command(std::make_unique<CompactCommand>());
    register_command(std::make_unique<UndoCommand>());
    register_command(std::make_unique<RetryCommand>());
    register_command(std::make_unique<ModelCommand>());
    register_command(std::make_unique<SettingsCommand>());
    register_command(std::make_unique<YoloCommand>());
    register_command(std::make_unique<CopyCommand>());
    register_command(std::make_unique<HistoryCommand>());
    register_command(std::make_unique<ReviewCommand>());
    register_command(std::make_unique<ExportCommand>());
    register_command(std::make_unique<ForkCommand>());
    register_command(std::make_unique<InitCommand>());
    register_command(std::make_unique<ShellCommand>());
}

void CommandExecutor::register_command(std::unique_ptr<Command> cmd) {
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
