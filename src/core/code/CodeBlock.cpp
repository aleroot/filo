#include "CodeBlock.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <ranges>

namespace core::code {
namespace {

struct Fence {
    char marker = '\0';
    std::size_t length = 0;
    std::string_view info;
};

[[nodiscard]] std::string_view trim(std::string_view value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.remove_suffix(1);
    }
    return value;
}

[[nodiscard]] std::optional<Fence> opening_fence(std::string_view line) {
    std::size_t indent = 0;
    while (indent < line.size() && indent < 4 && line[indent] == ' ') ++indent;
    if (indent > 3 || indent == line.size()) return std::nullopt;

    const char marker = line[indent];
    if (marker != '`' && marker != '~') return std::nullopt;
    std::size_t end = indent;
    while (end < line.size() && line[end] == marker) ++end;
    const std::size_t length = end - indent;
    if (length < 3) return std::nullopt;

    const auto info = trim(line.substr(end));
    if (marker == '`' && info.find('`') != std::string_view::npos) return std::nullopt;
    return Fence{.marker = marker, .length = length, .info = info};
}

[[nodiscard]] bool closing_fence(std::string_view line, const Fence& fence) {
    std::size_t indent = 0;
    while (indent < line.size() && indent < 4 && line[indent] == ' ') ++indent;
    if (indent > 3 || indent == line.size() || line[indent] != fence.marker) return false;
    std::size_t end = indent;
    while (end < line.size() && line[end] == fence.marker) ++end;
    return end - indent >= fence.length && trim(line.substr(end)).empty();
}

[[nodiscard]] std::string normalized_language(std::string_view info) {
    info = trim(info);
    const auto end = info.find_first_of(" \t{");
    std::string language(info.substr(0, end));
    std::ranges::transform(language, language.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return language;
}

[[nodiscard]] std::optional<std::filesystem::path> executable_in_path(
    std::string_view executable) {
    if (executable.empty()) return std::nullopt;
    const char* path_value = std::getenv("PATH");
    if (path_value == nullptr) return std::nullopt;
    std::string_view path(path_value);
    while (true) {
        const auto separator = path.find(':');
        const auto directory = path.substr(0, separator);
        std::error_code ec;
        const auto candidate = std::filesystem::path(
            directory.empty() ? "." : std::string(directory)) / executable;
        const auto permissions = std::filesystem::status(candidate, ec).permissions();
        if (!ec && std::filesystem::is_regular_file(candidate, ec)
            && permissions != std::filesystem::perms::unknown
            && (permissions & (std::filesystem::perms::owner_exec
                               | std::filesystem::perms::group_exec
                               | std::filesystem::perms::others_exec))
                != std::filesystem::perms::none) {
            auto resolved = std::filesystem::weakly_canonical(candidate, ec);
            if (ec) resolved = std::filesystem::absolute(candidate, ec);
            return ec ? candidate : resolved;
        }
        if (separator == std::string_view::npos) break;
        path.remove_prefix(separator + 1);
    }
    return std::nullopt;
}

[[nodiscard]] std::string strip_console_prompts(std::string_view source) {
    std::string result;
    while (!source.empty()) {
        const auto newline = source.find('\n');
        auto line = source.substr(0, newline);
        if (line.starts_with("$ ")) line.remove_prefix(2);
        result.append(line);
        if (newline == std::string_view::npos) break;
        result.push_back('\n');
        source.remove_prefix(newline + 1);
    }
    return result;
}

} // namespace

std::vector<FencedCodeBlock> extract_fenced_code_blocks(std::string_view markdown) {
    std::vector<FencedCodeBlock> blocks;
    std::optional<Fence> active;
    std::string source;
    std::size_t opening_line = 0;
    std::size_t line_number = 1;

    while (!markdown.empty()) {
        const auto newline = markdown.find('\n');
        std::string_view line = markdown.substr(0, newline);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

        if (!active.has_value()) {
            if (auto fence = opening_fence(line); fence.has_value()) {
                active = *fence;
                source.clear();
                opening_line = line_number;
            }
        } else if (closing_fence(line, *active)) {
            blocks.push_back(FencedCodeBlock{
                .ordinal = blocks.size() + 1,
                .first_line = opening_line + 1,
                .last_line = line_number > opening_line + 1 ? line_number - 1 : opening_line,
                .language = normalized_language(active->info),
                .info = std::string(active->info),
                .source = std::move(source),
            });
            active.reset();
            source.clear();
        } else {
            source.append(line);
            if (newline != std::string_view::npos) source.push_back('\n');
        }

        if (newline == std::string_view::npos) break;
        markdown.remove_prefix(newline + 1);
        ++line_number;
    }
    return blocks;
}

std::expected<ExecutionPlan, ExecutionPlanError> plan_execution(FencedCodeBlock block) {
    struct Runtime {
        std::string_view interpreter;
        std::string_view extension;
        std::vector<std::string> arguments;
    };

    const std::string& language = block.language;
    std::optional<Runtime> runtime;
    bool console = false;
    if (language.empty() || language == "sh" || language == "shell"
        || language == "bash" || language == "console" || language == "terminal") {
        runtime = Runtime{.interpreter = "bash", .extension = "sh",
                          .arguments = {"--noprofile", "--norc"}};
        console = language == "console" || language == "terminal";
    } else if (language == "zsh") {
        runtime = Runtime{.interpreter = "zsh", .extension = "zsh"};
    } else if (language == "python" || language == "python3" || language == "py") {
        runtime = Runtime{.interpreter = "python3", .extension = "py", .arguments = {"-u"}};
    } else if (language == "javascript" || language == "js" || language == "node") {
        runtime = Runtime{.interpreter = "node", .extension = "js"};
    } else if (language == "ruby" || language == "rb") {
        runtime = Runtime{.interpreter = "ruby", .extension = "rb"};
    } else if (language == "perl" || language == "pl") {
        runtime = Runtime{.interpreter = "perl", .extension = "pl"};
    } else if (language == "powershell" || language == "pwsh") {
        runtime = Runtime{.interpreter = "pwsh", .extension = "ps1",
                          .arguments = {"-NoLogo", "-NoProfile", "-File"}};
    }

    if (!runtime.has_value()) {
        const std::string reason = language.empty()
            ? "No executable language was declared."
            : "The language '" + language + "' is not executable by Filo yet.";
        return std::unexpected(ExecutionPlanError{
            .block = std::move(block),
            .reason = reason,
        });
    }
    const auto interpreter_path = executable_in_path(runtime->interpreter);
    if (!interpreter_path.has_value()) {
        return std::unexpected(ExecutionPlanError{
            .block = std::move(block),
            .reason = "Interpreter '" + std::string(runtime->interpreter) + "' was not found in PATH.",
        });
    }
    if (trim(block.source).empty()) {
        return std::unexpected(ExecutionPlanError{
            .block = std::move(block),
            .reason = "The code block is empty.",
        });
    }

    auto prepared = console ? strip_console_prompts(block.source) : block.source;
    auto arguments = std::move(runtime->arguments);
    return ExecutionPlan{
        .block = std::move(block),
        .interpreter = std::string(runtime->interpreter),
        .interpreter_path = *interpreter_path,
        .interpreter_arguments = std::move(arguments),
        .script_extension = std::string(runtime->extension),
        .prepared_source = std::move(prepared),
    };
}

std::string code_block_preview(std::string_view source, std::size_t max_characters) {
    std::string result;
    bool previous_space = false;
    for (const unsigned char ch : source) {
        const bool space = std::isspace(ch) != 0;
        if (space) {
            if (!result.empty() && !previous_space) result.push_back(' ');
        } else if (std::iscntrl(ch) == 0) {
            result.push_back(static_cast<char>(ch));
        }
        previous_space = space;
        if (result.size() >= max_characters) {
            if (max_characters >= 3) {
                result.resize(max_characters - 3);
                result += "...";
            }
            break;
        }
    }
    while (!result.empty() && result.back() == ' ') result.pop_back();
    return result.empty() ? "<empty>" : result;
}

std::string sanitize_terminal_output(std::string_view output) {
    enum class State { Text, Escape, Csi, Osc, OscEscape };
    State state = State::Text;
    std::string clean;
    clean.reserve(output.size());
    for (const unsigned char ch : output) {
        switch (state) {
        case State::Text:
            if (ch == 0x1b) state = State::Escape;
            else if (ch == '\n' || ch == '\t') clean.push_back(static_cast<char>(ch));
            else if (ch == '\r') {
                if (clean.empty() || clean.back() != '\n') clean.push_back('\n');
            } else if (std::iscntrl(ch) == 0) clean.push_back(static_cast<char>(ch));
            break;
        case State::Escape:
            if (ch == '[') state = State::Csi;
            else if (ch == ']') state = State::Osc;
            else state = State::Text;
            break;
        case State::Csi:
            if (ch >= 0x40 && ch <= 0x7e) state = State::Text;
            break;
        case State::Osc:
            if (ch == 0x07) state = State::Text;
            else if (ch == 0x1b) state = State::OscEscape;
            break;
        case State::OscEscape:
            state = ch == '\\' ? State::Text : State::Osc;
            break;
        }
    }
    return clean;
}

} // namespace core::code
