#include "SystemPromptEditor.hpp"

#include "core/tools/shell/ShellUtils.hpp"
#include "tui/Text.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <optional>
#include <vector>

#if !defined(_WIN32)
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace tui::editor {
namespace {

constexpr std::string_view kFallbackEditor = "vi";

// GUI editors return immediately unless asked to block, which would hand the
// user back an unedited buffer.
constexpr std::array<std::string_view, 7> kGuiEditors{
    "code", "code-insiders", "codium", "cursor", "subl", "zed", "windsurf",
};

// The vi family sources a user init file that could rewrite or relocate the
// buffer; -i NONE keeps the session hermetic.
constexpr std::array<std::string_view, 3> kViFamilyEditors{"vi", "vim", "nvim"};

// Splits the command for inspection without evaluating expansions or
// substitutions. The original command remains untouched for shell execution.
[[nodiscard]] std::optional<std::vector<std::string>> shell_words(std::string_view command) {
    std::vector<std::string> words;
    std::string word;
    char quote = '\0';
    bool word_started = false;

    for (std::size_t index = 0; index < command.size(); ++index) {
        const char character = command[index];
        if (quote == '\'') {
            if (character == '\'') {
                quote = '\0';
            } else {
                word += character;
            }
            continue;
        }
        if (quote == '"') {
            if (character == '"') {
                quote = '\0';
            } else if (character == '\\') {
                if (++index == command.size()) {
                    return std::nullopt;
                }
                word += command[index];
            } else {
                word += character;
            }
            continue;
        }

        if (std::isspace(static_cast<unsigned char>(character))) {
            if (word_started) {
                words.push_back(std::move(word));
                word.clear();
                word_started = false;
            }
        } else if (character == '\'' || character == '"') {
            quote = character;
            word_started = true;
        } else if (character == '\\') {
            if (++index == command.size()) {
                return std::nullopt;
            }
            word += command[index];
            word_started = true;
        } else {
            word += character;
            word_started = true;
        }
    }

    if (quote != '\0') {
        return std::nullopt;
    }
    if (word_started) {
        words.push_back(std::move(word));
    }
    return words;
}

// The program name without directory or extension, lowercased — the only part
// that identifies which editor family we are dealing with.
[[nodiscard]] std::string program_name(const std::vector<std::string>& words) {
    if (words.empty() || words.front().empty()) {
        return {};
    }
    return to_lower_ascii(std::filesystem::path(words.front()).stem().string());
}

// Invokes `visit` for each non-empty field of `text`, stopping early when it
// returns true. Returns whether `visit` ever accepted a field.
template <typename Visit>
[[nodiscard]] bool any_field(std::string_view text, char separator, Visit&& visit) {
    for (std::size_t begin = 0; begin <= text.size();) {
        const std::size_t end = std::min(text.find(separator, begin), text.size());
        if (end > begin && visit(text.substr(begin, end - begin))) {
            return true;
        }
        begin = end + 1;
    }
    return false;
}

[[nodiscard]] bool contains_flag(
    const std::vector<std::string>& words,
    std::string_view flag) {
    return std::ranges::contains(words, flag);
}

[[nodiscard]] std::string_view environment_editor() {
    for (const char* variable : {"VISUAL", "EDITOR"}) {
        if (const char* value = std::getenv(variable); value != nullptr && *value != '\0') {
            return value;
        }
    }
    return kFallbackEditor;
}

[[nodiscard]] int normalize_exit_code(int status) {
#if defined(_WIN32)
    return status;
#else
    if (status == -1) {
        return -1;
    }
    if (WIFEXITED(status)) {
        return WEXITSTATUS(status);
    }
    if (WIFSIGNALED(status)) {
        return 128 + WTERMSIG(status);
    }
    return status;
#endif
}

} // namespace

EditorDescriptor system_prompt_editor_descriptor() noexcept {
    return EditorDescriptor{
        .id = "system",
        .label = "System",
        .description = "Edit the draft in $VISUAL / $EDITOR (vim, nano, ...).",
        .launch = LaunchMode::Terminal,
    };
}

std::string resolve_system_editor_command(std::string_view configured_command) {
    return configured_command.empty()
        ? std::string(environment_editor())
        : std::string(configured_command);
}

std::string build_editor_invocation(std::string_view command, std::string_view file_path) {
    std::string invocation(command);
    const auto words = shell_words(command).value_or(std::vector<std::string>{});
    const std::string program = program_name(words);

    if (std::ranges::contains(kGuiEditors, program)
        && !contains_flag(words, "--wait")
        && !contains_flag(words, "-w")) {
        invocation += program == "subl" ? " -w" : " --wait";
    }
    if (std::ranges::contains(kViFamilyEditors, program) && !contains_flag(words, "-i")) {
        invocation += " -i NONE";
    }

    invocation += " '";
    invocation += core::tools::detail::shell_single_quote(file_path);
    invocation += "'";
    return invocation;
}

bool editor_command_is_executable(std::string_view command) {
    const auto words = shell_words(command);
    if (!words || words->empty() || words->front().empty()) {
        return false;
    }
    const std::string& program = words->front();

#if defined(_WIN32)
    return true;
#else
    if (program.contains('/')) {
        return ::access(program.c_str(), X_OK) == 0;
    }

    const char* path = std::getenv("PATH");
    if (path == nullptr) {
        return false;
    }
    return any_field(path, ':', [&program](std::string_view directory) {
        const std::filesystem::path candidate = std::filesystem::path(directory) / program;
        return ::access(candidate.c_str(), X_OK) == 0;
    });
#endif
}

EditorDescriptor SystemPromptEditor::descriptor() const noexcept {
    return system_prompt_editor_descriptor();
}

EditResult SystemPromptEditor::edit(const EditContext& context) {
    if (!context.with_terminal) {
        return EditResult::failed("No terminal is available for the system editor.");
    }
    if (context.cancellation.stop_requested()) {
        return EditResult::cancelled();
    }

    context.report_progress(EditProgress::Editing);
    const std::string invocation = build_editor_invocation(
        resolve_system_editor_command(command_),
        context.file.string());

    int status = -1;
    context.with_terminal([&invocation, &status]() { status = std::system(invocation.c_str()); });

    if (const int exit_code = normalize_exit_code(status); exit_code != 0) {
        return EditResult::failed(
            std::format("External editor exited with status {}.", exit_code));
    }
    return EditResult::saved();
}

} // namespace tui::editor
