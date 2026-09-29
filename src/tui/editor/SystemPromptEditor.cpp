#include "SystemPromptEditor.hpp"

#include "core/tools/shell/ShellUtils.hpp"
#include "tui/CommandLookup.hpp"
#include "tui/Text.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <optional>
#include <vector>

namespace tui::editor {
namespace {

using tui::shell_words;

constexpr std::string_view kFallbackEditor = "vi";

// GUI editors return immediately unless asked to block, which would hand the
// user back an unedited buffer.
constexpr std::array<std::string_view, 7> kGuiEditors{
    "code", "code-insiders", "codium", "cursor", "subl", "zed", "windsurf",
};

// The vi family sources a user init file that could rewrite or relocate the
// buffer; -i NONE keeps the session hermetic.
constexpr std::array<std::string_view, 3> kViFamilyEditors{"vi", "vim", "nvim"};

// The program name without directory or extension, lowercased — the only part
// that identifies which editor family we are dealing with.
[[nodiscard]] std::string program_name(const std::vector<std::string>& words) {
    if (words.empty() || words.front().empty()) {
        return {};
    }
    return to_lower_ascii(std::filesystem::path(words.front()).stem().string());
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

    if (const int exit_code = tui::child_exit_code(status); exit_code != 0) {
        return EditResult::failed(
            std::format("External editor exited with status {}.", exit_code));
    }
    return EditResult::saved();
}

} // namespace tui::editor
