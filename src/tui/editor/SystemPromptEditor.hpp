#pragma once

#include "PromptEditor.hpp"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace tui::editor {

// The portable backend: whatever the user already configured through the
// standard `VISUAL` / `EDITOR` convention, or an explicit command such as
// "vim", "nano" or "code --wait".
class SystemPromptEditor final : public PromptEditor {
public:
    // An empty command defers to $VISUAL, then $EDITOR, then "vi" — resolved at
    // edit time so the user can change it without restarting Filo.
    explicit SystemPromptEditor(std::string command = {}) : command_(std::move(command)) {}

    [[nodiscard]] EditorDescriptor descriptor() const noexcept override;
    [[nodiscard]] EditResult edit(const EditContext& context) override;

private:
    std::string command_;
};

[[nodiscard]] EditorDescriptor system_prompt_editor_descriptor() noexcept;

// ── Exposed for testing ──────────────────────────────────────────────────────

// Resolves the editor command, honouring $VISUAL then $EDITOR then "vi".
[[nodiscard]] std::string resolve_system_editor_command(std::string_view configured_command);

// Builds the shell invocation, adding the flags a well-behaved wrapper needs:
// --wait for GUI editors (so they block) and -i NONE for the vi family (so a
// user's init file cannot repurpose the buffer).
[[nodiscard]] std::string build_editor_invocation(
    std::string_view command,
    std::string_view file_path);

// True when the first token of `command` names an executable reachable from the
// current PATH (or an executable path).
[[nodiscard]] bool editor_command_is_executable(std::string_view command);

} // namespace tui::editor
