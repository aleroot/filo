#pragma once

#include "PromptEditor.hpp"

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tui::editor {

// The set of prompt-editor backends compiled into this build.
//
// Registration is the single extension point: a new IDE integration adds one
// translation unit and one register() call, and it automatically appears in
// /settings, in config validation, and in the key binding — no TUI change.
class PromptEditorCatalog {
public:
    using Factory = std::function<std::unique_ptr<PromptEditor>()>;

    void register_editor(EditorDescriptor descriptor, Factory factory);

    [[nodiscard]] std::span<const EditorDescriptor> descriptors() const noexcept {
        return descriptors_;
    }

    // Case-insensitive lookup of a configured id.
    [[nodiscard]] const EditorDescriptor* find(std::string_view id) const noexcept;

    // Returns nullptr when `id` is not a registered backend.
    [[nodiscard]] std::unique_ptr<PromptEditor> create(std::string_view id) const;

private:
    struct Entry {
        EditorDescriptor descriptor;
        Factory factory;
    };

    std::vector<Entry> entries_;
    std::vector<EditorDescriptor> descriptors_;
};

// Backends available on this platform. Lampo is present in macOS builds only.
[[nodiscard]] const PromptEditorCatalog& builtin_prompt_editors();

// The id used when `prompt_editor` is unset.
[[nodiscard]] constexpr std::string_view default_prompt_editor_id() noexcept { return "system"; }

// Outcome of interpreting a configured `prompt_editor` value.
struct EditorSelection {
    std::unique_ptr<PromptEditor> editor;  // Never null.
    std::string warning;                   // Non-empty when the value was not usable verbatim.
};

// Resolves a configured value to a usable backend:
//   * empty            → the default backend;
//   * a registered id  → that backend;
//   * any other value  → treated as an explicit editor command ("nvim",
//                        "code --wait", ...) when it resolves on PATH;
//   * otherwise        → the default backend, with a warning for the user.
[[nodiscard]] EditorSelection select_prompt_editor(
    const PromptEditorCatalog& catalog,
    std::string_view configured);

} // namespace tui::editor
