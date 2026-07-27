#include "PromptEditorCatalog.hpp"

#include "SystemPromptEditor.hpp"
#include "tui/Text.hpp"

#include <algorithm>
#include <format>
#include <utility>

#if defined(__APPLE__)
#include "LampoPromptEditor.hpp"
#endif

namespace tui::editor {
namespace {

[[nodiscard]] PromptEditorCatalog make_builtin_catalog() {
    PromptEditorCatalog catalog;
    catalog.register_editor(
        system_prompt_editor_descriptor(),
        [] { return std::make_unique<SystemPromptEditor>(); });
#if defined(__APPLE__)
    catalog.register_editor(lampo_prompt_editor_descriptor(), &make_lampo_prompt_editor);
#endif
    return catalog;
}

} // namespace

void PromptEditorCatalog::register_editor(EditorDescriptor descriptor, Factory factory) {
    entries_.push_back(Entry{descriptor, std::move(factory)});
    descriptors_.push_back(descriptor);
}

const EditorDescriptor* PromptEditorCatalog::find(std::string_view id) const noexcept {
    const std::string needle = to_lower_ascii(id);
    const auto match = std::ranges::find_if(
        entries_,
        [&needle](const Entry& entry) { return entry.descriptor.id == needle; });
    return match == entries_.end() ? nullptr : &match->descriptor;
}

std::unique_ptr<PromptEditor> PromptEditorCatalog::create(std::string_view id) const {
    const std::string needle = to_lower_ascii(id);
    const auto match = std::ranges::find_if(
        entries_,
        [&needle](const Entry& entry) { return entry.descriptor.id == needle; });
    return match == entries_.end() ? nullptr : match->factory();
}

const PromptEditorCatalog& builtin_prompt_editors() {
    static const PromptEditorCatalog catalog = make_builtin_catalog();
    return catalog;
}

EditorSelection select_prompt_editor(
    const PromptEditorCatalog& catalog,
    std::string_view configured) {
    const auto fallback = [&catalog](std::string warning) {
        auto editor = catalog.create(default_prompt_editor_id());
        return EditorSelection{
            .editor = editor ? std::move(editor) : std::make_unique<SystemPromptEditor>(),
            .warning = std::move(warning),
        };
    };

    if (configured.empty()) {
        return fallback({});
    }
    if (auto editor = catalog.create(configured); editor != nullptr) {
        return EditorSelection{.editor = std::move(editor), .warning = {}};
    }
    // Undocumented ids are treated as a literal editor command, so a user can
    // pin "nvim" or "code --wait" without waiting for a dedicated backend.
    if (editor_command_is_executable(configured)) {
        return EditorSelection{
            .editor = std::make_unique<SystemPromptEditor>(std::string(configured)),
            .warning = {},
        };
    }
    return fallback(std::format(
        "Unknown prompt editor '{}'; falling back to $VISUAL/$EDITOR.",
        configured));
}

} // namespace tui::editor
