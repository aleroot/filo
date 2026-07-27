#pragma once

// Lampo is a native macOS App Store application; the integration exists only in
// macOS builds. Every other platform simply never sees this backend, and the
// catalog never offers it.
#if defined(__APPLE__)

#include "PromptEditor.hpp"

#include <memory>

namespace tui::editor {

[[nodiscard]] EditorDescriptor lampo_prompt_editor_descriptor() noexcept;

// Creates the Lampo backend. The implementation (and all of its AppKit
// dependencies) is confined to LampoPromptEditor.mm.
[[nodiscard]] std::unique_ptr<PromptEditor> make_lampo_prompt_editor();

} // namespace tui::editor

#endif // defined(__APPLE__)
