#pragma once

// Lampo is a native macOS App Store application; the integration exists only in
// macOS builds. Every other platform simply never sees this comparer, and the
// catalog never offers it.
#if defined(__APPLE__)

#include "DiffViewer.hpp"

#include <memory>

namespace tui::viewer {

[[nodiscard]] ViewerDescriptor lampo_diff_viewer_descriptor() noexcept;

// Creates the Lampo comparer. All AppKit dependencies live in the shared Lampo
// CLI transport (tui/lampo/LampoCliSession.mm); the comparer itself is plain
// C++ and only writes the protocol's state machine.
[[nodiscard]] std::unique_ptr<DiffViewer> make_lampo_diff_viewer();

} // namespace tui::viewer

#endif // defined(__APPLE__)
