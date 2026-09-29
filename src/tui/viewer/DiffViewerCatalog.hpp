#pragma once

#include "DiffViewer.hpp"

#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace tui::viewer {

// The set of external diff comparers compiled into this build.
//
// Registration is the single extension point: a new comparer adds one
// translation unit and one register() call, and it automatically appears in
// /settings, in config validation, and in the key binding — no TUI change.
class DiffViewerCatalog {
public:
    using Factory = std::function<std::unique_ptr<DiffViewer>()>;

    void register_viewer(ViewerDescriptor descriptor, Factory factory);

    [[nodiscard]] std::span<const ViewerDescriptor> descriptors() const noexcept {
        return descriptors_;
    }

    // Returns nullptr when `id` is not a registered comparer.
    [[nodiscard]] std::unique_ptr<DiffViewer> create(std::string_view id) const;

private:
    struct Entry {
        ViewerDescriptor descriptor;
        Factory factory;
    };

    std::vector<Entry> entries_;
    std::vector<ViewerDescriptor> descriptors_;
};

// Comparers available on this platform. Lampo is present in macOS builds only.
[[nodiscard]] const DiffViewerCatalog& builtin_diff_comparers();

// The id used when `diff_comparer` is unset.
[[nodiscard]] constexpr std::string_view default_diff_comparer_id() noexcept { return "builtin"; }

// The transcript itself: every diff Filo shows inline, in the turn it belongs
// to. It is a choice in /settings but not a catalog entry, because nothing is
// launched for it — the host already has the changes and renders them.
[[nodiscard]] ViewerDescriptor builtin_diff_comparer_descriptor() noexcept;

// Outcome of interpreting a configured `diff_comparer` value.
struct ViewerSelection {
    std::unique_ptr<DiffViewer> viewer;  // Null when the host shows the diff.
    std::string warning;                 // Non-empty when the value was not usable verbatim.

    [[nodiscard]] bool inline_view() const noexcept { return viewer == nullptr; }
};

// Resolves a configured value to a usable comparer:
//   * empty or "builtin" → the transcript, which the host renders itself;
//   * a registered id    → that comparer;
//   * any other value    → treated as an explicit pager command ("delta",
//                          "difftastic --width=120", ...) when it resolves on
//                          PATH;
//   * otherwise          → the transcript, with a warning for the user.
[[nodiscard]] ViewerSelection select_diff_viewer(
    const DiffViewerCatalog& catalog,
    std::string_view configured);

} // namespace tui::viewer
