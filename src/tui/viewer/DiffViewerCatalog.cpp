#include "DiffViewerCatalog.hpp"

#include "TerminalDiffViewer.hpp"
#include "tui/CommandLookup.hpp"
#include "tui/Text.hpp"

#include <algorithm>
#include <format>
#include <utility>

#if defined(__APPLE__)
#include "LampoDiffViewer.hpp"
#endif

namespace tui::viewer {
namespace {

[[nodiscard]] DiffViewerCatalog make_builtin_catalog() {
    DiffViewerCatalog catalog;
    catalog.register_viewer(
        terminal_diff_viewer_descriptor(),
        [] { return std::make_unique<TerminalDiffViewer>(); });
#if defined(__APPLE__)
    catalog.register_viewer(lampo_diff_viewer_descriptor(), &make_lampo_diff_viewer);
#endif
    return catalog;
}

} // namespace

void DiffViewerCatalog::register_viewer(ViewerDescriptor descriptor, Factory factory) {
    entries_.push_back(Entry{descriptor, std::move(factory)});
    descriptors_.push_back(descriptor);
}

std::unique_ptr<DiffViewer> DiffViewerCatalog::create(std::string_view id) const {
    const std::string needle = to_lower_ascii(id);
    const auto match = std::ranges::find_if(
        entries_,
        [&needle](const Entry& entry) { return entry.descriptor.id == needle; });
    return match == entries_.end() ? nullptr : match->factory();
}

const DiffViewerCatalog& builtin_diff_comparers() {
    static const DiffViewerCatalog catalog = make_builtin_catalog();
    return catalog;
}

ViewerDescriptor builtin_diff_comparer_descriptor() noexcept {
    return ViewerDescriptor{
        .id = default_diff_comparer_id(),
        .label = "Built in",
        .description = "Read each diff in the transcript, under the turn that made it.",
        .launch = ViewerLaunch::Inline,
    };
}

ViewerSelection select_diff_viewer(
    const DiffViewerCatalog& catalog,
    std::string_view configured) {
    const auto inline_view = [](std::string warning) {
        return ViewerSelection{.viewer = nullptr, .warning = std::move(warning)};
    };

    if (configured.empty() || to_lower_ascii(configured) == default_diff_comparer_id()) {
        return inline_view({});
    }
    if (auto viewer = catalog.create(configured); viewer != nullptr) {
        return ViewerSelection{.viewer = std::move(viewer), .warning = {}};
    }
    // Undocumented ids are treated as a literal pager command, so a user can pin
    // "delta" or "diff-so-fancy | less -R" without waiting for a dedicated
    // backend. The command reads the patch on standard input.
    if (command_is_executable(configured)) {
        return ViewerSelection{
            .viewer = std::make_unique<TerminalDiffViewer>(std::string(configured)),
            .warning = {},
        };
    }
    return inline_view(std::format(
        "Unknown diff comparer '{}'; showing diffs in the transcript.",
        configured));
}

} // namespace tui::viewer
