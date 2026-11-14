#pragma once

#include "RemoteActivityPanel.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include <ftxui/dom/elements.hpp>

namespace tui {

/// How much of the "context left" reading the status bar spells out.
enum class ContextLeftLabel {
    full,     ///< " 83% Context Left "
    compact,  ///< " 83% ctx "
    percent,  ///< " 83% "
};

/// One way of drawing the status bar. The default is the complete footer;
/// every other value only takes detail away from it.
struct StatusFooterDensity {
    /// Trailing workspace path components kept after a leading "…/";
    /// 0 keeps the whole path. See `elide_workspace_path`.
    std::size_t workspace_kept_components = 0;
    bool show_workspace = true;
    bool show_provider = true;
    bool show_effort = true;
    bool show_usage_totals = true;
    RemoteFooterDetail remote_detail = RemoteFooterDetail::full;
    ContextLeftLabel context_label = ContextLeftLabel::full;
};

/// The densities to try, from the complete footer to the most compact one.
/// Each step gives up one more piece of detail, least useful first, so the
/// footer only loses what the terminal width actually forces it to:
///   1. leading workspace directories (the folder name always stays),
///   2. "Context Left" → "ctx",
///   3. the provider name in the model badge,
///   4. the session token / cost totals,
///   5. the remote MCP status detail (its subject and tone colour stay),
///   6. the effort level in the model badge,
///   7. "ctx" from the context reading (the percentage stays),
///   8. the remote MCP subject (the coloured glyph stays),
///   9. the workspace.
/// The mode, the model, rate-limit windows, the turn activity indicator and
/// warnings are never given up.
[[nodiscard]] std::vector<StatusFooterDensity> status_footer_densities(
    std::size_t workspace_path_components);

/// Number of components after the root of a displayed workspace path
/// ("~/a/b" and "/a/b" both have 2).
[[nodiscard]] std::size_t workspace_path_components(std::string_view path);

/// Replace all but the last `kept` components of `path` with "…", e.g.
/// ("~/Documents/Develop/Projects/Lampo", 2) → "…/Projects/Lampo". `kept`
/// of 0, or at least the number of components, returns the path unchanged.
[[nodiscard]] std::string elide_workspace_path(std::string_view path,
                                               std::size_t kept);

[[nodiscard]] std::string format_context_left_label(int percent,
                                                    ContextLeftLabel label);

/// Lay out the first candidate whose natural width fits the space it is
/// given, falling back to the last one. Candidates go from the richest to the
/// most compact, so the footer keeps everything when there is room and
/// sheds detail in a fixed order when there is not, instead of every item
/// being clipped a little.
[[nodiscard]] ftxui::Element first_that_fits(ftxui::Elements candidates);

} // namespace tui
