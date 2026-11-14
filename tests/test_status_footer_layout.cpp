#include <catch2/catch_test_macros.hpp>

#include "tui/PromptComponents.hpp"
#include "tui/RemoteActivityPanel.hpp"
#include "tui/StatusFooterLayout.hpp"

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>

#include <string>
#include <vector>

using namespace tui;

namespace {

constexpr std::string_view kWorkspace = "~/Documents/Develop/Projects/Lampo";

/// Plain text of a one-row render, trailing blanks removed.
std::string render_row(const ftxui::Element& element, int width) {
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(width),
                                        ftxui::Dimension::Fixed(1));
    ftxui::Render(screen, element);
    // ToString() writes the blank cells a layout leaves behind as spaces;
    // reading CellAt() directly would drop them. Strip its colour codes.
    const std::string styled = screen.ToString();
    std::string row;
    for (std::size_t i = 0; i < styled.size(); ++i) {
        if (styled[i] == '\x1b') {
            while (i < styled.size() && styled[i] != 'm') {
                ++i;
            }
            continue;
        }
        row += styled[i];
    }
    while (row.ends_with(' ')) {
        row.pop_back();
    }
    return row;
}

/// The status bar of the reported screenshot, composed the way MainApp
/// composes it: left items, the remote MCP pill centered, then the right.
ftxui::Element screenshot_status_bar(const StatusFooterDensity& density) {
    using namespace ftxui;
    auto left = hbox({
        text(" BUILD "),
        text(format_model_status_badge(density.show_provider ? "claude" : "",
                                       "claude-opus-5-5",
                                       "")) | xflex_shrink,
        text(density.show_usage_totals ? "  ↑12.4M ↓81.4K" : ""),
        text(" 5h:21% 7d:38%") | xflex_shrink,
        text(" ◔"),
    });
    auto middle = render_remote_footer_status(
        {.label = "MCP · waiting", .tone = RemoteFooterTone::neutral},
        density.remote_detail);
    Elements right_items;
    if (density.show_workspace) {
        right_items.push_back(text(format_workspace_status_label(
            elide_workspace_path(kWorkspace, density.workspace_kept_components)
            + " (+1)")));
    }
    right_items.push_back(text(format_context_left_label(83, density.context_label)));
    return render_centered_footer(std::move(left), std::move(middle),
                                  hbox(std::move(right_items)));
}

ftxui::Element adaptive_status_bar() {
    ftxui::Elements candidates;
    for (const auto& density :
         status_footer_densities(workspace_path_components(kWorkspace))) {
        candidates.push_back(screenshot_status_bar(density));
    }
    return first_that_fits(std::move(candidates));
}

} // namespace

TEST_CASE("Workspace paths are elided from the root, keeping the folder name",
          "[tui][status-footer]") {
    CHECK(workspace_path_components(kWorkspace) == 4);
    CHECK(workspace_path_components("/Users/me/code") == 3);
    CHECK(workspace_path_components("~") == 0);
    CHECK(workspace_path_components("/") == 0);

    CHECK(elide_workspace_path(kWorkspace, 0) == kWorkspace);
    CHECK(elide_workspace_path(kWorkspace, 4) == kWorkspace);
    CHECK(elide_workspace_path(kWorkspace, 3) == "…/Develop/Projects/Lampo");
    CHECK(elide_workspace_path(kWorkspace, 2) == "…/Projects/Lampo");
    CHECK(elide_workspace_path(kWorkspace, 1) == "…/Lampo");
    CHECK(elide_workspace_path("/Users/me/code", 2) == "…/me/code");
    CHECK(elide_workspace_path("~", 1) == "~");
}

TEST_CASE("Status footer densities give detail up one step at a time",
          "[tui][status-footer]") {
    const auto densities = status_footer_densities(4);
    REQUIRE(densities.size() == 3 + 8 + 1);

    // The first density is the complete footer.
    const StatusFooterDensity full;
    CHECK(densities.front().workspace_kept_components == 0);
    CHECK(densities.front().show_workspace);
    CHECK(densities.front().show_provider == full.show_provider);
    CHECK(densities.front().context_label == ContextLeftLabel::full);
    CHECK(densities.front().remote_detail == RemoteFooterDetail::full);

    // Leading directories go first, one per step, the folder name never.
    CHECK(densities[1].workspace_kept_components == 3);
    CHECK(densities[2].workspace_kept_components == 2);
    CHECK(densities[3].workspace_kept_components == 1);
    CHECK(densities[3].context_label == ContextLeftLabel::full);
    CHECK(densities[3].show_provider);

    CHECK(densities[4].context_label == ContextLeftLabel::compact);
    CHECK_FALSE(densities[5].show_provider);
    CHECK_FALSE(densities[6].show_usage_totals);
    CHECK(densities[7].remote_detail == RemoteFooterDetail::subject);
    CHECK_FALSE(densities[8].show_effort);
    CHECK(densities[9].context_label == ContextLeftLabel::percent);
    CHECK(densities[10].remote_detail == RemoteFooterDetail::glyph);
    CHECK_FALSE(densities[11].show_workspace);

    // Steps are cumulative: nothing given up comes back later.
    for (std::size_t i = 4; i < densities.size(); ++i) {
        CHECK(densities[i].workspace_kept_components == 1);
    }

    // A path with nothing to elide adds no workspace steps.
    CHECK(status_footer_densities(1).size() == 1 + 8);
}

TEST_CASE("first_that_fits lays out only the first candidate that fits",
          "[tui][status-footer]") {
    using namespace ftxui;
    Box wide_box{0, -1, 0, -1};
    Box narrow_box{0, -1, 0, -1};
    const auto make = [&] {
        return first_that_fits({
            text("a rather wide candidate") | reflect(wide_box),
            text("narrow") | reflect(narrow_box),
        });
    };

    // The chosen candidate gets the whole box; the others are never laid out.
    CHECK(render_row(make(), 30) == "a rather wide candidate");
    CHECK(wide_box.x_max == 29);
    CHECK(narrow_box.x_max == -1);

    wide_box = narrow_box = Box{0, -1, 0, -1};
    CHECK(render_row(make(), 10) == "narrow");
    CHECK(wide_box.x_max == -1);
    CHECK(narrow_box.x_max == 9);

    // Nothing fits: the most compact candidate is drawn, clipped.
    CHECK(render_row(make(), 4) == "narr");
}

TEST_CASE("The status bar keeps everything when the terminal is wide enough",
          "[tui][status-footer]") {
    const std::string row = render_row(adaptive_status_bar(), 150);
    CHECK(row.find(" claude · claude-opus-5-5 ") != std::string::npos);
    CHECK(row.find("↑12.4M ↓81.4K") != std::string::npos);
    CHECK(row.find("⚡ MCP · waiting") != std::string::npos);
    CHECK(row.find(std::string(kWorkspace) + " (+1)") != std::string::npos);
    CHECK(row.ends_with(" 83% Context Left"));
}

TEST_CASE("A narrow status bar sheds whole details instead of clipping every item",
          "[tui][status-footer]") {
    // The reported terminal: 130 columns, 128 inside the window frame. The
    // old layout clipped every item ("claude-opus-5-", "7d:38", "Lampo (",
    // "Context Lef") and dropped the activity glyph.
    const std::string row = render_row(adaptive_status_bar(), 128);
    CHECK(row.find(" claude · claude-opus-5-5 ") != std::string::npos);
    CHECK(row.find("↑12.4M ↓81.4K") != std::string::npos);
    CHECK(row.find(" 5h:21% 7d:38% ◔ ⚡ MCP · waiting ") != std::string::npos);
    CHECK(row.find(" …/Projects/Lampo (+1)  83% Context Left") != std::string::npos);
    CHECK(row.ends_with(" 83% Context Left"));

    // Much narrower: the mode, model, rate windows and activity stay whole.
    const std::string narrow = render_row(adaptive_status_bar(), 70);
    CHECK(narrow.starts_with(" BUILD  claude-opus-5-5"));
    CHECK(narrow.find(" 5h:21% 7d:38% ◔") != std::string::npos);
    CHECK(narrow.find("⚡") != std::string::npos);
    CHECK(narrow.find("83%") != std::string::npos);
    CHECK(narrow.find("↑12.4M") == std::string::npos);
}

TEST_CASE("Below the most compact footer only the model and rate windows are cut",
          "[tui][status-footer]") {
    // Nothing is left to give up, so something must be clipped. The mode,
    // the activity glyph, the MCP glyph and the context reading stay whole;
    // left to FTXUI's default every item would lose a share of its cells.
    const std::string row = render_row(adaptive_status_bar(), 40);
    CHECK(row.starts_with(" BUILD  claude"));
    CHECK(row.find(" ◔ ⚡") != std::string::npos);
    CHECK(row.ends_with(" 83%"));
    CHECK(row.find("claude-opus-5-5") == std::string::npos);
}

TEST_CASE("Status footer pieces have compact forms", "[tui][status-footer]") {
    CHECK(format_context_left_label(83, ContextLeftLabel::full) == " 83% Context Left ");
    CHECK(format_context_left_label(83, ContextLeftLabel::compact) == " 83% ctx ");
    CHECK(format_context_left_label(83, ContextLeftLabel::percent) == " 83% ");

    CHECK(format_model_status_badge("", "gpt-5", "high") == " gpt-5 · high ");

    const RemoteFooterStatus status{.label = "Lampo · seen 6m 16s ago"};
    CHECK(render_row(render_remote_footer_status(status), 40)
          == "⚡ Lampo · seen 6m 16s ago");
    CHECK(render_row(render_remote_footer_status(status, RemoteFooterDetail::subject), 40)
          == "⚡ Lampo");
    CHECK(render_row(render_remote_footer_status(status, RemoteFooterDetail::glyph), 40)
          == "⚡");
}
