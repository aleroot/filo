#pragma once

#include "DocumentReader.hpp"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <ftxui/component/event.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/box.hpp>

namespace tui {

/// The document pane beside the transcript: a DocumentReader, the find
/// prompt, and what the last frame drew (so a click lands where it was aimed).
struct DocumentPaneState {
    DocumentReader reader;
    DocumentFind find;
    bool open = false;
    /// The pane holds the keyboard; otherwise the prompt does.
    bool focused = false;

    // Written by render_document_pane, read by handle_document_pane_event.
    ScrollMetrics metrics;
    ftxui::Box pane_box{0, -1, 0, -1};
    std::vector<ftxui::Box> tab_boxes;
    std::vector<ftxui::Box> close_boxes;
    std::unordered_map<std::string, ftxui::Box> link_boxes;

    /// The rendered document of the selected tab, reused while nothing that
    /// changes it changes: a long Markdown file is not parsed every frame.
    struct Cache {
        std::filesystem::path tab;
        std::size_t generation = 0;
        std::optional<std::size_t> selected_link;
        ftxui::Element element;
    } cache;
    /// The tab @c metrics describes.
    std::filesystem::path metrics_tab;

    /// Opens the pane on @p target, focused.
    void show(const DocumentTarget& target);
    /// Where a file the agent asked to show went.
    enum class Placement {
        front,   ///< In view.
        behind,  ///< A tab behind the one the user is reading.
        hidden,  ///< A tab of the closed pane; F6 shows it.
    };
    /// Shows @p target for the agent without taking anything from the user:
    /// the keyboard stays where it is, a tab the user is reading in the
    /// focused pane stays in front, and a closed pane opens only when
    /// @p can_split, so the transcript is never replaced unasked.
    Placement show_for_agent(const DocumentTarget& target, bool can_split);
    void close();
};

/// What a key or click in the pane asks of its host.
struct DocumentPaneAction {
    enum class Kind {
        /// Not for the pane; let it reach the rest of the UI.
        ignored,
        handled,
        /// Handled; open @c url in the browser.
        open_url,
    };
    Kind kind = Kind::ignored;
    std::string url;
};

/// Keys while the pane is focused, and the mouse over it:
/// ↑/↓ j/k PgUp/PgDn Space Home/End scroll · ←/→ or [ ] tab · Tab/Shift+Tab
/// link · Enter open link · / find, n/N next/previous match · x close tab ·
/// q hide the pane · Esc or F6 back to the prompt. F6 from the prompt focuses
/// the pane, opening it again when it was hidden with tabs left. A link to a
/// local file opens in the pane (relative to the document it is in); any
/// other link asks the host to open it.
[[nodiscard]] DocumentPaneAction handle_document_pane_event(DocumentPaneState& state,
                                                            const ftxui::Event& event);

/// The pane, filling the space its parent gives it.
[[nodiscard]] ftxui::Element render_document_pane(DocumentPaneState& state);

/// Width of the pane beside a transcript @p total columns wide, or nothing
/// when the terminal is too narrow to split and the pane takes the whole
/// width instead.
[[nodiscard]] std::optional<int> document_pane_split_width(int total);

} // namespace tui
