#pragma once

#include "DocumentLink.hpp"
#include "DocumentView.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace tui {

/// One file open in the reader, as last read from disk.
struct DocumentTab {
    /// Absolute.
    std::filesystem::path path;
    DocumentFormat format = DocumentFormat::Text;
    /// Safe to draw (see sanitize_terminal_text). Empty when @c problem is set.
    std::string content;
    /// Why the file is not shown ("not found", "binary"), or empty.
    std::string problem;
    /// The file is longer than the reader keeps; @c content is its start.
    bool truncated = false;

    /// The first drawn row in view.
    int offset = 0;
    /// A one-based line to bring into view once the reader knows its width.
    std::optional<int> pending_line;
    /// A one-based line drawn marked: the line a link named, or a find match.
    std::optional<int> marked_line;
    /// The link that holds the keyboard, by ordinal.
    std::optional<std::size_t> selected_link;

    /// Bumped every time the file is read, so a view knows its cache is stale.
    std::size_t generation = 0;

    /// What the file looked like on disk when it was read.
    std::filesystem::file_time_type modified{};
    std::uintmax_t size = 0;
};

/// The files a reader has open, as tabs. It reads them, reads them again when
/// they change on disk (keeping the reader's place), and finds text in them.
/// Drawing and keys belong to the view that shows it (DocumentPane).
class DocumentReader {
public:
    /// Kept from one file: enough for any document a person reads.
    static constexpr std::uintmax_t kMaxBytes = 2 * 1024 * 1024;
    static constexpr std::size_t kMaxTabs = 12;

    /// Shows @p target (an absolute path): its tab when it is open already,
    /// else a new one. A line in @p target is brought into view and marked.
    /// Past kMaxTabs, the oldest tab that is not selected closes. With
    /// @p select false the tab is opened or updated without coming to the
    /// front (the first tab is always selected).
    void open(const DocumentTarget& target, bool select = true);

    void close_selected();
    void select(std::size_t index);
    /// The tab @p step places away, wrapping around.
    void select_relative(int step);

    [[nodiscard]] bool empty() const noexcept { return tabs_.empty(); }
    [[nodiscard]] const std::vector<DocumentTab>& tabs() const noexcept { return tabs_; }
    [[nodiscard]] std::size_t selected_index() const noexcept { return selected_; }
    [[nodiscard]] DocumentTab* selected() noexcept;
    [[nodiscard]] const DocumentTab* selected() const noexcept;

    /// Reads again every tab whose file changed on disk since it was read, at
    /// most once per @p interval. True when a tab changed.
    bool refresh(std::chrono::steady_clock::time_point now,
                 std::chrono::milliseconds interval = std::chrono::milliseconds{750});

private:
    std::vector<DocumentTab> tabs_;
    std::size_t selected_ = 0;
    std::chrono::steady_clock::time_point last_refresh_{};
};

/// @p tab with its file read again from disk; the place in it is kept.
void load_document(DocumentTab& tab);

/// Text search within the selected document.
struct DocumentFind {
    /// The query is being typed.
    bool editing = false;
    std::string query;
    /// One-based lines that match, and the one shown.
    std::vector<int> matches;
    std::size_t current = 0;
    /// The tab version the matches were found in (DocumentTab::generation).
    std::size_t generation = 0;

    [[nodiscard]] bool has_matches() const noexcept { return !matches.empty(); }
};

/// Runs @p find's query over @p tab and marks the first match at or after
/// the line in view. False when nothing matches.
bool find_in_document(DocumentFind& find, DocumentTab& tab, int first_visible_line);

/// Moves to the next (@p step 1) or previous (-1) match, wrapping. When the
/// file was read again since the matches were found, they are found again
/// first, from the line in view.
void step_document_find(DocumentFind& find, DocumentTab& tab, int step);

} // namespace tui
