#include "DocumentReader.hpp"

#include "Text.hpp"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <system_error>

namespace tui {
namespace {

constexpr std::size_t kBinaryProbeBytes = 8 * 1024;

[[nodiscard]] bool looks_binary(std::string_view bytes) {
    return bytes.substr(0, kBinaryProbeBytes).find('\0') != std::string_view::npos;
}

struct DiskState {
    std::filesystem::file_time_type modified{};
    std::uintmax_t size = 0;
    bool exists = false;
};

/// Whether the file is as @p tab last read it. A file that is missing reads
/// as no time and no size both times, so it is unchanged too.
[[nodiscard]] bool unchanged(const DocumentTab& tab, const DiskState& state) {
    return tab.modified == state.modified && tab.size == state.size;
}

[[nodiscard]] DiskState disk_state(const std::filesystem::path& path) {
    std::error_code ec;
    DiskState state;
    if (!std::filesystem::is_regular_file(path, ec)) return state;
    state.size = std::filesystem::file_size(path, ec);
    if (ec) return {};
    state.modified = std::filesystem::last_write_time(path, ec);
    state.exists = !ec;
    return state;
}

/// Brings @p line into view on the next frame and marks it.
void target_line(DocumentTab& tab, std::optional<int> line) {
    tab.marked_line = line;
    tab.pending_line = line;
    if (!line) tab.offset = 0;
}

} // namespace

void load_document(DocumentTab& tab) {
    const auto state = disk_state(tab.path);
    tab.modified = state.modified;
    tab.size = state.size;
    tab.content.clear();
    tab.problem.clear();
    tab.truncated = false;
    tab.selected_link.reset();
    ++tab.generation;
    if (!state.exists) {
        std::error_code ec;
        tab.problem = std::filesystem::is_directory(tab.path, ec) ? "This is a directory."
                                                                   : "The file does not exist.";
        return;
    }
    std::ifstream in(tab.path, std::ios::binary);
    if (!in) {
        tab.problem = "The file cannot be read.";
        return;
    }
    const auto keep = static_cast<std::size_t>(std::min(state.size, DocumentReader::kMaxBytes));
    std::string bytes(keep, '\0');
    in.read(bytes.data(), static_cast<std::streamsize>(keep));
    bytes.resize(static_cast<std::size_t>(std::max<std::streamsize>(in.gcount(), 0)));
    if (looks_binary(bytes)) {
        tab.problem = "This is a binary file.";
        return;
    }
    tab.truncated = state.size > DocumentReader::kMaxBytes;
    tab.content = sanitize_terminal_text(bytes);
}

void DocumentReader::open(const DocumentTarget& target, bool select) {
    const auto found = std::ranges::find(tabs_, target.path, &DocumentTab::path);
    if (found != tabs_.end()) {
        if (select) selected_ = static_cast<std::size_t>(found - tabs_.begin());
        // Opening a file again shows it as it is now.
        if (!unchanged(*found, disk_state(found->path))) load_document(*found);
        if (target.line) target_line(*found, target.line);
        return;
    }
    if (tabs_.size() >= kMaxTabs) {
        // The oldest tab goes; the one in front never does, since it was the
        // last one shown before this.
        const std::size_t victim = selected_ == 0 ? 1 : 0;
        tabs_.erase(tabs_.begin() + static_cast<std::ptrdiff_t>(victim));
        if (selected_ > victim) --selected_;
    }
    DocumentTab tab{.path = target.path, .format = document_format(target.path)};
    load_document(tab);
    target_line(tab, target.line);
    tabs_.push_back(std::move(tab));
    if (select || tabs_.size() == 1) selected_ = tabs_.size() - 1;
}

void DocumentReader::close_selected() {
    if (tabs_.empty()) return;
    tabs_.erase(tabs_.begin() + static_cast<std::ptrdiff_t>(selected_));
    if (selected_ >= tabs_.size() && selected_ > 0) --selected_;
}

void DocumentReader::select(std::size_t index) {
    if (index < tabs_.size()) selected_ = index;
}

void DocumentReader::select_relative(int step) {
    if (tabs_.empty()) return;
    const auto count = static_cast<long long>(tabs_.size());
    const auto next = ((static_cast<long long>(selected_) + step) % count + count) % count;
    selected_ = static_cast<std::size_t>(next);
}

DocumentTab* DocumentReader::selected() noexcept {
    return tabs_.empty() ? nullptr : &tabs_[selected_];
}

const DocumentTab* DocumentReader::selected() const noexcept {
    return tabs_.empty() ? nullptr : &tabs_[selected_];
}

bool DocumentReader::refresh(std::chrono::steady_clock::time_point now,
                             std::chrono::milliseconds interval) {
    if (tabs_.empty() || now - last_refresh_ < interval) return false;
    last_refresh_ = now;
    bool changed = false;
    for (auto& tab : tabs_) {
        if (unchanged(tab, disk_state(tab.path))) continue;
        // The offset stays: a file edited while it is read keeps its place.
        load_document(tab);
        changed = true;
    }
    return changed;
}

bool find_in_document(DocumentFind& find, DocumentTab& tab, int first_visible_line) {
    find.matches = find_document_lines(tab.content, find.query);
    find.current = 0;
    find.generation = tab.generation;
    if (find.matches.empty()) {
        tab.marked_line.reset();
        return false;
    }
    const auto at = std::ranges::lower_bound(find.matches, first_visible_line);
    find.current = at == find.matches.end()
        ? 0
        : static_cast<std::size_t>(at - find.matches.begin());
    target_line(tab, find.matches[find.current]);
    return true;
}

void step_document_find(DocumentFind& find, DocumentTab& tab, int step) {
    if (find.generation != tab.generation) {
        // The lines found belong to the file as it was; search what it is now.
        find_in_document(find, tab, std::max(1, tab.offset + 1));
        return;
    }
    if (find.matches.empty()) return;
    const auto count = static_cast<long long>(find.matches.size());
    find.current = static_cast<std::size_t>(
        ((static_cast<long long>(find.current) + step) % count + count) % count);
    target_line(tab, find.matches[find.current]);
}

} // namespace tui
