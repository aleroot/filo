#include "FileSystemPicker.hpp"

#include "KeyInput.hpp"

#include "core/net/NetworkTraffic.hpp"  // format_bytes: unit-agnostic size text.
#include "core/utils/PathUtils.hpp"
#include "core/utils/StringUtils.hpp"

#include <algorithm>
#include <format>
#include <ranges>
#include <system_error>
#include <utility>

namespace tui {
namespace {

using core::utils::path::abbreviate_user_path;
using core::utils::path::expand_user_path;
using core::utils::str::to_lower_ascii_copy;

constexpr int kPageJump = 10;

/// Rows the fuzzy filter is allowed to hide. Synthetic navigation rows stay
/// pinned while the filter is empty and disappear entirely once it is not, so
/// the listing never mixes "places" with search results.
[[nodiscard]] bool is_real_entry(FileSystemRowRole role) noexcept {
    return role == FileSystemRowRole::Directory || role == FileSystemRowRole::File;
}

[[nodiscard]] bool is_printable_character(const ftxui::Event& event) {
    if (!event.is_character()) {
        return false;
    }
    const std::string& character = event.character();
    if (character.size() != 1) {
        return true;  // Multi-byte UTF-8 input is always printable text.
    }
    const auto byte = static_cast<unsigned char>(character.front());
    return byte >= 32 && byte != 127;
}

/// Absolute, `.`/`..`-free form of @p path without touching the filesystem.
/// `weakly_canonical` would resolve symlinks, which makes breadcrumbs jump
/// unexpectedly when browsing through linked project directories.
[[nodiscard]] std::filesystem::path normalize(const std::filesystem::path& path) {
    std::error_code ec;
    auto absolute = path.is_absolute() ? path : std::filesystem::absolute(path, ec);
    if (ec) {
        absolute = path;
    }
    auto normalized = absolute.lexically_normal();
    // `lexically_normal` leaves a trailing separator on directory-shaped input
    // ("/tmp/foo/"), which would make `filename()` empty and break the parent
    // and focus comparisons below.
    if (normalized.has_filename() || normalized == normalized.root_path()) {
        return normalized;
    }
    return normalized.parent_path();
}

[[nodiscard]] bool is_hidden_name(std::string_view name) noexcept {
    return name.size() > 1 && name.front() == '.';
}

[[nodiscard]] std::string lower_extension(const std::filesystem::path& path) {
    return to_lower_ascii_copy(path.extension().string());
}

[[nodiscard]] bool extension_accepted(const FileSystemPickerRequest& request,
                                      const std::filesystem::path& path) {
    return request.extensions.empty()
        || std::ranges::contains(request.extensions, lower_extension(path));
}

[[nodiscard]] std::string describe_size(const std::filesystem::directory_entry& entry) {
    std::error_code ec;
    const auto size = entry.file_size(ec);
    return ec ? std::string{} : core::net::format_bytes(size);
}

/// Directories first, then files; each group sorted case-insensitively so the
/// order matches what a human expects from a file manager.
[[nodiscard]] bool row_precedes(const FileSystemRow& lhs, const FileSystemRow& rhs) {
    const bool lhs_dir = lhs.role == FileSystemRowRole::Directory;
    const bool rhs_dir = rhs.role == FileSystemRowRole::Directory;
    if (lhs_dir != rhs_dir) {
        return lhs_dir;
    }
    const auto lhs_key = to_lower_ascii_copy(lhs.label);
    const auto rhs_key = to_lower_ascii_copy(rhs.label);
    return lhs_key != rhs_key ? lhs_key < rhs_key : lhs.label < rhs.label;
}

void append_synthetic_rows(const std::filesystem::path& directory,
                           const FileSystemPickerRequest& request,
                           std::vector<FileSystemRow>& rows) {
    if (request.target == FileSystemPickerTarget::Directory) {
        rows.push_back(FileSystemRow{
            .role = FileSystemRowRole::ConfirmCurrent,
            .label = "Use this folder",
            .detail = abbreviate_user_path(directory),
            .path = directory,
            .selectable = true,
        });
    }

    if (const auto parent = directory.parent_path();
        !parent.empty() && parent != directory) {
        rows.push_back(FileSystemRow{
            .role = FileSystemRowRole::Parent,
            .label = "..",
            .detail = abbreviate_user_path(parent),
            .path = parent,
            .navigable = true,
        });
    }

    for (const auto& root : request.quick_roots) {
        const auto target = normalize(root.path);
        if (target == directory) {
            continue;  // Already here: the shortcut would be a no-op.
        }
        const bool duplicate = std::ranges::any_of(rows, [&](const FileSystemRow& row) {
            return row.role == FileSystemRowRole::QuickRoot && row.path == target;
        });
        std::error_code ec;
        if (duplicate || !std::filesystem::is_directory(target, ec) || ec) {
            continue;
        }
        rows.push_back(FileSystemRow{
            .role = FileSystemRowRole::QuickRoot,
            .label = root.label,
            .detail = abbreviate_user_path(target),
            .path = target,
            .navigable = true,
        });
    }
}

/// First real (non-synthetic) row, or -1 when the directory has no entries.
[[nodiscard]] int first_real_entry(const std::vector<FileSystemRow>& rows) {
    const auto match = std::ranges::find_if(
        rows, [](const FileSystemRow& row) { return is_real_entry(row.role); });
    return match == rows.end()
        ? -1
        : static_cast<int>(std::ranges::distance(rows.begin(), match));
}

/// Index the cursor should land on after descending into a directory: the row
/// named by @p focus when present, otherwise "Use this folder".
///
/// Landing on the confirm row is what makes drilling down feel right: you went
/// *into* this folder deliberately, so a second Enter takes it, while ↓ still
/// continues deeper.
[[nodiscard]] int default_selection(const std::vector<FileSystemRow>& rows,
                                    const std::filesystem::path& focus) {
    if (!focus.empty()) {
        const auto match = std::ranges::find_if(rows, [&](const FileSystemRow& row) {
            return is_real_entry(row.role) && row.path == focus;
        });
        if (match != rows.end()) {
            return static_cast<int>(std::ranges::distance(rows.begin(), match));
        }
    }

    const auto first_useful = std::ranges::find_if(rows, [](const FileSystemRow& row) {
        return row.role != FileSystemRowRole::Parent;
    });
    return first_useful == rows.end()
        ? 0
        : static_cast<int>(std::ranges::distance(rows.begin(), first_useful));
}

/// Split typed path text into the directory part (through the last separator)
/// and the leaf being completed.
struct SplitPath {
    std::string directory_text;  ///< Retains the user's `~`/relative spelling.
    std::string leaf;
};

[[nodiscard]] SplitPath split_for_completion(std::string_view partial) {
    const auto separator = partial.find_last_of('/');
    if (separator == std::string_view::npos) {
        return SplitPath{.directory_text = {}, .leaf = std::string(partial)};
    }
    return SplitPath{
        .directory_text = std::string(partial.substr(0, separator + 1)),
        .leaf = std::string(partial.substr(separator + 1)),
    };
}

/// Space-separated preview of the first few completion candidates.
[[nodiscard]] std::string summarize_candidates(const std::vector<std::string>& candidates,
                                              std::size_t limit) {
    std::string summary;
    for (const auto& name : candidates | std::views::take(limit)) {
        if (!summary.empty()) {
            summary += "  ";
        }
        summary += name;
    }
    if (candidates.size() > limit) {
        summary += "  …";
    }
    return summary;
}

[[nodiscard]] std::string longest_common_prefix(const std::vector<std::string>& values) {
    if (values.empty()) {
        return {};
    }
    std::string prefix = values.front();
    for (const auto& value : values | std::views::drop(1)) {
        const auto shared = std::ranges::mismatch(prefix, value);
        prefix.erase(shared.in1, prefix.end());
        if (prefix.empty()) {
            break;
        }
    }
    return prefix;
}

void enter_path_entry_mode(FileSystemPickerState& state) {
    state.mode = FileSystemPickerMode::PathEntry;
    state.path_buffer = abbreviate_user_path(state.directory);
    if (!state.path_buffer.ends_with('/')) {
        state.path_buffer += '/';
    }
    state.path_buffer += state.filter;
    state.status = "Type a path — Tab completes, Enter goes there, Esc returns.";
}

void leave_path_entry_mode(FileSystemPickerState& state) {
    state.mode = FileSystemPickerMode::Browse;
    state.path_buffer.clear();
    state.status.clear();
}

[[nodiscard]] FileSystemPickerEventResult confirm(FileSystemPickerState& state,
                                                 const std::filesystem::path& path) {
    state.active = false;
    return {
        .handled = true,
        .action = FileSystemPickerAction::Confirm,
        .path = path,
    };
}

/// Resolve the PathEntry buffer. Directories are navigated into (never
/// confirmed) so that "Enter" keeps meaning "go here", and the caller's choice
/// is always an explicit second keystroke on a highlighted row.
[[nodiscard]] FileSystemPickerEventResult submit_path_entry(FileSystemPickerState& state) {
    FileSystemPickerEventResult result{.handled = true};

    const auto resolved = normalize(expand_user_path(state.path_buffer));
    std::error_code ec;
    if (std::filesystem::is_directory(resolved, ec) && !ec) {
        leave_path_entry_mode(state);
        navigate_file_system_picker(state, resolved);
        return result;
    }

    const bool is_file = std::filesystem::is_regular_file(resolved, ec) && !ec;
    if (is_file && state.request.target == FileSystemPickerTarget::File
        && extension_accepted(state.request, resolved)) {
        return confirm(state, resolved);
    }

    if (is_file) {
        // Land the user next to the file so the mistake is easy to see and fix.
        const auto parent = resolved.parent_path();
        leave_path_entry_mode(state);
        navigate_file_system_picker(state, parent, resolved);
        state.status = state.request.target == FileSystemPickerTarget::Directory
            ? std::format("'{}' is a file — pick a folder.", resolved.filename().string())
            : std::format("'{}' does not match the expected file type.",
                          resolved.filename().string());
        return result;
    }

    state.status = std::format("'{}' does not exist.", state.path_buffer);
    return result;
}

/// Enter on the focused row. Opening beats choosing: a folder picker must let
/// you pass *through* a folder without adopting it, which is the whole point of
/// drilling down. Rows that cannot be opened confirm instead.
[[nodiscard]] FileSystemPickerEventResult activate_focused_row(FileSystemPickerState& state) {
    FileSystemPickerEventResult result{.handled = true};

    const FileSystemRow* row = focused_file_system_row(state);
    if (row == nullptr) {
        return result;
    }

    if (row->navigable) {
        navigate_file_system_picker(state, row->path);
        return result;
    }
    if (row->selectable) {
        return confirm(state, row->path);
    }

    state.status = state.request.target == FileSystemPickerTarget::File
        ? std::format("'{}' is not a selectable file.", row->label)
        : std::format("'{}' is not a folder.", row->label);
    return result;
}

/// Tab on the focused row: take it as the answer without opening it. Silently
/// inert on `..` and quick roots, which are waypoints rather than candidates.
[[nodiscard]] FileSystemPickerEventResult choose_focused_row(FileSystemPickerState& state) {
    const FileSystemRow* row = focused_file_system_row(state);
    if (row == nullptr || !row->selectable) {
        return {.handled = true};
    }
    return confirm(state, row->path);
}

void move_selection(FileSystemPickerState& state, int delta, bool wrap) {
    const int count = static_cast<int>(state.visible.size());
    if (count == 0) {
        state.selected = 0;
        return;
    }
    if (wrap) {
        state.selected = ((state.selected + delta) % count + count) % count;
        return;
    }
    state.selected = std::clamp(state.selected + delta, 0, count - 1);
}

void ascend(FileSystemPickerState& state) {
    const auto parent = state.directory.parent_path();
    if (parent.empty() || parent == state.directory) {
        state.status = "Already at the filesystem root.";
        return;
    }
    navigate_file_system_picker(state, parent, state.directory);
}

} // namespace

std::optional<int> score_file_system_match(std::string_view label,
                                           std::string_view lower_query) {
    if (lower_query.empty()) {
        return 0;
    }

    // Directory labels carry a trailing '/' for display; matching against it
    // would make "src/" fail a "src" exact-match test.
    if (label.ends_with('/')) {
        label.remove_suffix(1);
    }
    const std::string name = to_lower_ascii_copy(label);

    if (name == lower_query) {
        return 0;
    }
    if (name.starts_with(lower_query)) {
        return 1;
    }
    if (const auto at = name.find(lower_query); at != std::string::npos) {
        return 10 + static_cast<int>(std::min<std::size_t>(at, 89));
    }

    // Subsequence fallback ("fsp" → "FileSystemPicker"). The span between the
    // first and last matched character ranks tighter matches higher.
    std::size_t cursor = 0;
    std::size_t first = std::string::npos;
    std::size_t last = 0;
    for (const char wanted : lower_query) {
        const auto at = name.find(wanted, cursor);
        if (at == std::string::npos) {
            return std::nullopt;
        }
        if (first == std::string::npos) {
            first = at;
        }
        last = at;
        cursor = at + 1;
    }
    const auto span = last - first;
    return 100 + static_cast<int>(std::min<std::size_t>(span, 899));
}

FileSystemListing list_file_system_rows(const std::filesystem::path& directory,
                                       const FileSystemPickerRequest& request) {
    FileSystemListing listing;

    std::error_code ec;
    if (!std::filesystem::is_directory(directory, ec) || ec) {
        listing.error = std::format("'{}' is not a readable directory.",
                                    abbreviate_user_path(directory));
        return listing;
    }

    std::filesystem::directory_iterator it(directory, ec);
    if (ec) {
        listing.error = std::format("Cannot read '{}': {}",
                                    abbreviate_user_path(directory), ec.message());
        return listing;
    }

    std::vector<FileSystemRow> entries;
    for (const std::filesystem::directory_iterator end; it != end; it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;  // Skip individual unreadable entries, keep the listing.
        }
        if (entries.size() >= kFileSystemPickerMaxEntries) {
            listing.truncated = true;
            break;
        }

        const auto& entry = *it;
        const auto path = entry.path();
        const auto name = path.filename().string();
        if (name.empty() || (!request.show_hidden && is_hidden_name(name))) {
            continue;
        }

        std::error_code kind_ec;
        if (entry.is_directory(kind_ec) && !kind_ec) {
            entries.push_back(FileSystemRow{
                .role = FileSystemRowRole::Directory,
                .label = name + "/",
                .path = path,
                .selectable = request.target == FileSystemPickerTarget::Directory,
                .navigable = true,
            });
            continue;
        }
        if (kind_ec) {
            continue;
        }

        const bool selectable = request.target == FileSystemPickerTarget::File
            && extension_accepted(request, path);
        if (!selectable && !request.show_context_files) {
            continue;
        }
        entries.push_back(FileSystemRow{
            .role = FileSystemRowRole::File,
            .label = name,
            .detail = describe_size(entry),
            .path = path,
            .selectable = selectable,
        });
    }
    std::ranges::sort(entries, row_precedes);

    listing.rows.reserve(entries.size() + request.quick_roots.size() + 2);
    append_synthetic_rows(directory, request, listing.rows);
    std::ranges::move(entries, std::back_inserter(listing.rows));
    return listing;
}

FileSystemPathCompletion complete_file_system_path(std::string_view partial) {
    FileSystemPathCompletion completion;
    if (partial.empty()) {
        return completion;
    }
    if (partial == "~") {
        completion.text = "~/";
        return completion;
    }

    const auto [directory_text, leaf] = split_for_completion(partial);
    const auto directory = normalize(
        directory_text.empty() ? expand_user_path(".") : expand_user_path(directory_text));

    std::error_code ec;
    std::filesystem::directory_iterator it(directory, ec);
    if (ec) {
        return completion;
    }

    const auto lower_leaf = to_lower_ascii_copy(leaf);
    std::vector<std::string> names;
    std::vector<std::string> display;
    for (const std::filesystem::directory_iterator end; it != end; it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }
        auto name = it->path().filename().string();
        if (name.empty()) {
            continue;
        }
        if (!to_lower_ascii_copy(name).starts_with(lower_leaf)) {
            continue;
        }
        // Hidden entries only surface once the user commits to a dot prefix.
        if (leaf.empty() && is_hidden_name(name)) {
            continue;
        }
        std::error_code kind_ec;
        const bool is_directory = it->is_directory(kind_ec) && !kind_ec;
        names.push_back(name);
        display.push_back(is_directory ? name + "/" : std::move(name));
    }

    if (names.empty()) {
        return completion;
    }

    completion.candidates = std::move(display);
    if (names.size() == 1) {
        completion.text = directory_text + completion.candidates.front();
        return completion;
    }
    completion.text = directory_text + longest_common_prefix(names);
    return completion;
}

void refresh_file_system_picker_filter(FileSystemPickerState& state) {
    if (state.filter.empty()) {
        state.visible = state.rows;
        state.selected = std::clamp(state.selected, 0,
                                    std::max(0, static_cast<int>(state.visible.size()) - 1));
        return;
    }

    struct Ranked {
        int score = 0;
        const FileSystemRow* row = nullptr;
    };

    const auto lower_query = to_lower_ascii_copy(state.filter);
    std::vector<Ranked> ranked;
    ranked.reserve(state.rows.size());
    for (const auto& row : state.rows) {
        if (!is_real_entry(row.role)) {
            continue;
        }
        if (const auto score = score_file_system_match(row.label, lower_query)) {
            ranked.push_back(Ranked{.score = *score, .row = &row});
        }
    }

    std::ranges::stable_sort(ranked, [](const Ranked& lhs, const Ranked& rhs) {
        if (lhs.score != rhs.score) {
            return lhs.score < rhs.score;
        }
        return row_precedes(*lhs.row, *rhs.row);
    });

    state.visible.clear();
    state.visible.reserve(ranked.size());
    for (const auto& candidate : ranked) {
        state.visible.push_back(*candidate.row);
    }
    state.selected = std::clamp(state.selected, 0,
                                std::max(0, static_cast<int>(state.visible.size()) - 1));
}

void navigate_file_system_picker(FileSystemPickerState& state,
                                const std::filesystem::path& directory,
                                const std::filesystem::path& focus) {
    // Both arguments are routinely aliases into `state` (ascending passes
    // `state.directory` as the focus), so resolve them before mutating it.
    const auto target = normalize(directory);
    const auto focus_path = focus.empty() ? std::filesystem::path{} : normalize(focus);

    auto listing = list_file_system_rows(target, state.request);
    if (!listing.error.empty()) {
        // Refuse the move so the user is never stranded in an unreadable place.
        state.status = std::move(listing.error);
        return;
    }

    state.directory = target;
    state.rows = std::move(listing.rows);
    state.truncated = listing.truncated;
    state.filter.clear();
    state.status.clear();
    state.visible = state.rows;
    state.selected = default_selection(state.visible, focus_path);
}

void reload_file_system_picker(FileSystemPickerState& state) {
    auto listing = list_file_system_rows(state.directory, state.request);
    if (!listing.error.empty()) {
        state.status = std::move(listing.error);
        return;
    }
    const FileSystemRow* previous = focused_file_system_row(state);
    const auto previous_path = previous == nullptr ? std::filesystem::path{} : previous->path;

    state.rows = std::move(listing.rows);
    state.truncated = listing.truncated;
    refresh_file_system_picker_filter(state);

    if (!previous_path.empty()) {
        const auto match = std::ranges::find_if(state.visible, [&](const FileSystemRow& row) {
            return row.path == previous_path;
        });
        if (match != state.visible.end()) {
            state.selected =
                static_cast<int>(std::ranges::distance(state.visible.begin(), match));
        }
    }
}

void open_file_system_picker(FileSystemPickerState& state,
                            FileSystemPickerRequest request) {
    state = FileSystemPickerState{};
    state.request = std::move(request);
    state.active = true;

    // First readable candidate wins; `/` is the guaranteed-existing fallback.
    std::error_code ec;
    const std::filesystem::path candidates[] = {
        state.request.start_directory,
        std::filesystem::current_path(ec),
        core::utils::path::home_directory(),
        std::filesystem::path("/"),
    };
    for (const auto& candidate : candidates) {
        if (candidate.empty()) {
            continue;
        }
        const auto target = normalize(candidate);
        auto listing = list_file_system_rows(target, state.request);
        if (!listing.error.empty()) {
            continue;
        }
        state.directory = target;
        state.rows = std::move(listing.rows);
        state.truncated = listing.truncated;
        state.visible = state.rows;
        // On the *first* listing the user has not chosen anything yet, so focus
        // the first real entry: browsing is the overwhelmingly likely intent,
        // and starting on "Use this folder" would put two synthetic rows between
        // the cursor and the first folder worth opening.
        state.selected = std::max(0, first_real_entry(state.visible));
        return;
    }
    state.status = "No readable directory to browse.";
}

const FileSystemRow* focused_file_system_row(const FileSystemPickerState& state) {
    if (state.visible.empty()) {
        return nullptr;
    }
    const auto index = static_cast<std::size_t>(
        std::clamp(state.selected, 0, static_cast<int>(state.visible.size()) - 1));
    return &state.visible[index];
}

FileSystemPickerEventResult handle_file_system_picker_event(
    FileSystemPickerState& state,
    const ftxui::Event& event) {
    if (!state.active) {
        return {};
    }

    FileSystemPickerEventResult result{.handled = true};

    // ── Direct path entry ────────────────────────────────────────────────────
    if (state.mode == FileSystemPickerMode::PathEntry) {
        if (event == ftxui::Event::Escape) {
            leave_path_entry_mode(state);
            return result;
        }
        if (event == ftxui::Event::Return) {
            return submit_path_entry(state);
        }
        if (event == ftxui::Event::Tab) {
            const auto completion = complete_file_system_path(state.path_buffer);
            if (completion.text.empty()) {
                state.status = "No completion found.";
                return result;
            }
            const bool advanced = completion.text.size() > state.path_buffer.size();
            state.path_buffer = completion.text;
            if (completion.candidates.size() > 1) {
                state.status = std::format("{} matches: {}",
                                           completion.candidates.size(),
                                           summarize_candidates(completion.candidates, 6));
            } else {
                state.status = advanced ? std::string{} : std::string("Already complete.");
            }
            return result;
        }
        if (event == ftxui::Event::Backspace) {
            if (state.path_buffer.empty()) {
                leave_path_entry_mode(state);
            } else {
                state.path_buffer.pop_back();
                state.status.clear();
            }
            return result;
        }
        if (is_printable_character(event)) {
            state.path_buffer += event.character();
            state.status.clear();
        }
        return result;
    }

    // ── Browse ───────────────────────────────────────────────────────────────
    if (event == ftxui::Event::Escape) {
        state.active = false;
        result.action = FileSystemPickerAction::Cancel;
        return result;
    }
    if (event == ftxui::Event::ArrowUp || is_ctrl_p_event(event)) {
        move_selection(state, -1, /*wrap=*/true);
        return result;
    }
    if (event == ftxui::Event::ArrowDown || is_ctrl_n_event(event)) {
        move_selection(state, 1, /*wrap=*/true);
        return result;
    }
    if (event == ftxui::Event::PageUp) {
        move_selection(state, -kPageJump, /*wrap=*/false);
        return result;
    }
    if (event == ftxui::Event::PageDown) {
        move_selection(state, kPageJump, /*wrap=*/false);
        return result;
    }
    if (event == ftxui::Event::Home) {
        state.selected = 0;
        return result;
    }
    if (event == ftxui::Event::End) {
        state.selected = std::max(0, static_cast<int>(state.visible.size()) - 1);
        return result;
    }
    if (event == ftxui::Event::ArrowLeft) {
        ascend(state);
        return result;
    }
    if (event == ftxui::Event::ArrowRight) {
        if (const FileSystemRow* row = focused_file_system_row(state);
            row != nullptr && row->navigable) {
            navigate_file_system_picker(state, row->path);
        }
        return result;
    }
    if (event == ftxui::Event::Return) {
        return activate_focused_row(state);
    }
    if (event == ftxui::Event::Backspace) {
        if (state.filter.empty()) {
            ascend(state);
        } else {
            state.filter.pop_back();
            state.status.clear();
            refresh_file_system_picker_filter(state);
        }
        return result;
    }
    // Ctrl+A rather than the conventional Ctrl+H: terminals normalize Ctrl+H to
    // Backspace unless an enhanced keyboard protocol is active (see KeyInput).
    if (is_ctrl_letter_event(event, 'a')) {
        state.request.show_hidden = !state.request.show_hidden;
        reload_file_system_picker(state);
        state.status = state.request.show_hidden ? "Showing hidden entries."
                                                 : "Hiding hidden entries.";
        return result;
    }
    if (event == ftxui::Event::Tab) {
        return choose_focused_row(state);
    }
    if (is_ctrl_letter_event(event, 'e')) {
        enter_path_entry_mode(state);
        return result;
    }
    if (is_printable_character(event)) {
        // A leading '/' or '~' unambiguously means "I want to type a path", and
        // neither can start a filename fragment worth filtering on.
        if (state.filter.empty()
            && (event == ftxui::Event::Character('/')
                || event == ftxui::Event::Character('~'))) {
            state.mode = FileSystemPickerMode::PathEntry;
            state.path_buffer = event.character();
            state.status = "Type a path — Tab completes, Enter goes there, Esc returns.";
            return result;
        }
        state.filter += event.character();
        state.status.clear();
        state.selected = 0;
        refresh_file_system_picker_filter(state);
    }
    return result;
}

} // namespace tui
