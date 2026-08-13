#pragma once

// A reusable, framework-agnostic filesystem browser for the TUI.
//
// FTXUI 7.x ships no file/directory chooser, so every feature that needs one
// used to grow its own ad-hoc listing loop inside MainApp. This component owns
// that job once:
//
//   * `FileSystemPickerRequest`  — what the caller wants (folder vs. file,
//     where to start, which extensions matter, which shortcuts to offer).
//   * `FileSystemPickerState`    — the browser's observable state.
//   * `handle_file_system_picker_event` — the complete keyboard contract.
//
// Deliberately split three ways so each part is independently testable:
// this header holds the model plus event handling (no `ftxui/dom` dependency),
// `FileSystemPickerView.hpp` holds rendering, and MainApp only wires callers to
// outcomes.
//
// Interaction contract, which follows terminal file-browser convention rather
// than inventing its own:
//
//   Enter / →   Open a folder. Navigation always wins, so drilling down never
//               risks committing to a folder you were only passing through.
//   Enter       Choose, on rows that cannot be opened: the pinned
//               "Use this folder" row, and matching files in a file picker.
//   Tab         Choose the highlighted folder *without* entering it — the
//               accelerator for when the target is already on screen.
//   ← / ⌫      Go up (⌫ clears the filter first, if there is one).

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include <ftxui/component/event.hpp>

namespace tui {

/// What the caller ultimately wants back.
enum class FileSystemPickerTarget : std::uint8_t {
    Directory,  ///< Only directories can be confirmed.
    File,       ///< Only files can be confirmed; directories stay navigable.
};

/// A labelled jump target rendered at the top of the listing (home, workspace
/// root, current working directory, …). Cheaper to discover than typing a path
/// and it costs no extra key binding.
struct FileSystemQuickRoot {
    std::string label;
    std::filesystem::path path;
};

/// Configuration of one browsing session. Everything here is fixed for the
/// lifetime of the picker except `show_hidden`, which the user can flip while
/// browsing (Ctrl+A) — it is a view option, not a caller policy.
struct FileSystemPickerRequest {
    std::string title = "SELECT FOLDER";  ///< Panel badge, upper-case by convention.
    std::string hint;                     ///< One line explaining the consequence.
    FileSystemPickerTarget target = FileSystemPickerTarget::Directory;
    std::filesystem::path start_directory;  ///< Empty → process working directory.
    /// Lower-case, dot-prefixed (".gguf"). Empty accepts any file. Files that
    /// do not match stay visible but unselectable so the user keeps context.
    std::vector<std::string> extensions;
    std::vector<FileSystemQuickRoot> quick_roots;
    bool show_hidden = false;
    /// List files that cannot be confirmed (dimmed). Directory pickers read far
    /// better with them: an empty-looking folder is otherwise indistinguishable
    /// from a folder full of files.
    bool show_context_files = true;
};

/// Why a row exists. Roles drive affordances and rendering, and keep synthetic
/// navigation rows out of the fuzzy filter.
enum class FileSystemRowRole : std::uint8_t {
    ConfirmCurrent,  ///< Confirms the directory currently being browsed.
    Parent,          ///< `..` — ascends one level.
    QuickRoot,       ///< Labelled jump target.
    Directory,       ///< Real subdirectory.
    File,            ///< Real file.
};

struct FileSystemRow {
    FileSystemRowRole role = FileSystemRowRole::File;
    std::string label;                   ///< Primary text (directories keep a `/`).
    std::string detail;                  ///< Trailing hint: size, target path, …
    std::filesystem::path path;          ///< Absolute target.
    bool selectable = false;             ///< This row's path is a valid answer (Tab).
    bool navigable = false;              ///< Enter/→ descends into this row.

    /// True when Enter is unambiguous because the row cannot be opened. A row
    /// that is selectable *and* navigable (a folder in a folder picker) opens
    /// on Enter and is chosen with Tab, so browsing never commits by accident.
    [[nodiscard]] bool confirms_on_enter() const noexcept {
        return selectable && !navigable;
    }
};

/// Browse uses type-to-filter; PathEntry accepts a literal path with
/// Tab completion for users who already know where they are going.
enum class FileSystemPickerMode : std::uint8_t {
    Browse,
    PathEntry,
};

struct FileSystemPickerState {
    bool active = false;
    FileSystemPickerRequest request;
    FileSystemPickerMode mode = FileSystemPickerMode::Browse;

    std::filesystem::path directory;   ///< Absolute directory being listed.
    std::vector<FileSystemRow> rows;   ///< Full listing (synthetic + real).
    std::vector<FileSystemRow> visible;///< `rows` after the fuzzy filter.
    int selected = 0;                  ///< Index into `visible`.

    std::string filter;                ///< Browse-mode query.
    std::string path_buffer;           ///< PathEntry-mode buffer.
    std::string status;                ///< Error or transient hint.
    bool truncated = false;            ///< Listing hit the entry cap.
};

enum class FileSystemPickerAction : std::uint8_t {
    None,
    Confirm,  ///< `path` holds the user's choice.
    Cancel,
};

struct FileSystemPickerEventResult {
    bool handled = false;
    FileSystemPickerAction action = FileSystemPickerAction::None;
    std::filesystem::path path;  ///< Meaningful only for `Confirm`.
};

/// Upper bound on rows read from a single directory. Protects the render loop
/// from pathological directories without needing a background thread.
inline constexpr std::size_t kFileSystemPickerMaxEntries = 4096;

/// Result of reading one directory from disk.
struct FileSystemListing {
    std::vector<FileSystemRow> rows;
    std::string error;       ///< Non-empty when the directory is unreadable.
    bool truncated = false;
};

/// Read @p directory and build the row list described by @p request, including
/// the synthetic confirm/parent/quick-root rows. Pure with respect to state,
/// which makes it directly testable against a temporary directory tree.
[[nodiscard]] FileSystemListing list_file_system_rows(
    const std::filesystem::path& directory,
    const FileSystemPickerRequest& request);

/// Rank @p label against an already-lower-cased @p query. Lower is better;
/// `std::nullopt` means "no match". Prefix beats substring beats subsequence,
/// so typing `sr` finds `src/` before `assets/resources`.
[[nodiscard]] std::optional<int> score_file_system_match(std::string_view label,
                                                        std::string_view lower_query);

/// Tab-completion for a partially typed path.
struct FileSystemPathCompletion {
    std::string text;                     ///< Longest unambiguous completion.
    std::vector<std::string> candidates;  ///< Ambiguous continuations, if any.
};

[[nodiscard]] FileSystemPathCompletion complete_file_system_path(std::string_view partial);

/// Reset @p state and list the requested start directory.
void open_file_system_picker(FileSystemPickerState& state,
                            FileSystemPickerRequest request);

/// Move to @p directory, clearing the filter. When @p focus names a child of
/// @p directory that row becomes selected, so ascending with `←` lands back on
/// the folder just left.
void navigate_file_system_picker(FileSystemPickerState& state,
                                const std::filesystem::path& directory,
                                const std::filesystem::path& focus = {});

/// Re-read the current directory from disk (after toggling hidden files, or to
/// pick up external changes) and re-apply the filter.
void reload_file_system_picker(FileSystemPickerState& state);

/// Re-apply the fuzzy filter to the cached listing and clamp the selection.
void refresh_file_system_picker_filter(FileSystemPickerState& state);

/// The row under the cursor, or `nullptr` when the listing is empty.
[[nodiscard]] const FileSystemRow* focused_file_system_row(
    const FileSystemPickerState& state);

/// Apply one key event. Returns `handled == false` only when the picker is
/// inactive, so callers can treat a handled event as fully consumed.
[[nodiscard]] FileSystemPickerEventResult handle_file_system_picker_event(
    FileSystemPickerState& state,
    const ftxui::Event& event);

} // namespace tui
