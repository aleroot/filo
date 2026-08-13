#include <catch2/catch_test_macros.hpp>

#include "tui/FileSystemPicker.hpp"
#include "tui/FileSystemPickerView.hpp"

#include <ftxui/component/event.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/screen/screen.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

using tui::FileSystemPickerAction;
using tui::FileSystemPickerMode;
using tui::FileSystemPickerRequest;
using tui::FileSystemPickerState;
using tui::FileSystemPickerTarget;
using tui::FileSystemRowRole;

namespace {

/// Self-cleaning directory tree:
///
///     root/
///       alpha/
///         nested/
///       beta/
///       .hidden/
///       notes.md
///       model.gguf
class TempTree {
public:
    TempTree() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        root_ = std::filesystem::temp_directory_path()
            / std::filesystem::path("filo-fs-picker-"
                                    + std::to_string(stamp)
                                    + "-"
                                    + std::to_string(counter()++));
        std::filesystem::create_directories(root_ / "alpha" / "nested");
        std::filesystem::create_directories(root_ / "beta");
        std::filesystem::create_directories(root_ / ".hidden");
        write(root_ / "notes.md", "notes");
        write(root_ / "model.gguf", "weights");
    }

    TempTree(const TempTree&) = delete;
    TempTree& operator=(const TempTree&) = delete;

    ~TempTree() {
        std::error_code ec;
        std::filesystem::remove_all(root_, ec);
    }

    [[nodiscard]] const std::filesystem::path& root() const noexcept { return root_; }

private:
    static int& counter() {
        static int value = 0;
        return value;
    }

    static void write(const std::filesystem::path& path, std::string_view content) {
        std::ofstream out(path);
        out << content;
    }

    std::filesystem::path root_;
};

[[nodiscard]] FileSystemPickerRequest folder_request(const std::filesystem::path& start) {
    return FileSystemPickerRequest{
        .title = "SELECT FOLDER",
        .target = FileSystemPickerTarget::Directory,
        .start_directory = start,
    };
}

[[nodiscard]] std::vector<std::string> labels(const FileSystemPickerState& state) {
    std::vector<std::string> out;
    out.reserve(state.visible.size());
    for (const auto& row : state.visible) {
        out.push_back(row.label);
    }
    return out;
}

[[nodiscard]] int index_of(const FileSystemPickerState& state, std::string_view label) {
    const auto match = std::ranges::find_if(
        state.visible, [&](const auto& row) { return row.label == label; });
    return match == state.visible.end()
        ? -1
        : static_cast<int>(std::distance(state.visible.begin(), match));
}

[[nodiscard]] bool contains(const std::vector<std::string>& values, std::string_view needle) {
    return std::ranges::find(values, needle) != values.end();
}

/// Type a whole string as individual character events.
void type(FileSystemPickerState& state, std::string_view text) {
    for (const char ch : text) {
        (void)tui::handle_file_system_picker_event(state, ftxui::Event::Character(ch));
    }
}

} // namespace

// ── Listing ─────────────────────────────────────────────────────────────────

TEST_CASE("Folder picker pins a confirm row, a parent row, and quick roots",
          "[tui][file_system_picker]") {
    const TempTree tree;
    auto request = folder_request(tree.root());
    request.quick_roots = {
        {.label = "Workspace", .path = tree.root() / "beta"},
        {.label = "Missing", .path = tree.root() / "does-not-exist"},
        // Pointing at the directory being browsed would be a no-op shortcut.
        {.label = "Here", .path = tree.root()},
    };

    const auto listing = tui::list_file_system_rows(tree.root(), request);
    REQUIRE(listing.error.empty());
    REQUIRE(listing.rows.size() >= 3);

    CHECK(listing.rows[0].role == FileSystemRowRole::ConfirmCurrent);
    CHECK(listing.rows[0].selectable);
    CHECK(listing.rows[0].path == tree.root());
    CHECK(listing.rows[1].role == FileSystemRowRole::Parent);
    CHECK(listing.rows[1].navigable);
    CHECK_FALSE(listing.rows[1].selectable);
    CHECK(listing.rows[2].role == FileSystemRowRole::QuickRoot);
    CHECK(listing.rows[2].label == "Workspace");

    // Only reachable, non-redundant shortcuts survive.
    const auto quick_root_count = std::ranges::count_if(
        listing.rows, [](const auto& row) { return row.role == FileSystemRowRole::QuickRoot; });
    CHECK(quick_root_count == 1);
}

TEST_CASE("Folder picker lists directories first and files as context only",
          "[tui][file_system_picker]") {
    const TempTree tree;
    const auto listing = tui::list_file_system_rows(tree.root(), folder_request(tree.root()));

    std::vector<std::string> real_labels;
    for (const auto& row : listing.rows) {
        if (row.role == FileSystemRowRole::Directory || row.role == FileSystemRowRole::File) {
            real_labels.push_back(row.label);
        }
    }
    REQUIRE(real_labels == std::vector<std::string>{"alpha/", "beta/", "model.gguf", "notes.md"});

    for (const auto& row : listing.rows) {
        if (row.role == FileSystemRowRole::Directory) {
            CHECK(row.selectable);
            CHECK(row.navigable);
        }
        if (row.role == FileSystemRowRole::File) {
            CHECK_FALSE(row.selectable);  // Never a valid answer for a folder picker.
            CHECK_FALSE(row.navigable);
            CHECK_FALSE(row.detail.empty());  // Size is shown for orientation.
        }
    }
}

TEST_CASE("File picker honours the extension filter without hiding context",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerRequest request{
        .target = FileSystemPickerTarget::File,
        .start_directory = tree.root(),
        .extensions = {".gguf"},
    };

    const auto listing = tui::list_file_system_rows(tree.root(), request);
    REQUIRE(listing.error.empty());

    // A file picker has nothing to confirm about the directory itself.
    CHECK(std::ranges::none_of(listing.rows, [](const auto& row) {
        return row.role == FileSystemRowRole::ConfirmCurrent;
    }));

    for (const auto& row : listing.rows) {
        if (row.label == "model.gguf") {
            CHECK(row.selectable);
        }
        if (row.label == "notes.md") {
            CHECK_FALSE(row.selectable);
        }
        if (row.role == FileSystemRowRole::Directory) {
            CHECK(row.navigable);
            CHECK_FALSE(row.selectable);  // Directories are never a file answer.
        }
    }

    request.show_context_files = false;
    const auto filtered = tui::list_file_system_rows(tree.root(), request);
    CHECK(std::ranges::none_of(filtered.rows,
                               [](const auto& row) { return row.label == "notes.md"; }));
}

TEST_CASE("Hidden entries stay out of the listing until asked for",
          "[tui][file_system_picker]") {
    const TempTree tree;
    auto request = folder_request(tree.root());

    auto listing = tui::list_file_system_rows(tree.root(), request);
    CHECK(std::ranges::none_of(listing.rows,
                               [](const auto& row) { return row.label == ".hidden/"; }));

    request.show_hidden = true;
    listing = tui::list_file_system_rows(tree.root(), request);
    CHECK(std::ranges::any_of(listing.rows,
                              [](const auto& row) { return row.label == ".hidden/"; }));
}

TEST_CASE("Unreadable directories report an error instead of an empty listing",
          "[tui][file_system_picker]") {
    const TempTree tree;
    const auto listing =
        tui::list_file_system_rows(tree.root() / "nope", folder_request(tree.root()));
    CHECK(listing.rows.empty());
    CHECK_FALSE(listing.error.empty());
}

// ── Fuzzy matching ──────────────────────────────────────────────────────────

TEST_CASE("Match scoring prefers exact, then prefix, then substring, then subsequence",
          "[tui][file_system_picker]") {
    const auto exact = tui::score_file_system_match("src/", "src");
    const auto prefix = tui::score_file_system_match("srcgen", "src");
    const auto substring = tui::score_file_system_match("my-src-dir", "src");
    const auto subsequence = tui::score_file_system_match("SessionRecorder", "src");

    REQUIRE(exact.has_value());
    REQUIRE(prefix.has_value());
    REQUIRE(substring.has_value());
    REQUIRE(subsequence.has_value());
    CHECK(*exact < *prefix);
    CHECK(*prefix < *substring);
    CHECK(*substring < *subsequence);

    CHECK_FALSE(tui::score_file_system_match("beta", "zzz").has_value());
    CHECK(tui::score_file_system_match("anything", "") == 0);
}

TEST_CASE("Filtering hides the synthetic rows and ranks matches",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root()));
    REQUIRE(state.active);

    type(state, "a");
    CHECK(state.filter == "a");
    // Synthetic rows are "places", not search results.
    for (const auto& row : state.visible) {
        CHECK((row.role == FileSystemRowRole::Directory
               || row.role == FileSystemRowRole::File));
    }
    CHECK(contains(labels(state), "alpha/"));
    CHECK(contains(labels(state), "beta/"));
    CHECK_FALSE(contains(labels(state), "Use this folder"));
    CHECK(state.visible.front().label == "alpha/");  // Prefix beats substring.

    // Backspace clears the query one character at a time before it navigates.
    (void)tui::handle_file_system_picker_event(state, ftxui::Event::Backspace);
    CHECK(state.filter.empty());
    CHECK(state.directory == tree.root());
    CHECK(state.visible.front().role == FileSystemRowRole::ConfirmCurrent);
}

// ── Navigation ──────────────────────────────────────────────────────────────

TEST_CASE("Opening focuses the first real entry so browsing starts immediately",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root()));

    // Not the pinned confirm row: the user came here to look around.
    REQUIRE(tui::focused_file_system_row(state) != nullptr);
    CHECK(tui::focused_file_system_row(state)->label == "alpha/");
    CHECK(state.visible[0].role == FileSystemRowRole::ConfirmCurrent);
}

TEST_CASE("Enter opens a folder instead of choosing it",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root()));

    state.selected = index_of(state, "alpha/");
    REQUIRE(state.selected >= 0);

    // The regression this guards: Enter used to commit the first folder touched,
    // making it impossible to drill down.
    auto result = tui::handle_file_system_picker_event(state, ftxui::Event::Return);
    CHECK(result.handled);
    CHECK(result.action == FileSystemPickerAction::None);
    CHECK(state.directory == tree.root() / "alpha");
    CHECK(contains(labels(state), "nested/"));

    // Descending lands on "Use this folder", so a second Enter takes it.
    REQUIRE(tui::focused_file_system_row(state) != nullptr);
    CHECK(tui::focused_file_system_row(state)->role == FileSystemRowRole::ConfirmCurrent);
    result = tui::handle_file_system_picker_event(state, ftxui::Event::Return);
    CHECK(result.action == FileSystemPickerAction::Confirm);
    CHECK(result.path == tree.root() / "alpha");
    CHECK_FALSE(state.active);
}

TEST_CASE("Right opens and Left ascends back onto the folder just left",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root()));

    state.selected = index_of(state, "alpha/");
    auto result = tui::handle_file_system_picker_event(state, ftxui::Event::ArrowRight);
    CHECK(result.action == FileSystemPickerAction::None);
    CHECK(state.directory == tree.root() / "alpha");

    (void)tui::handle_file_system_picker_event(state, ftxui::Event::ArrowLeft);
    CHECK(state.directory == tree.root());
    REQUIRE(tui::focused_file_system_row(state) != nullptr);
    CHECK(tui::focused_file_system_row(state)->label == "alpha/");
}

TEST_CASE("Tab takes the highlighted folder without entering it",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root()));

    state.selected = index_of(state, "beta/");
    REQUIRE(state.selected >= 0);
    const auto result = tui::handle_file_system_picker_event(state, ftxui::Event::Tab);
    CHECK(result.action == FileSystemPickerAction::Confirm);
    CHECK(result.path == tree.root() / "beta");
    CHECK_FALSE(state.active);
}

TEST_CASE("Tab is inert on waypoint rows", "[tui][file_system_picker]") {
    const TempTree tree;
    auto request = folder_request(tree.root() / "alpha");
    request.quick_roots = {{.label = "Root", .path = tree.root()}};
    FileSystemPickerState state;
    tui::open_file_system_picker(state, std::move(request));

    // `..` and quick roots are places to go, not answers to give.
    for (const std::string_view waypoint : {"..", "Root"}) {
        state.selected = index_of(state, waypoint);
        REQUIRE(state.selected >= 0);
        const auto result = tui::handle_file_system_picker_event(state, ftxui::Event::Tab);
        CHECK(result.handled);
        CHECK(result.action == FileSystemPickerAction::None);
        CHECK(state.active);
        CHECK(state.directory == tree.root() / "alpha");
    }
}

TEST_CASE("Enter on the confirm row returns the directory being browsed",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root() / "beta"));

    // `beta` is empty, so the confirm row is the only thing to focus.
    state.selected = index_of(state, "Use this folder");
    REQUIRE(state.selected >= 0);
    const auto result = tui::handle_file_system_picker_event(state, ftxui::Event::Return);
    CHECK(result.action == FileSystemPickerAction::Confirm);
    CHECK(result.path == tree.root() / "beta");
}

TEST_CASE("A file picker descends into directories and confirms only files",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, FileSystemPickerRequest{
                                           .target = FileSystemPickerTarget::File,
                                           .start_directory = tree.root(),
                                       });

    state.selected = index_of(state, "alpha/");
    REQUIRE(state.selected >= 0);
    auto result = tui::handle_file_system_picker_event(state, ftxui::Event::Return);
    CHECK(result.action == FileSystemPickerAction::None);
    CHECK(state.directory == tree.root() / "alpha");

    (void)tui::handle_file_system_picker_event(state, ftxui::Event::ArrowLeft);
    state.selected = index_of(state, "notes.md");
    REQUIRE(state.selected >= 0);
    result = tui::handle_file_system_picker_event(state, ftxui::Event::Return);
    CHECK(result.action == FileSystemPickerAction::Confirm);
    CHECK(result.path == tree.root() / "notes.md");
}

TEST_CASE("Navigation failures leave the browser where it was",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root()));

    tui::navigate_file_system_picker(state, tree.root() / "missing");
    CHECK(state.directory == tree.root());
    CHECK_FALSE(state.status.empty());
}

TEST_CASE("Escape cancels the browser", "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root()));

    const auto result = tui::handle_file_system_picker_event(state, ftxui::Event::Escape);
    CHECK(result.handled);
    CHECK(result.action == FileSystemPickerAction::Cancel);
    CHECK_FALSE(state.active);

    // An inactive picker consumes nothing, so callers can chain handlers.
    CHECK_FALSE(tui::handle_file_system_picker_event(state, ftxui::Event::Return).handled);
}

TEST_CASE("Ctrl+A toggles hidden entries and keeps the cursor on its row",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root()));

    state.selected = index_of(state, "beta/");
    REQUIRE(state.selected >= 0);

    const auto ctrl_a = ftxui::Event::Special({1});
    (void)tui::handle_file_system_picker_event(state, ctrl_a);
    CHECK(state.request.show_hidden);
    CHECK(contains(labels(state), ".hidden/"));
    REQUIRE(tui::focused_file_system_row(state) != nullptr);
    CHECK(tui::focused_file_system_row(state)->label == "beta/");

    (void)tui::handle_file_system_picker_event(state, ctrl_a);
    CHECK_FALSE(state.request.show_hidden);
    CHECK_FALSE(contains(labels(state), ".hidden/"));
}

TEST_CASE("Selection wraps and clamps", "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root()));
    const int count = static_cast<int>(state.visible.size());
    REQUIRE(count > 1);

    state.selected = 0;
    (void)tui::handle_file_system_picker_event(state, ftxui::Event::ArrowUp);
    CHECK(state.selected == count - 1);
    (void)tui::handle_file_system_picker_event(state, ftxui::Event::ArrowDown);
    CHECK(state.selected == 0);

    (void)tui::handle_file_system_picker_event(state, ftxui::Event::End);
    CHECK(state.selected == count - 1);
    (void)tui::handle_file_system_picker_event(state, ftxui::Event::PageUp);
    CHECK(state.selected == 0);  // Paging clamps rather than wrapping.
}

// ── Direct path entry ───────────────────────────────────────────────────────

TEST_CASE("Path completion resolves unique names and common prefixes",
          "[tui][file_system_picker]") {
    const TempTree tree;
    const auto prefix = tree.root().string();

    const auto unique = tui::complete_file_system_path(prefix + "/alp");
    CHECK(unique.text == prefix + "/alpha/");  // Directories gain a separator.
    CHECK(unique.candidates.size() == 1);

    const auto ambiguous = tui::complete_file_system_path(prefix + "/");
    CHECK(ambiguous.candidates.size() == 4);  // Hidden entries stay hidden.
    CHECK(ambiguous.text == prefix + "/");    // No shared prefix to add.

    CHECK(tui::complete_file_system_path(prefix + "/zzz").text.empty());
    CHECK(tui::complete_file_system_path("~").text == "~/");
    CHECK(tui::complete_file_system_path("").text.empty());
}

TEST_CASE("Typing a leading slash switches to path entry", "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root()));

    (void)tui::handle_file_system_picker_event(state, ftxui::Event::Character('/'));
    REQUIRE(state.mode == FileSystemPickerMode::PathEntry);
    CHECK(state.path_buffer == "/");

    (void)tui::handle_file_system_picker_event(state, ftxui::Event::Escape);
    CHECK(state.mode == FileSystemPickerMode::Browse);
    CHECK(state.active);  // Escape leaves path entry before it closes the panel.
}

TEST_CASE("Ctrl+E enters path entry seeded with the current directory and filter",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root()));

    type(state, "al");
    (void)tui::handle_file_system_picker_event(state, ftxui::Event::Special({5}));
    REQUIRE(state.mode == FileSystemPickerMode::PathEntry);
    CHECK(state.path_buffer == tree.root().string() + "/al");

    // Tab only means "complete" once the buffer is what is being edited.
    (void)tui::handle_file_system_picker_event(state, ftxui::Event::Tab);
    CHECK(state.path_buffer == tree.root().string() + "/alpha/");

    (void)tui::handle_file_system_picker_event(state, ftxui::Event::Return);
    CHECK(state.mode == FileSystemPickerMode::Browse);
    CHECK(state.directory == tree.root() / "alpha");
}

TEST_CASE("Path entry never confirms a directory implicitly",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root()));

    state.mode = FileSystemPickerMode::PathEntry;
    state.path_buffer = (tree.root() / "beta").string();
    const auto result = tui::handle_file_system_picker_event(state, ftxui::Event::Return);
    CHECK(result.action == FileSystemPickerAction::None);
    CHECK(state.directory == tree.root() / "beta");
    // …but the confirm row is focused, so one more Enter answers.
    REQUIRE(tui::focused_file_system_row(state) != nullptr);
    CHECK(tui::focused_file_system_row(state)->role == FileSystemRowRole::ConfirmCurrent);
}

TEST_CASE("Path entry reports missing paths and mismatched kinds",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, folder_request(tree.root()));

    state.mode = FileSystemPickerMode::PathEntry;
    state.path_buffer = (tree.root() / "nope").string();
    (void)tui::handle_file_system_picker_event(state, ftxui::Event::Return);
    CHECK(state.mode == FileSystemPickerMode::PathEntry);
    CHECK_FALSE(state.status.empty());

    // A file typed into a folder picker lands beside the file with an
    // explanation rather than silently failing.
    state.path_buffer = (tree.root() / "notes.md").string();
    (void)tui::handle_file_system_picker_event(state, ftxui::Event::Return);
    CHECK(state.mode == FileSystemPickerMode::Browse);
    CHECK(state.directory == tree.root());
    CHECK_FALSE(state.status.empty());
}

TEST_CASE("A file picker confirms a matching path typed directly",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    tui::open_file_system_picker(state, FileSystemPickerRequest{
                                           .target = FileSystemPickerTarget::File,
                                           .start_directory = tree.root(),
                                           .extensions = {".gguf"},
                                       });

    state.mode = FileSystemPickerMode::PathEntry;
    state.path_buffer = (tree.root() / "model.gguf").string();
    const auto result = tui::handle_file_system_picker_event(state, ftxui::Event::Return);
    CHECK(result.action == FileSystemPickerAction::Confirm);
    CHECK(result.path == tree.root() / "model.gguf");
    CHECK_FALSE(state.active);
}

// ── Rendering ───────────────────────────────────────────────────────────────

TEST_CASE("The panel renders the breadcrumb, rows, and hint",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    auto request = folder_request(tree.root());
    request.hint = "Pick a project folder.";
    tui::open_file_system_picker(state, std::move(request));

    auto element = tui::render_file_system_picker_panel(state);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(120),
                                       ftxui::Dimension::Fixed(24));
    ftxui::Render(screen, element);
    const auto output = screen.ToString();

    CHECK(output.find("SELECT FOLDER") != std::string::npos);
    CHECK(output.find("Use this folder") != std::string::npos);
    CHECK(output.find("alpha/") != std::string::npos);
    CHECK(output.find("Pick a project folder.") != std::string::npos);
    CHECK(output.find(tree.root().filename().string()) != std::string::npos);
}

TEST_CASE("The panel surfaces the status message over the static hint",
          "[tui][file_system_picker]") {
    const TempTree tree;
    FileSystemPickerState state;
    auto request = folder_request(tree.root());
    request.hint = "Static hint.";
    tui::open_file_system_picker(state, std::move(request));
    tui::navigate_file_system_picker(state, tree.root() / "missing");

    auto element = tui::render_file_system_picker_panel(state);
    // Wide enough that the wrapped status stays on one line for the assertion.
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(220),
                                       ftxui::Dimension::Fixed(24));
    ftxui::Render(screen, element);
    const auto output = screen.ToString();

    CHECK(output.find("Static hint.") == std::string::npos);
    CHECK(output.find("is not a readable directory") != std::string::npos);
}
