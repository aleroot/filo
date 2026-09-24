#include "tui/WorkspaceDetailsPanel.hpp"
#include "core/workspace/SessionWorkspace.hpp"

#include <catch2/catch_test_macros.hpp>
#include <ftxui/screen/screen.hpp>
#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <unistd.h>

namespace {

std::string render_panel(const core::workspace::WorkspaceSnapshot& workspace,
                         std::size_t selected = 0, int width = 100,
                         int max_lines = 12) {
    auto element = tui::render_workspace_details_panel(workspace, selected, max_lines);
    auto screen = ftxui::Screen::Create(
        ftxui::Dimension::Fixed(width), ftxui::Dimension::Fit(element));
    ftxui::Render(screen, element);
    return screen.ToString();
}

// Unique per process and per call: ctest runs each test case as its own
// process in parallel, and Catch re-enters a TEST_CASE once per SECTION.
struct TempDir {
    std::filesystem::path path;
    explicit TempDir(std::string_view label) {
        static std::atomic<unsigned> counter{0};
        path = std::filesystem::temp_directory_path()
            / std::format("{}_{}_{}_{}", label, ::getpid(), std::random_device{}(),
                          counter.fetch_add(1));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

void write_file(const std::filesystem::path& path) {
    std::ofstream(path, std::ios::binary) << "png-bytes";
    REQUIRE(std::filesystem::is_regular_file(path));
}

} // namespace

TEST_CASE("Workspace details show loaded roots in precedence order", "[workspace_details][tui]") {
    core::workspace::WorkspaceSnapshot workspace{
        .primary = "/projects/app",
        .additional = {"/projects/library", {}, "/other/app"},
    };
    const auto output = render_panel(workspace);
    CHECK(output.find("Workspaces (3)") != std::string::npos);
    CHECK(output.find("1. app · Primary") != std::string::npos);
    CHECK(output.find("2. library · Additional") != std::string::npos);
    CHECK(output.find("3. app · Additional") != std::string::npos);
    CHECK(output.find("/projects/app") < output.find("/projects/library"));
    CHECK(output.find("/projects/library") < output.find("/other/app"));

    SECTION("Updated snapshots replace the displayed workspace list") {
        workspace.primary = "/new/thread";
        workspace.additional.clear();
        const auto updated = render_panel(workspace);
        CHECK(updated.find("Workspaces (1)") != std::string::npos);
        CHECK(updated.find("/new/thread") != std::string::npos);
        CHECK(updated.find("/projects/app") == std::string::npos);
    }
}

TEST_CASE("Workspace details handle empty and root-only snapshots", "[workspace_details][tui]") {
    CHECK(render_panel({}).find("No workspaces loaded") != std::string::npos);
    CHECK(render_panel({.primary = "/"}).find("1. / · Primary") != std::string::npos);
    CHECK(render_panel({.additional = {"/extra"}}).find("1. extra · Additional") != std::string::npos);
}

TEST_CASE("Workspace details keep every root reachable in a bounded viewport", "[workspace_details][tui]") {
    core::workspace::WorkspaceSnapshot workspace{.primary = "/projects/main"};
    for (int i = 1; i <= 30; ++i) {
        workspace.additional.emplace_back("/projects/extra-" + std::to_string(i));
    }
    const auto top = render_panel(workspace, 0, 80, 6);
    CHECK(top.find("/projects/main") != std::string::npos);
    CHECK(top.find("/projects/extra-30") == std::string::npos);
    const auto bottom = render_panel(workspace, 100, 80, 6);
    CHECK(bottom.find("/projects/extra-30") != std::string::npos);
    CHECK(bottom.find("/projects/main") == std::string::npos);
}

TEST_CASE("Workspace paths wrap without losing the tail on narrow terminals", "[workspace_details][tui]") {
    const auto output = render_panel({
        .primary = "/a-very-long-directory-without-spaces/another-long-directory/unique-tail",
    }, 0, 38);
    CHECK(output.find("/a-very-long-directory") != std::string::npos);
    CHECK(output.find("unique-tail") != std::string::npos);
}

TEST_CASE("Workspace details navigation stays in bounds and dismisses consistently", "[workspace_details][tui]") {
    using namespace ftxui;
    tui::WorkspaceDetailsPanelState state;
    CHECK_FALSE(tui::handle_workspace_details_event(state, Event::ArrowDown, 10));
    state.active = true;
    CHECK(tui::handle_workspace_details_event(state, Event::ArrowUp, 10));
    CHECK(state.selected == 0);
    CHECK(tui::handle_workspace_details_event(state, Event::ArrowDown, 10));
    CHECK(state.selected == 1);
    CHECK(tui::handle_workspace_details_event(state, Event::PageDown, 10));
    CHECK(state.selected == 6);
    CHECK(tui::handle_workspace_details_event(state, Event::End, 10));
    CHECK(state.selected == 9);
    CHECK(tui::handle_workspace_details_event(state, Event::ArrowDown, 10));
    CHECK(state.selected == 9);
    CHECK(tui::handle_workspace_details_event(state, Event::PageUp, 10));
    CHECK(state.selected == 4);

    Mouse mouse;
    mouse.button = Mouse::WheelDown;
    CHECK(tui::handle_workspace_details_event(state, Event::Mouse("", mouse), 10));
    CHECK(state.selected == 5);
    mouse.button = Mouse::WheelUp;
    CHECK(tui::handle_workspace_details_event(state, Event::Mouse("", mouse), 10));
    CHECK(state.selected == 4);
    CHECK(tui::handle_workspace_details_event(state, Event::Home, 10));
    CHECK(state.selected == 0);

    state.selected = 9;
    CHECK(tui::handle_workspace_details_event(state, Event::ArrowDown, 2));
    CHECK(state.selected == 1);
    CHECK(tui::handle_workspace_details_event(state, Event::ArrowDown, 0));
    CHECK(state.selected == 0);
    CHECK(tui::handle_workspace_details_event(state, Event::Character('x'), 0));
    CHECK(tui::handle_workspace_details_event(state, Event::Return, 0));
    CHECK(tui::handle_workspace_details_event(state, Event::CtrlU, 0));
    CHECK_FALSE(tui::handle_workspace_details_event(state, Event::CtrlC, 0));
    CHECK_FALSE(tui::handle_workspace_details_event(state, Event::Custom, 0));

    for (const auto& event : {Event::Escape, Event::Character('q'), Event::Character('Q')}) {
        state.active = true;
        CHECK(tui::handle_workspace_details_event(state, event, 0));
        CHECK_FALSE(state.active);
    }
}

TEST_CASE("Attached files never count as workspaces, even after the file is gone",
          "[workspace_details][tui][attachments]") {
    namespace fs = std::filesystem;
    const TempDir sandbox{"filo_workspace_attachments"};
    fs::create_directories(sandbox.path / "project");
    fs::create_directories(sandbox.path / "library");
    fs::create_directories(sandbox.path / "shots");
    const fs::path screenshot = sandbox.path / "shots" / "Screenshot.png";
    const fs::path second = sandbox.path / "shots" / "second.png";
    write_file(screenshot);
    write_file(second);

    core::workspace::SessionWorkspace workspace{core::workspace::WorkspaceSnapshot{
        .primary = sandbox.path / "project",
        .enforce = true,
    }};
    REQUIRE(workspace.add_additional_paths({screenshot, second, sandbox.path / "library"}) == 3);
    REQUIRE(workspace.snapshot().attached_files.size() == 2);
    CHECK(tui::workspace_details_workspace_count(workspace.snapshot()) == 2);
    CHECK(tui::workspace_details_entry_count(workspace.snapshot()) == 4);

    SECTION("a moved or deleted attachment stays an attachment") {
        // macOS screenshot thumbnails are temporary files deleted right after
        // the drop; a filesystem probe then reported them as workspaces.
        REQUIRE(fs::remove(screenshot));
        CHECK(tui::workspace_details_workspace_count(workspace.snapshot()) == 2);
        const auto output = render_panel(workspace.snapshot());
        CHECK(output.find("Workspaces (2)") != std::string::npos);
        CHECK(output.find("Attachments (2)") != std::string::npos);
    }

    SECTION("re-attaching the same file does not duplicate it") {
        CHECK(workspace.add_additional_paths({screenshot}) == 0);
        CHECK(workspace.snapshot().attached_files.size() == 2);
        CHECK(tui::workspace_details_entry_count(workspace.snapshot()) == 4);
    }

    SECTION("granting the containing directory absorbs its attachments") {
        REQUIRE(workspace.add_additional_paths({sandbox.path / "shots"}) == 1);
        CHECK(workspace.snapshot().attached_files.empty());
        CHECK(tui::workspace_details_workspace_count(workspace.snapshot()) == 3);
        CHECK(tui::workspace_details_entry_count(workspace.snapshot()) == 3);
    }

    SECTION("changing the primary to a parent drops nested attachments") {
        REQUIRE(workspace.set_primary(sandbox.path));
        CHECK(workspace.snapshot().additional.empty());
        CHECK(workspace.snapshot().attached_files.empty());
        CHECK(tui::workspace_details_workspace_count(workspace.snapshot()) == 1);
    }

    SECTION("copied and re-normalized snapshots keep the grant-time record") {
        REQUIRE(fs::remove(screenshot));
        const core::workspace::SessionWorkspace copy{workspace.snapshot()};
        CHECK(copy.snapshot().attached_files.size() == 2);
        CHECK(tui::workspace_details_workspace_count(copy.snapshot()) == 2);
    }
}

TEST_CASE("Unrecorded roots are classified by probing the filesystem",
          "[workspace_details][tui][attachments]") {
    const TempDir sandbox{"filo_workspace_unrecorded"};
    write_file(sandbox.path / "notes.txt");

    // Roots composed outside SessionWorkspace carry no grant-time record: an
    // existing file is still an attachment, while an unavailable root stays a
    // workspace because it may be an unmounted directory.
    const core::workspace::WorkspaceSnapshot snapshot{
        .primary = sandbox.path / "project",
        .additional = {sandbox.path / "notes.txt", sandbox.path / "unmounted" / "repo"},
    };
    CHECK(tui::workspace_details_workspace_count(snapshot) == 2);
    CHECK(tui::workspace_details_entry_count(snapshot) == 3);
}
