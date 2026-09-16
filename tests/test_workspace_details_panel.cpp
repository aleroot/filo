#include "tui/WorkspaceDetailsPanel.hpp"

#include <catch2/catch_test_macros.hpp>
#include <ftxui/screen/screen.hpp>
#include <string>

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
