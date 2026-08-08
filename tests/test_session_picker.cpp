#include <catch2/catch_test_macros.hpp>

#include "tui/SessionPicker.hpp"

#include <ftxui/component/event.hpp>
#include <string>
#include <vector>

using tui::SessionPickerAction;
using tui::SessionPickerMode;
using tui::SessionPickerResource;
using tui::SessionPickerState;

namespace {

core::session::SessionInfo make_session(std::string_view id, std::string_view name) {
    core::session::SessionInfo info;
    info.session_id = std::string{id};
    info.name = std::string{name};
    info.last_active_at = "2026-03-22T12:00:00Z";
    info.preview = std::string{name};
    return info;
}

} // namespace

TEST_CASE("SessionPicker opens focused on the current session", "[tui][session_picker]") {
    SessionPickerState state;
    std::vector<core::session::SessionInfo> sessions{
        make_session("aaa11111", "first"),
        make_session("bbb22222", "current"),
        make_session("ccc33333", "third"),
    };
    tui::open_session_picker(state, std::move(sessions), "bbb22222");

    REQUIRE(state.active);
    REQUIRE(state.filtered.size() == 3);
    CHECK(state.selected == 1);
    CHECK(state.current_session_id == "bbb22222");
    CHECK(state.resource == SessionPickerResource::SavedSessions);
}

TEST_CASE("SessionPicker supports filter mode", "[tui][session_picker]") {
    SessionPickerState state;
    tui::open_session_picker(
        state,
        {make_session("aaa11111", "oauth"), make_session("bbb22222", "flaky")},
        "aaa11111");

    auto enter_filter = tui::handle_session_picker_event(
        state, ftxui::Event::Character('/'));
    CHECK(enter_filter.handled);
    CHECK(state.mode == SessionPickerMode::Filter);

    auto type_o = tui::handle_session_picker_event(
        state, ftxui::Event::Character('o'));
    CHECK(type_o.handled);
    REQUIRE(state.filtered.size() == 1);
    CHECK(state.filtered[0].session_id == "aaa11111");
}

TEST_CASE("SessionPicker browse actions", "[tui][session_picker]") {
    SessionPickerState state;
    tui::open_session_picker(
        state,
        {make_session("aaa11111", "one"), make_session("bbb22222", "two")},
        "aaa11111");

    auto open = tui::handle_session_picker_event(state, ftxui::Event::Return);
    CHECK(open.action == SessionPickerAction::Open);
    CHECK(open.filtered_index == 0);
    CHECK_FALSE(state.active);

    tui::open_session_picker(
        state,
        {make_session("aaa11111", "one"), make_session("bbb22222", "two")},
        "aaa11111");
    auto del = tui::handle_session_picker_event(
        state, ftxui::Event::Delete);
    CHECK(del.action == SessionPickerAction::Delete);
    CHECK(del.filtered_index == 0);
    CHECK(state.active); // delete does not auto-close

    auto no_new = tui::handle_session_picker_event(
        state, ftxui::Event::Character('n'));
    CHECK(no_new.action == SessionPickerAction::None);
    CHECK(state.active);

    auto no_rename = tui::handle_session_picker_event(
        state, ftxui::Event::Character('r'));
    CHECK(no_rename.action == SessionPickerAction::None);
    CHECK(state.mode == SessionPickerMode::Browse);
}

TEST_CASE("Thread picker offers new, rename, and close but never session deletion",
          "[tui][session_picker][threads]") {
    SessionPickerState state;
    tui::open_thread_picker(
        state,
        {make_session("aaa11111", "one"), make_session("bbb22222", "two")},
        "aaa11111");
    CHECK(state.resource == SessionPickerResource::ActiveThreads);

    auto no_delete = tui::handle_session_picker_event(
        state, ftxui::Event::Delete);
    CHECK(no_delete.action == SessionPickerAction::None);

    auto close = tui::handle_session_picker_event(
        state, ftxui::Event::Character('c'));
    CHECK(close.action == SessionPickerAction::CloseThread);
    CHECK(close.filtered_index == 0);
    CHECK(state.active);

    auto rename_start = tui::handle_session_picker_event(
        state, ftxui::Event::Character('r'));
    CHECK(rename_start.handled);
    CHECK(state.mode == SessionPickerMode::Rename);

    state.rename_buffer.clear();
    static_cast<void>(tui::handle_session_picker_event(
        state, ftxui::Event::Character('x')));
    auto rename_commit = tui::handle_session_picker_event(state, ftxui::Event::Return);
    CHECK(rename_commit.action == SessionPickerAction::RenameCommit);
    CHECK(rename_commit.rename_name == "x");

    tui::open_thread_picker(
        state,
        {make_session("aaa11111", "one"), make_session("bbb22222", "two")},
        "aaa11111");
    auto neu = tui::handle_session_picker_event(state, ftxui::Event::Character('n'));
    CHECK(neu.action == SessionPickerAction::NewThread);
    CHECK_FALSE(state.active);
}

TEST_CASE("Thread picker opens even when catalogue is empty", "[tui][session_picker]") {
    SessionPickerState state;
    tui::open_thread_picker(state, {}, "deadbeef");
    REQUIRE(state.active);
    CHECK(state.filtered.empty());

    auto neu = tui::handle_session_picker_event(state, ftxui::Event::Character('N'));
    CHECK(neu.action == SessionPickerAction::NewThread);
}
