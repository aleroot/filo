#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <format>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>

#include "core/mcp/RemoteActivity.hpp"
#include "tui/RemoteActivityPanel.hpp"
#include "tui/TuiTheme.hpp"

using core::mcp::RemoteActivityHub;
using core::mcp::RemoteServerState;
using core::mcp::RemoteToolStatus;

namespace {

std::string strip_ansi(std::string_view input) {
    std::string output;
    output.reserve(input.size());
    for (std::size_t i = 0; i < input.size();) {
        if (input[i] == '\x1b' && i + 1 < input.size() && input[i + 1] == '[') {
            i += 2;
            while (i < input.size()) {
                const char ch = input[i++];
                if (ch >= '@' && ch <= '~') break;
            }
            continue;
        }
        output.push_back(input[i++]);
    }
    return output;
}

std::string_view line_containing(std::string_view input,
                                 std::string_view needle) {
    const std::size_t match = input.find(needle);
    if (match == std::string_view::npos) return {};
    const std::size_t line_start = input.rfind('\n', match);
    const std::size_t line_end = input.find('\n', match);
    const std::size_t start = line_start == std::string_view::npos
        ? 0
        : line_start + 1;
    return input.substr(
        start,
        line_end == std::string_view::npos
            ? std::string_view::npos
            : line_end - start);
}

} // namespace

TEST_CASE("remote MCP client names are safe and have a stable fallback",
          "[mcp][remote-activity]") {
    CHECK(core::mcp::sanitize_remote_client_name("") == "Client");
    CHECK(core::mcp::sanitize_remote_client_name("  Lampo\n\t1.0  ") == "Lampo 1.0");
    CHECK(core::mcp::sanitize_remote_client_name("\x01\x02") == "Client");
}

TEST_CASE("remote activity hub tracks client identity and tool lifecycle",
          "[mcp][remote-activity]") {
    auto& hub = RemoteActivityHub::get_instance();
    hub.reset_for_testing();
    hub.server_starting();
    hub.server_listening("127.0.0.1:8080");
    hub.client_initialized("session-a", "Lampo", "1.4.0");
    hub.client_ready("session-a");

    const auto activity_id = hub.tool_started(
        "session-a",
        "read_file",
        R"({"path":"/tmp/example"})");

    auto snapshot = hub.snapshot();
    REQUIRE(snapshot.server_state == RemoteServerState::listening);
    REQUIRE(snapshot.clients.size() == 1);
    CHECK(snapshot.clients.front().name == "Lampo");
    CHECK(snapshot.clients.front().version == "1.4.0");
    CHECK(snapshot.clients.front().ready);
    REQUIRE(snapshot.activities.size() == 1);
    CHECK(snapshot.activities.front().status == RemoteToolStatus::running);

    hub.tool_finished(activity_id, R"({"content":"ok"})", false);
    snapshot = hub.snapshot();
    CHECK(snapshot.activities.front().status == RemoteToolStatus::succeeded);
    CHECK(snapshot.unacknowledged_errors == 0);

    const auto failed_id = hub.tool_started("session-a", "write_file", "{}");
    hub.tool_finished(failed_id, R"({"error":"denied"})", true);
    snapshot = hub.snapshot();
    CHECK(snapshot.activities.front().status == RemoteToolStatus::failed);
    CHECK(snapshot.unacknowledged_errors == 1);

    hub.acknowledge_errors();
    CHECK(hub.snapshot().unacknowledged_errors == 0);
    hub.clear_completed();
    CHECK(hub.snapshot().activities.empty());
}

TEST_CASE("stateless MCP requests refresh identity without reset",
          "[mcp][remote-activity]") {
    auto& hub = RemoteActivityHub::get_instance();
    hub.reset_for_testing();
    hub.server_starting();
    hub.server_listening("127.0.0.1:8080");

    hub.client_identified("", "Lampo", "1.0");
    const auto activity_id = hub.tool_started("", "read_file", "{}");
    hub.tool_finished(activity_id, R"({"content":"ok"})", false);

    auto snapshot = hub.snapshot();
    REQUIRE(snapshot.clients.size() == 1);
    CHECK(snapshot.clients.front().name == "Lampo");
    CHECK(snapshot.clients.front().version == "1.0");
    CHECK(snapshot.clients.front().ready);

    hub.client_identified("", "", "");
    snapshot = hub.snapshot();
    CHECK(snapshot.clients.front().name == "Client");
    CHECK(snapshot.clients.front().ready);
}

TEST_CASE("remote footer uses protocol client name and generic fallback",
          "[tui][mcp][remote-activity]") {
    CHECK(tui::format_remote_footer_status({}).label.empty());

    auto& hub = RemoteActivityHub::get_instance();
    hub.reset_for_testing();
    hub.server_starting();
    hub.server_listening("127.0.0.1:8080");
    hub.client_initialized("known", "Lampo", "2.0");
    hub.client_ready("known");

    auto status = tui::format_remote_footer_status(hub.snapshot());
    CHECK(status.label.find("Lampo") != std::string::npos);
    CHECK(status.label.find("ready") != std::string::npos);

    hub.server_starting();
    hub.server_listening("127.0.0.1:8080");
    hub.client_initialized("unknown", "", "");
    hub.client_ready("unknown");

    status = tui::format_remote_footer_status(hub.snapshot());
    CHECK(status.label.find("Client") != std::string::npos);
    CHECK(status.label.find("Lampo") == std::string::npos);
}

TEST_CASE("remote footer humanizes long elapsed times",
          "[tui][mcp][remote-activity]") {
    using namespace std::chrono;

    const auto now = steady_clock::time_point{hours{1000}};
    core::mcp::RemoteActivitySnapshot snapshot;
    snapshot.server_state = RemoteServerState::listening;
    snapshot.clients.push_back({
        .session_id = "session-a",
        .name = "Lampo",
        .ready = true,
    });

    const auto footer_after = [&](steady_clock::duration elapsed) {
        snapshot.clients.front().last_seen = now - elapsed;
        return tui::format_remote_footer_status(snapshot, now).label;
    };

    CHECK(footer_after(minutes{59} + seconds{59})
          == "Lampo · seen 59m 59s ago");
    CHECK(footer_after(hours{1}) == "Lampo · seen 1h ago");
    CHECK(footer_after(minutes{253} + seconds{49})
          == "Lampo · seen 4h 13m ago");
    CHECK(footer_after(hours{49}) == "Lampo · seen 2d 1h ago");
}

TEST_CASE("remote footer pill has no embedded gap and uses foreground color only",
          "[tui][mcp][remote-activity][rendering]") {
    const tui::RemoteFooterStatus status{
        .label = "Lampo · seen 6m 16s ago",
        .tone = tui::RemoteFooterTone::neutral,
    };
    // The pill is hosted in its own centered status-bar slot, so any embedded
    // padding would visibly offset it from the middle of the bar.
    auto screen = ftxui::Screen::Create(
        ftxui::Dimension::Fixed(40),
        ftxui::Dimension::Fixed(1));
    ftxui::Render(screen, tui::render_remote_footer_status(status));

    const std::string rendered = strip_ansi(screen.ToString());
    CHECK(rendered.starts_with("⚡ Lampo · seen 6m 16s ago "));
    CHECK(screen.CellAt(0, 0).background_color == ftxui::Color::Default);
    CHECK(screen.CellAt(4, 0).background_color == ftxui::Color::Default);

    using ToneColor = std::pair<tui::RemoteFooterTone, ftxui::Color>;
    const std::array<ToneColor, 5> tone_colors{{
        {tui::RemoteFooterTone::neutral, ftxui::Color::GrayLight},
        {tui::RemoteFooterTone::ready, ftxui::Color::Green},
        {tui::RemoteFooterTone::running,
         static_cast<ftxui::Color>(tui::ColorYellowBright)},
        {tui::RemoteFooterTone::success, ftxui::Color::Green},
        {tui::RemoteFooterTone::error,
         static_cast<ftxui::Color>(tui::ColorToolFail)},
    }};
    for (const auto& [tone, expected_color] : tone_colors) {
        auto tone_screen = ftxui::Screen::Create(
            ftxui::Dimension::Fixed(24),
            ftxui::Dimension::Fixed(1));
        ftxui::Render(
            tone_screen,
            tui::render_remote_footer_status({.label = "MCP", .tone = tone}));

        CHECK(tone_screen.CellAt(0, 0).foreground_color == expected_color);
        for (int x = 0; x < tone_screen.dimx(); ++x) {
            CHECK(tone_screen.CellAt(x, 0).background_color
                  == ftxui::Color::Default);
        }
    }
}

TEST_CASE("remote activity summaries use the available terminal width",
          "[tui][mcp][remote-activity][rendering]") {
    auto& hub = RemoteActivityHub::get_instance();
    hub.reset_for_testing();
    hub.server_starting();
    hub.server_listening("127.0.0.1:8080");
    hub.client_initialized("session-a", "Lampo", "1.2");
    hub.client_ready("session-a");

    constexpr std::string_view kWorkingDir =
        "/Users/alessiopollero/Develop/Projects/coreconnect-modulith";
    const auto activity_id = hub.tool_started(
        "session-a",
        "run_terminal_command",
        R"({"command":"git status --short","working_dir":"/Users/alessiopollero/Develop/Projects/coreconnect-modulith"})");
    REQUIRE(activity_id != 0);

    auto wide_screen = ftxui::Screen::Create(
        ftxui::Dimension::Fixed(180),
        ftxui::Dimension::Fixed(22));
    ftxui::Render(
        wide_screen,
        tui::render_remote_activity_panel(hub.snapshot(), 0));
    const std::string wide = strip_ansi(wide_screen.ToString());
    const std::string_view wide_row = line_containing(wide, "terminal");
    REQUIRE_FALSE(wide_row.empty());
    CHECK(wide_row.find(kWorkingDir) != std::string_view::npos);
    CHECK(wide_row.find("cmd: git status --short") != std::string_view::npos);
    CHECK(wide_row.find("Lampo") != std::string_view::npos);

    auto narrow_screen = ftxui::Screen::Create(
        ftxui::Dimension::Fixed(72),
        ftxui::Dimension::Fixed(22));
    ftxui::Render(
        narrow_screen,
        tui::render_remote_activity_panel(hub.snapshot(), 0));
    const std::string narrow = strip_ansi(narrow_screen.ToString());
    const std::string_view narrow_row = line_containing(narrow, "terminal");
    REQUIRE_FALSE(narrow_row.empty());
    CHECK(narrow_row.find("cwd: /Users/alessiopollero")
          != std::string_view::npos);
    CHECK(narrow_row.find("…") != std::string_view::npos);
    CHECK(narrow_row.find("Lampo") != std::string_view::npos);

    hub.reset_for_testing();
}

TEST_CASE("long remote activity histories expose and scroll their viewport",
          "[tui][mcp][remote-activity][rendering]") {
    auto& hub = RemoteActivityHub::get_instance();
    hub.reset_for_testing();
    hub.server_starting();
    hub.server_listening("127.0.0.1:8080");
    hub.client_initialized("session-a", "Lampo", "1.2");
    hub.client_ready("session-a");

    for (int i = 1; i <= 12; ++i) {
        const auto activity_id = hub.tool_started(
            "session-a",
            "run_terminal_command",
            std::format(R"({{"command":"command-{}"}})", i));
        hub.tool_finished(activity_id, R"({"output":"ok"})", false);
    }
    const auto snapshot = hub.snapshot();

    const auto render_at = [&](std::size_t selected) {
        auto screen = ftxui::Screen::Create(
            ftxui::Dimension::Fixed(100),
            ftxui::Dimension::Fixed(28));
        ftxui::Render(
            screen,
            tui::render_remote_activity_panel(snapshot, selected));
        return strip_ansi(screen.ToString());
    };

    const std::string newest = render_at(0);
    CHECK(newest.find("command-12") != std::string::npos);
    CHECK(newest.find("command-5") != std::string::npos);
    CHECK(newest.find("command-4") == std::string::npos);
    CHECK(newest.find("↓ 4 older activities") != std::string::npos);
    CHECK(newest.find("newer activit") == std::string::npos);
    CHECK(newest.find("1 of 12 · newest first") != std::string::npos);

    const std::string middle = render_at(6);
    CHECK(middle.find("↑ 3 newer activities") != std::string::npos);
    CHECK(middle.find("↓ 1 older activity") != std::string::npos);
    CHECK(middle.find("7 of 12 · newest first") != std::string::npos);

    const std::string oldest = render_at(11);
    CHECK(oldest.find("↑ 4 newer activities") != std::string::npos);
    CHECK(oldest.find("older activit") == std::string::npos);
    CHECK(oldest.find("command-1") != std::string::npos);
    CHECK(oldest.find("12 of 12 · newest first") != std::string::npos);

    hub.reset_for_testing();
}

TEST_CASE("remote activity history stays bounded even when nothing finishes",
          "[mcp][remote-activity]") {
    auto& hub = RemoteActivityHub::get_instance();
    hub.reset_for_testing();
    hub.server_starting();
    hub.server_listening("127.0.0.1:8080");

    // More running entries than the retention cap: a stuck producer must not
    // grow the history without limit.
    constexpr std::size_t kFlood = 150;
    for (std::size_t i = 0; i < kFlood; ++i) {
        const std::uint64_t id = hub.tool_started("session-a", "read_file", "{}");
        CHECK(id != 0);
    }
    const auto snapshot = hub.snapshot(false);
    CHECK(snapshot.activities.size() == 100);
    CHECK(std::ranges::all_of(snapshot.activities,
                              [](const core::mcp::RemoteToolActivity& activity) {
                                  return activity.status
                                      == RemoteToolStatus::running;
                              }));
    // Late completions for evicted ids are dropped silently.
    hub.tool_finished(1, "{}", false);
    CHECK(hub.snapshot(false).activities.size() == 100);
}

TEST_CASE("remote activity payloads carry a single truncation marker",
          "[mcp][remote-activity]") {
    auto& hub = RemoteActivityHub::get_instance();
    hub.reset_for_testing();
    hub.server_starting();
    hub.server_listening("127.0.0.1:8080");

    const std::string oversized(40 * 1024, 'x');
    const auto activity_id = hub.tool_started("session-a", "read_file", "{}");
    hub.tool_finished(activity_id, oversized, false);

    const auto snapshot = hub.snapshot();
    REQUIRE(snapshot.activities.size() == 1);
    const std::string& result = snapshot.activities.front().result;
    CHECK(result.size() < oversized.size());
    CHECK(result.ends_with("[truncated]"));
    CHECK(result.find("…") == std::string::npos);
}

TEST_CASE("evicting a failed activity does not strand the error badge",
          "[mcp][remote-activity]") {
    auto& hub = RemoteActivityHub::get_instance();
    hub.reset_for_testing();
    hub.server_starting();
    hub.server_listening("127.0.0.1:8080");

    const auto failed_id = hub.tool_started("session-a", "write_file", "{}");
    hub.tool_finished(failed_id, R"({"error":"denied"})", true);
    REQUIRE(hub.snapshot(false).unacknowledged_errors == 1);

    // Push the failure out of the retention window; the badge must follow the
    // history rather than keep counting an entry the user can no longer reach.
    for (std::size_t i = 0; i < 120; ++i) {
        const auto id = hub.tool_started("session-a", "read_file", "{}");
        hub.tool_finished(id, "{}", false);
    }

    const auto snapshot = hub.snapshot(false);
    CHECK(std::ranges::none_of(
        snapshot.activities,
        [](const core::mcp::RemoteToolActivity& activity) {
            return activity.status == RemoteToolStatus::failed;
        }));
    CHECK(snapshot.unacknowledged_errors == 0);
}

TEST_CASE("acknowledging errors only clears failures already recorded",
          "[mcp][remote-activity]") {
    auto& hub = RemoteActivityHub::get_instance();
    hub.reset_for_testing();
    hub.server_starting();
    hub.server_listening("127.0.0.1:8080");

    const auto first = hub.tool_started("session-a", "write_file", "{}");
    hub.tool_finished(first, R"({"error":"denied"})", true);
    hub.acknowledge_errors();
    REQUIRE(hub.snapshot(false).unacknowledged_errors == 0);

    // A failure arriving after the acknowledgement must light the badge again.
    const auto second = hub.tool_started("session-a", "write_file", "{}");
    hub.tool_finished(second, R"({"error":"denied"})", true);
    CHECK(hub.snapshot(false).unacknowledged_errors == 1);
}

TEST_CASE("client retention is bounded and prefers evicting closed sessions",
          "[mcp][remote-activity]") {
    auto& hub = RemoteActivityHub::get_instance();
    hub.reset_for_testing();
    hub.server_starting();
    hub.server_listening("127.0.0.1:8080");

    // MCP sessions only end on an explicit DELETE that clients may never send,
    // so a long-lived daemon must not accumulate records without limit.
    constexpr std::size_t kSessions = 500;
    for (std::size_t i = 0; i < kSessions; ++i) {
        const auto session = std::format("session-{}", i);
        hub.client_initialized(session, "Lampo", "1.0");
        hub.client_closed(session);
    }
    CHECK(hub.client_count() <= 64);

    // An open session must survive a flood of closed ones.
    hub.client_initialized("keeper", "Keeper", "1.0");
    hub.client_ready("keeper");
    for (std::size_t i = 0; i < kSessions; ++i) {
        const auto session = std::format("closed-{}", i);
        hub.client_initialized(session, "Lampo", "1.0");
        hub.client_closed(session);
    }
    CHECK(hub.client_count() <= 64);

    const auto snapshot = hub.snapshot(false);
    const auto keeper = std::ranges::find(
        snapshot.clients, std::string{"keeper"},
        &core::mcp::RemoteClientActivity::session_id);
    REQUIRE(keeper != snapshot.clients.end());
    CHECK(keeper->ready);
    CHECK_FALSE(keeper->closed);
}

TEST_CASE("detaching the notify callback waits for in-flight notifications",
          "[mcp][remote-activity]") {
    auto& hub = RemoteActivityHub::get_instance();
    hub.reset_for_testing();
    hub.server_starting();
    hub.server_listening("127.0.0.1:8080");

    std::atomic_bool inside_callback{false};
    std::atomic_bool callback_finished{false};
    std::atomic_bool release{false};
    std::atomic_int callbacks{0};

    hub.set_notify_callback([&] {
        callbacks.fetch_add(1, std::memory_order_relaxed);
        inside_callback.store(true, std::memory_order_release);
        while (!release.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        // Re-entering the hub from the callback must not deadlock against a
        // concurrent detach.
        (void)hub.snapshot(false);
        callback_finished.store(true, std::memory_order_release);
    });

    std::thread producer([&] { hub.client_seen("session-a"); });
    while (!inside_callback.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    std::atomic_bool detach_returned{false};
    std::thread detacher([&] {
        hub.set_notify_callback({});
        detach_returned.store(true, std::memory_order_release);
    });

    // The detach must block while the callback is still running: this is what
    // makes it safe for the TUI to destroy the state the callback captured.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK_FALSE(detach_returned.load(std::memory_order_acquire));

    release.store(true, std::memory_order_release);
    detacher.join();
    producer.join();

    CHECK(detach_returned.load(std::memory_order_acquire));
    CHECK(callback_finished.load(std::memory_order_acquire));

    // Once detached, no further notification may be delivered.
    const int observed = callbacks.load(std::memory_order_relaxed);
    (void)hub.tool_started("session-a", "read_file", "{}");
    CHECK(callbacks.load(std::memory_order_relaxed) == observed);

    hub.reset_for_testing();
}

TEST_CASE("repeated liveness touches do not wake the UI once per request",
          "[mcp][remote-activity]") {
    auto& hub = RemoteActivityHub::get_instance();
    hub.reset_for_testing();
    hub.server_starting();
    hub.server_listening("127.0.0.1:8080");
    hub.client_initialized("session-a", "Lampo", "1.0");

    std::atomic_int wakes{0};
    hub.set_notify_callback([&] { wakes.fetch_add(1, std::memory_order_relaxed); });

    for (std::size_t i = 0; i < 1000; ++i) {
        hub.client_seen("session-a");
    }
    // The session is already open and was just seen, so these are coalesced.
    CHECK(wakes.load(std::memory_order_relaxed) == 0);

    // A genuine state change still refreshes immediately.
    hub.client_closed("session-a");
    hub.client_seen("session-a");
    CHECK(wakes.load(std::memory_order_relaxed) >= 2);

    hub.reset_for_testing();
}
