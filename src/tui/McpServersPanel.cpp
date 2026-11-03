#include "McpServersPanel.hpp"

#include "KeyInput.hpp"
#include "TextLayout.hpp"
#include "TuiTheme.hpp"

#include <ftxui/dom/flexbox_config.hpp>
#include <algorithm>
#include <ctime>
#include <format>
#include <ranges>
#include <string>
#include <string_view>
#include <utility>

namespace tui {
namespace {

using namespace ftxui;
using core::mcp::McpServerState;
using core::mcp::McpServerStatus;
using Clock = std::chrono::system_clock;

constexpr int kLabelWidth = 11;
constexpr std::size_t kMaxListedTools = 60;

[[nodiscard]] std::string plural(std::uint64_t count, std::string_view noun) {
    return std::format("{} {}{}", count, noun, count == 1 ? "" : "s");
}

[[nodiscard]] std::string ago(Clock::time_point then, Clock::time_point now) {
    if (then == Clock::time_point{}) return {};
    const auto seconds = std::max<std::int64_t>(
        0, std::chrono::duration_cast<std::chrono::seconds>(now - then).count());
    if (seconds < 5) return "just now";
    if (seconds < 60) return std::format("{}s ago", seconds);
    if (seconds < 3600) return std::format("{}m ago", seconds / 60);
    if (seconds < 86400) return std::format("{}h {}m ago", seconds / 3600, seconds % 3600 / 60);
    return std::format("{}d ago", seconds / 86400);
}

[[nodiscard]] std::string clock_time(Clock::time_point when) {
    const auto tt = Clock::to_time_t(when);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    return std::format("{:02d}:{:02d}:{:02d}", tm.tm_hour, tm.tm_min, tm.tm_sec);
}

[[nodiscard]] std::string duration_label(std::chrono::milliseconds duration) {
    if (duration.count() < 1000) return std::format("{} ms", duration.count());
    return std::format("{:.1f} s", static_cast<double>(duration.count()) / 1000.0);
}

[[nodiscard]] std::string_view state_glyph(const McpServerStatus& server) {
    switch (server.state) {
    case McpServerState::connected: return server.activity.in_flight > 0 ? "◉" : "●";
    case McpServerState::connecting: return "◌";
    case McpServerState::failed: return "✗";
    }
    return "?";
}

[[nodiscard]] Color state_color(const McpServerStatus& server) {
    switch (server.state) {
    case McpServerState::connected:
        return server.activity.in_flight > 0 ? Color(ColorYellowBright) : Color(Color::Green);
    case McpServerState::connecting: return ColorYellowDark;
    case McpServerState::failed: return ColorToolFail;
    }
    return ColorTextMuted;
}

[[nodiscard]] std::string list_summary(const McpServerStatus& server, Clock::time_point now) {
    switch (server.state) {
    case McpServerState::connecting: return "connecting…";
    case McpServerState::failed: return server.needs_login ? "needs /mcp login" : "connection failed";
    case McpServerState::connected: break;
    }
    const auto& activity = server.activity;
    std::string summary = plural(server.tools.size(), "tool");
    if (activity.in_flight > 0) {
        return summary + " · running " + activity.last_tool;
    }
    if (activity.calls > 0) {
        summary += " · " + plural(activity.calls, "call");
        if (activity.failures > 0) summary += std::format(" · {} failed", activity.failures);
        summary += " · " + ago(activity.last_call_at, now);
    }
    return summary;
}

[[nodiscard]] Element field(std::string_view label, Element value) {
    return hbox({
        text(" " + fit_column(label, kLabelWidth - 1)) | color(ColorTextMuted),
        std::move(value) | xflex,
    });
}

[[nodiscard]] Element field(std::string_view label, std::string value, Color value_color = ColorText) {
    return field(label, paragraph(std::move(value)) | color(value_color));
}

[[nodiscard]] Element capability(std::string_view name, bool enabled) {
    return text(std::format("{} {}  ", enabled ? "✓" : "–", name))
        | color(enabled ? Color(ColorText) : Color(ColorTextMuted));
}

[[nodiscard]] Element render_server_row(const McpServerStatus& server,
                                        bool selected,
                                        Clock::time_point now) {
    const std::string kind = server.transport.empty() ? "?" : server.transport;
    auto row = hbox({
        text(selected ? " › " : "   ") | color(ColorYellowBright),
        text(std::string(state_glyph(server)) + " ") | color(state_color(server)) | bold,
        text(fit_column(server.name, 18))
            | color(selected ? Color(ColorYellowBright) : Color(ColorText)) | bold,
        text(" " + fit_column(kind, 6)) | color(ColorTextSecondary),
        elided_text(server.auth) | color(ColorTextMuted) | xflex,
        text("  " + list_summary(server, now) + " ")
            | color(server.state == McpServerState::failed ? Color(ColorToolFail)
                                                            : Color(ColorTextSecondary)),
    });
    if (selected) row = std::move(row) | focus;
    return row;
}

[[nodiscard]] Element render_tool_chips(const std::vector<std::string>& tools) {
    auto chips = tools | std::views::take(kMaxListedTools)
        | std::views::transform([](const std::string& tool) { return text(tool) | color(ColorQuestionCyan); })
        | std::ranges::to<Elements>();
    if (tools.size() > kMaxListedTools) {
        chips.push_back(text(std::format("+{} more", tools.size() - kMaxListedTools))
                        | color(ColorTextMuted));
    }
    return flexbox(std::move(chips), FlexboxConfig().SetGap(2, 0));
}

[[nodiscard]] Element render_server_details(const McpServerStatus& server, Clock::time_point now) {
    Elements rows;

    const auto& info = server.server_info;
    if (!info.name.empty() || !info.protocol_version.empty()) {
        std::string reported = info.name;
        if (!info.version.empty()) reported += (reported.empty() ? "v" : " v") + info.version;
        if (!info.protocol_version.empty()) {
            reported += (reported.empty() ? "" : " · ") + std::string("protocol ") + info.protocol_version;
        }
        rows.push_back(field("Server", std::move(reported)));
    }
    rows.push_back(field(server.transport == "stdio" ? "Command" : "Endpoint",
                         server.endpoint.empty() ? "—" : server.endpoint));
    rows.push_back(field("Auth", server.auth));

    switch (server.state) {
    case McpServerState::connecting:
        rows.push_back(field("Status", "connecting…", ColorYellowDark));
        break;
    case McpServerState::failed: {
        rows.push_back(field(
            "Status",
            std::format("failed after {}", duration_label(server.connect_duration)),
            ColorToolFail));
        rows.push_back(field("Error", server.error.empty() ? "unknown error" : server.error,
                             ColorToolFail));
        rows.push_back(field(
            "Next step",
            server.needs_login
                ? std::format("Run /mcp login {} — Filo reconnects when sign-in succeeds.", server.name)
                : std::string("Check the endpoint and credentials with /mcp list."),
            ColorYellowBright));
        break;
    }
    case McpServerState::connected: {
        rows.push_back(field(
            "Status",
            hbox({
                text("connected") | color(Color::Green) | bold,
                text(std::format(" since {} ({}) · handshake {}",
                                 clock_time(server.connected_at),
                                 ago(server.connected_at, now),
                                 duration_label(server.connect_duration)))
                    | color(ColorTextSecondary),
            })));
        rows.push_back(field("Supports", hbox({
            capability(plural(server.tools.size(), "tool"), !server.tools.empty()),
            capability("resources", server.resources),
            capability("prompts", server.prompts),
            capability("sampling", server.sampling),
        })));

        const auto& activity = server.activity;
        if (activity.calls == 0 && activity.in_flight == 0) {
            rows.push_back(field("Activity", "No tool calls yet this session.", ColorTextSecondary));
        } else {
            std::string totals = plural(activity.calls, "call");
            if (activity.failures > 0) totals += std::format(" · {} failed", activity.failures);
            Elements line{text(totals) | color(ColorText)};
            if (activity.in_flight > 0) {
                line.push_back(text(std::format("  ▶ running {}", activity.last_tool))
                                   | color(ColorYellowBright) | bold);
            }
            rows.push_back(field("Activity", hbox(std::move(line))));
            if (activity.calls > 0 && activity.in_flight == 0) {
                rows.push_back(field("Last call", hbox({
                    text(activity.last_failed ? "✗ " : "✓ ")
                        | color(activity.last_failed ? Color(ColorToolFail) : Color(Color::Green))
                        | bold,
                    text(activity.last_tool) | color(ColorQuestionCyan),
                    text(std::format(" · {} · {}", duration_label(activity.last_duration),
                                     ago(activity.last_call_at, now)))
                        | color(ColorTextSecondary),
                })));
                if (activity.last_failed && !activity.last_error.empty()) {
                    rows.push_back(field("", activity.last_error, ColorToolFail));
                }
            }
        }
        if (!server.tools.empty()) {
            rows.push_back(field("Tools", render_tool_chips(server.tools)));
        }
        break;
    }
    }

    return UiWindow(text(std::format(" {} ", server.name)) | color(ColorYellowDark) | bold,
                    vbox(std::move(rows)) | xflex);
}

[[nodiscard]] std::string panel_title(const std::vector<McpServerStatus>& servers) {
    const auto count = [&](McpServerState state) {
        return std::ranges::count(servers, state, &McpServerStatus::state);
    };
    const auto connecting = count(McpServerState::connecting);
    const auto failed = count(McpServerState::failed);
    std::string title = std::format(" MCP servers · {} connected", count(McpServerState::connected));
    if (connecting > 0) title += std::format(" · {} connecting", connecting);
    if (failed > 0) title += std::format(" · {} failed", failed);
    return title + " ";
}

} // namespace

bool handle_mcp_servers_panel_event(McpServersPanelState& state,
                                    ftxui::Event event,
                                    std::size_t server_count,
                                    const ftxui::Box& popover_box) {
    if (!state.active) return false;

    const auto last = server_count == 0 ? 0 : server_count - 1;
    state.selected = std::min(state.selected, last);
    const auto select_previous = [&] { state.selected -= std::min<std::size_t>(state.selected, 1); };
    const auto select_next = [&] { state.selected += std::min<std::size_t>(last - state.selected, 1); };

    if (event.is_mouse()) {
        auto& mouse = event.mouse();
        if (!popover_box.Contain(mouse.x, mouse.y)) {
            // A press elsewhere dismisses the popover and still lands where it
            // was aimed; the wheel keeps scrolling whatever is under it.
            if (mouse.button == Mouse::Left && mouse.motion == Mouse::Pressed) {
                state.active = false;
            }
            return false;
        }
        if (mouse.button == Mouse::WheelUp) select_previous();
        if (mouse.button == Mouse::WheelDown) select_next();
        return true;
    }

    if (is_panel_dismiss_event(event)) {
        state.active = false;
    } else if (event == Event::Home) {
        state.selected = 0;
    } else if (event == Event::End) {
        state.selected = last;
    } else if (event == Event::ArrowUp || event == Event::Character('k')) {
        select_previous();
    } else if (event == Event::ArrowDown || event == Event::Character('j')) {
        select_next();
    } else {
        // Preserve Ctrl+C and app events, but keep keystrokes out of the
        // prompt while the popover holds the keyboard.
        return event != Event::Custom && !is_ctrl_c_event(event);
    }
    return true;
}

Element render_mcp_servers_panel(const std::vector<McpServerStatus>& servers,
                                 std::size_t selected,
                                 Clock::time_point now,
                                 int max_visible_lines) {
    Elements body;
    if (servers.empty()) {
        body.push_back(text(" No MCP servers configured.") | color(ColorTextSecondary));
        body.push_back(text(""));
        body.push_back(paragraph(" Add one with /mcp add, e.g. "
                                 "/mcp add --global linear http https://mcp.linear.app/mcp")
                       | color(ColorTextMuted));
    } else {
        const auto active = std::min(selected, servers.size() - 1);
        Elements list;
        for (std::size_t i = 0; i < servers.size(); ++i) {
            list.push_back(render_server_row(servers[i], i == active, now));
        }
        body.push_back(vbox(std::move(list)));
        body.push_back(text(""));
        body.push_back(render_server_details(servers[active], now));
    }

    const std::string hint = servers.size() > 1
        ? "↑/↓/Wheel: select server · Esc, q, or click outside to close"
        : "Esc, q, or click outside to close";
    return UiWindow(
        text(panel_title(servers)) | color(ColorYellowBright) | bold,
        vbox({
            vbox(std::move(body)) | vscroll_indicator | yframe
                | size(HEIGHT, LESS_THAN, std::max(1, max_visible_lines)),
            text(""),
            paragraph(hint) | color(ColorTextMuted) | dim,
        }) | xflex);
}

} // namespace tui
