#include "RemoteActivityPanel.hpp"

#include "Conversation.hpp"
#include "Text.hpp"
#include "TuiTheme.hpp"
#include "core/tools/ToolNames.hpp"

#include <algorithm>
#include <format>
#include <string_view>
#include <unordered_set>

#include <ftxui/dom/elements.hpp>

namespace tui {

namespace {

using core::mcp::RemoteActivitySnapshot;
using core::mcp::RemoteClientActivity;
using core::mcp::RemoteServerState;
using core::mcp::RemoteToolActivity;
using core::mcp::RemoteToolStatus;
using namespace ftxui;

/// A client that has not sent any request within this window is presented as
/// idle: MCP has no liveness probe beyond the optional ping, and sessions
/// only close via an explicit HTTP DELETE, so silence is the only signal.
constexpr std::chrono::minutes kClientStaleAfter{1};

[[nodiscard]] bool client_is_stale(const RemoteClientActivity& client,
                                   std::chrono::steady_clock::time_point now) {
    return !client.closed && now - client.last_seen > kClientStaleAfter;
}

[[nodiscard]] const RemoteClientActivity* client_for(
    const RemoteActivitySnapshot& snapshot,
    std::string_view session_id) {
    const auto it = std::ranges::find(snapshot.clients, session_id,
                                      &RemoteClientActivity::session_id);
    return it == snapshot.clients.end() ? nullptr : &*it;
}

[[nodiscard]] std::string client_name_for(const RemoteActivitySnapshot& snapshot,
                                          std::string_view session_id) {
    if (const auto* client = client_for(snapshot, session_id)) {
        return client->name.empty() ? "Client" : client->name;
    }
    return "Client";
}

[[nodiscard]] std::string format_duration(
    std::chrono::steady_clock::duration duration) {
    using namespace std::chrono;
    if (duration < seconds{1}) {
        return std::format("{}ms", std::max<int64_t>(
            duration_cast<milliseconds>(duration).count(), 0));
    }
    const auto total_seconds = duration_cast<seconds>(duration).count();
    if (total_seconds < 60) return std::format("{}s", total_seconds);
    return std::format("{}m {:02}s", total_seconds / 60, total_seconds % 60);
}

[[nodiscard]] std::string elapsed_label(
    std::chrono::steady_clock::time_point timestamp,
    std::chrono::steady_clock::time_point now) {
    if (timestamp == std::chrono::steady_clock::time_point{}) return "never";
    const auto elapsed = std::max(now - timestamp,
                                  std::chrono::steady_clock::duration::zero());
    if (elapsed < std::chrono::seconds{2}) return "now";
    return format_duration(elapsed) + " ago";
}

[[nodiscard]] std::string compact_tool_name(std::string_view tool_name) {
    if (core::tools::names::is_terminal_tool(tool_name)) return "terminal";
    std::string label(tool_name);
    std::ranges::replace(label, '_', ' ');
    return label;
}

[[nodiscard]] std::string activity_summary(const RemoteToolActivity& activity) {
    const std::string summary = summarize_tool_arguments(
        activity.tool_name, activity.arguments);
    if (!summary.empty()) return compact_single_line(summary, 46);
    return compact_tool_name(activity.tool_name);
}

[[nodiscard]] Color status_color(RemoteToolStatus status) {
    switch (status) {
        case RemoteToolStatus::running:
            return ColorYellowBright;
        case RemoteToolStatus::succeeded:
            return Color::Green;
        case RemoteToolStatus::failed:
            return ColorToolFail;
        case RemoteToolStatus::cancelled:
            return Color::GrayLight;
    }
    return Color::GrayLight;
}

[[nodiscard]] std::string_view status_icon(RemoteToolStatus status) {
    switch (status) {
        case RemoteToolStatus::running:
            return "◐";
        case RemoteToolStatus::succeeded:
            return "✓";
        case RemoteToolStatus::failed:
            return "!";
        case RemoteToolStatus::cancelled:
            return "○";
    }
    return " ";
}

[[nodiscard]] std::string activity_duration(
    const RemoteToolActivity& activity,
    std::chrono::steady_clock::time_point now) {
    const auto end = activity.status == RemoteToolStatus::running
        ? now
        : activity.finished_at;
    return format_duration(std::max(
        end - activity.started_at,
        std::chrono::steady_clock::duration::zero()));
}

[[nodiscard]] std::string result_summary(const RemoteToolActivity& activity) {
    if (activity.result.empty()) return {};
    auto tool = make_tool_activity(
        std::to_string(activity.id),
        activity.tool_name,
        activity.arguments,
        {},
        false);
    apply_tool_result(tool, activity.result);
    return tool.result.summary.empty()
        ? compact_single_line(activity.result, 240)
        : tool.result.summary;
}

} // namespace

RemoteFooterStatus format_remote_footer_status(
    const RemoteActivitySnapshot& snapshot,
    std::chrono::steady_clock::time_point now) {
    constexpr std::string_view icon = "⚡";

    switch (snapshot.server_state) {
        case RemoteServerState::disabled:
            return {};
        case RemoteServerState::starting:
            return {std::format("{} MCP · starting", icon),
                    RemoteFooterTone::running,
                    true};
        case RemoteServerState::failed:
            return {std::format("{} MCP · unavailable", icon),
                    RemoteFooterTone::error,
                    false};
        case RemoteServerState::stopped:
            return {std::format("{} MCP · stopped", icon),
                    RemoteFooterTone::error,
                    false};
        case RemoteServerState::listening:
            break;
    }

    std::vector<const RemoteToolActivity*> running;
    for (const auto& activity : snapshot.activities) {
        if (activity.status == RemoteToolStatus::running) {
            running.push_back(&activity);
        }
    }

    auto active_client_label = [&]() -> std::string {
        if (!running.empty()) {
            std::unordered_set<std::string_view> sessions;
            for (const auto* activity : running) {
                sessions.insert(activity->session_id);
            }
            if (sessions.size() > 1) {
                return std::format("{} clients", sessions.size());
            }
            const std::string first =
                client_name_for(snapshot, running.front()->session_id);
            const bool same_client = std::ranges::all_of(running, [&](const auto* activity) {
                return client_name_for(snapshot, activity->session_id) == first;
            });
            if (same_client) return first;
        }

        std::size_t open_clients = 0;
        const RemoteClientActivity* latest = nullptr;
        for (const auto& client : snapshot.clients) {
            if (client.closed) continue;
            ++open_clients;
            if (latest == nullptr || client.last_seen > latest->last_seen) {
                latest = &client;
            }
        }
        if (open_clients > 1) return std::format("{} clients", open_clients);
        return latest == nullptr ? "MCP" : latest->name;
    };

    const std::string client_label = active_client_label();
    if (snapshot.unacknowledged_errors > 0) {
        return {
            std::format(
                "{} {} · {} issue{}",
                icon,
                client_label,
                snapshot.unacknowledged_errors,
                snapshot.unacknowledged_errors == 1 ? "" : "s"),
            RemoteFooterTone::error,
            false,
        };
    }

    if (!running.empty()) {
        if (running.size() == 1) {
            return {
                std::format(
                    "{} {} · {} · {}",
                    icon,
                    client_label,
                    compact_tool_name(running.front()->tool_name),
                    activity_duration(*running.front(), now)),
                RemoteFooterTone::running,
                true,
            };
        }
        return {
            std::format("{} {} · {} running", icon, client_label, running.size()),
            RemoteFooterTone::running,
            true,
        };
    }

    if (!snapshot.activities.empty()) {
        const auto& latest = snapshot.activities.front();
        if (latest.finished_at != std::chrono::steady_clock::time_point{}
            && now - latest.finished_at <= std::chrono::seconds{2}) {
            return {
                std::format(
                    "{} {} · {} · {}",
                    icon,
                    client_name_for(snapshot, latest.session_id),
                    compact_tool_name(latest.tool_name),
                    activity_duration(latest, now)),
                latest.status == RemoteToolStatus::succeeded
                    ? RemoteFooterTone::success
                    : RemoteFooterTone::neutral,
                false,
            };
        }
    }

    const RemoteClientActivity* latest_open = nullptr;
    for (const auto& client : snapshot.clients) {
        if (!client.closed
            && (latest_open == nullptr || client.last_seen > latest_open->last_seen)) {
            latest_open = &client;
        }
    }
    if (latest_open == nullptr) {
        return {std::format("{} MCP · waiting", icon),
                RemoteFooterTone::neutral,
                false};
    }

    const auto since_seen = now - latest_open->last_seen;
    if (since_seen > std::chrono::seconds{30}) {
        return {
            std::format(
                "{} {} · seen {}",
                icon,
                latest_open->name,
                elapsed_label(latest_open->last_seen, now)),
            RemoteFooterTone::neutral,
            false,
        };
    }
    return {
        std::format(
            "{} {} · {}",
            icon,
            latest_open->name,
            latest_open->ready ? "ready" : "connecting"),
        latest_open->ready ? RemoteFooterTone::ready : RemoteFooterTone::neutral,
        false,
    };
}

Element render_remote_activity_panel(const RemoteActivitySnapshot& snapshot,
                                     std::size_t selected_activity) {
    const auto now = std::chrono::steady_clock::now();
    Elements rows;

    if (snapshot.server_state == RemoteServerState::failed) {
        rows.push_back(
            paragraph(snapshot.server_detail.empty()
                          ? "The MCP server could not be started."
                          : snapshot.server_detail)
            | color(ColorToolFail));
        rows.push_back(text(""));
    }

    if (snapshot.clients.empty()) {
        rows.push_back(
            text("  Waiting for an MCP client to initialize…")
            | color(Color::GrayLight)
            | dim);
    } else {
        constexpr std::size_t kVisibleClients = 4;
        const std::size_t visible_clients =
            std::min(snapshot.clients.size(), kVisibleClients);
        for (std::size_t i = 0; i < visible_clients; ++i) {
            const auto& client = snapshot.clients[i];
            const std::string version =
                client.version.empty() ? std::string{} : " " + client.version;
            const bool stale = client_is_stale(client, now);
            const std::string state = client.closed
                ? "closed"
                : stale ? "idle"
                : client.ready ? "ready" : "connecting";
            rows.push_back(
                hbox({
                    text(client.closed ? "  ○ " : stale ? "  ◌ " : "  ● ")
                        | color(client.closed ? Color::GrayDark
                                : stale ? Color::GrayLight : Color::Green),
                    text(client.name + version) | bold,
                    filler(),
                    text(std::format(
                        "{} · seen {}",
                        state,
                        elapsed_label(client.last_seen, now)))
                        | color(Color::GrayDark)
                        | dim,
                }) | xflex);
        }
        if (snapshot.clients.size() > visible_clients) {
            rows.push_back(
                text(std::format(
                    "    … {} more client{}",
                    snapshot.clients.size() - visible_clients,
                    snapshot.clients.size() - visible_clients == 1 ? "" : "s"))
                | color(Color::GrayDark)
                | dim);
        }
    }

    rows.push_back(text(""));
    rows.push_back(text("  Activity") | color(ColorYellowDark) | bold);

    if (snapshot.activities.empty()) {
        rows.push_back(
            text("  No remote tools have been used yet.")
            | color(Color::GrayLight)
            | dim);
    } else {
        constexpr std::size_t kVisibleRows = 8;
        const std::size_t selected =
            std::min(selected_activity, snapshot.activities.size() - 1);
        const std::size_t start = selected >= kVisibleRows
            ? selected - kVisibleRows + 1
            : 0;
        const std::size_t end =
            std::min(snapshot.activities.size(), start + kVisibleRows);

        for (std::size_t i = start; i < end; ++i) {
            const auto& activity = snapshot.activities[i];
            const bool is_selected = i == selected;
            rows.push_back(
                hbox({
                    text(is_selected ? "  › " : "    ")
                        | color(ColorYellowBright),
                    text(std::string(status_icon(activity.status)) + " ")
                        | color(status_color(activity.status))
                        | bold,
                    text(compact_tool_name(activity.tool_name))
                        | color(ColorYellowBright)
                        | bold,
                    text("  " + activity_summary(activity))
                        | color(Color::GrayLight)
                        | xflex,
                    text("  " + client_name_for(snapshot, activity.session_id))
                        | color(Color::GrayDark)
                        | dim,
                    text("  " + activity_duration(activity, now))
                        | color(status_color(activity.status)),
                }) | xflex);
        }

        const auto& selected_item = snapshot.activities[selected];
        rows.push_back(text(""));
        rows.push_back(
            text("  Details") | color(ColorYellowDark) | bold);
        rows.push_back(
            paragraph(std::format(
                "  {} · {} · {}",
                client_name_for(snapshot, selected_item.session_id),
                selected_item.tool_name,
                activity_duration(selected_item, now)))
            | color(Color::GrayLight));
        if (const std::string result = result_summary(selected_item); !result.empty()) {
            rows.push_back(
                paragraph("  " + compact_single_line(result, 240))
                | color(selected_item.status == RemoteToolStatus::failed
                            ? static_cast<Color>(ColorToolFail)
                            : Color{Color::GrayLight}));
        }
    }

    rows.push_back(text(""));
    rows.push_back(
        text("  ↑/↓ select   C clear completed   Esc close")
        | color(Color::GrayDark)
        | dim);

    return UiWindow(
        text(" ⚡ Remote MCP activity ") | color(ColorYellowBright) | bold,
        vbox(std::move(rows)) | xflex);
}

} // namespace tui
