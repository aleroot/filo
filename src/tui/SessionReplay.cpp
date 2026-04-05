#include "SessionReplay.hpp"

#include <format>
#include <string>

namespace tui {

std::vector<UiMessage> build_resumed_ui_messages(
    const core::session::SessionData& data,
    SessionReplayOptions options) {
    // Format timestamp for display: "YYYY-MM-DDTHH:MM" -> "YYYY-MM-DD HH:MM"
    std::string ts = data.last_active_at.empty() ? data.created_at : data.last_active_at;
    if (ts.size() >= 16 && ts[10] == 'T') {
        ts[10] = ' ';
    }
    ts = ts.substr(0, 16);

    std::string header = std::format(
        "Resumed session {}  (last active: {}  provider: {}  model: {}  {} turns)\n",
        data.session_id,
        ts,
        data.provider,
        data.model,
        data.stats.turn_count);
    if (options.include_continue_hint) {
        header += "Type a message to continue, or /sessions to manage sessions.\n";
    }

    std::vector<UiMessage> ui_messages;
    ui_messages.reserve(data.messages.size() + 1);
    ui_messages.push_back(make_system_message(std::move(header)));

    // Track the current assistant UI message index so that tool result messages
    // can update the matching tool activity's status.
    int current_asst_idx = -1;
    for (const auto& msg : data.messages) {
        if (msg.role == "user") {
            ui_messages.push_back(make_user_message(msg.content, ""));
            current_asst_idx = -1;
            continue;
        }

        if (msg.role == "assistant") {
            UiMessage asst_msg = make_assistant_message(msg.content, "", false);
            for (const auto& tc : msg.tool_calls) {
                asst_msg.tools.push_back(make_tool_activity(
                    tc.id,
                    tc.function.name,
                    tc.function.arguments,
                    summarize_tool_arguments(tc.function.name, tc.function.arguments)));
            }
            current_asst_idx = static_cast<int>(ui_messages.size());
            ui_messages.push_back(std::move(asst_msg));
            continue;
        }

        if (msg.role == "tool" && current_asst_idx >= 0) {
            auto* tool = find_tool_activity(
                ui_messages[static_cast<std::size_t>(current_asst_idx)],
                msg.tool_call_id);
            if (tool != nullptr) {
                apply_tool_result(*tool, msg.content);
            }
        }
    }

    return ui_messages;
}

} // namespace tui
