#include "RewindActions.hpp"

#include "core/agent/Agent.hpp"

namespace tui {

void rewind_last_turn(core::agent::Agent& agent,
                      std::mutex& ui_mutex,
                      std::vector<UiMessage>& ui_messages,
                      const AppendHistoryCallback& append_history,
                      const SaveSessionSnapshotCallback& save_session_snapshot) {
    const auto before = agent.get_history().size();
    agent.undo_last();
    const auto after = agent.get_history().size();

    bool removed_visible_turn = false;
    {
        std::lock_guard lock(ui_mutex);
        removed_visible_turn = remove_latest_ui_turn(ui_messages);
    }

    if (after < before || removed_visible_turn) {
        save_session_snapshot();
        append_history("\n↶  Rewound the latest turn. Continue from here.\n");
    } else {
        append_history("\nℹ  Nothing to rewind yet.\n");
    }
}

void compact_history_from_rewind(core::agent::Agent& agent,
                                 bool assistant_turn_active,
                                 const AppendHistoryCallback& append_history,
                                 const SaveSessionSnapshotCallback& save_session_snapshot) {
    if (assistant_turn_active) {
        append_history("\nℹ  Stop the active turn before compacting history.\n");
        return;
    }

    if (!agent.has_user_turn()) {
        append_history("\nℹ  Nothing to summarize yet.\n");
        return;
    }

    agent.compact_history_async(append_history, save_session_snapshot);
}

} // namespace tui
