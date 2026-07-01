#pragma once

#include "Conversation.hpp"

#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace core::agent {
class Agent;
}

namespace tui {

using AppendHistoryCallback = std::function<void(const std::string&)>;
using SaveSessionSnapshotCallback = std::function<void()>;

void rewind_last_turn(core::agent::Agent& agent,
                      std::mutex& ui_mutex,
                      std::vector<UiMessage>& ui_messages,
                      const AppendHistoryCallback& append_history,
                      const SaveSessionSnapshotCallback& save_session_snapshot);

void compact_history_from_rewind(core::agent::Agent& agent,
                                 bool assistant_turn_active,
                                 const AppendHistoryCallback& append_history,
                                 const SaveSessionSnapshotCallback& save_session_snapshot);

} // namespace tui
