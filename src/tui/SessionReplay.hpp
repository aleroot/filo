#pragma once

#include "Conversation.hpp"
#include "core/session/SessionData.hpp"

#include <vector>

namespace tui {

struct SessionReplayOptions {
    bool include_continue_hint = false;
};

std::vector<UiMessage> build_resumed_ui_messages(
    const core::session::SessionData& data,
    SessionReplayOptions options = {});

} // namespace tui
