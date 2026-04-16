#pragma once

#include "../context/SessionContext.hpp"

#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace core::hooks {

enum class HookEvent {
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
};

[[nodiscard]] std::string_view to_string(HookEvent event) noexcept;

void dispatch(HookEvent event,
              std::string payload_json,
              const core::context::SessionContext& session_context,
              std::vector<std::pair<std::string, std::string>> extra_env = {});

} // namespace core::hooks
