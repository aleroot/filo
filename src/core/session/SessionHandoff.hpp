#pragma once

#include "SessionData.hpp"
#include <string>
#include <vector>

namespace core::session {

[[nodiscard]] std::string build_handoff_summary(const SessionData& data);

[[nodiscard]] std::string build_handoff_summary(
    const std::vector<core::llm::Message>& messages,
    std::string_view existing_context_summary,
    std::string_view mode);

[[nodiscard]] bool has_handoff_summary(const SessionData& data) noexcept;

} // namespace core::session
