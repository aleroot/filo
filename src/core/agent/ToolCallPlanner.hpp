#pragma once

#include "ToolAccess.hpp"
#include "../context/SessionContext.hpp"
#include "../llm/Models.hpp"

namespace core::agent {

struct PlannedToolCall {
    core::llm::ToolCall call;
    ToolAccessSet accesses;
};

[[nodiscard]] PlannedToolCall plan_tool_call(
    const core::llm::ToolCall& call,
    const core::context::SessionContext& context);

} // namespace core::agent
