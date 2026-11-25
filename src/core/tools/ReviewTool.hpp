#pragma once

#include "Tool.hpp"

#include <functional>
#include <string>
#include <string_view>

namespace core::agent {
class Agent;
}

namespace core::tools {

[[nodiscard]] ToolDefinition review_tool_definition();

// Runs the review campaign on the session checkout. `target` in the JSON is
// "uncommitted" (default) or "staged". The queue reviews other checkouts
// through review::review_workspace directly; this tool does not take a path.
[[nodiscard]] std::string execute_review_tool(
    core::agent::Agent& agent,
    std::string_view json_args,
    const std::function<bool()>& cancellation_requested);

} // namespace core::tools
