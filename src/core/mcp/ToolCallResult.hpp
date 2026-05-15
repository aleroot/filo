#pragma once

#include <string>
#include <string_view>

namespace core::mcp {

struct ToolCallResultClassification {
    bool is_error = false;
    bool is_structured_object = false;
};

[[nodiscard]] ToolCallResultClassification classify_tool_call_payload(std::string_view payload);

[[nodiscard]] std::string build_call_tool_result_from_payload(
    std::string_view payload,
    std::string_view related_task_id = {});

} // namespace core::mcp
