#pragma once

#include "../../core/llm/Models.hpp"

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace exec::gateway::openai {

struct StreamChunkContext {
    std::string_view response_id;
    int64_t created = 0;
    std::string_view model;
    bool include_usage = false;
};

[[nodiscard]] std::string error_body(std::string_view message);
[[nodiscard]] std::string role_chunk(const StreamChunkContext& context);
[[nodiscard]] std::string content_chunk(const StreamChunkContext& context,
                                        std::string_view content);
[[nodiscard]] std::string tool_calls_chunk(
    const StreamChunkContext& context,
    const std::vector<core::llm::ToolCall>& tool_calls);
[[nodiscard]] std::string finish_chunk(const StreamChunkContext& context,
                                       std::string_view finish_reason);
[[nodiscard]] std::string usage_chunk(const StreamChunkContext& context,
                                      const core::llm::TokenUsage& usage);

} // namespace exec::gateway::openai
