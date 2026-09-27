#pragma once

/**
 * @file
 * @brief Provider-neutral normalization, repair, and validation of tool-call history.
 *
 * Tool-call fragments are intentionally incomplete while a response streams.
 * Once the terminal event arrives, this module owns the stricter invariants
 * required for a transcript to be persisted and replayed by any provider.
 */

#include "Models.hpp"

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace core::llm {

struct CompletedToolCallNormalization {
    std::size_t assigned_ids = 0;
    std::size_t discarded_calls = 0;
};

/// Removes tool calls that cannot be represented in a provider-neutral replay
/// transcript. Use this for complete externally supplied calls; streamed calls
/// must first be assembled, and terminal calls may use the normalizer below.
[[nodiscard]] std::size_t discard_non_replayable_tool_calls(
    std::vector<ToolCall>& tool_calls);

/// Normalizes a terminal, fully assembled tool-call batch. Named calls that
/// lack an id receive one from @p make_tool_call_id; calls still lacking either
/// required identity field are discarded. Do not apply this to stream chunks.
[[nodiscard]] CompletedToolCallNormalization normalize_completed_tool_calls(
    std::vector<ToolCall>& completed_tool_calls,
    const std::function<std::string()>& make_tool_call_id);

struct ToolTranscriptRepair {
    std::size_t removed_calls = 0;
    std::size_t removed_results = 0;
    std::size_t removed_empty_messages = 0;
};

/// Repairs legacy or externally supplied history without losing valid sibling
/// calls or later conversation. A removed tool call also removes only a result
/// in its immediately following tool-result run.
[[nodiscard]] ToolTranscriptRepair repair_tool_transcript(
    std::vector<Message>& messages);

struct ToolTranscriptIssue {
    std::size_t message_index = 0;
    std::string reason;
};

/// Validates the provider-neutral tool-call/result ordering and identities.
/// The returned issue is suitable for surfacing at a host boundary; this layer
/// deliberately has no UI or provider dependencies.
[[nodiscard]] std::optional<ToolTranscriptIssue> validate_tool_transcript(
    const std::vector<Message>& messages);

} // namespace core::llm
