#include "ToolTranscript.hpp"

#include <algorithm>
#include <format>
#include <iterator>
#include <ranges>
#include <unordered_set>
#include <utility>

namespace core::llm {
namespace {

[[nodiscard]] bool has_replayable_identity(const ToolCall& tool_call) noexcept {
    return !tool_call.id.empty() && !tool_call.function.name.empty();
}

[[nodiscard]] bool is_empty_assistant_message(const Message& message) noexcept {
    return message.role == "assistant"
        && message.content.empty()
        && message.content_parts.empty()
        && message.tool_calls.empty()
        && message.reasoning_content.empty()
        && message.continuation_items.empty();
}

} // namespace

std::size_t discard_non_replayable_tool_calls(
    std::vector<ToolCall>& tool_calls) {
    return std::erase_if(tool_calls, [](const ToolCall& tool_call) {
        return !has_replayable_identity(tool_call);
    });
}

CompletedToolCallNormalization normalize_completed_tool_calls(
    std::vector<ToolCall>& completed_tool_calls,
    const std::function<std::string()>& make_tool_call_id) {
    CompletedToolCallNormalization normalization;

    for (auto& tool_call : completed_tool_calls) {
        if (tool_call.function.name.empty() || !tool_call.id.empty()
            || !make_tool_call_id) {
            continue;
        }
        if (auto id = make_tool_call_id(); !id.empty()) {
            tool_call.id = std::move(id);
            ++normalization.assigned_ids;
        }
    }

    normalization.discarded_calls =
        discard_non_replayable_tool_calls(completed_tool_calls);
    return normalization;
}

ToolTranscriptRepair repair_tool_transcript(std::vector<Message>& messages) {
    ToolTranscriptRepair repair;
    std::vector<Message> repaired;
    repaired.reserve(messages.size());

    for (std::size_t index = 0; index < messages.size();) {
        auto message = std::move(messages[index++]);
        if (message.role != "assistant") {
            repaired.push_back(std::move(message));
            continue;
        }

        std::unordered_set<std::string> removed_call_ids;
        bool removed_idless_call = false;
        for (const auto& tool_call : message.tool_calls) {
            if (has_replayable_identity(tool_call)) continue;

            ++repair.removed_calls;
            if (tool_call.id.empty()) {
                removed_idless_call = true;
            } else {
                removed_call_ids.insert(tool_call.id);
            }
        }
        std::erase_if(message.tool_calls, [](const ToolCall& tool_call) {
            return !has_replayable_identity(tool_call);
        });

        const bool removed_calls_from_message =
            !removed_call_ids.empty() || removed_idless_call;
        if (removed_calls_from_message && is_empty_assistant_message(message)) {
            ++repair.removed_empty_messages;
        } else {
            repaired.push_back(std::move(message));
        }

        // Tool results are meaningful only as the contiguous run immediately
        // following their assistant tool-use message. Restrict removal to that
        // run so a malformed legacy id cannot affect later history.
        if (!removed_calls_from_message) continue;
        while (index < messages.size() && messages[index].role == "tool") {
            auto result = std::move(messages[index++]);
            const bool belongs_to_removed_call =
                (!result.tool_call_id.empty()
                 && removed_call_ids.contains(result.tool_call_id))
                || (result.tool_call_id.empty() && removed_idless_call);
            if (belongs_to_removed_call) {
                ++repair.removed_results;
            } else {
                repaired.push_back(std::move(result));
            }
        }
    }

    messages = std::move(repaired);
    return repair;
}

std::optional<ToolTranscriptIssue> validate_tool_transcript(
    const std::vector<Message>& messages) {
    for (std::size_t index = 0; index < messages.size();) {
        const auto& message = messages[index];
        if (message.role == "tool") {
            return ToolTranscriptIssue{
                .message_index = index,
                .reason = std::format(
                    "tool result '{}' does not immediately follow an assistant tool call",
                    message.tool_call_id),
            };
        }
        if (message.role != "assistant" || message.tool_calls.empty()) {
            ++index;
            continue;
        }

        for (const auto& tool_call : message.tool_calls) {
            if (!has_replayable_identity(tool_call)) {
                return ToolTranscriptIssue{
                    .message_index = index,
                    .reason = "assistant tool call is missing its required id or name",
                };
            }
        }

        std::vector<bool> matched(message.tool_calls.size(), false);
        std::size_t result_count = 0;
        std::size_t result_index = index + 1;
        while (result_index < messages.size()
               && messages[result_index].role == "tool") {
            const auto& result = messages[result_index];
            if (result.tool_call_id.empty()) {
                return ToolTranscriptIssue{
                    .message_index = index,
                    .reason = std::format(
                        "tool result at history index {} is missing its tool call id",
                        result_index),
                };
            }
            const auto call = std::ranges::find_if(
                message.tool_calls,
                [&](const ToolCall& tool_call) {
                    return tool_call.id == result.tool_call_id;
                });
            if (call == message.tool_calls.end()) {
                return ToolTranscriptIssue{
                    .message_index = index,
                    .reason = std::format(
                        "tool result '{}' does not belong to the preceding assistant message at history index {}",
                        result.tool_call_id,
                        index),
                };
            }
            const auto call_index = static_cast<std::size_t>(
                std::distance(message.tool_calls.begin(), call));
            if (matched[call_index]) {
                return ToolTranscriptIssue{
                    .message_index = index,
                    .reason = std::format(
                        "tool call '{}' has more than one result after history index {}",
                        result.tool_call_id,
                        index),
                };
            }
            matched[call_index] = true;
            ++result_count;
            ++result_index;
        }

        if (result_count != message.tool_calls.size()) {
            std::string missing;
            for (std::size_t call_index = 0;
                 call_index < message.tool_calls.size();
                 ++call_index) {
                if (matched[call_index]) continue;
                if (!missing.empty()) missing += ", ";
                missing += message.tool_calls[call_index].id;
            }
            return ToolTranscriptIssue{
                .message_index = index,
                .reason = std::format(
                    "assistant tool calls at history index {} are not immediately followed by all results; missing: {}",
                    index,
                    missing),
            };
        }
        index = result_index;
    }
    return std::nullopt;
}

} // namespace core::llm
