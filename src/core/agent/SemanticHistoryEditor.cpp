#include "SemanticHistoryEditor.hpp"

#include "ToolCallDeduplicator.hpp"

#include <format>
#include <string>
#include <unordered_map>

namespace core::agent {
namespace {

[[nodiscard]] bool is_successful_result(const core::llm::Message& message) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(message.content);
    simdjson::dom::object object;
    if (parser.parse(padded).get(object) != simdjson::SUCCESS) return false;

    simdjson::dom::element error;
    if (object["error"].get(error) == simdjson::SUCCESS) return false;
    for (const std::string_view field : {"isError", "is_error"}) {
        bool failed = false;
        if (object[field].get(failed) == simdjson::SUCCESS && failed) return false;
    }
    return true;
}

} // namespace

SemanticHistoryEditResult SemanticHistoryEditor::edit(
    const std::vector<core::llm::Message>& history,
    SemanticHistoryEditPolicy policy) {
    SemanticHistoryEditResult result{.messages = history};
    if (history.empty()) return result;

    std::unordered_map<std::string, std::string> key_by_call_id;
    for (const auto& message : history) {
        if (message.role != "assistant") continue;
        for (const auto& call : message.tool_calls) {
            const std::string key = ToolCallDeduplicator::key_for_call(call);
            if (!call.id.empty() && !key.empty()) key_by_call_id[call.id] = key;
        }
    }

    std::unordered_map<std::string, std::size_t> latest_result_by_key;
    for (std::size_t i = 0; i < history.size(); ++i) {
        const auto& message = history[i];
        if (message.role != "tool" || !is_successful_result(message)) continue;
        if (const auto found = key_by_call_id.find(message.tool_call_id);
            found != key_by_call_id.end()) {
            latest_result_by_key[found->second] = i;
        }
    }

    const std::size_t editable_end = history.size() > policy.protected_tail_messages
        ? history.size() - policy.protected_tail_messages
        : 0;
    for (std::size_t i = 0; i < editable_end; ++i) {
        auto& message = result.messages[i];
        if (message.role != "tool"
            || message.content.size() < policy.minimum_result_chars
            || !message.continuation_items.empty()
            || !is_successful_result(message)) {
            continue;
        }
        const auto key = key_by_call_id.find(message.tool_call_id);
        if (key == key_by_call_id.end()) continue;
        const auto latest = latest_result_by_key.find(key->second);
        if (latest == latest_result_by_key.end() || latest->second <= i) continue;

        const std::size_t original_chars = message.content.size();
        message.content = std::format(
            R"({{"context_edit":"superseded_tool_result","original_chars":{},"newer_message_index":{}}})",
            original_chars,
            latest->second);
        ++result.superseded_results;
        result.characters_removed += original_chars - message.content.size();
    }
    return result;
}

} // namespace core::agent
