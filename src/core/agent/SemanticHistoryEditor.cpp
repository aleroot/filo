#include "SemanticHistoryEditor.hpp"

#include "ToolCallDeduplicator.hpp"

#include <format>
#include <string>
#include <unordered_map>
#include <utility>

namespace core::agent {
namespace {

[[nodiscard]] bool is_successful_result(
    const core::llm::Message& message, simdjson::dom::parser& parser) {
    simdjson::dom::object object;
    if (parser.parse(message.content).get(object) != simdjson::SUCCESS) return false;

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
    std::vector<core::llm::Message> history,
    SemanticHistoryEditPolicy policy) {
    SemanticHistoryEditResult result{.messages = std::move(history)};
    auto& messages = result.messages;
    const std::size_t editable_end = messages.size() > policy.protected_tail_messages
        ? messages.size() - policy.protected_tail_messages
        : 0;
    if (editable_end == 0) return result;

    std::unordered_map<std::string, std::string> key_by_call_id;
    for (const auto& message : messages) {
        if (message.role != "assistant") continue;
        for (const auto& call : message.tool_calls) {
            const std::string key = ToolCallDeduplicator::key_for_call(call);
            if (!call.id.empty() && !key.empty()) key_by_call_id[call.id] = key;
        }
    }
    if (key_by_call_id.empty()) return result;

    // Walk newest first: once a successful observation is known, only older
    // editable results need parsing. Non-observation output is never parsed.
    // The retained index always names the newest successful result, including
    // protected or signed results that must themselves remain untouched.
    std::unordered_map<std::string, std::size_t> latest_result_by_key;
    simdjson::dom::parser parser;
    for (std::size_t next = messages.size(); next > 0; --next) {
        const std::size_t i = next - 1;
        auto& message = messages[i];
        if (message.role != "tool") continue;
        const auto key = key_by_call_id.find(message.tool_call_id);
        if (key == key_by_call_id.end()) continue;
        const auto latest = latest_result_by_key.find(key->second);
        if (latest != latest_result_by_key.end()
            && (i >= editable_end
                || message.content.size() < policy.minimum_result_chars
                || !message.continuation_items.empty())) continue;
        if (!is_successful_result(message, parser)) continue;
        if (latest == latest_result_by_key.end()) {
            latest_result_by_key.emplace(key->second, i);
            continue;
        }

        const std::size_t original_chars = message.content.size();
        auto replacement = std::format(
            R"({{"context_edit":"superseded_tool_result","original_chars":{},"newer_message_index":{}}})",
            original_chars,
            latest->second);
        if (replacement.size() >= original_chars) continue;
        message.content = std::move(replacement);
        ++result.superseded_results;
        result.characters_removed += original_chars - message.content.size();
    }
    return result;
}

} // namespace core::agent
