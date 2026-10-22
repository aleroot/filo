#include "ToolCallResult.hpp"

#include "../utils/JsonWriter.hpp"
#include "../utils/JsonUtils.hpp"

namespace core::mcp {

ToolCallResultClassification classify_tool_call_payload(std::string_view payload) {
    ToolCallResultClassification out{};

    // Every tool result passes through here, and results can be megabytes. A
    // DOM parse touched several times the payload to read two top-level keys;
    // find_members validates it in one pass without allocating.
    constexpr std::string_view kKeys[] = {"isError", "error"};
    core::utils::json::RawMember found[2];
    if (!core::utils::json::find_members(payload, kKeys, found)) {
        out.is_error = true;
        return out;
    }
    out.is_structured_object = true;

    using core::utils::json::ValueKind;
    out.is_error = found[0].kind == ValueKind::True
        || (found[1].kind != ValueKind::Absent && found[1].kind != ValueKind::Null);
    return out;
}

std::string build_call_tool_result_from_payload(std::string_view payload,
                                                std::string_view related_task_id) {
    return build_call_tool_result_from_payload(
        payload,
        classify_tool_call_payload(payload),
        related_task_id);
}

std::string build_call_tool_result_from_payload(
    std::string_view payload,
    ToolCallResultClassification classification,
    std::string_view related_task_id) {
    const bool is_error = classification.is_error;
    const bool embeds_structured = !is_error && classification.is_structured_object;

    // The payload is written exactly once. A structured result travels only as
    // structuredContent; repeating it as escaped text doubled every response
    // (and the client's parse of it) for no information. "content" stays, empty,
    // because the schema requires it. Errors and non-object payloads carry no
    // structuredContent, so they keep the text block — a result always has one
    // of the two. Escaping grows text by its quotes and backslashes; the
    // reserve covers that so a multi-megabyte buffer never regrows.
    core::utils::JsonWriter writer(
        (embeds_structured ? payload.size() : payload.size() + payload.size() / 8)
        + related_task_id.size() + 192);
    {
        auto root = writer.object();
        writer.key("content");
        {
            auto content = writer.array();
            if (!embeds_structured) {
                auto item = writer.object();
                writer.kv_str("type", "text").comma().kv_str("text", payload);
            }
        }
        writer.comma().kv_bool("isError", is_error);
        if (embeds_structured) {
            writer.comma().kv_raw("structuredContent", payload);
        }
        if (!related_task_id.empty()) {
            writer.comma().key("_meta");
            {
                auto meta = writer.object();
                writer.key("io.modelcontextprotocol/related-task");
                {
                    auto related = writer.object();
                    writer.kv_str("taskId", related_task_id);
                }
            }
        }
    }
    return std::move(writer).take();
}

} // namespace core::mcp
