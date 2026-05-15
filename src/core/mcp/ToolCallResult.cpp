#include "ToolCallResult.hpp"

#include "../utils/JsonWriter.hpp"

#include <simdjson.h>

namespace core::mcp {

ToolCallResultClassification classify_tool_call_payload(std::string_view payload) {
    ToolCallResultClassification out{};

    if (payload.empty()) {
        out.is_error = true;
        return out;
    }

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string padded(payload);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        out.is_error = true;
        return out;
    }

    simdjson::dom::object object;
    if (doc.get(object) != simdjson::SUCCESS) {
        out.is_error = true;
        return out;
    }
    out.is_structured_object = true;

    bool is_error_flag = false;
    if (object["isError"].get(is_error_flag) == simdjson::SUCCESS && is_error_flag) {
        out.is_error = true;
        return out;
    }

    simdjson::dom::element error_value;
    if (object["error"].get(error_value) == simdjson::SUCCESS) {
        const auto type = error_value.type();
        if (type == simdjson::dom::element_type::NULL_VALUE) {
            return out;
        }

        out.is_error = true;
        return out;
    }

    return out;
}

std::string build_call_tool_result_from_payload(std::string_view payload,
                                                std::string_view related_task_id) {
    const ToolCallResultClassification classification = classify_tool_call_payload(payload);
    const bool is_error = classification.is_error;

    core::utils::JsonWriter writer(payload.size() + related_task_id.size() + 192);
    {
        auto root = writer.object();
        writer.key("content");
        {
            auto content = writer.array();
            auto item = writer.object();
            writer.kv_str("type", "text").comma().kv_str("text", payload);
        }
        writer.comma().kv_bool("isError", is_error);
        if (!is_error && classification.is_structured_object) {
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
