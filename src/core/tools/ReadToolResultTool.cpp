#include "ReadToolResultTool.hpp"

#include "ToolNames.hpp"
#include "../utils/JsonWriter.hpp"

#include <simdjson.h>

#include <cstdint>

namespace core::tools {

ReadToolResultTool::ReadToolResultTool(core::agent::ToolResultStore& store) noexcept
    : store_(store) {}

ToolDefinition ReadToolResultTool::get_definition() const {
    return {
        .name = std::string(names::kReadToolResult),
        .title = "Read Tool Result",
        .description =
            "Reads a byte range from a complete oversized tool result previously offloaded by "
            "Filo. Use the opaque reference returned in the original tool response and paginate "
            "with next_offset until complete.",
        .input_schema =
            R"({"type":"object","properties":{"reference":{"type":"string","minLength":3,"maxLength":192},"offset":{"type":"integer","minimum":0,"description":"Byte offset; omit for the first chunk."},"limit":{"type":"integer","minimum":4,"maximum":4096,"description":"Maximum bytes to return."}},"required":["reference"],"additionalProperties":false})",
        .output_schema =
            R"({"type":"object","properties":{"reference":{"type":"string"},"content":{"type":"string"},"offset":{"type":"integer"},"next_offset":{"type":"integer"},"total_bytes":{"type":"integer"},"complete":{"type":"boolean"}},"required":["reference","content","offset","next_offset","total_bytes","complete"],"additionalProperties":false})",
        .annotations = {
            .read_only_hint = true,
            .destructive_hint = false,
            .idempotent_hint = true,
            .open_world_hint = false,
        },
    };
}

std::string ReadToolResultTool::execute(
    const std::string& json_args,
    const core::context::SessionContext& context) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded{json_args};
    simdjson::dom::object object;
    if (parser.parse(padded).get(object) != simdjson::SUCCESS) {
        return R"({"error":"Invalid JSON arguments for read_tool_result."})";
    }

    std::string_view reference;
    if (object["reference"].get(reference) != simdjson::SUCCESS) {
        return R"({"error":"read_tool_result requires a reference string."})";
    }
    std::int64_t offset = 0;
    std::int64_t limit = static_cast<std::int64_t>(
        core::agent::ToolResultStore::kDefaultReadChars);
    (void)object["offset"].get(offset);
    (void)object["limit"].get(limit);
    if (offset < 0 || limit < 4) {
        return R"({"error":"read_tool_result requires offset >= 0 and limit >= 4."})";
    }

    const auto chunk = store_.read(
        context.session_id,
        reference,
        static_cast<std::uint64_t>(offset),
        static_cast<std::size_t>(limit));
    if (!chunk.has_value()) {
        core::utils::JsonWriter error(128 + chunk.error().size());
        {
            auto object_scope = error.object();
            error.kv_str("error", chunk.error());
        }
        return std::move(error).take();
    }

    core::utils::JsonWriter writer(192 + chunk->content.size());
    {
        auto object_scope = writer.object();
        writer.kv_str("reference", reference).comma()
              .kv_str("content", chunk->content).comma()
              .kv_num("offset", static_cast<std::int64_t>(chunk->offset)).comma()
              .kv_num("next_offset", static_cast<std::int64_t>(chunk->next_offset)).comma()
              .kv_num("total_bytes", static_cast<std::int64_t>(chunk->total_bytes)).comma()
              .kv_bool("complete", chunk->complete);
    }
    return std::move(writer).take();
}

} // namespace core::tools
