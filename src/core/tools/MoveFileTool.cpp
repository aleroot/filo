#include "MoveFileTool.hpp"
#include "../utils/JsonUtils.hpp"
#include "ToolArgumentUtils.hpp"
#include <simdjson.h>
#include <filesystem>
#include <format>

namespace core::tools {

ToolDefinition MoveFileTool::get_definition() const {
    return {
        .name  = "move_file",
        .title = "Move File",
        .description =
            "Moves or renames a file or directory. "
            "Destination parent directories are created automatically. "
            "Falls back to copy + delete if source and destination are on different filesystems. "
            "Returns 'from' and 'to' paths on success.",
        .parameters = {
            {"source",      "string", "Current absolute or relative path of the file or directory.", true},
            {"destination", "string", "Target path — the new name or location.",                     true}
        },
        .output_schema =
            R"({"type":"object","properties":{"success":{"type":"boolean","description":"Whether the move completed successfully."},"from":{"type":"string","description":"The original source path."},"to":{"type":"string","description":"The final destination path."}},"required":["success","from","to"],"additionalProperties":false})",
        .annotations = {
            .destructive_hint = true,  // removes the source path
        },
    };
}

std::string MoveFileTool::execute(const std::string& json_args) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return R"({"error":"Invalid JSON arguments for move_file."})";
    }

    std::string_view src_v, dst_v;
    if (doc["source"].get(src_v) != simdjson::SUCCESS) {
        return R"({"error":"Missing required argument 'source'."})";
    }
    if (doc["destination"].get(dst_v) != simdjson::SUCCESS) {
        return R"({"error":"Missing required argument 'destination'."})";
    }

    const std::string src_str(src_v);
    const std::string dst_str(dst_v);
    std::filesystem::path src(src_str);
    std::filesystem::path dst(dst_str);
    if (const auto access_error = detail::check_workspace_access(src, src_str)) {
        return *access_error;
    }
    if (const auto access_error = detail::check_workspace_access(dst, dst_str)) {
        return *access_error;
    }
    std::error_code ec;

    if (!std::filesystem::exists(src, ec)) {
        return std::format(R"({{"error":"Source does not exist: {}"}})",
                           core::utils::escape_json_string(src_str));
    }

    // Create destination parent directories if needed
    if (dst.has_parent_path()) {
        std::filesystem::create_directories(dst.parent_path(), ec);
        if (ec) {
            return std::format(R"({{"error":"Cannot create destination parent: {}"}})",
                               core::utils::escape_json_string(ec.message()));
        }
    }

    std::filesystem::rename(src, dst, ec);
    if (ec) {
        // rename() fails across filesystems; fall back to copy + remove
        std::filesystem::copy(src, dst,
            std::filesystem::copy_options::recursive
          | std::filesystem::copy_options::overwrite_existing, ec);
        if (ec) {
            return std::format(R"({{"error":"Failed to move '{}' to '{}': {}"}})",
                               core::utils::escape_json_string(src_str),
                               core::utils::escape_json_string(dst_str),
                               core::utils::escape_json_string(ec.message()));
        }
        std::filesystem::remove_all(src, ec);
    }

    return std::format(R"({{"success":true,"from":"{}","to":"{}"}})",
                       core::utils::escape_json_string(src_str),
                       core::utils::escape_json_string(dst_str));
}

} // namespace core::tools
