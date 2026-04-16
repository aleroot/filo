#include "ListDirectoryTool.hpp"
#include "ToolArgumentUtils.hpp"
#include "ToolNames.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <format>
#include <filesystem>

namespace core::tools {

ToolDefinition ListDirectoryTool::get_definition() const {
    return {
        .name  = std::string(names::kListDirectory),
        .title = "List Directory",
        .description =
            "Lists the immediate contents of a directory. "
            "Returns a JSON array of objects with 'type' ('file' or 'dir') and 'name' fields.",
        .parameters = {
            {"path", "string", "Absolute or relative path to the directory to list.", true}
        },
        .output_schema =
            R"({"type":"object","properties":{"entries":{"type":"array","items":{"type":"object","properties":{"type":{"type":"string","enum":["file","dir"]},"name":{"type":"string"}},"required":["type","name"],"additionalProperties":false},"description":"Immediate entries in the directory."}},"required":["entries"],"additionalProperties":false})",
        .annotations = {
            .read_only_hint  = true,
            .idempotent_hint = true,
        },
    };
}

std::string ListDirectoryTool::execute(const std::string& json_args, const core::context::SessionContext& context) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    auto error = parser.parse(json_args).get(doc);
    
    if (error) {
        return "{\"error\": \"Invalid JSON arguments provided to list_directory.\"}";
    }

    if (const auto validation_error =
            detail::validate_object_arguments(doc, names::kListDirectory, {"path"})) {
        return *validation_error;
    }

    std::string_view dir_path;
    if (doc["path"].get(dir_path)) {
        return "{\"error\": \"Missing or invalid 'path' argument.\"}";
    }

    std::string path_str(dir_path);
    std::filesystem::path resolved_path;
    if (const auto access_error =
            detail::check_workspace_access(
                path_str,
                path_str,
                context,
                &resolved_path,
                names::kListDirectory)) {
        return *access_error;
    }
    std::error_code ec;
    if (!std::filesystem::exists(resolved_path, ec)
        || !std::filesystem::is_directory(resolved_path, ec)) {
        return std::format("{{\"error\": \"Directory not found: {}\"}}", core::utils::escape_json_string(path_str));
    }

    core::utils::JsonWriter w(1024);
    {
        auto _root = w.object();
        w.key("entries");
        {
            auto _arr = w.array();
            bool first = true;
            for (const auto& entry : std::filesystem::directory_iterator(resolved_path, ec)) {
                if (!first) w.comma();
                first = false;
                auto _item = w.object();
                w.kv_str("type", entry.is_directory(ec) ? "dir" : "file").comma()
                 .kv_str("name", entry.path().filename().string());
            }
        }
    }
    return std::move(w).take();
}

} // namespace core::tools
