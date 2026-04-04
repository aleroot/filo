#include "ListDirectoryTool.hpp"
#include "ToolArgumentUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <format>
#include <filesystem>

namespace core::tools {

ToolDefinition ListDirectoryTool::get_definition() const {
    return {
        .name  = "list_directory",
        .title = "List Directory",
        .description =
            "Lists the immediate contents of a directory. "
            "Returns a JSON array of objects with 'type' ('file' or 'dir') and 'name' fields.",
        .parameters = {
            {"path", "string", "Absolute or relative path to the directory to list.", true}
        },
        .annotations = {
            .read_only_hint  = true,
            .idempotent_hint = true,
        },
    };
}

std::string ListDirectoryTool::execute(const std::string& json_args) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    auto error = parser.parse(json_args).get(doc);
    
    if (error) {
        return "{\"error\": \"Invalid JSON arguments provided to list_directory.\"}";
    }

    if (const auto validation_error =
            detail::validate_object_arguments(doc, "list_directory", {"path"})) {
        return *validation_error;
    }

    std::string_view dir_path;
    if (doc["path"].get(dir_path)) {
        return "{\"error\": \"Missing or invalid 'path' argument.\"}";
    }

    std::string path_str(dir_path);
    if (const auto access_error = detail::check_workspace_access(path_str, path_str)) return *access_error;
    std::error_code ec;
    if (!std::filesystem::exists(path_str, ec) || !std::filesystem::is_directory(path_str, ec)) {
        return std::format("{{\"error\": \"Directory not found: {}\"}}", core::utils::escape_json_string(path_str));
    }

    core::utils::JsonWriter w(1024);
    {
        auto _root = w.object();
        w.key("entries");
        {
            auto _arr = w.array();
            bool first = true;
            for (const auto& entry : std::filesystem::directory_iterator(path_str, ec)) {
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
