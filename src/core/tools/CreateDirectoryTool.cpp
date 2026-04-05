#include "CreateDirectoryTool.hpp"
#include "../utils/JsonUtils.hpp"
#include "ToolArgumentUtils.hpp"
#include <simdjson.h>
#include <filesystem>
#include <format>

namespace core::tools {

ToolDefinition CreateDirectoryTool::get_definition() const {
    return {
        .name  = "create_directory",
        .title = "Create Directory",
        .description =
            "Creates a directory (and any missing parent directories) at the given path. "
            "Succeeds silently if the directory already exists. "
            "Returns the created path in 'path' on success.",
        .parameters = {
            {"dir_path", "string", "Absolute or relative path of the directory to create.", true}
        },
        .output_schema =
            R"({"type":"object","properties":{"success":{"type":"boolean","description":"Whether the directory exists after the call."},"path":{"type":"string","description":"The created directory path."}},"required":["success","path"],"additionalProperties":false})",
        .annotations = {
            .idempotent_hint = true,  // creating an existing directory is a no-op
        },
    };
}

std::string CreateDirectoryTool::execute(const std::string& json_args) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return R"({"error":"Invalid JSON arguments for create_directory."})";
    }

    std::string_view path_v;
    if (doc["dir_path"].get(path_v) != simdjson::SUCCESS) {
        return R"({"error":"Missing required argument 'dir_path'."})";
    }

    const std::string path_str(path_v);
    std::filesystem::path p(path_str);
    if (const auto access_error = detail::check_workspace_access(p, path_str)) {
        return *access_error;
    }
    std::error_code ec;
    std::filesystem::create_directories(p, ec);
    if (ec) {
        return std::format(R"({{"error":"Failed to create directory '{}': {}"}})",
                           core::utils::escape_json_string(path_str),
                           core::utils::escape_json_string(ec.message()));
    }

    return std::format(R"({{"success":true,"path":"{}"}})",
                       core::utils::escape_json_string(path_str));
}

} // namespace core::tools
