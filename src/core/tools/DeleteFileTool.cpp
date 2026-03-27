#include "DeleteFileTool.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <filesystem>
#include <format>

namespace core::tools {

ToolDefinition DeleteFileTool::get_definition() const {
    return {
        .name  = "delete_file",
        .title = "Delete File",
        .description =
            "Permanently deletes a file or empty directory from the filesystem. "
            "This action cannot be undone. "
            "Returns the deleted path in 'deleted' on success.",
        .parameters = {
            {"file_path", "string",
             "Absolute or relative path to the file or empty directory to delete.", true}
        },
        .annotations = {
            .destructive_hint = true,  // permanently removes data
            .idempotent_hint  = true,  // deleting an already-deleted path is effectively the same end state
        },
    };
}

std::string DeleteFileTool::execute(const std::string& json_args) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return R"({"error":"Invalid JSON arguments for delete_file."})";
    }

    std::string_view path_v;
    if (doc["file_path"].get(path_v) != simdjson::SUCCESS) {
        return R"({"error":"Missing required argument 'file_path'."})";
    }

    std::filesystem::path p(path_v);
    std::error_code ec;

    if (!std::filesystem::exists(p, ec)) {
        return std::format(R"({{"error":"Path does not exist: {}"}})",
                           core::utils::escape_json_string(std::string(path_v)));
    }

    std::filesystem::remove(p, ec);
    if (ec) {
        return std::format(R"({{"error":"Failed to delete '{}': {}"}})",
                           core::utils::escape_json_string(std::string(path_v)),
                           core::utils::escape_json_string(ec.message()));
    }

    return std::format(R"({{"success":true,"deleted":"{}"}})",
                       core::utils::escape_json_string(std::string(path_v)));
}

} // namespace core::tools
