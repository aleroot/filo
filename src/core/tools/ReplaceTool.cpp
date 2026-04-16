#include "ReplaceTool.hpp"
#include "ToolArgumentUtils.hpp"
#include "ToolNames.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <fstream>
#include <sstream>
#include <format>
#include <filesystem>

namespace core::tools {

ToolDefinition ReplaceTool::get_definition() const {
    return {
        .name  = std::string(names::kReplace),
        .title = "Replace in File",
        .description =
            "Replaces the first occurrence of an exact literal string within a file. "
            "Fails if the string is not found or appears more than once (use more context). "
            "Returns the 1-based line number of the replacement for UI display.",
        .parameters = {
            {"file_path",  "string", "Absolute or relative path to the file to modify.",       true},
            {"old_string", "string", "The exact literal text to find and replace.",             true},
            {"new_string", "string", "The exact literal text to insert in place of old_string.", true}
        },
        .output_schema =
            R"({"type":"object","properties":{"success":{"type":"boolean","description":"Whether the replacement completed successfully."},"file_path":{"type":"string","description":"The modified file path."},"replaced_at_line":{"type":"integer","description":"1-based line number where the replacement began."}},"required":["success","file_path","replaced_at_line"],"additionalProperties":false})",
        .annotations = {
            .destructive_hint = true,  // modifies file content on disk
        },
    };
}

std::string ReplaceTool::execute(const std::string& json_args, const core::context::SessionContext& context) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return R"({"error":"Invalid JSON arguments provided to replace."})";
    }

    std::string_view file_path, old_string, new_string;
    if (doc["file_path"].get(file_path)   != simdjson::SUCCESS)
        return R"({"error":"Missing 'file_path' argument."})";
    if (doc["old_string"].get(old_string) != simdjson::SUCCESS)
        return R"({"error":"Missing 'old_string' argument."})";
    if (doc["new_string"].get(new_string) != simdjson::SUCCESS)
        return R"({"error":"Missing 'new_string' argument."})";

    const std::string path_str(file_path);
    std::filesystem::path resolved_path;
    if (const auto access_error =
            detail::check_workspace_access(
                path_str,
                path_str,
                context,
                &resolved_path,
                names::kReplace)) {
        return *access_error;
    }
    std::error_code ec;
    if (!std::filesystem::exists(resolved_path, ec)
        || !std::filesystem::is_regular_file(resolved_path, ec)) {
        return std::format(R"({{"error":"File not found: '{}'."}})",
                           core::utils::escape_json_string(path_str));
    }

    // Read entire file.
    std::string content;
    {
        std::ifstream ifs(resolved_path, std::ios::binary);
        if (!ifs) {
            return std::format(R"({{"error":"Failed to open file for reading: '{}'."}})",
                               core::utils::escape_json_string(path_str));
        }
        std::ostringstream ss;
        ss << ifs.rdbuf();
        content = ss.str();
    }

    const std::string old_str(old_string);
    const std::size_t pos = content.find(old_str);
    if (pos == std::string::npos) {
        return R"({"error":"old_string not found in file."})";
    }

    // Reject ambiguous replacements — require the caller to add more context.
    if (content.find(old_str, pos + old_str.size()) != std::string::npos) {
        return R"({"error":"old_string matches multiple locations. Provide more surrounding context to make it unique."})";
    }

    // Compute 1-based line number of the replacement start for the UI.
    int64_t replaced_at_line = 1;
    for (std::size_t i = 0; i < pos; ++i) {
        if (content[i] == '\n') ++replaced_at_line;
    }

    content.replace(pos, old_str.size(), new_string);

    // Write back.
    std::ofstream ofs(resolved_path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return std::format(R"({{"error":"Failed to open file for writing: '{}'."}})",
                           core::utils::escape_json_string(path_str));
    }
    ofs << content;
    ofs.close();

    return std::format(R"({{"success":true,"file_path":"{}","replaced_at_line":{}}})",
                       core::utils::escape_json_string(resolved_path.string()),
                       replaced_at_line);
}

} // namespace core::tools
