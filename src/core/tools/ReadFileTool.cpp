#include "ReadFileTool.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <fstream>
#include <sstream>
#include <format>
#include <filesystem>

namespace core::tools {

ToolDefinition ReadFileTool::get_definition() const {
    return {
        .name  = "read_file",
        .title = "Read File",
        .description =
            "Reads the contents of a file. Optionally specify offset_line (1-based) and "
            "limit_lines to read a slice of the file without loading the entire thing. "
            "Returns the file text in the 'content' field. "
            "Files larger than 1 MB are automatically truncated.",
        .parameters = {
            {"file_path",   "string",  "The absolute or relative path to the file to read.", true},
            {"offset_line", "integer", "First line to return (1-based). Defaults to 1.",      false},
            {"limit_lines", "integer", "Maximum number of lines to return. Defaults to all.", false}
        },
        .annotations = {
            .read_only_hint   = true,
            .idempotent_hint  = true,
        },
    };
}

std::string ReadFileTool::execute(const std::string& json_args) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    auto error = parser.parse(json_args).get(doc);

    if (error) {
        return "{\"error\": \"Invalid JSON arguments provided to read_file.\"}";
    }

    std::string_view file_path;
    if (doc["file_path"].get(file_path)) {
        return "{\"error\": \"Missing 'file_path' argument.\"}";
    }

    int64_t offset_line = 1;
    int64_t limit_lines = -1; // -1 = no limit

    int64_t tmp = 0;
    if (!doc["offset_line"].get(tmp) && tmp >= 1) offset_line = tmp;
    if (!doc["limit_lines"].get(tmp) && tmp >= 1) limit_lines = tmp;

    std::string path_str(file_path);
    std::error_code ec;
    if (!std::filesystem::exists(path_str, ec) || !std::filesystem::is_regular_file(path_str, ec)) {
        return std::format("{{\"error\": \"File not found or is not a regular file: {}\"}}",
                           core::utils::escape_json_string(path_str));
    }

    std::ifstream ifs(path_str);
    if (!ifs) {
        return std::format("{{\"error\": \"Failed to open file for reading: {}\"}}",
                           core::utils::escape_json_string(path_str));
    }

    std::string content;

    if (offset_line == 1 && limit_lines == -1) {
        // Fast path: read entire file at once
        std::ostringstream buf;
        buf << ifs.rdbuf();
        content = buf.str();
        if (content.size() > 1024 * 1024) {
            content = content.substr(0, 1024 * 1024) + "\n\n... [TRUNCATED DUE TO SIZE] ...";
        }
    } else {
        // Sliced read: skip (offset_line - 1) lines, then collect limit_lines
        std::string line;
        int64_t current_line = 0;
        int64_t collected = 0;
        constexpr std::size_t kMaxBytes = 512 * 1024;

        while (std::getline(ifs, line)) {
            ++current_line;
            if (current_line < offset_line) continue;
            if (limit_lines != -1 && collected >= limit_lines) break;

            content += line;
            content += '\n';
            ++collected;

            if (content.size() > kMaxBytes) {
                content += "\n... [TRUNCATED DUE TO SIZE] ...";
                break;
            }
        }
    }

    return std::format("{{\"content\": \"{}\"}}", core::utils::escape_json_string(content));
}

} // namespace core::tools
