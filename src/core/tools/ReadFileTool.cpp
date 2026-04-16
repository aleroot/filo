#include "ReadFileTool.hpp"
#include "ToolArgumentUtils.hpp"
#include "ToolNames.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <fstream>
#include <sstream>
#include <format>
#include <filesystem>
#include <cerrno>
#include <system_error>

namespace {

[[nodiscard]] std::string file_type_name(std::filesystem::file_type type) {
    switch (type) {
        case std::filesystem::file_type::none:       return "none";
        case std::filesystem::file_type::not_found:  return "not_found";
        case std::filesystem::file_type::regular:    return "regular";
        case std::filesystem::file_type::directory:  return "directory";
        case std::filesystem::file_type::symlink:    return "symlink";
        case std::filesystem::file_type::block:      return "block_device";
        case std::filesystem::file_type::character:  return "character_device";
        case std::filesystem::file_type::fifo:       return "fifo";
        case std::filesystem::file_type::socket:     return "socket";
        case std::filesystem::file_type::unknown:    return "unknown";
    }
    return "unknown";
}

[[nodiscard]] std::string format_read_error(
    std::string_view requested_path,
    std::string_view resolved_path,
    std::string_view reason)
{
    return std::format(
        "{{\"error\":\"Cannot read path '{}': {} (resolved: '{}')\"}}",
        core::utils::escape_json_string(requested_path),
        core::utils::escape_json_string(reason),
        core::utils::escape_json_string(resolved_path));
}

} // namespace

namespace core::tools {

ToolDefinition ReadFileTool::get_definition() const {
    return {
        .name  = std::string(names::kReadFile),
        .title = "Read File",
        .description =
            "Reads the contents of a file. Optionally specify offset_line (1-based) and "
            "limit_lines to read a slice of the file without loading the entire thing. "
            "Returns the file text in the 'content' field. "
            "Files larger than 1 MB are automatically truncated.",
        .parameters = {
            {"path",        "string",  "The absolute or relative path to the file to read.", true},
            {"offset_line", "integer", "First line to return (1-based). Defaults to 1.",      false},
            {"limit_lines", "integer", "Maximum number of lines to return. Defaults to all.", false}
        },
        .output_schema =
            R"({"type":"object","properties":{"content":{"type":"string","description":"The requested file contents, possibly truncated."}},"required":["content"],"additionalProperties":false})",
        .annotations = {
            .read_only_hint   = true,
            .idempotent_hint  = true,
        },
    };
}

std::string ReadFileTool::execute(const std::string& json_args, const core::context::SessionContext& context) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    auto error = parser.parse(json_args).get(doc);

    if (error) {
        return "{\"error\": \"Invalid JSON arguments provided to read_file.\"}";
    }

    if (const auto validation_error =
            detail::validate_object_arguments(doc, names::kReadFile, {"path", "offset_line", "limit_lines"})) {
        return *validation_error;
    }

    std::string_view file_path;
    if (doc["path"].get(file_path)) {
        return "{\"error\": \"Missing or invalid 'path' argument.\"}";
    }

    int64_t offset_line = 1;
    int64_t limit_lines = -1; // -1 = no limit

    int64_t tmp = 0;
    if (!doc["offset_line"].get(tmp) && tmp >= 1) offset_line = tmp;
    if (!doc["limit_lines"].get(tmp) && tmp >= 1) limit_lines = tmp;

    std::string path_str(file_path);
    const std::filesystem::path requested_path(path_str);
    std::filesystem::path resolved_path;
    if (const auto access_error =
            detail::check_workspace_access(
                requested_path,
                path_str,
                context,
                &resolved_path,
                names::kReadFile)) {
        return *access_error;
    }
    const std::string resolved_path_string = resolved_path.string();

    std::error_code status_ec;
    const auto status = std::filesystem::status(resolved_path, status_ec);
    if (status_ec) {
        if (status_ec == std::errc::no_such_file_or_directory
            || status_ec == std::errc::not_a_directory) {
            return format_read_error(path_str, resolved_path_string, "path does not exist");
        }
        return format_read_error(path_str, resolved_path_string,
                                 std::format("failed to inspect path ({})", status_ec.message()));
    }
    if (status.type() == std::filesystem::file_type::not_found) {
        return format_read_error(path_str, resolved_path_string, "path does not exist");
    }
    if (status.type() != std::filesystem::file_type::regular) {
        return format_read_error(path_str, resolved_path_string,
                                 std::format("path is not a regular file (type={})",
                                             file_type_name(status.type())));
    }

    errno = 0;
    std::ifstream ifs(resolved_path, std::ios::binary);
    const int open_errno = errno;
    if (!ifs) {
        const std::string reason = open_errno != 0
            ? std::error_code(open_errno, std::generic_category()).message()
            : std::string("unknown reason");
        return format_read_error(path_str, resolved_path_string,
                                 std::format("failed to open file for reading ({})", reason));
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
