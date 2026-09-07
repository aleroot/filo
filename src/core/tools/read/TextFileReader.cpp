#include "ReadTypes.hpp"
#include "../ToolArgumentUtils.hpp"
#include "../ToolNames.hpp"
#include "../../utils/JsonUtils.hpp"
#include <fstream>
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

namespace core::tools::read {
std::string read_text_file(const Options& options, const core::context::SessionContext& context) {
    const auto& file_path = options.paths.front();
    const int64_t offset_line = options.offset_line;
    const int64_t limit_lines = options.limit_lines ? options.limit_lines : -1;
    std::string path_str(file_path);
    const std::filesystem::path requested_path(path_str);
    std::filesystem::path resolved_path;
    if (const auto access_error =
            detail::check_workspace_access(
                requested_path,
                path_str,
                context,
                &resolved_path,
                names::kRead)) {
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
        content = read_prefix(ifs, 1024 * 1024 + 1);
        if (content.size() > 1024 * 1024) {
            content = content.substr(0, 1024 * 1024) + "\n\n... [TRUNCATED DUE TO SIZE] ...";
        }
    } else {
        // Sliced read: skip (offset_line - 1) lines, then collect limit_lines
        std::string line;
        int64_t current_line = 0;
        int64_t collected = 0;
        constexpr std::size_t kMaxBytes = kMaxSliceChars;

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

} // namespace core::tools::read
