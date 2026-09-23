#include "ReadTypes.hpp"
#include "../ToolArgumentUtils.hpp"
#include "../ToolNames.hpp"
#include "../../utils/JsonUtils.hpp"
#include <fstream>
#include <format>
#include <filesystem>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
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

// Stream a line window without ever holding a skipped line. getline() would
// allocate a multi-gigabyte line before the size cap could reject it, and a
// per-byte skip loop would walk every character; memchr spans keep both the
// skip and the copy proportional to the bytes actually moved.
[[nodiscard]] std::string read_line_window(
    std::istream& input,
    std::int64_t offset_line,
    std::int64_t limit_lines,
    bool& io_error)
{
    constexpr std::size_t kMaxBytes = core::tools::read::kMaxSliceChars;
    constexpr std::string_view kMarker = "\n... [TRUNCATED DUE TO SIZE] ...";
    std::string content;
    content.reserve(std::min<std::size_t>(kMaxBytes, 64 * 1024));
    std::string line;
    line.reserve(256);
    std::int64_t current_line = 0;
    std::int64_t collected = 0;
    bool oversized = false;
    bool stop = false;
    std::array<char, 64 * 1024> buffer{};

    auto finish_line = [&] {
        ++current_line;
        if (current_line < offset_line) {
            line.clear();
            oversized = false;
            return;
        }
        if (limit_lines != -1 && collected >= limit_lines) {
            stop = true;
            return;
        }
        if (oversized || line.size() > kMaxBytes) {
            content += core::tools::read::bounded_prefix(line, kMaxBytes);
            content += '\n';
            content += kMarker;
            stop = true;
            return;
        }
        content += line;
        content += '\n';
        ++collected;
        line.clear();
        if (content.size() > kMaxBytes) {
            content += kMarker;
            stop = true;
            return;
        }
        // The window closed on this line's own newline: stop now instead of
        // scanning the bytes of the line after it.
        if (limit_lines != -1 && collected >= limit_lines) stop = true;
    };

    while (!stop) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto got = input.gcount();
        if (got <= 0) break;
        const auto available = static_cast<std::size_t>(got);
        std::size_t i = 0;
        while (i < available && !stop) {
            const bool keep = current_line + 1 >= offset_line
                && (limit_lines == -1 || collected < limit_lines);
            const void* found = std::memchr(buffer.data() + i, '\n', available - i);
            const std::size_t newline = found != nullptr
                ? static_cast<std::size_t>(static_cast<const char*>(found) - buffer.data())
                : available;
            if (keep && !oversized) {
                const std::size_t span = newline - i;
                if (line.size() + span <= kMaxBytes)
                    line.append(buffer.data() + i, span);
                else
                    oversized = true;
            }
            i = newline;
            if (i < available) {
                ++i;
                finish_line();
            }
        }
        if (input.bad()) {
            io_error = true;
            return {};
        }
    }
    if (!stop && (!line.empty() || oversized)) finish_line();
    return content;
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
        constexpr std::size_t kCap = 1024 * 1024;
        // Size from the already-open descriptor: two lseeks on this fd beat a
        // second path resolution, and size and bytes read cannot disagree.
        ifs.seekg(0, std::ios::end);
        const std::streamoff end = ifs.tellg();
        ifs.seekg(0);
        if (end != 0) {
            const std::size_t request = end > 0
                ? static_cast<std::size_t>(std::min<std::streamoff>(
                      end, static_cast<std::streamoff>(kCap) + 1))
                : kCap + 1;
            content = read_prefix(ifs, request);
            if (ifs.bad()) {
                return format_read_error(path_str, resolved_path_string, "failed to read file");
            }
            if (content.size() > kCap) {
                content = bounded_prefix(content, kCap) + "\n\n... [TRUNCATED DUE TO SIZE] ...";
            }
        }
    } else {
        bool io_error = false;
        content = read_line_window(ifs, offset_line, limit_lines, io_error);
        if (io_error) {
            return format_read_error(path_str, resolved_path_string, "failed to read file");
        }
    }

    return std::format("{{\"content\": \"{}\"}}", core::utils::escape_json_string(content));
}

} // namespace core::tools::read
