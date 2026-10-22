#include "WriteFileTool.hpp"
#include "ToolArgumentUtils.hpp"
#include "ToolDiffUtils.hpp"
#include "ToolNames.hpp"
#include "read/ReadTypes.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include <algorithm>
#include <array>
#include <fstream>
#include <sstream>
#include <format>
#include <filesystem>
#include <utility>

namespace core::tools {

WriteFileTool::WriteFileTool() = default;

ToolDefinition WriteFileTool::get_definition() const {
    return {
        .name  = std::string(names::kWriteFile),
        .title = "Write",
        .description =
            "Write complete file content, creating parents. Overwrites existing content and "
            "returns bounded previous_content for diff display.",
        .parameters = {
            {"file_path", "string", "File path.", true},
            {"content",   "string", "Complete new content.", true}
        },
        .output_schema =
            R"({"type":"object","properties":{"success":{"type":"boolean","description":"Whether the write completed successfully."},"file_path":{"type":"string","description":"The path that was written."},"created":{"type":"boolean","description":"True if the file did not exist before this write."},"bytes_written":{"type":"integer","description":"Number of bytes written to disk."},"previous_content":{"type":"string","description":"The prior file content, capped for diff display."},"previous_content_truncated":{"type":"boolean","description":"True when previous_content is not the full prior file."}},"required":["success","file_path","created","bytes_written","previous_content","previous_content_truncated"],"additionalProperties":false})",
        .annotations = {
            .destructive_hint = true,  // overwrites existing file content
            .idempotent_hint  = true,  // writing the same content twice is a no-op in effect
        },
        .primary_argument = "file_path",
    };
}

std::string WriteFileTool::execute(const std::string& json_args, const core::context::SessionContext& context) {
    // The content is never materialized: it is decoded from the argument
    // payload straight into the file below, so a multi-megabyte write holds
    // no copy of it beyond the request itself.
    namespace json = core::utils::json;
    constexpr std::string_view kKeys[] = {"file_path", "content"};
    json::RawMember found[std::size(kKeys)];
    if (!json::find_members(json_args, kKeys, found)) {
        return R"({"error":"Invalid JSON arguments provided to write_file."})";
    }
    if (found[0].kind != json::ValueKind::String) {
        return R"({"error":"Missing 'file_path' argument."})";
    }
    if (found[1].kind != json::ValueKind::String) {
        return R"({"error":"Missing 'content' argument."})";
    }

    const std::string path_str = json::decode_string(found[0].value);
    const std::filesystem::path requested_path(path_str);
    std::filesystem::path resolved_path;
    if (const auto access_error =
            detail::check_workspace_access(
                requested_path,
                path_str,
                context,
                &resolved_path,
                names::kWriteFile)) {
        return *access_error;
    }

    // -------------------------------------------------------------------------
    // Read previous content BEFORE overwriting so the client can render a diff.
    // We cap at 512 KB to keep the MCP response size reasonable.
    // -------------------------------------------------------------------------
    std::error_code ec;
    const bool file_existed = std::filesystem::exists(resolved_path, ec);

    std::string previous_content;
    bool previous_truncated = false;
    if (file_existed) {
        std::ifstream prev_ifs(resolved_path, std::ios::binary);
        if (!prev_ifs) {
            // The file is there, but its old bytes are unknown. Say so instead
            // of letting a diff treat the file as if it used to be empty.
            previous_truncated = true;
        } else {
            // The diff budget: returning more than a diff can be built from
            // would only inflate the tool result. Sized from the descriptor so
            // a small file does not zero-fill the whole budget.
            constexpr std::size_t kMaxPrev = detail::kMaxToolDiffInputBytes;
            prev_ifs.seekg(0, std::ios::end);
            const std::streamoff size = prev_ifs.tellg();
            prev_ifs.seekg(0);
            const std::size_t request = size >= 0
                ? static_cast<std::size_t>(std::min<std::streamoff>(size, kMaxPrev))
                : kMaxPrev;
            previous_content = read::read_prefix(prev_ifs, request);
            if (previous_content.size() == kMaxPrev) {
                char extra = 0;
                if (prev_ifs.read(&extra, 1) && prev_ifs.gcount() == 1) {
                    previous_truncated = true;
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // Create parent directories if needed.
    // -------------------------------------------------------------------------
    if (resolved_path.has_parent_path()) {
        std::filesystem::create_directories(resolved_path.parent_path(), ec);
        // Ignore ec: if create_directories fails the open below will also fail
        // and we'll report that error instead.
    }

    // -------------------------------------------------------------------------
    // Write the new content.
    // -------------------------------------------------------------------------
    std::ofstream ofs(resolved_path, std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return std::format(R"({{"error":"Failed to open file for writing: '{}'."}})",
                           core::utils::escape_json_string(path_str));
    }
    std::string_view pending = found[1].value.substr(1, found[1].value.size() - 2);
    std::array<char, 64 * 1024> chunk;
    std::size_t bytes_written = 0;
    while (!pending.empty()) {
        const std::size_t length = json::decode_string_part(pending, chunk);
        ofs.write(chunk.data(), static_cast<std::streamsize>(length));
        bytes_written += length;
    }
    ofs.close();

    // -------------------------------------------------------------------------
    // Build rich response for Lampo's diff UI, escaping straight into it.
    // -------------------------------------------------------------------------
    const std::string resolved = resolved_path.string();
    core::utils::JsonWriter writer(
        resolved.size() + previous_content.size() + previous_content.size() / 8 + 160);
    {
        auto object = writer.object();
        writer.kv_bool("success", true)
            .comma().kv_str("file_path", resolved)
            .comma().kv_bool("created", !file_existed)
            .comma().kv_num("bytes_written", bytes_written)
            .comma().kv_str("previous_content", previous_content)
            .comma().kv_bool("previous_content_truncated", previous_truncated);
    }
    return std::move(writer).take();
}

} // namespace core::tools
