#include "WriteFileTool.hpp"
#include "../utils/JsonUtils.hpp"
#include "ToolArgumentUtils.hpp"
#include <simdjson.h>
#include <fstream>
#include <sstream>
#include <format>
#include <filesystem>

namespace core::tools {

ToolDefinition WriteFileTool::get_definition() const {
    return {
        .name  = "write_file",
        .title = "Write File",
        .description =
            "Writes complete content to a file, overwriting any existing content. "
            "Parent directories are created automatically if they do not exist. "
            "Returns the previous file content in 'previous_content' so the caller can "
            "display a diff; 'created' is true when the file did not exist before.",
        .parameters = {
            {"file_path", "string", "Absolute or relative path to the file to write.", true},
            {"content",   "string", "The complete new content to write to the file.",  true}
        },
        .output_schema =
            R"({"type":"object","properties":{"success":{"type":"boolean","description":"Whether the write completed successfully."},"file_path":{"type":"string","description":"The path that was written."},"created":{"type":"boolean","description":"True if the file did not exist before this write."},"bytes_written":{"type":"integer","description":"Number of bytes written to disk."},"previous_content":{"type":"string","description":"The prior file content, capped for diff display."}},"required":["success","file_path","created","bytes_written","previous_content"],"additionalProperties":false})",
        .annotations = {
            .destructive_hint = true,  // overwrites existing file content
            .idempotent_hint  = true,  // writing the same content twice is a no-op in effect
        },
    };
}

std::string WriteFileTool::execute(const std::string& json_args, const core::context::SessionContext& context) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return R"({"error":"Invalid JSON arguments provided to write_file."})";
    }

    std::string_view file_path, content;
    if (doc["file_path"].get(file_path) != simdjson::SUCCESS) {
        return R"({"error":"Missing 'file_path' argument."})";
    }
    if (doc["content"].get(content) != simdjson::SUCCESS) {
        return R"({"error":"Missing 'content' argument."})";
    }

    const std::string path_str(file_path);
    const std::filesystem::path requested_path(path_str);
    std::filesystem::path resolved_path;
    if (const auto access_error =
            detail::check_workspace_access(requested_path, path_str, context, &resolved_path)) {
        return *access_error;
    }

    // -------------------------------------------------------------------------
    // Read previous content BEFORE overwriting so the client can render a diff.
    // We cap at 512 KB to keep the MCP response size reasonable.
    // -------------------------------------------------------------------------
    std::error_code ec;
    const bool file_existed = std::filesystem::exists(resolved_path, ec);

    std::string previous_content;
    if (file_existed) {
        std::ifstream prev_ifs(resolved_path, std::ios::binary);
        if (prev_ifs) {
            constexpr std::size_t kMaxPrev = 512 * 1024;
            std::string buf(kMaxPrev, '\0');
            prev_ifs.read(buf.data(), static_cast<std::streamsize>(kMaxPrev));
            const auto n = static_cast<std::size_t>(prev_ifs.gcount());
            buf.resize(n);
            previous_content = std::move(buf);
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
    ofs << content;
    ofs.close();

    // -------------------------------------------------------------------------
    // Build rich response for Lampo's diff UI.
    // -------------------------------------------------------------------------
    return std::format(
        R"({{"success":true,"file_path":"{}","created":{},"bytes_written":{},"previous_content":"{}"}})",
        core::utils::escape_json_string(resolved_path.string()),
        file_existed ? "false" : "true",
        content.size(),
        core::utils::escape_json_string(previous_content));
}

} // namespace core::tools
