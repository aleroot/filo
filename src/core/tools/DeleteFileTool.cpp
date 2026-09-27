#include "DeleteFileTool.hpp"
#include "ToolArgumentUtils.hpp"
#include "ToolNames.hpp"
#include "../utils/JsonUtils.hpp"
#include "../workspace/SessionWorkspace.hpp"
#include <simdjson.h>
#include <algorithm>
#include <filesystem>
#include <format>

namespace core::tools {

ToolDefinition DeleteFileTool::get_definition() const {
    return {
        .name  = std::string(names::kDeleteFile),
        .title = "Delete File or Folder",
        .description =
            "Permanently delete a file or directory; nonempty directories require recursive=true.",
        .parameters = {
            {"file_path", "string", "Path to delete.", true},
            {"recursive", "boolean", "Delete contents (default false).", false}
        },
        .output_schema =
            R"({"type":"object","properties":{"success":{"type":"boolean","description":"Whether the deletion completed successfully."},"deleted":{"type":"string","description":"The path that was removed."}},"required":["success","deleted"],"additionalProperties":false})",
        .annotations = {
            .destructive_hint = true,  // permanently removes data
            .idempotent_hint  = true,  // deleting an already-deleted path is effectively the same end state
        },
    };
}

std::string DeleteFileTool::execute(const std::string& json_args, const core::context::SessionContext& context) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS) {
        return R"({"error":"Invalid JSON arguments for delete_file."})";
    }

    std::string_view path_v;
    if (doc["file_path"].get(path_v) != simdjson::SUCCESS) {
        return R"({"error":"Missing required argument 'file_path'."})";
    }
    bool recursive = false;
    if (simdjson::dom::element option; doc["recursive"].get(option) == simdjson::SUCCESS
        && option.get(recursive) != simdjson::SUCCESS) {
        return R"({"error":"Argument 'recursive' must be a boolean."})";
    }

    const std::string path_str(path_v);
    if (path_str.empty()) {
        return R"({"error":"Argument 'file_path' must not be empty."})";
    }
    const std::filesystem::path requested_path(path_str);
    std::filesystem::path resolved_path;
    if (const auto access_error =
            detail::check_workspace_access(
                requested_path,
                path_str,
                context,
                &resolved_path,
                names::kDeleteFile)) {
        return *access_error;
    }
    std::error_code ec;

    if (!std::filesystem::exists(resolved_path, ec)) {
        return std::format(R"({{"error":"Path does not exist: {}"}})",
                           core::utils::escape_json_string(path_str));
    }

    if (recursive) {
        const auto& workspace = context.workspace_view();
        std::filesystem::path literal_path = requested_path.is_absolute()
            ? requested_path
            : workspace.primary().empty()
                ? std::filesystem::absolute(requested_path, ec)
                : workspace.primary() / requested_path;
        if (ec) {
            return std::format(R"({{"error":"Failed to resolve '{}': {}"}})",
                               core::utils::escape_json_string(path_str),
                               core::utils::escape_json_string(ec.message()));
        }
        literal_path = literal_path.lexically_normal();
        if (literal_path.filename().empty()) literal_path = literal_path.parent_path();
        if (std::filesystem::is_symlink(std::filesystem::symlink_status(literal_path, ec))) {
            return R"({"error":"Recursive deletion of a symlink is not allowed."})";
        }
        if (ec) {
            return std::format(R"({{"error":"Failed to inspect '{}': {}"}})",
                               core::utils::escape_json_string(path_str),
                               core::utils::escape_json_string(ec.message()));
        }
        if (!std::filesystem::is_directory(resolved_path, ec) || ec) {
            return R"({"error":"Recursive deletion requires a directory."})";
        }
        const auto contains_root = [&](const std::filesystem::path& root) {
            return !root.empty()
                && core::workspace::SessionWorkspace::is_subpath(resolved_path, root);
        };
        if (resolved_path == resolved_path.root_path()
            || contains_root(workspace.primary())
            || std::ranges::any_of(workspace.additional(), contains_root)
            || std::ranges::any_of(workspace.scratch().writable_roots(), contains_root)) {
            return R"({"error":"Cannot recursively delete a workspace or scratch root."})";
        }

        // A parent can be permitted while one of its children is excluded by
        // workspace scope, .agentignore, steering, or a per-tool path policy.
        // Check every entry before changing the tree.
        std::filesystem::recursive_directory_iterator it(resolved_path, ec), end;
        if (ec) {
            return std::format(R"({{"error":"Failed to inspect '{}': {}"}})",
                               core::utils::escape_json_string(path_str),
                               core::utils::escape_json_string(ec.message()));
        }
        for (; it != end; it.increment(ec)) {
            if (ec) break;
            const auto& entry = *it;
            if (entry.is_symlink(ec)) {
                return std::format(R"({{"error":"Recursive deletion contains a symlink: {}"}})",
                                   core::utils::escape_json_string(entry.path().string()));
            }
            if (ec) break;
            if (const auto access_error = detail::check_workspace_access(
                    entry.path(), entry.path().string(), context, nullptr,
                    names::kDeleteFile, true, &entry)) {
                return *access_error;
            }
        }
        if (ec) {
            return std::format(R"({{"error":"Failed to inspect '{}': {}"}})",
                               core::utils::escape_json_string(path_str),
                               core::utils::escape_json_string(ec.message()));
        }
        std::filesystem::remove_all(resolved_path, ec);
    } else {
        std::filesystem::remove(resolved_path, ec);
    }
    if (ec) {
        return std::format(R"({{"error":"Failed to delete '{}': {}"}})",
                           core::utils::escape_json_string(path_str),
                           core::utils::escape_json_string(ec.message()));
    }

    return std::format(R"({{"success":true,"deleted":"{}"}})",
                       core::utils::escape_json_string(resolved_path.string()));
}

} // namespace core::tools
