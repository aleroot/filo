#include "FileSearchTool.hpp"
#include "ToolArgumentUtils.hpp"
#include "shell/FsUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include <simdjson.h>
#include <algorithm>
#include <filesystem>
#include <optional>

namespace core::tools {

using detail::glob_match;
using detail::should_skip_dir;

namespace {

[[nodiscard]] std::string normalize_glob_pattern(std::string_view pattern) {
    std::string normalized(pattern);
    std::replace(normalized.begin(), normalized.end(), '\\', '/');
    return normalized;
}

[[nodiscard]] bool is_subpath(const std::filesystem::path& root,
                              const std::filesystem::path& target) {
    const auto normalized_root = root.lexically_normal();
    const auto normalized_target = target.lexically_normal();

    auto root_it = normalized_root.begin();
    auto target_it = normalized_target.begin();
    while (root_it != normalized_root.end() && target_it != normalized_target.end()) {
        if (*root_it != *target_it) {
            return false;
        }
        ++root_it;
        ++target_it;
    }
    return root_it == normalized_root.end();
}

[[nodiscard]] std::string relative_generic_path(const std::filesystem::path& file,
                                                const std::filesystem::path& root) {
    std::error_code ec;
    const auto relative = std::filesystem::relative(file, root, ec);
    if (!ec) {
        return relative.generic_string();
    }
    return file.generic_string();
}

} // namespace

ToolDefinition FileSearchTool::get_definition() const {
    return {
        .name  = "file_search",
        .title = "File Search",
        .description =
            "Searches for files that match a glob pattern, recursively. "
            "Pure C++ — no external tools required, works on all platforms. "
            "Supports * and ? wildcards. Skips .git, node_modules, build, and similar directories. "
            "Returns up to 100 matching file paths.",
        .parameters = {
            {"pattern", "string", "Glob pattern to match (e.g. '*.cpp', '**/example/**/*.kt', 'src/core').", true},
            {"path",    "string", "Root directory to search. Defaults to '.'.", false}
        },
        .output_schema =
            R"({"type":"object","properties":{"files":{"type":"array","items":{"type":"string"},"description":"Matching file paths."}},"required":["files"],"additionalProperties":false})",
        .annotations = {
            .read_only_hint  = true,
            .idempotent_hint = true,
        },
    };
}

std::string FileSearchTool::execute(const std::string& json_args, const core::context::SessionContext& context) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS)
        return R"({"error":"Invalid JSON arguments provided to file_search."})";

    if (const auto validation_error =
            detail::validate_object_arguments(doc, "file_search", {"pattern", "path"})) {
        return *validation_error;
    }

    std::string_view pattern_view;
    if (doc["pattern"].get(pattern_view) != simdjson::SUCCESS)
        return R"({"error":"Missing 'pattern' argument."})";
    const std::string pattern = normalize_glob_pattern(pattern_view);
    const bool pattern_has_separator = pattern.find('/') != std::string::npos;
    const bool pattern_has_wildcards = pattern.find_first_of("*?") != std::string::npos;

    std::string dir = ".";
    std::string_view dir_view;
    if (doc["path"].get(dir_view) == simdjson::SUCCESS)
        dir = std::string(dir_view);

    std::filesystem::path resolved_dir;
    if (const auto access_error =
            detail::check_workspace_access(dir, dir, context, &resolved_dir)) {
        return *access_error;
    }

    std::error_code ec;
    if (!std::filesystem::is_directory(resolved_dir, ec))
        return R"({"error":"'path' does not exist or is not a directory."})";

    std::optional<std::filesystem::path> directory_filter;
    if (!pattern_has_wildcards) {
        std::filesystem::path candidate = std::filesystem::path(pattern);
        if (!candidate.is_absolute()) {
            candidate = resolved_dir / candidate;
        }
        if (std::filesystem::is_directory(candidate, ec)
            && is_subpath(resolved_dir, candidate)) {
            directory_filter = candidate.lexically_normal();
        }
    }

    constexpr size_t kMaxResults = 100;
    core::utils::JsonWriter w(512);
    {
        auto _root = w.object();
        w.key("files");
        {
            auto _arr = w.array();
            bool first = true;
            size_t count = 0;

            std::filesystem::recursive_directory_iterator it(
                resolved_dir,
                std::filesystem::directory_options::skip_permission_denied,
                ec
            );
            if (!ec) {
                const auto end = std::filesystem::recursive_directory_iterator{};
                for (; it != end; ++it) {
                    if (count >= kMaxResults) break;
                    const auto& entry = *it;
                    ec.clear();
                    if (entry.is_directory(ec)) {
                        if (should_skip_dir(entry.path()))
                            it.disable_recursion_pending();
                        continue;
                    }
                    if (!entry.is_regular_file(ec)) continue;
                    const std::string relative_path = relative_generic_path(entry.path(), resolved_dir);
                    bool matches = false;
                    if (directory_filter.has_value()) {
                        matches = is_subpath(*directory_filter, entry.path());
                    } else if (pattern_has_separator) {
                        matches = glob_match(pattern, relative_path);
                    } else {
                        matches = glob_match(pattern, entry.path().filename().string());
                    }

                    if (matches) {
                        if (!first) w.comma();
                        first = false;
                        w.str(entry.path().string());
                        ++count;
                    }
                }
            }
        }
    }
    return std::move(w).take();
}

} // namespace core::tools
