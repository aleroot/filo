#include "FileSearchTool.hpp"
#include "shell/FsUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include <simdjson.h>
#include <filesystem>

namespace core::tools {

using detail::glob_match;
using detail::should_skip_dir;

ToolDefinition FileSearchTool::get_definition() const {
    return {
        .name  = "file_search",
        .title = "File Search",
        .description =
            "Searches for files whose name matches a glob pattern, recursively. "
            "Pure C++ — no external tools required, works on all platforms. "
            "Supports * and ? wildcards. Skips .git, node_modules, build, and similar directories. "
            "Returns up to 100 matching file paths.",
        .parameters = {
            {"pattern", "string", "Filename glob pattern to match (e.g. '*.cpp', 'CMakeLists.txt').", true},
            {"dir",     "string", "Root directory to search. Defaults to '.'.", false}
        },
        .annotations = {
            .read_only_hint  = true,
            .idempotent_hint = true,
        },
    };
}

std::string FileSearchTool::execute(const std::string& json_args) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS)
        return R"({"error":"Invalid JSON arguments provided to file_search."})";

    std::string_view pattern_view;
    if (doc["pattern"].get(pattern_view) != simdjson::SUCCESS)
        return R"({"error":"Missing 'pattern' argument."})";

    std::string dir = ".";
    std::string_view dir_view;
    if (doc["dir"].get(dir_view) == simdjson::SUCCESS)
        dir = std::string(dir_view);

    std::error_code ec;
    if (!std::filesystem::is_directory(dir, ec))
        return R"({"error":"'dir' does not exist or is not a directory."})";

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
                dir,
                std::filesystem::directory_options::skip_permission_denied,
                ec
            );
            if (!ec) {
                for (auto& entry : it) {
                    if (count >= kMaxResults) break;
                    ec.clear();
                    if (entry.is_directory(ec)) {
                        if (should_skip_dir(entry.path()))
                            it.disable_recursion_pending();
                        continue;
                    }
                    if (!entry.is_regular_file(ec)) continue;
                    if (glob_match(pattern_view, entry.path().filename().string())) {
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
