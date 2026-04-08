#include "GrepSearchTool.hpp"
#include "ToolArgumentUtils.hpp"
#include "shell/FsUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include <simdjson.h>
#include <filesystem>
#include <regex>
#include <string>
#include <string_view>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <optional>

// POSIX mmap for zero-copy file reading
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace core::tools {

using detail::glob_match;
using detail::should_skip_dir;

namespace {

struct MatchResult {
    std::string path;
    int64_t     line{};
    std::string text;
};

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

// Returns true when 'pattern' contains no ECMAScript metacharacters, making
// it safe to treat as a literal string and search with string_view::find
// (which libc will typically vectorise with SIMD).
[[nodiscard]] bool is_literal_pattern(std::string_view pattern) noexcept {
    // Every character that has special meaning in ECMAScript regex.
    static constexpr std::string_view kMeta = R"(\.[]{}()*+?^$|)";
    return pattern.find_first_of(kMeta) == std::string_view::npos;
}

// Searches one file for matches of either a literal string or a compiled regex.
// Results are appended to 'out'. Returns early once total_found >= max_results.
//
// Uses mmap for file reading:
//   - Avoids userspace buffering overhead of ifstream / getline.
//   - Lets memchr scan for newlines using SIMD (via libc).
//   - MADV_SEQUENTIAL hints the kernel to prefetch pages ahead of the cursor.
void search_file(
    const std::filesystem::path& fpath,
    bool                         literal_mode,
    std::string_view             literal_str,
    const std::regex&            re,
    size_t                       max_results,
    std::atomic<size_t>&         total_found,
    std::vector<MatchResult>&    out
) {
    const int fd = ::open(fpath.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) return;

    struct stat st{};
    if (::fstat(fd, &st) < 0 || st.st_size <= 0) {
        ::close(fd);
        return;
    }

    const size_t sz  = static_cast<size_t>(st.st_size);
    void* const  raw = ::mmap(nullptr, sz, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (raw == MAP_FAILED) return;

    ::madvise(raw, sz, MADV_SEQUENTIAL);
    const char* const buf = static_cast<const char*>(raw);

    // Binary-file heuristic: any null byte in the first 512 bytes → skip.
    const size_t probe = std::min(sz, size_t{512});
    if (::memchr(buf, '\0', probe) != nullptr) {
        ::munmap(raw, sz);
        return;
    }

    const char* p     = buf;
    const char* end   = buf + sz;
    int64_t     lineno = 0;
    std::string path_str = fpath.string();

    while (p < end && total_found.load(std::memory_order_relaxed) < max_results) {
        const char* nl       = static_cast<const char*>(::memchr(p, '\n', static_cast<size_t>(end - p)));
        const char* line_end = nl ? nl : end;
        ++lineno;

        std::string_view line(p, static_cast<size_t>(line_end - p));
        // Strip trailing CR for CRLF line endings.
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

        bool matched;
        if (literal_mode) {
            // string_view::find on modern libc gets SIMD-optimised (SSE4.2 / AVX2).
            matched = (line.find(literal_str) != std::string_view::npos);
        } else {
            // regex_search on const regex is thread-safe per the C++ standard.
            matched = std::regex_search(line.cbegin(), line.cend(), re);
        }

        if (matched) {
            out.push_back({path_str, lineno, std::string(line)});
            total_found.fetch_add(1, std::memory_order_relaxed);
        }

        p = nl ? nl + 1 : end;
    }

    ::munmap(raw, sz);
}

} // namespace

ToolDefinition GrepSearchTool::get_definition() const {
    return {
        .name  = "grep_search",
        .title = "Grep Search",
        .description =
            "Searches for a regular expression pattern within file contents, recursively. "
            "Pure C++ — no external tools required, works on all platforms. "
            "Skips .git, node_modules, build, and similar directories. "
            "Returns up to 100 matching lines, each with 'path', 'line' (1-based), and 'text'.",
        .parameters = {
            {"pattern",         "string", "ECMAScript regex pattern to search for.", true},
            {"path",            "string", "Root directory to search. Defaults to '.'.", false},
            {"include_pattern", "string",
             "Glob pattern to restrict searched files (e.g. '*.cpp', '**/example/**/*.kt', 'modules/core'). "
             "Patterns with path separators match against relative paths; plain directory patterns include files beneath that directory.",
             false}
        },
        .output_schema =
            R"({"type":"object","properties":{"matches":{"type":"array","items":{"type":"object","properties":{"path":{"type":"string"},"line":{"type":"integer"},"text":{"type":"string"}},"required":["path","line","text"],"additionalProperties":false},"description":"Matching lines with file path and 1-based line number."}},"required":["matches"],"additionalProperties":false})",
        .annotations = {
            .read_only_hint  = true,
            .idempotent_hint = true,
        },
    };
}

std::string GrepSearchTool::execute(const std::string& json_args, const core::context::SessionContext& context) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS)
        return R"({"error":"Invalid JSON arguments provided to grep_search."})";

    if (const auto validation_error =
            detail::validate_object_arguments(doc, "grep_search", {"pattern", "path", "include_pattern"})) {
        return *validation_error;
    }

    std::string_view pattern;
    if (doc["pattern"].get(pattern) != simdjson::SUCCESS)
        return R"({"error":"Missing 'pattern' argument."})";

    std::string dir_path = ".";
    std::string_view dir_v;
    if (doc["path"].get(dir_v) == simdjson::SUCCESS)
        dir_path = std::string(dir_v);

    std::string_view include_v;
    const bool has_include = (doc["include_pattern"].get(include_v) == simdjson::SUCCESS);
    const std::string include_pattern = has_include ? normalize_glob_pattern(include_v) : std::string{};
    const bool include_has_separator = has_include && include_pattern.find('/') != std::string::npos;
    const bool include_has_wildcards =
        has_include && include_pattern.find_first_of("*?") != std::string::npos;

    std::filesystem::path resolved_dir;
    if (const auto access_error =
            detail::check_workspace_access(dir_path, dir_path, context, &resolved_dir)) {
        return *access_error;
    }

    std::error_code ec;
    if (!std::filesystem::is_directory(resolved_dir, ec))
        return R"({"error":"'path' does not exist or is not a directory."})";

    std::optional<std::filesystem::path> include_directory_filter;
    if (has_include && !include_has_wildcards) {
        std::filesystem::path candidate = std::filesystem::path(include_pattern);
        if (!candidate.is_absolute()) {
            candidate = resolved_dir / candidate;
        }
        if (std::filesystem::is_directory(candidate, ec)
            && is_subpath(resolved_dir, candidate)) {
            include_directory_filter = candidate.lexically_normal();
        }
    }

    // ── Regex or literal? ───────────────────────────────────────────────────
    const bool literal_mode = is_literal_pattern(pattern);
    const std::string literal_str(pattern);

    std::regex re;
    if (!literal_mode) {
        try {
            re = std::regex(literal_str, std::regex::ECMAScript | std::regex::optimize);
        } catch (const std::regex_error& e) {
            return std::string(R"({"error":"Invalid regex: )") + e.what() + "\"}";
        }
    }

    // ── Phase 1: collect candidate files (single-threaded enumeration) ──────
    std::vector<std::filesystem::path> files;
    files.reserve(512);

    std::filesystem::recursive_directory_iterator it(
        resolved_dir, std::filesystem::directory_options::skip_permission_denied, ec
    );
    if (!ec) {
        const auto end = std::filesystem::recursive_directory_iterator{};
        for (; it != end; ++it) {
            const auto& entry = *it;
            ec.clear();
            if (entry.is_directory(ec)) {
                if (should_skip_dir(entry.path()))
                    it.disable_recursion_pending();
                continue;
            }
            if (!entry.is_regular_file(ec)) continue;
            if (has_include) {
                bool include_match = false;
                if (include_directory_filter.has_value()) {
                    include_match = is_subpath(*include_directory_filter, entry.path());
                } else if (include_has_separator) {
                    include_match = glob_match(
                        include_pattern,
                        relative_generic_path(entry.path(), resolved_dir));
                } else {
                    include_match = glob_match(include_pattern, entry.path().filename().string());
                }
                if (!include_match) continue;
            }
            files.push_back(entry.path());
        }
    }

    // Sort once so output order is deterministic regardless of thread scheduling.
    std::sort(files.begin(), files.end());

    // ── Phase 2: parallel search ─────────────────────────────────────────────
    // Each thread grabs the next unprocessed file via an atomic index (work-stealing).
    // Per-thread result vectors avoid any mutex on the hot path.
    constexpr size_t kMaxResults = 100;
    std::atomic<size_t> total_found{0};
    std::atomic<size_t> next_idx{0};

    const size_t N = std::clamp<size_t>(std::thread::hardware_concurrency(), 1u, 16u);
    std::vector<std::vector<MatchResult>> per_thread(N);

    {
        std::vector<std::thread> workers;
        workers.reserve(N);
        for (size_t tid = 0; tid < N; ++tid) {
            workers.emplace_back([&, tid] {
                auto& local = per_thread[tid];
                for (;;) {
                    const size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                    if (idx >= files.size()) break;
                    if (total_found.load(std::memory_order_relaxed) >= kMaxResults) break;
                    search_file(files[idx], literal_mode, literal_str, re,
                                kMaxResults, total_found, local);
                }
            });
        }
        for (auto& w : workers) w.join();
    }

    // ── Phase 3: merge, sort, trim, serialize ───────────────────────────────
    // Collect pointers into a single flat view, sort by (path, line), then trim.
    std::vector<const MatchResult*> all;
    {
        size_t total = 0;
        for (const auto& v : per_thread) total += v.size();
        all.reserve(total);
        for (const auto& v : per_thread)
            for (const auto& r : v) all.push_back(&r);
    }

    std::sort(all.begin(), all.end(), [](const MatchResult* a, const MatchResult* b) {
        if (a->path != b->path) return a->path < b->path;
        return a->line < b->line;
    });
    if (all.size() > kMaxResults) all.resize(kMaxResults);

    core::utils::JsonWriter w(2048);
    {
        auto _root = w.object();
        w.key("matches");
        {
            auto _arr = w.array();
            bool first = true;
            for (const auto* r : all) {
                if (!first) w.comma();
                first = false;
                auto _item = w.object();
                w.kv_str("path", r->path).comma()
                 .kv_num("line", r->line).comma()
                 .kv_str("text", r->text);
            }
        }
    }
    return std::move(w).take();
}

} // namespace core::tools
