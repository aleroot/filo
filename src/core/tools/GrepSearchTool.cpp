#include "GrepSearchTool.hpp"
#include "ToolArgumentUtils.hpp"
#include "ToolNames.hpp"
#include "shell/FsUtils.hpp"
#include "../utils/JsonUtils.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"
#include "../workspace/PathVisibility.hpp"
#include <simdjson.h>
#include <filesystem>
#include <regex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <thread>
#include <atomic>
#include <cstdint>
#include <algorithm>
#include <array>
#include <cctype>
#include <optional>
#include <format>
#include <system_error>

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
    std::size_t file_index{};
    int64_t     line{};
    std::string text;
};

/// Longest text returned for one matching line. Minified and generated files
/// put megabytes on a single line; returned whole, one match cost megabytes of
/// memory and of response. Longer lines are clipped to a window around the
/// match, so ordinary source lines are never affected.
constexpr std::size_t kMaxMatchTextBytes = 1024;
constexpr std::string_view kClipMarker = "\u2026";

[[nodiscard]] bool is_utf8_continuation(char c) noexcept {
    return (static_cast<unsigned char>(c) & 0xC0u) == 0x80u;
}

[[nodiscard]] std::string match_text(std::string_view line, std::size_t match_offset) {
    if (line.size() <= kMaxMatchTextBytes) return std::string(line);

    // Keep a little context before the match and the rest after it.
    std::size_t begin = match_offset > kMaxMatchTextBytes / 4
        ? match_offset - kMaxMatchTextBytes / 4
        : 0;
    begin = std::min(begin, line.size() - kMaxMatchTextBytes);
    std::size_t end = begin + kMaxMatchTextBytes;
    // Snap inwards to code point boundaries so the clip never splits one.
    while (begin < end && is_utf8_continuation(line[begin])) ++begin;
    while (end > begin && end < line.size() && is_utf8_continuation(line[end])) --end;

    std::string out;
    out.reserve(end - begin + 2 * kClipMarker.size());
    if (begin > 0) out += kClipMarker;
    out.append(line.substr(begin, end - begin));
    if (end < line.size()) out += kClipMarker;
    return out;
}

struct SearchScope {
    std::filesystem::path traversal_root;
    std::filesystem::path include_root;
    std::optional<std::filesystem::path> single_file;
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

[[nodiscard]] std::string searchable_path_error(std::string_view path,
                                                std::string_view reason) {
    return std::format(
        R"({{"error":"Cannot search path '{}': {}."}})",
        core::utils::escape_json_string(path),
        core::utils::escape_json_string(reason));
}

[[nodiscard]] std::optional<std::string> resolve_search_scope(
    std::string_view requested_path,
    const std::filesystem::path& resolved_path,
    const std::filesystem::path& workspace_root,
    SearchScope& scope) {
    std::error_code ec;
    const auto status = std::filesystem::status(resolved_path, ec);
    if (ec) {
        if (ec == std::errc::no_such_file_or_directory
            || ec == std::errc::not_a_directory) {
            return searchable_path_error(requested_path, "path does not exist");
        }
        return searchable_path_error(
            requested_path,
            std::format("failed to inspect path ({})", ec.message()));
    }

    if (status.type() == std::filesystem::file_type::not_found) {
        return searchable_path_error(requested_path, "path does not exist");
    }

    if (std::filesystem::is_directory(status)) {
        scope.traversal_root = resolved_path;
        scope.include_root = resolved_path;
        scope.single_file.reset();
        return std::nullopt;
    }

    if (std::filesystem::is_regular_file(status)) {
        scope.traversal_root = resolved_path.parent_path();
        if (scope.traversal_root.empty()) {
            scope.traversal_root = ".";
        }
        scope.include_root = !workspace_root.empty() && is_subpath(workspace_root, resolved_path)
            ? workspace_root
            : scope.traversal_root;
        scope.single_file = resolved_path;
        return std::nullopt;
    }

    return searchable_path_error(requested_path, "path is not a regular file or directory");
}

// Returns true when 'pattern' contains no ECMAScript metacharacters, making
// it safe to treat as a literal string and search with string_view::find
// (which libc will typically vectorise with SIMD).
[[nodiscard]] bool is_literal_pattern(std::string_view pattern) noexcept {
    // Every character that has special meaning in ECMAScript regex.
    static constexpr std::string_view kMeta = R"(\.[]{}()*+?^$|)";
    return pattern.find_first_of(kMeta) == std::string_view::npos;
}

// `nosubs` lets standard-library regex engines omit capture bookkeeping when
// callers only need a boolean result. It cannot be used when the expression
// contains a numeric backreference because those depend on captured text.
//
// ECMAScript backreferences are decimal escapes outside bracket expressions.
// Keep this deliberately conservative: any \1..\9 disables the optimization,
// even if the complete decimal escape would ultimately be rejected.
[[nodiscard]] bool regex_uses_backreference(std::string_view pattern) noexcept {
    bool in_bracket_expression = false;
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        const char current = pattern[i];
        if (current == '\\') {
            if (i + 1 < pattern.size()) {
                const char escaped = pattern[++i];
                if (!in_bracket_expression && escaped >= '1' && escaped <= '9') {
                    return true;
                }
            }
            continue;
        }
        if (current == '[' && !in_bracket_expression) {
            in_bracket_expression = true;
        } else if (current == ']' && in_bracket_expression) {
            in_bracket_expression = false;
        }
    }
    return false;
}

// Searches one file for matches of either a literal string or a compiled regex.
/// The first max_results matches in sorted-file order are the answer, so a
/// file only matters while the files before it hold fewer than that. Match
/// counts per file live in a Fenwick tree of atomics: counts only grow, so any
/// prefix sum read is a lower bound on what those files will finally hold.
/// Once it reaches max_results for the files before j, file j is skipped or
/// abandoned mid-search. A file that can still contribute is always searched
/// in full, which keeps the result identical however the threads were
/// scheduled, while stopping as early as the scheduling allows.
class MatchCounts {
public:
    MatchCounts(std::size_t file_count, std::size_t max_results)
        : tree_(file_count + 1), max_results_(max_results) {}

    void record(std::size_t file_index) noexcept {
        for (std::size_t i = file_index + 1; i < tree_.size(); i += i & (~i + 1)) {
            tree_[i].fetch_add(1, std::memory_order_relaxed);
        }
    }

    /// False once the files before @p file_index already hold max_results.
    [[nodiscard]] bool can_contribute(std::size_t file_index) const noexcept {
        std::size_t found = 0;
        for (std::size_t i = file_index; i > 0; i -= i & (~i + 1)) {
            found += tree_[i].load(std::memory_order_relaxed);
        }
        return found < max_results_;
    }

private:
    std::vector<std::atomic<std::uint32_t>> tree_;
    const std::size_t max_results_;
};

/// Literals of which every match contains at least one, found with a
/// vectorised substring search before any per-line work. Literal mode holds
/// the pattern itself; a regex holds one literal per top-level alternative.
class NeedleSet {
public:
    static constexpr std::size_t kMaxNeedles = 8;

    NeedleSet() = default;
    NeedleSet(std::vector<std::string_view> texts, bool ignore_case) : texts_(std::move(texts)) {
        if (ignore_case) {
            caseless_.reserve(texts_.size());
            for (const auto text : texts_) caseless_.emplace_back(text);
        }
    }

    [[nodiscard]] bool empty() const noexcept { return texts_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return texts_.size(); }
    [[nodiscard]] std::string_view text(std::size_t i) const noexcept { return texts_[i]; }

    [[nodiscard]] std::size_t find(std::size_t i, std::string_view data, std::size_t from) const noexcept {
        return caseless_.empty() ? data.find(texts_[i], from) : caseless_[i].find(data, from);
    }

private:
    std::vector<std::string_view> texts_;
    std::vector<core::utils::str::CaseInsensitiveAsciiSearcher> caseless_;
};

/// Earliest occurrence of any needle in one file. Each needle's next position
/// is cached, so the buffer is scanned once per needle rather than once per
/// needle per candidate line.
class NeedleCursor {
public:
    NeedleCursor(const NeedleSet& needles, std::string_view data) : needles_(needles), data_(data) {
        for (std::size_t i = 0; i < needles_.size(); ++i) next_[i] = needles_.find(i, data_, 0);
    }

    /// Position of the earliest needle at or after @p from, and which one.
    [[nodiscard]] std::pair<std::size_t, std::size_t> find(std::size_t from) noexcept {
        std::size_t best = std::string_view::npos;
        std::size_t which = 0;
        for (std::size_t i = 0; i < needles_.size(); ++i) {
            if (next_[i] != std::string_view::npos && next_[i] < from) {
                next_[i] = needles_.find(i, data_, from);
            }
            if (next_[i] < best) {
                best = next_[i];
                which = i;
            }
        }
        return {best, which};
    }

private:
    const NeedleSet& needles_;
    std::string_view data_;
    std::array<std::size_t, NeedleSet::kMaxNeedles> next_{};
};

[[nodiscard]] bool is_plain_regex_char(char c) noexcept {
    const auto uc = static_cast<unsigned char>(c);
    if (uc >= 0x80) return false;
    if (std::isalnum(uc) != 0) return true;
    return std::string_view(" _:,-=<>!@#%&~'\";/`").find(c) != std::string_view::npos;
}

/// The literal every match of one alternative must contain: its leading run of
/// plain characters after any zero-width `^`, `\b` or `\B`. A quantifier on
/// the run's last character drops that character. Empty when none is proven.
[[nodiscard]] std::string_view required_branch_literal(std::string_view branch) {
    std::size_t begin = 0;
    for (;;) {
        if (branch.substr(begin).starts_with('^')) {
            begin += 1;
        } else if (branch.substr(begin).starts_with("\\b") || branch.substr(begin).starts_with("\\B")) {
            begin += 2;
        } else {
            break;
        }
    }
    std::size_t end = begin;
    while (end < branch.size() && is_plain_regex_char(branch[end])) ++end;
    if (end < branch.size() && std::string_view("*+?{").find(branch[end]) != std::string_view::npos) {
        if (end == begin) return {};
        --end;
    }
    return branch.substr(begin, end - begin);
}

/// Splits @p pattern at alternations outside groups, classes and escapes, and
/// returns the literal each alternative requires. Empty when any alternative
/// has none or the pattern does not parse cleanly, so the result is
/// conservative: a line containing none of the literals cannot match. This
/// lets std::regex, which allocates on every call, run only on candidates.
[[nodiscard]] std::vector<std::string_view> required_regex_literals(std::string_view pattern) {
    std::vector<std::string_view> literals;
    int depth = 0;
    bool in_class = false;
    std::size_t branch_start = 0;
    const auto close_branch = [&](std::size_t branch_end) {
        const auto literal = required_branch_literal(
            pattern.substr(branch_start, branch_end - branch_start));
        if (literal.empty() || literals.size() == NeedleSet::kMaxNeedles) return false;
        literals.push_back(literal);
        branch_start = branch_end + 1;
        return true;
    };
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        const char c = pattern[i];
        if (c == '\\') {
            ++i;
        } else if (in_class) {
            if (c == ']') in_class = false;
        } else if (c == '[') {
            // Whether a leading ']' closes the class or belongs to it is
            // exactly where a misreading could hide an alternation.
            if (pattern.substr(i + 1).starts_with(']') || pattern.substr(i + 1).starts_with("^]")) {
                return {};
            }
            in_class = true;
        } else if (c == '(') {
            ++depth;
        } else if (c == ')') {
            if (--depth < 0) return {};
        } else if (c == '|' && depth == 0) {
            if (!close_branch(i)) return {};
        }
    }
    if (in_class || depth != 0 || !close_branch(pattern.size())) return {};
    return literals;
}

/// Where @p re matches in @p line, or nullopt. The exact position is only
/// computed for overlong lines, which are clipped around it.
[[nodiscard]] std::optional<std::size_t> regex_match_offset(std::string_view line,
                                                            const std::regex& re) {
    // regex_search on const regex is thread-safe per the C++ standard.
    if (!std::regex_search(line.cbegin(), line.cend(), re)) return std::nullopt;
    if (line.size() <= kMaxMatchTextBytes) return 0;
    std::match_results<std::string_view::const_iterator> found;
    if (std::regex_search(line.cbegin(), line.cend(), found, re)) {
        return static_cast<std::size_t>(found.position(0));
    }
    return 0;
}

// Results are appended to 'out'. A file stops as soon as it and the files
// before it are known to hold enough matches; see MatchCounts.
//
// Uses mmap for file reading:
//   - Avoids userspace buffering overhead of ifstream / getline.
//   - Lets memchr scan for newlines using SIMD (via libc).
//   - MADV_SEQUENTIAL hints the kernel to prefetch pages ahead of the cursor.
void search_file(
    const std::filesystem::path& fpath,
    std::size_t                  file_index,
    bool                         literal_mode,
    const NeedleSet&             needles,
    const std::regex&            re,
    MatchCounts&                 counts,
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

    // True while this file's matches can still be among the results. Counts
    // include this file's own, so this also caps a single file.
    const auto wanted = [&counts, file_index] { return counts.can_contribute(file_index + 1); };
    const auto record = [&](int64_t lineno, std::string_view line, std::size_t offset) {
        out.push_back({file_index, lineno, match_text(line, offset)});
        counts.record(file_index);
    };

    // Search the needles across the mapped buffer instead of restarting a
    // search for every line. Matches are normally sparse, so this avoids both
    // newline discovery and matcher setup for every non-matching line. A
    // literal containing a newline could never match a line, either.
    if (!needles.empty()) {
        if (literal_mode && needles.text(0).find('\n') != std::string_view::npos) {
            ::munmap(raw, sz);
            return;
        }

        const std::string_view data(buf, sz);
        NeedleCursor cursor(needles, data);
        size_t search_from = 0;
        size_t line_start = 0;
        int64_t lineno = 1;

        while (search_from < sz && wanted()) {
            const auto [match, which] = cursor.find(search_from);
            if (match == std::string_view::npos) break;

            while (line_start < match) {
                const char* nl = static_cast<const char*>(
                    ::memchr(buf + line_start, '\n', match - line_start));
                if (nl == nullptr) break;
                line_start = static_cast<size_t>(nl - buf) + 1;
                ++lineno;
            }

            const char* nl = static_cast<const char*>(
                ::memchr(buf + match, '\n', sz - match));
            size_t line_end = nl == nullptr ? sz : static_cast<size_t>(nl - buf);

            // A needle containing CR can still begin before the CR in a CRLF
            // line.  Match against the same CR-stripped view as the general
            // line path below.
            size_t text_end = line_end;
            if (text_end > line_start && buf[text_end - 1] == '\r') --text_end;
            const std::string_view line(buf + line_start, text_end - line_start);

            std::optional<std::size_t> offset;
            if (literal_mode) {
                if (match + needles.text(which).size() <= text_end) offset = match - line_start;
            } else {
                offset = regex_match_offset(line, re);
            }
            if (offset.has_value()) record(lineno, line, *offset);

            // A regex has judged the whole line; a literal only this occurrence.
            if (offset.has_value() || !literal_mode) {
                if (line_end == sz) break;
                search_from = line_end + 1;
                line_start = search_from;
                ++lineno;
            } else {
                search_from = match + 1;
            }
        }

        ::munmap(raw, sz);
        return;
    }

    const char* p     = buf;
    const char* end   = buf + sz;
    int64_t     lineno = 0;
    while (p < end) {
        // Checked every 64 lines: the prefix sum costs a few atomic loads.
        if ((lineno & 63) == 0 && !wanted()) break;
        const char* nl       = static_cast<const char*>(::memchr(p, '\n', static_cast<size_t>(end - p)));
        const char* line_end = nl ? nl : end;
        ++lineno;

        std::string_view line(p, static_cast<size_t>(line_end - p));
        // Strip trailing CR for CRLF line endings.
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);

        std::optional<std::size_t> offset;
        if (literal_mode) {
            // Only an empty literal reaches this loop; it matches every line.
            offset = 0;
        } else {
            offset = regex_match_offset(line, re);
        }
        if (offset.has_value()) {
            record(lineno, line, *offset);
            if (!wanted()) break;
        }

        p = nl ? nl + 1 : end;
    }

    ::munmap(raw, sz);
}

} // namespace

ToolDefinition GrepSearchTool::get_definition() const {
    return {
        .name  = std::string(names::kGrepSearch),
        .title = "Grep Search",
        .description =
            "Find up to 100 matching lines recursively, skipping generated and VCS directories. "
            "Regex syntax is C++ ECMAScript only; inline flags such as (?i) are unsupported.",
        .parameters = {
            {"pattern",         "string",
             "Literal text or C++ ECMAScript-only regular expression. Use ignore_case instead of inline flags such as (?i).",
             true},
            {"path",            "string", "File or search root; defaults to the workspace.", false},
            {"include_pattern", "string",
             "Optional file glob such as '*.cpp' or '**/tests/*.swift'.",
             false},
            {"ignore_case",     "boolean",
             "Case-insensitive matching; defaults to false.",
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
            detail::validate_object_arguments(
                doc,
                names::kGrepSearch,
                {"pattern", "path", "include_pattern", "ignore_case"})) {
        return *validation_error;
    }

    std::string_view pattern;
    if (doc["pattern"].get(pattern) != simdjson::SUCCESS)
        return R"({"error":"Missing 'pattern' argument."})";

    bool ignore_case = false;
    core::utils::json::ignore_error(doc["ignore_case"].get(ignore_case));

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

    std::filesystem::path resolved_path;
    if (const auto access_error =
            detail::check_workspace_access(
                dir_path,
                dir_path,
                context,
                &resolved_path,
                names::kGrepSearch)) {
        return *access_error;
    }

    SearchScope scope;
    if (const auto scope_error = resolve_search_scope(
            dir_path,
            resolved_path,
            context.workspace_view().primary(),
            scope)) {
        return *scope_error;
    }

    std::optional<std::filesystem::path> include_directory_filter;
    if (has_include && !include_has_wildcards) {
        std::filesystem::path candidate = std::filesystem::path(include_pattern);
        if (!candidate.is_absolute()) {
            candidate = scope.include_root / candidate;
        }
        std::error_code ec;
        if (std::filesystem::is_directory(candidate, ec)
            && is_subpath(scope.include_root, candidate)) {
            include_directory_filter = candidate.lexically_normal();
        }
    }

    // ── Regex or literal? ───────────────────────────────────────────────────
    // Plain ASCII patterns can avoid std::regex even when matching without
    // case. Non-ASCII case folding remains on std::regex to preserve its
    // locale-aware behavior.
    const bool literal_mode = is_literal_pattern(pattern)
        && (!ignore_case || core::utils::ascii::is_ascii(pattern));
    const std::string literal_str(pattern);
    // Literal mode searches for the pattern itself; a regex for the literals
    // one of which each of its matches must contain, when they can be proven.
    const NeedleSet needles(
        literal_mode
            ? (literal_str.empty() ? std::vector<std::string_view>{}
                                   : std::vector<std::string_view>{literal_str})
            : required_regex_literals(literal_str),
        ignore_case);

    std::regex re;
    if (!literal_mode) {
        try {
            auto flags = std::regex::ECMAScript | std::regex::optimize;
            if (ignore_case) {
                flags |= std::regex::icase;
            }
            if (!regex_uses_backreference(pattern)) {
                flags |= std::regex::nosubs;
            }
            re = std::regex(literal_str, flags);
        } catch (const std::regex_error& e) {
            return std::format(
                R"({{"error":"Invalid C++ ECMAScript regex: {} Inline flags such as (?i) are unsupported; set ignore_case to true for case-insensitive matching."}})",
                core::utils::escape_json_string(e.what()));
        }
    }

    // ── Phase 1: collect candidate files (single-threaded enumeration) ──────
    std::vector<std::filesystem::path> files;
    files.reserve(512);

    auto add_candidate = [&](const std::filesystem::path& file) {
        if (has_include) {
            bool include_match = false;
            if (include_directory_filter.has_value()) {
                include_match = is_subpath(*include_directory_filter, file);
            } else if (include_has_separator) {
                include_match = glob_match(
                    include_pattern,
                    relative_generic_path(file, scope.include_root));
            } else {
                include_match = glob_match(include_pattern, file.filename().string());
            }
            if (!include_match) return;
        }
        files.push_back(file);
    };

    if (scope.single_file.has_value()) {
        add_candidate(*scope.single_file);
    } else {
        const auto visible_files = core::workspace::collect_visible_regular_files(
            scope.traversal_root,
            context,
            [](const std::filesystem::path& directory) {
                return should_skip_dir(directory);
            });
        for (const auto& file : visible_files) {
            add_candidate(file);
        }
    }

    // Sort once so output order is deterministic regardless of thread
    // scheduling. Plain string order: it is the order results are reported
    // in, and it avoids re-parsing path components on every comparison.
    std::sort(files.begin(), files.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.native() < rhs.native();
    });

    // ── Phase 2: parallel search ─────────────────────────────────────────────
    // Each thread grabs the next unprocessed file via an atomic index (work-stealing).
    // Per-thread result vectors avoid any mutex on the hot path.
    constexpr size_t kMaxResults = 100;
    MatchCounts counts(files.size(), kMaxResults);
    std::atomic<size_t> next_idx{0};

    const size_t N = std::clamp<size_t>(
        std::min<size_t>(std::thread::hardware_concurrency(), files.size()), 1u, 16u);
    std::vector<std::vector<MatchResult>> per_thread(N);

    {
        std::vector<std::thread> workers;
        workers.reserve(N);
        for (size_t tid = 0; tid < N; ++tid) {
            workers.emplace_back([&, tid] {
                auto& local = per_thread[tid];
                for (;;) {
                    const size_t idx = next_idx.fetch_add(1, std::memory_order_relaxed);
                    // Claims and counts only grow, so once a file cannot
                    // contribute no later claim can either.
                    if (idx >= files.size() || !counts.can_contribute(idx)) break;
                    search_file(files[idx], idx, literal_mode, needles, re, counts, local);
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
        if (a->file_index != b->file_index) return a->file_index < b->file_index;
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
                w.kv_str("path", files[r->file_index].native()).comma()
                 .kv_num("line", r->line).comma()
                 .kv_str("text", r->text);
            }
        }
    }
    return std::move(w).take();
}

} // namespace core::tools
