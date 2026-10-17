#include "SearchReplaceTool.hpp"
#include "shell/FsUtils.hpp"
#include "ToolArgumentUtils.hpp"
#include "ToolDiffUtils.hpp"
#include "ToolNames.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <format>

namespace core::tools {

// ---------------------------------------------------------------------------
// Edit pair
// ---------------------------------------------------------------------------

struct SRBlock {
    std::string search;
    std::string replace;
};

// ---------------------------------------------------------------------------
// Fuzzy matching (only used in error messages to suggest near-misses)
// ---------------------------------------------------------------------------

// Similarity ratio 2*LCS(a,b)/(|a|+|b|), the metric of Python's
// difflib.SequenceMatcher.ratio(). Each text is capped at kLcsCap bytes for
// the LCS itself, while the ratio's denominator keeps the full lengths.
constexpr size_t kLcsCap = 1500;

// LCS length of one fixed text against many others, bit-parallel (Hyyrö):
// each character of the other text costs |a|/64 word operations instead of
// |a| cell updates, with exactly the result of the dynamic program.
namespace {
class LcsMatcher {
public:
    explicit LcsMatcher(std::string_view a)
        : n_(std::min(a.size(), kLcsCap)),
          words_((n_ + 63) / 64),
          masks_(256 * words_, 0),
          row_(words_) {
        for (size_t i = 0; i < n_; ++i) {
            masks_[static_cast<unsigned char>(a[i]) * words_ + i / 64] |= uint64_t{1} << (i % 64);
        }
    }

    [[nodiscard]] size_t length(std::string_view b) {
        if (b.size() > kLcsCap) b = b.substr(0, kLcsCap);
        // A zero bit in row_ marks a position of a that the LCS uses.
        std::fill(row_.begin(), row_.end(), ~uint64_t{0});
        for (const char ch : b) {
            const uint64_t* match = &masks_[static_cast<unsigned char>(ch) * words_];
            uint64_t carry = 0;
            for (size_t w = 0; w < words_; ++w) {
                const uint64_t v = row_[w];
                const uint64_t u = v & match[w];
                const uint64_t partial = v + u;
                const uint64_t sum = partial + carry;
                carry = static_cast<uint64_t>(partial < v) | static_cast<uint64_t>(sum < partial);
                row_[w] = sum | (v & ~match[w]);
            }
        }
        size_t used = 0;
        for (size_t w = 0; w < words_; ++w) {
            const size_t bits = std::min<size_t>(64, n_ - w * 64);
            const uint64_t live = bits == 64 ? ~uint64_t{0} : (uint64_t{1} << bits) - 1;
            used += bits - static_cast<size_t>(std::popcount(row_[w] & live));
        }
        return used;
    }

private:
    size_t n_;
    size_t words_;
    std::vector<uint64_t> masks_;
    std::vector<uint64_t> row_;
};
} // namespace

[[nodiscard]] static double similarity_ratio(size_t lcs, size_t total) {
    return total == 0 ? 1.0 : 2.0 * static_cast<double>(lcs) / static_cast<double>(total);
}

struct FuzzyMatch {
    double similarity;
    size_t start_line; // 1-based
    size_t end_line;
    std::string text;
};

// Lines of @p s without their terminators; a trailing CR is dropped too.
static std::vector<std::string_view> split_lines(std::string_view s) {
    std::vector<std::string_view> lines;
    size_t start = 0;
    while (start < s.size()) {
        size_t nl = s.find('\n', start);
        if (nl == std::string_view::npos) { lines.push_back(s.substr(start)); break; }
        auto line = s.substr(start, nl - start);
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        lines.push_back(line);
        start = nl + 1;
    }
    return lines;
}

// Windows whose similarity is actually computed, at most. A cheap upper
// bound skips nearly every window first, so this is only reached on very
// large files full of near-duplicates; it keeps such a miss bounded.
constexpr size_t kMaxScoredWindows = 4000;

static std::optional<FuzzyMatch> find_fuzzy_match(
    const std::string& content,
    const std::string& search_text,
    double threshold = 0.85)
{
    const auto content_lines = split_lines(content);
    const size_t window = split_lines(search_text).size();
    if (window == 0 || content_lines.size() < window) return std::nullopt;

    // The LCS can use each byte value at most as often as both texts hold
    // it, so the summed minimum of their byte counts bounds it from above.
    // The window's counts slide with it, keeping that sum current in O(1)
    // per byte; a window whose bound cannot beat the best so far is skipped
    // without changing the result.
    const std::string_view capped_search =
        std::string_view(search_text).substr(0, std::min(search_text.size(), kLcsCap));
    std::array<size_t, 256> search_counts{};
    for (const char ch : capped_search) ++search_counts[static_cast<unsigned char>(ch)];
    std::array<size_t, 256> window_counts{};
    size_t bound = 0;
    size_t window_size = 0;
    const auto add = [&](char ch) {
        const auto c = static_cast<unsigned char>(ch);
        if (window_counts[c]++ < search_counts[c]) ++bound;
        ++window_size;
    };
    const auto remove = [&](char ch) {
        const auto c = static_cast<unsigned char>(ch);
        if (--window_counts[c] < search_counts[c]) --bound;
        --window_size;
    };
    for (size_t i = 0; i < window; ++i) {
        if (i > 0) add('\n');
        for (const char ch : content_lines[i]) add(ch);
    }

    LcsMatcher matcher(search_text);
    FuzzyMatch best{0.0, 0, 0, {}};
    std::string window_text;
    size_t scored = 0;
    const size_t last_start = content_lines.size() - window;
    for (size_t start = 0;; ++start) {
        const size_t total = search_text.size() + window_size;
        const double ceiling = similarity_ratio(bound, total);
        if (ceiling >= threshold && ceiling > best.similarity) {
            window_text.clear();
            for (size_t i = start; i < start + window; ++i) {
                if (i > start) window_text += '\n';
                window_text += content_lines[i];
            }
            const double sim = similarity_ratio(matcher.length(window_text), total);
            if (sim >= threshold && sim > best.similarity) {
                best = {sim, start + 1, start + window, window_text};
            }
            if (++scored == kMaxScoredWindows) break;
        }
        if (start == last_start) break;
        for (const char ch : content_lines[start]) remove(ch);
        for (const char ch : content_lines[start + window]) add(ch);
    }

    if (best.similarity == 0.0) return std::nullopt;
    return best;
}

// ---------------------------------------------------------------------------
// Block application
// ---------------------------------------------------------------------------

struct ApplyResult {
    std::string content;
    int applied{0};
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
};

static ApplyResult apply_blocks(
    const std::string& original,
    const std::vector<SRBlock>& blocks,
    std::string_view filepath)
{
    ApplyResult r;
    r.content = original;

    for (size_t i = 0; i < blocks.size(); ++i) {
        const auto& [search, replace] = blocks[i];
        const size_t idx = i + 1;

        if (search.empty()) {
            r.errors.push_back(std::format("Edit {}: old_string is empty.", idx));
            continue;
        }

        const size_t pos = r.content.find(search);
        if (pos == std::string::npos) {
            std::string msg = std::format(
                "Edit {}: old_string not found in {}.\n"
                "Search text was:\n{}\n",
                idx, filepath, search);

            auto match = find_fuzzy_match(r.content, search);
            if (match) {
                msg += std::format(
                    "Closest match ({:.1f}% similarity) at lines {}–{}:\n{}\n",
                    match->similarity * 100.0,
                    match->start_line, match->end_line,
                    match->text);
            }
            msg += "Tips: ensure old_string is copied verbatim from the file, "
                   "preserving all indentation (spaces/tabs), punctuation, and "
                   "line endings. Check that previous edits haven't already modified the text.";
            r.errors.push_back(std::move(msg));
            continue;
        }

        // Warn if the search text appears more than once; replace only the first.
        const size_t next = r.content.find(search, pos + search.size());
        if (next != std::string::npos) {
            r.warnings.push_back(std::format(
                "Edit {}: old_string appears more than once — only the first occurrence was replaced. "
                "Include more surrounding context to make it unique.", idx));
        }

        r.content.replace(pos, search.size(), replace);
        ++r.applied;
    }
    return r;
}

// ---------------------------------------------------------------------------
// Tool definition & execute
// ---------------------------------------------------------------------------

// JSON Schema for each element of the "edits" array.
// Emitted verbatim as the "items" value in the serialized tool schema.
static constexpr std::string_view kEditsItemsSchema =
    R"({"type":"object","properties":{"old_string":{"type":"string","description":"Unique verbatim text to replace."},"new_string":{"type":"string","description":"Replacement text."}},"required":["old_string","new_string"],"additionalProperties":false})";

ToolDefinition SearchReplaceTool::get_definition() const {
    return {
        .name  = std::string(names::kSearchReplace),
        .title = "Edit",
        .description =
            "Apply ordered exact-text edits to one file. Copy old_string verbatim with enough "
            "context to be unique; later edits see earlier edits. Returns a bounded diff.",
        .parameters = {
            {
                .name = "file_path",
                .type = "string",
                .description = "File path.",
                .required = true,
            },
            {
                .name = "edits",
                .type = "array",
                .description =
                    "Ordered {old_string,new_string} edits, as a JSON array value "
                    "(never a string containing JSON).",
                .required = true,
                .items_schema = std::string{kEditsItemsSchema},
            },
        },
        .output_schema =
            R"({"type":"object","properties":{"success":{"type":"boolean","description":"Whether all edits were applied successfully."},"file_path":{"type":"string","description":"The modified file path."},"blocks_applied":{"type":"integer","description":"Number of search-and-replace blocks applied."},"lines_changed":{"type":"integer","description":"Net line-count change after applying the edits."},"diff":{"type":"string","description":"Minimal contextual unified diff. Omitted for oversized, binary-like, or computationally expensive changes."},"warnings":{"type":"array","items":{"type":"string"},"description":"Optional non-fatal warnings about ambiguous or repeated matches."}},"required":["success","file_path","blocks_applied","lines_changed"],"additionalProperties":false})",
        .annotations = {
            .destructive_hint = true,  // modifies file content on disk
        },
    };
}

std::string SearchReplaceTool::execute(const std::string& json_args, const core::context::SessionContext& context) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(json_args).get(doc) != simdjson::SUCCESS)
        return R"({"error":"Invalid JSON arguments provided to search_replace."})";

    std::string_view file_path_v;
    if (doc["file_path"].get(file_path_v) != simdjson::SUCCESS)
        return R"({"error":"Missing 'file_path' argument."})";

    simdjson::dom::array edits_arr;
    if (doc["edits"].get(edits_arr) != simdjson::SUCCESS)
        return R"({"error":"Missing or invalid 'edits' argument. Expected an array of {\"old_string\", \"new_string\"} objects."})";

    // Parse edits array into SRBlock vector.
    std::vector<SRBlock> blocks;
    {
        size_t idx = 0;
        for (simdjson::dom::element edit : edits_arr) {
            ++idx;
            std::string_view old_sv, new_sv;
            if (edit["old_string"].get(old_sv) != simdjson::SUCCESS)
                return std::format(R"({{"error":"Edit {} is missing 'old_string'."}})", idx);
            if (edit["new_string"].get(new_sv) != simdjson::SUCCESS)
                return std::format(R"({{"error":"Edit {} is missing 'new_string'."}})", idx);
            blocks.push_back({std::string(old_sv), std::string(new_sv)});
        }
    }

    if (blocks.empty())
        return R"({"error":"'edits' array is empty. Provide at least one {\"old_string\", \"new_string\"} edit."})";

    const std::string path_str(file_path_v);
    std::filesystem::path resolved_path;
    if (const auto access_error =
            detail::check_workspace_access(
                path_str,
                path_str,
                context,
                &resolved_path,
                names::kSearchReplace)) {
        return *access_error;
    }
    std::error_code ec;
    if (!std::filesystem::is_regular_file(resolved_path, ec))
        return std::format(R"({{"error":"File not found: {}"}})",
                           core::utils::escape_json_string(path_str));

    // Read file.
    auto file = detail::read_whole_file(resolved_path);
    if (!file)
        return std::format(R"({{"error":"Cannot open file for reading: {}"}})",
                           core::utils::escape_json_string(path_str));
    const std::string file_content = std::move(*file);

    const int original_lines = static_cast<int>(
        std::count(file_content.begin(), file_content.end(), '\n'));

    const auto result = apply_blocks(file_content, blocks, file_path_v);

    if (!result.errors.empty()) {
        std::string combined;
        for (size_t i = 0; i < result.errors.size(); ++i) {
            if (i) combined += "\n\n";
            combined += result.errors[i];
        }
        return std::format(R"({{"error":"{}"}})",
                           core::utils::escape_json_string(combined));
    }

    // Write back only if content changed.
    if (result.content != file_content) {
        std::ofstream ofs(resolved_path, std::ios::binary | std::ios::trunc);
        if (!ofs)
            return std::format(R"({{"error":"Cannot open file for writing: {}"}})",
                               core::utils::escape_json_string(path_str));
        ofs << result.content;
    }

    const int new_lines = static_cast<int>(
        std::count(result.content.begin(), result.content.end(), '\n'));

    // Build success response, optionally with warnings.
    std::string resp;
    resp.reserve(128);
    resp += std::format(
        R"({{"success":true,"file_path":"{}","blocks_applied":{},"lines_changed":{})",
        core::utils::escape_json_string(resolved_path.string()),
        result.applied,
        new_lines - original_lines);
    if (const auto diff = detail::build_unified_diff(
            resolved_path.string(),
            file_content,
            result.content)) {
        resp += detail::json_diff_field(*diff);
    }
    if (!result.warnings.empty()) {
        resp += R"(,"warnings":[)";
        for (size_t i = 0; i < result.warnings.size(); ++i) {
            if (i) resp += ',';
            resp += '"';
            resp += core::utils::escape_json_string(result.warnings[i]);
            resp += '"';
        }
        resp += ']';
    }
    resp += '}';
    return resp;
}

} // namespace core::tools
