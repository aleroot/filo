#include "SearchReplaceTool.hpp"
#include "ToolArgumentUtils.hpp"
#include "ToolNames.hpp"
#include "../utils/JsonUtils.hpp"
#include <simdjson.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>
#include <optional>
#include <algorithm>
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

// LCS-based similarity ratio: 2*LCS(a,b)/(|a|+|b|).
// Same metric as Python's difflib.SequenceMatcher.ratio().
static double lcs_ratio(std::string_view a, std::string_view b) {
    if (a.empty() && b.empty()) return 1.0;
    const size_t total = a.size() + b.size();

    // Cap to keep O(n*m) bounded; search blocks are usually short.
    constexpr size_t kCap = 1500;
    if (a.size() > kCap) a = a.substr(0, kCap);
    if (b.size() > kCap) b = b.substr(0, kCap);

    const size_t n = a.size(), m = b.size();
    std::vector<size_t> dp(m + 1, 0);
    for (size_t i = 0; i < n; ++i) {
        size_t prev = 0;
        for (size_t j = 0; j < m; ++j) {
            const size_t temp = dp[j + 1];
            if (a[i] == b[j]) dp[j + 1] = prev + 1;
            else               dp[j + 1] = std::max(dp[j + 1], dp[j]);
            prev = temp;
        }
    }
    return 2.0 * dp[m] / total;
}

struct FuzzyMatch {
    double similarity;
    size_t start_line; // 1-based
    size_t end_line;
    std::string text;
};

static std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> lines;
    size_t start = 0;
    while (start < s.size()) {
        size_t nl = s.find('\n', start);
        if (nl == std::string::npos) { lines.push_back(s.substr(start)); break; }
        auto line = s.substr(start, nl - start);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(std::move(line));
        start = nl + 1;
    }
    return lines;
}

static std::optional<FuzzyMatch> find_fuzzy_match(
    const std::string& content,
    const std::string& search_text,
    double threshold = 0.85)
{
    const auto content_lines = split_lines(content);
    const auto search_lines  = split_lines(search_text);
    const size_t window = search_lines.size();
    if (window == 0 || content_lines.size() < window) return std::nullopt;

    // Try at most 500 windows to keep this bounded.
    const size_t limit = std::min(content_lines.size() - window + 1, size_t{500});
    FuzzyMatch best{0.0, 0, 0, {}};

    for (size_t start = 0; start < limit; ++start) {
        std::string window_text;
        window_text.reserve(search_text.size() + 32);
        for (size_t i = start; i < start + window; ++i) {
            if (i > start) window_text += '\n';
            window_text += content_lines[i];
        }
        const double sim = lcs_ratio(search_text, window_text);
        if (sim >= threshold && sim > best.similarity) {
            best = {sim, start + 1, start + window, std::move(window_text)};
        }
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
    R"({"type":"object","properties":{"old_string":{"type":"string","description":"Exact literal text to find in the file, copied verbatim with all indentation and line endings."},"new_string":{"type":"string","description":"Replacement text to insert in place of old_string."}},"required":["old_string","new_string"],"additionalProperties":false})";

ToolDefinition SearchReplaceTool::get_definition() const {
    return {
        .name  = std::string(names::kSearchReplace),
        .title = "Search & Replace",
        .description =
            "Applies one or more search-and-replace edits to a file in a single call. "
            "Each edit replaces the first occurrence of old_string with new_string. "
            "old_string must be copied verbatim from the file, preserving all indentation, "
            "punctuation, and line endings — do NOT paraphrase or reformat. "
            "Include 3-5 lines of surrounding context to ensure uniqueness. "
            "Edits are applied sequentially, so later edits see the result of earlier ones. "
            "Fails if any old_string is not found in the file.",
        .parameters = {
            {
                .name = "file_path",
                .type = "string",
                .description = "Absolute or relative path to the file to modify.",
                .required = true,
            },
            {
                .name = "edits",
                .type = "array",
                .description =
                    "Array of edit objects, each with 'old_string' (verbatim text to find) "
                    "and 'new_string' (replacement text). Applied sequentially in order.",
                .required = true,
                .items_schema = std::string{kEditsItemsSchema},
            },
        },
        .output_schema =
            R"({"type":"object","properties":{"success":{"type":"boolean","description":"Whether all edits were applied successfully."},"blocks_applied":{"type":"integer","description":"Number of search-and-replace blocks applied."},"lines_changed":{"type":"integer","description":"Net line-count change after applying the edits."},"warnings":{"type":"array","items":{"type":"string"},"description":"Optional non-fatal warnings about ambiguous or repeated matches."}},"required":["success","blocks_applied","lines_changed"],"additionalProperties":false})",
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
    std::string file_content;
    {
        std::ifstream ifs(resolved_path, std::ios::binary);
        if (!ifs)
            return std::format(R"({{"error":"Cannot open file for reading: {}"}})",
                               core::utils::escape_json_string(path_str));
        std::ostringstream ss;
        ss << ifs.rdbuf();
        file_content = ss.str();
    }

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
    resp += std::format(R"({{"success":true,"blocks_applied":{},"lines_changed":{})",
                        result.applied, new_lines - original_lines);
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
