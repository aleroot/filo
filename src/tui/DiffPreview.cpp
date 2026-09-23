#include "DiffPreview.hpp"

#include "Constants.hpp"
#include "core/tools/ToolDiffUtils.hpp"
#include "core/tools/ToolNames.hpp"
#include "core/utils/JsonUtils.hpp"
#include "core/utils/StringUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace tui {
namespace {

template <typename Fn>
void for_each_line(std::string_view text, Fn&& fn) {
    if (text.empty()) {
        fn(std::string_view{});
        return;
    }
    std::size_t start = 0;
    while (start < text.size()) {
        const std::size_t end = text.find('\n', start);
        if (end == std::string_view::npos) {
            fn(text.substr(start));
            break;
        }
        fn(text.substr(start, end - start));
        start = end + 1;
    }
}

/// Splits `text` into lines, preserving blank lines. A trailing newline is a
/// terminator rather than an extra line, so "alpha\nbeta\n" is two lines.
std::vector<std::string> split_lines_keep_empty(std::string_view text) {
    std::vector<std::string> lines;
    for_each_line(text, [&](std::string_view line) {
        lines.emplace_back(line);
    });
    if (lines.empty()) {
        lines.emplace_back();
    }
    return lines;
}

bool parse_positive_int(std::string_view text, std::size_t& pos, int& out) {
    if (pos >= text.size() || !std::isdigit(static_cast<unsigned char>(text[pos]))) {
        return false;
    }

    int value = 0;
    while (pos < text.size() && std::isdigit(static_cast<unsigned char>(text[pos]))) {
        const int digit = static_cast<int>(text[pos] - '0');
        if (value > (std::numeric_limits<int>::max() - digit) / 10) {
            return false;
        }
        value = value * 10 + digit;
        ++pos;
    }
    out = value;
    return true;
}

bool parse_hunk_header(std::string_view line,
                       int& old_start,
                       int& old_count,
                       int& new_start,
                       int& new_count) {
    if (!line.starts_with("@@ -")) {
        return false;
    }

    std::size_t pos = 4;
    if (!parse_positive_int(line, pos, old_start)) {
        return false;
    }

    old_count = 1;
    if (pos < line.size() && line[pos] == ',') {
        ++pos;
        if (!parse_positive_int(line, pos, old_count)) {
            return false;
        }
    }

    if (pos >= line.size() || line[pos] != ' ') {
        return false;
    }
    ++pos;

    if (pos >= line.size() || line[pos] != '+') {
        return false;
    }
    ++pos;

    if (!parse_positive_int(line, pos, new_start)) {
        return false;
    }

    new_count = 1;
    if (pos < line.size() && line[pos] == ',') {
        ++pos;
        if (!parse_positive_int(line, pos, new_count)) {
            return false;
        }
    }

    return true;
}

bool is_patch_metadata_line(std::string_view line) {
    return line.starts_with("diff --git ")
        || line.starts_with("index ")
        || line.starts_with("--- ")
        || line.starts_with("+++ ")
        || line.starts_with("new file mode ")
        || line.starts_with("deleted file mode ")
        || line.starts_with("rename from ")
        || line.starts_with("rename to ")
        || line.starts_with("*** Begin Patch")
        || line.starts_with("*** End Patch")
        || line.starts_with("*** Update File: ")
        || line.starts_with("*** Add File: ")
        || line.starts_with("*** Delete File: ")
        || line.starts_with("*** Move to: ");
}

std::string normalize_diff_path(std::string_view path) {
    auto cleaned = core::utils::str::trim_ascii_copy(path);
    if (cleaned == "/dev/null") {
        return {};
    }
    if (cleaned.size() > 2 && cleaned[1] == '/' && (cleaned[0] == 'a' || cleaned[0] == 'b')) {
        cleaned.erase(cleaned.begin(), cleaned.begin() + 2);
    }
    return cleaned;
}

DiffLinePreview make_diff_line(DiffLineKind kind,
                               std::string content,
                               std::optional<int> old_line = std::nullopt,
                               std::optional<int> new_line = std::nullopt) {
    DiffLinePreview line;
    line.kind = kind;
    line.old_line = old_line;
    line.new_line = new_line;
    line.content = std::move(content);
    return line;
}

std::string first_non_empty_path_from_patch(std::string_view patch) {
    constexpr std::array prefixes = {
        std::string_view{"*** Update File: "},
        std::string_view{"*** Add File: "},
        std::string_view{"*** Delete File: "},
        std::string_view{"*** Move to: "},
        std::string_view{"+++ "},
        std::string_view{"--- "}
    };

    std::string resolved;
    for_each_line(patch, [&](std::string_view line) {
        if (!resolved.empty()) {
            return;
        }

        for (auto prefix : prefixes) {
            if (!line.starts_with(prefix)) {
                continue;
            }
            auto path = normalize_diff_path(line.substr(prefix.size()));
            if (!path.empty()) {
                resolved = std::move(path);
            }
            return;
        }
    });
    return resolved;
}

std::vector<DiffLinePreview> parse_patch_lines(std::string_view patch) {
    std::vector<DiffLinePreview> lines;
    lines.reserve(64);

    int old_line = 0;
    int new_line = 0;
    int old_remaining = 0;
    int new_remaining = 0;
    bool in_hunk = false;

    const auto consume_line = [&](std::string_view raw_line) {
        // Unified diff records are LF-delimited even when their source file
        // uses CRLF. Keep source bytes in the tool response, but avoid handing
        // a carriage return to the terminal text renderer.
        std::string_view line = raw_line;
        if (line.ends_with('\r')) {
            line.remove_suffix(1);
        }
        if (line.empty()) {
            lines.push_back(make_diff_line(DiffLineKind::Other, {}));
            return;
        }

        int parsed_old = 0;
        int parsed_old_count = 0;
        int parsed_new = 0;
        int parsed_new_count = 0;
        if (parse_hunk_header(line,
                              parsed_old,
                              parsed_old_count,
                              parsed_new,
                              parsed_new_count)) {
            old_line = std::max(0, parsed_old - 1);
            new_line = std::max(0, parsed_new - 1);
            old_remaining = parsed_old_count;
            new_remaining = parsed_new_count;
            in_hunk = old_remaining > 0 || new_remaining > 0;
            lines.push_back(make_diff_line(DiffLineKind::Hunk, std::string(line)));
            return;
        }

        if (!in_hunk && is_patch_metadata_line(line)) {
            if (line.starts_with("*** Update File: ")
                    || line.starts_with("*** Add File: ")
                    || line.starts_with("*** Delete File: ")
                    || line.starts_with("*** Move to: ")) {
                in_hunk = false;
            }
            lines.push_back(make_diff_line(DiffLineKind::Header, std::string(line)));
            return;
        }

        if (line.starts_with("\\ No newline at end of file")) {
            lines.push_back(make_diff_line(DiffLineKind::Other, std::string(line)));
            return;
        }

        const char marker = line.front();
        if (marker == '+') {
            DiffLinePreview out = make_diff_line(DiffLineKind::Add, std::string(line.substr(1)));
            if (in_hunk) {
                ++new_line;
                --new_remaining;
                out.new_line = new_line;
            }
            lines.push_back(std::move(out));
        } else if (marker == '-') {
            DiffLinePreview out = make_diff_line(DiffLineKind::Delete, std::string(line.substr(1)));
            if (in_hunk) {
                ++old_line;
                --old_remaining;
                out.old_line = old_line;
            }
            lines.push_back(std::move(out));
        } else if (marker == ' ') {
            DiffLinePreview out = make_diff_line(DiffLineKind::Context, std::string(line.substr(1)));
            if (in_hunk) {
                ++old_line;
                ++new_line;
                --old_remaining;
                --new_remaining;
                out.old_line = old_line;
                out.new_line = new_line;
            }
            lines.push_back(std::move(out));
        } else {
            lines.push_back(make_diff_line(DiffLineKind::Other, std::string(line)));
        }

        if (in_hunk && old_remaining <= 0 && new_remaining <= 0) {
            in_hunk = false;
        }
    };

    std::size_t start = 0;
    while (start < patch.size()) {
        const std::size_t end = patch.find('\n', start);
        if (end == std::string_view::npos) {
            consume_line(patch.substr(start));
            break;
        }
        consume_line(patch.substr(start, end - start));
        start = end + 1;
    }

    return lines;
}

/// A built diff before it becomes a `ToolDiffPreview`.
struct DiffBuildResult {
    std::string                  title;
    std::vector<DiffLinePreview> lines;
};

/// Turns a built diff into the immutable model object: records the true totals,
/// then applies the model-level ceiling. Counts are taken *before* the ceiling
/// so a pathological change still reports its real size in the header even
/// though the transcript refuses to carry every line of it.
ToolDiffPreview finish_preview(DiffBuildResult built) {
    ToolDiffPreview preview;
    preview.title = std::move(built.title);
    preview.total_line_count = built.lines.size();
    for (const auto& line : built.lines) {
        preview.added_count += line.kind == DiffLineKind::Add ? 1 : 0;
        preview.deleted_count += line.kind == DiffLineKind::Delete ? 1 : 0;
    }

    if (kToolDiffModelMaxLines != 0 && built.lines.size() > kToolDiffModelMaxLines) {
        built.lines.resize(kToolDiffModelMaxLines);
        preview.truncated_at_source = true;
    }

    preview.set_lines(std::move(built.lines));
    return preview;
}

DiffBuildResult build_replace_preview(const simdjson::dom::object& object) {
    DiffBuildResult preview;

    const auto file_path = core::utils::json::first_string_field(object, {"file_path", "path"});
    const auto old_string = core::utils::json::first_string_field(object, {"old_string"});
    const auto new_string = core::utils::json::first_string_field(object, {"new_string"});
    if (!file_path || !old_string || !new_string) {
        return preview;
    }

    if (const auto diff = core::tools::detail::build_unified_diff(
            *file_path, *old_string, *new_string)) {
        preview.title = *file_path;
        preview.lines = parse_patch_lines(*diff);
    }

    return preview;
}

DiffBuildResult build_search_replace_preview(const simdjson::dom::object& object) {
    DiffBuildResult preview;
    const auto file_path = core::utils::json::first_string_field(object, {"file_path", "path"});
    if (!file_path) {
        return preview;
    }

    simdjson::dom::array edits;
    if (object["edits"].get(edits) != simdjson::SUCCESS) {
        return preview;
    }

    preview.title = *file_path;
    std::size_t edit_number = 0;
    for (const simdjson::dom::element edit : edits) {
        simdjson::dom::object edit_object;
        if (edit.get(edit_object) != simdjson::SUCCESS) {
            continue;
        }
        const auto old_string = core::utils::json::first_string_field(
            edit_object, {"old_string"});
        const auto new_string = core::utils::json::first_string_field(
            edit_object, {"new_string"});
        if (!old_string || !new_string) {
            continue;
        }
        const auto diff = core::tools::detail::build_unified_diff(
            *file_path, *old_string, *new_string);
        if (!diff) {
            continue;
        }

        auto snippet = parse_patch_lines(*diff);
        if (snippet.empty()) {
            continue;
        }
        ++edit_number;

        if (preview.lines.empty()) {
            // The first two patch lines identify the file. Later edits share
            // those headers and get their own clearly scoped preview hunk.
            preview.lines.push_back(snippet[0]);
            if (snippet.size() > 1) {
                preview.lines.push_back(snippet[1]);
            }
        }
        std::string hunk_header = snippet.size() > 2
            ? snippet[2].content
            : "@@ -0,0 +0,0 @@";
        hunk_header += std::format(" (edit {}; location determined at apply time)",
                                   edit_number);
        preview.lines.push_back(make_diff_line(DiffLineKind::Hunk,
                                                std::move(hunk_header)));
        const auto body_start = std::min<std::size_t>(3, snippet.size());
        preview.lines.insert(
            preview.lines.end(),
            std::make_move_iterator(snippet.begin() + static_cast<std::ptrdiff_t>(body_start)),
            std::make_move_iterator(snippet.end()));
    }

    return preview;
}

DiffBuildResult build_write_file_preview(const simdjson::dom::object& object) {
    DiffBuildResult preview;

    const auto file_path = core::utils::json::first_string_field(object, {"file_path", "path"});
    const auto content = core::utils::json::first_string_field(object, {"content"});
    if (!file_path || !content) {
        return preview;
    }

    preview.title = *file_path;
    preview.lines.push_back(make_diff_line(
        DiffLineKind::Header,
        std::string("+++ b/") + *file_path));
    preview.lines.push_back(make_diff_line(
        DiffLineKind::Hunk,
        "@@ file content @@"));

    int new_line = 0;
    for (const auto& line : split_lines_keep_empty(*content)) {
        ++new_line;
        preview.lines.push_back(make_diff_line(
            DiffLineKind::Add,
            line,
            std::nullopt,
            new_line));
    }

    return preview;
}

} // namespace

const std::vector<DiffLinePreview>& ToolDiffPreview::lines() const noexcept {
    static const std::vector<DiffLinePreview> kEmpty;
    return lines_ ? *lines_ : kEmpty;
}

void ToolDiffPreview::set_lines(std::vector<DiffLinePreview> value) {
    lines_ = std::make_shared<const std::vector<DiffLinePreview>>(std::move(value));
}

ToolDiffPreview build_tool_diff_preview(std::string_view tool_name,
                                        std::string_view tool_args_json) {
    if (tool_args_json.empty()) {
        return {};
    }

    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(tool_args_json).get(document) != simdjson::SUCCESS) {
        return {};
    }

    simdjson::dom::object object;
    if (document.get(object) != simdjson::SUCCESS) {
        return {};
    }

    if (tool_name == core::tools::names::kApplyPatch) {
        const auto patch = core::utils::json::first_string_field(object, {"patch"});
        if (!patch || patch->empty()) {
            return {};
        }
        return finish_preview({
            .title = first_non_empty_path_from_patch(*patch),
            .lines = parse_patch_lines(*patch),
        });
    }

    if (core::tools::names::is_replace_tool(tool_name)) {
        return finish_preview(build_replace_preview(object));
    }

    if (tool_name == core::tools::names::kSearchReplace) {
        return finish_preview(build_search_replace_preview(object));
    }

    if (tool_name == core::tools::names::kWriteFile) {
        return finish_preview(build_write_file_preview(object));
    }

    return {};
}

ToolDiffPreview build_tool_diff_preview_from_unified_diff(std::string_view patch) {
    if (patch.empty()) {
        return {};
    }

    const std::size_t first_line_end = patch.find('\n');
    if (first_line_end == std::string_view::npos
        || !patch.substr(0, first_line_end).starts_with("--- ")) {
        return {};
    }
    const std::size_t second_line_start = first_line_end + 1;
    const std::size_t second_line_end = patch.find('\n', second_line_start);
    if (second_line_end == std::string_view::npos
        || !patch.substr(second_line_start,
                         second_line_end - second_line_start).starts_with("+++ ")) {
        return {};
    }

    auto preview = finish_preview({
        .title = first_non_empty_path_from_patch(patch),
        .lines = parse_patch_lines(patch),
    });
    const bool has_hunk = std::ranges::any_of(preview.lines(), [](const auto& line) {
        return line.kind == DiffLineKind::Hunk;
    });
    if (!has_hunk || (preview.added_count == 0 && preview.deleted_count == 0)) {
        return {};
    }
    return preview;
}

ToolDiffPreview clamp_diff_preview(const ToolDiffPreview& preview, std::size_t max_lines) {
    const auto& lines = preview.lines();
    const std::size_t shown = max_lines == 0
        ? lines.size()
        : std::min(lines.size(), max_lines);

    // Measured against the true total, not against what survived the model
    // ceiling: a diff that was already cut at build time still owes the reader
    // an honest count, even when this clamp itself drops nothing.
    const std::size_t hidden = preview.total_line_count - shown;
    if (hidden == preview.hidden_line_count && shown == lines.size()) {
        return preview;
    }

    ToolDiffPreview clamped = preview;
    clamped.hidden_line_count = hidden;
    if (shown < lines.size()) {
        clamped.set_lines(
            std::vector<DiffLinePreview>(lines.begin(),
                                         lines.begin() + static_cast<std::ptrdiff_t>(shown)));
    }
    return clamped;
}

std::size_t diff_line_number_width(const ToolDiffPreview& preview) {
    int max_line = 0;
    for (const auto& line : preview.lines()) {
        if (line.old_line) {
            max_line = std::max(max_line, *line.old_line);
        }
        if (line.new_line) {
            max_line = std::max(max_line, *line.new_line);
        }
    }

    std::size_t width = 1;
    while (max_line >= 10) {
        max_line /= 10;
        ++width;
    }
    return width;
}

} // namespace tui
