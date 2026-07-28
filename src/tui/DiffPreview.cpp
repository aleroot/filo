#include "DiffPreview.hpp"

#include "Constants.hpp"
#include "core/tools/ToolNames.hpp"
#include "core/utils/JsonUtils.hpp"
#include "core/utils/StringUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace tui {
namespace {

template <typename Fn>
void for_each_line(std::string_view text, Fn&& fn) {
    std::size_t start = 0;
    while (start <= text.size()) {
        const std::size_t end = text.find('\n', start);
        if (end == std::string_view::npos) {
            fn(text.substr(start));
            break;
        }
        fn(text.substr(start, end - start));
        start = end + 1;
        if (start == text.size()) {
            fn(std::string_view{});
            break;
        }
    }
}

/// Splits `text` into lines, preserving interior blanks. A single trailing
/// newline is treated as a terminator rather than the start of another line, so
/// "alpha\nbeta\n" is two lines — matching how editors count them and keeping the
/// transcript's "+N -M" stats and the rendered diff free of a phantom last line.
std::vector<std::string> split_lines_keep_empty(std::string_view text) {
    std::vector<std::string> lines;
    for_each_line(text, [&](std::string_view line) {
        lines.emplace_back(line);
    });
    if (lines.size() > 1 && lines.back().empty()) {
        lines.pop_back();
    }
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
        value = value * 10 + static_cast<int>(text[pos] - '0');
        ++pos;
    }
    out = value;
    return true;
}

bool parse_hunk_header(std::string_view line, int& old_start, int& new_start) {
    if (!line.starts_with("@@ -")) {
        return false;
    }

    std::size_t pos = 4;
    if (!parse_positive_int(line, pos, old_start)) {
        return false;
    }

    if (pos < line.size() && line[pos] == ',') {
        ++pos;
        int ignored = 0;
        if (!parse_positive_int(line, pos, ignored)) {
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
    bool in_hunk = false;

    for_each_line(patch, [&](std::string_view line) {
        if (line.empty()) {
            if (in_hunk) {
                ++old_line;
                ++new_line;
                lines.push_back(make_diff_line(DiffLineKind::Context, {}, old_line, new_line));
            }
            return;
        }

        int parsed_old = 0;
        int parsed_new = 0;
        if (parse_hunk_header(line, parsed_old, parsed_new)) {
            old_line = std::max(0, parsed_old - 1);
            new_line = std::max(0, parsed_new - 1);
            in_hunk = true;
            lines.push_back(make_diff_line(DiffLineKind::Hunk, std::string(line)));
            return;
        }

        if (is_patch_metadata_line(line)) {
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
                out.new_line = new_line;
            }
            lines.push_back(std::move(out));
            return;
        }

        if (marker == '-') {
            DiffLinePreview out = make_diff_line(DiffLineKind::Delete, std::string(line.substr(1)));
            if (in_hunk) {
                ++old_line;
                out.old_line = old_line;
            }
            lines.push_back(std::move(out));
            return;
        }

        if (marker == ' ') {
            DiffLinePreview out = make_diff_line(DiffLineKind::Context, std::string(line.substr(1)));
            if (in_hunk) {
                ++old_line;
                ++new_line;
                out.old_line = old_line;
                out.new_line = new_line;
            }
            lines.push_back(std::move(out));
            return;
        }

        lines.push_back(make_diff_line(DiffLineKind::Other, std::string(line)));
    });

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

    preview.title = *file_path;
    preview.lines.push_back(make_diff_line(
        DiffLineKind::Header,
        std::string("--- a/") + *file_path));
    preview.lines.push_back(make_diff_line(
        DiffLineKind::Header,
        std::string("+++ b/") + *file_path));
    preview.lines.push_back(make_diff_line(
        DiffLineKind::Hunk,
        "@@ replacement @@"));

    int old_line = 0;
    for (const auto& line : split_lines_keep_empty(*old_string)) {
        ++old_line;
        preview.lines.push_back(make_diff_line(
            DiffLineKind::Delete,
            line,
            old_line,
            std::nullopt));
    }

    int new_line = 0;
    for (const auto& line : split_lines_keep_empty(*new_string)) {
        ++new_line;
        preview.lines.push_back(make_diff_line(
            DiffLineKind::Add,
            line,
            std::nullopt,
            new_line));
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

    if (tool_name == core::tools::names::kWriteFile) {
        return finish_preview(build_write_file_preview(object));
    }

    return {};
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
