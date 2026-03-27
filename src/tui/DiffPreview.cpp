#include "DiffPreview.hpp"

#include <simdjson.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <string>
#include <string_view>

namespace tui {
namespace {

std::string trim_copy(std::string_view text) {
    const auto start = text.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return std::string(text.substr(start, end - start + 1));
}

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

std::optional<std::string> extract_string_field(const simdjson::dom::object& object,
                                                std::initializer_list<std::string_view> keys) {
    for (const auto key : keys) {
        std::string_view value;
        if (object[key].get(value) == simdjson::SUCCESS) {
            return std::string(value);
        }
    }
    return std::nullopt;
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
    auto cleaned = trim_copy(path);
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

void clamp_preview_lines(ToolDiffPreview& preview, std::size_t max_lines) {
    if (max_lines == 0 || preview.lines.size() <= max_lines) {
        preview.hidden_line_count = 0;
        return;
    }

    preview.hidden_line_count = preview.lines.size() - max_lines;
    preview.lines.resize(max_lines);
}

ToolDiffPreview build_replace_preview(const simdjson::dom::object& object) {
    ToolDiffPreview preview;

    const auto file_path = extract_string_field(object, {"file_path", "path"});
    const auto old_string = extract_string_field(object, {"old_string"});
    const auto new_string = extract_string_field(object, {"new_string"});
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

ToolDiffPreview build_write_file_preview(const simdjson::dom::object& object) {
    ToolDiffPreview preview;

    const auto file_path = extract_string_field(object, {"file_path", "path"});
    const auto content = extract_string_field(object, {"content"});
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

ToolDiffPreview build_tool_diff_preview(std::string_view tool_name,
                                        std::string_view tool_args_json,
                                        std::size_t max_lines) {
    ToolDiffPreview preview;
    if (tool_args_json.empty()) {
        return preview;
    }

    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(tool_args_json).get(document) != simdjson::SUCCESS) {
        return preview;
    }

    simdjson::dom::object object;
    if (document.get(object) != simdjson::SUCCESS) {
        return preview;
    }

    if (tool_name == "apply_patch") {
        const auto patch = extract_string_field(object, {"patch"});
        if (!patch || patch->empty()) {
            return preview;
        }

        preview.title = first_non_empty_path_from_patch(*patch);
        preview.lines = parse_patch_lines(*patch);
        clamp_preview_lines(preview, max_lines);
        return preview;
    }

    if (tool_name == "replace" || tool_name == "replace_in_file") {
        preview = build_replace_preview(object);
        clamp_preview_lines(preview, max_lines);
        return preview;
    }

    if (tool_name == "write_file") {
        preview = build_write_file_preview(object);
        clamp_preview_lines(preview, max_lines);
        return preview;
    }

    return preview;
}

std::size_t diff_line_number_width(const ToolDiffPreview& preview) {
    int max_line = 0;
    for (const auto& line : preview.lines) {
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
