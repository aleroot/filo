#include "PromptComponents.hpp"
#include "Constants.hpp"
#include "StringUtils.hpp"
#include "TuiTheme.hpp"
#include "core/session/SessionStore.hpp"

#include <ftxui/component/screen_interactive.hpp>
#include <ftxui/dom/elements.hpp>
#include <simdjson.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <format>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace ftxui;

namespace tui {

namespace {

int estimated_prompt_lines(std::string_view input_text) {
    const auto explicit_lines =
        static_cast<int>(std::count(input_text.begin(), input_text.end(), '\n')) + 1;
    const auto wrapped_lines =
        std::max(1, static_cast<int>((input_text.size() + kEstimatedLineWidthChars - 1)
                                      / kEstimatedLineWidthChars));
    return std::max(explicit_lines, wrapped_lines);
}

int prompt_body_height(std::string_view input_text, int base_height) {
    const auto lines = estimated_prompt_lines(input_text);
    return std::clamp(std::max(base_height, lines + 1), base_height, base_height + 4);
}

std::string trim_copy(std::string_view text) {
    const auto start = text.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto end = text.find_last_not_of(" \t\r\n");
    return std::string(text.substr(start, end - start + 1));
}

std::string collapse_whitespace(std::string_view text) {
    std::string out;
    out.reserve(text.size());

    bool last_was_space = false;
    for (const unsigned char ch : text) {
        if (std::isspace(ch)) {
            if (!last_was_space) {
                out.push_back(' ');
                last_was_space = true;
            }
            continue;
        }
        out.push_back(static_cast<char>(ch));
        last_was_space = false;
    }
    return trim_copy(out);
}

std::string truncate_preview(std::string_view text, std::size_t max_len = 180) {
    std::string cleaned = collapse_whitespace(text);
    if (cleaned.size() <= max_len) {
        return cleaned;
    }
    if (max_len <= 1) {
        return cleaned.substr(0, max_len);
    }
    return cleaned.substr(0, max_len - 1) + "…";
}

std::string to_lower_ascii_copy(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const unsigned char ch : text) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

std::optional<std::size_t> find_case_insensitive(std::string_view haystack,
                                                 std::string_view needle) {
    if (needle.empty()) {
        return std::nullopt;
    }
    const std::string hay = to_lower_ascii_copy(haystack);
    const std::string ned = to_lower_ascii_copy(needle);
    const auto pos = hay.find(ned);
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    return pos;
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

std::string humanize_key(std::string_view key) {
    std::string out;
    out.reserve(key.size());

    bool capitalize = true;
    for (char ch : key) {
        if (ch == '_' || ch == '-') {
            out.push_back(' ');
            capitalize = true;
            continue;
        }

        if (capitalize) {
            out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(ch))));
            capitalize = false;
        } else {
            out.push_back(ch);
        }
    }
    return out;
}

std::string extract_patch_target(std::string_view patch) {
    constexpr std::array prefixes{
        std::string_view{"*** Update File: "},
        std::string_view{"*** Add File: "},
        std::string_view{"*** Delete File: "},
        std::string_view{"*** Move to: "},
        std::string_view{"--- "},
        std::string_view{"+++ "}
    };

    std::size_t start = 0;
    while (start < patch.size()) {
        const std::size_t end = patch.find('\n', start);
        const std::string_view line = patch.substr(
            start,
            end == std::string_view::npos ? std::string_view::npos : end - start);

        const std::string trimmed = trim_copy(line);
        if (!trimmed.empty()) {
            for (const auto prefix : prefixes) {
                if (line.starts_with(prefix)) {
                    return truncate_preview(line.substr(prefix.size()), 120);
                }
            }
            return truncate_preview(trimmed, 120);
        }

        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }

    return "patch";
}

struct PermissionField {
    std::string label;
    std::string value;
};

void add_field(std::vector<PermissionField>& fields,
               std::string label,
               std::string value) {
    value = trim_copy(value);
    if (value.empty()) {
        return;
    }
    fields.push_back(PermissionField{
        .label = std::move(label),
        .value = std::move(value),
    });
}

std::vector<PermissionField> describe_permission_arguments(std::string_view tool_name,
                                                           std::string_view args_json) {
    std::vector<PermissionField> fields;
    if (trim_copy(args_json).empty()) {
        add_field(fields, "Arguments", "No arguments.");
        return fields;
    }

    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(args_json).get(document) != simdjson::SUCCESS) {
        add_field(fields, "Arguments", "Could not parse tool arguments.");
        add_field(fields, "Raw", truncate_preview(args_json, 120));
        return fields;
    }

    simdjson::dom::object object;
    if (document.get(object) != simdjson::SUCCESS) {
        add_field(fields, "Arguments", "Unsupported argument format.");
        return fields;
    }

    if (tool_name == "run_terminal_command") {
        if (const auto command = extract_string_field(object, {"command"})) {
            add_field(fields, "Command", *command);
        }
        if (const auto working_dir = extract_string_field(object, {"working_dir"})) {
            add_field(fields, "Working Dir", *working_dir);
        }
    } else if (tool_name == "write_file") {
        if (const auto file_path = extract_string_field(object, {"file_path", "path"})) {
            add_field(fields, "File", *file_path);
        }
        if (const auto content = extract_string_field(object, {"content"})) {
            add_field(fields, "Content", std::format("{} chars", content->size()));
        }
    } else if (tool_name == "delete_file") {
        if (const auto path = extract_string_field(object, {"path", "file_path"})) {
            add_field(fields, "Path", *path);
        }
    } else if (tool_name == "move_file") {
        if (const auto source = extract_string_field(object, {"source", "source_path", "from_path"})) {
            add_field(fields, "From", *source);
        }
        if (const auto destination = extract_string_field(object, {"destination", "destination_path", "to_path"})) {
            add_field(fields, "To", *destination);
        }
    } else if (tool_name == "apply_patch") {
        if (const auto working_dir = extract_string_field(object, {"working_dir"})) {
            add_field(fields, "Working Dir", *working_dir);
        }
        if (const auto patch = extract_string_field(object, {"patch"})) {
            add_field(fields, "Patch", extract_patch_target(*patch));
            add_field(fields, "Patch Size", std::format("{} chars", patch->size()));
        }
    } else if (tool_name == "replace" || tool_name == "replace_in_file") {
        if (const auto file_path = extract_string_field(object, {"file_path", "path"})) {
            add_field(fields, "File", *file_path);
        }
        if (const auto old_string = extract_string_field(object, {"old_string"})) {
            add_field(fields, "Find", truncate_preview(*old_string, 80));
        }
        if (const auto new_string = extract_string_field(object, {"new_string"})) {
            add_field(fields, "Replace", truncate_preview(*new_string, 80));
        }
    }

    if (!fields.empty()) {
        return fields;
    }

    // Generic fallback: display scalar object fields in a readable way.
    std::size_t emitted = 0;
    for (const auto field : object) {
        if (emitted >= 6) {
            break;
        }

        std::string value;
        std::string_view string_value;
        bool bool_value = false;
        int64_t int_value = 0;
        uint64_t uint_value = 0;
        double double_value = 0.0;

        if (field.value.get(string_value) == simdjson::SUCCESS) {
            value = truncate_preview(string_value, 100);
        } else if (field.value.get(bool_value) == simdjson::SUCCESS) {
            value = bool_value ? "true" : "false";
        } else if (field.value.get(int_value) == simdjson::SUCCESS) {
            value = std::to_string(int_value);
        } else if (field.value.get(uint_value) == simdjson::SUCCESS) {
            value = std::to_string(uint_value);
        } else if (field.value.get(double_value) == simdjson::SUCCESS) {
            value = std::format("{:.3f}", double_value);
        } else {
            value = "<complex value>";
        }

        add_field(fields, humanize_key(field.key), value);
        ++emitted;
    }

    if (fields.empty()) {
        add_field(fields, "Arguments", "No displayable fields.");
    }
    return fields;
}

Element render_permission_fields(std::string_view tool_name,
                                 std::string_view args_json) {
    const auto fields = describe_permission_arguments(tool_name, args_json);

    std::vector<Element> rows;
    rows.reserve(fields.size() * 2);
    for (std::size_t i = 0; i < fields.size(); ++i) {
        rows.push_back(
            hbox({
                text(fields[i].label + ":") | dim | size(WIDTH, EQUAL, 12),
                paragraph(fields[i].value) | color(Color::White) | xflex
            }));
        if (i + 1 < fields.size()) {
            rows.push_back(separator());
        }
    }

    return vbox(std::move(rows));
}

std::string permission_intent_message(std::string_view tool_name) {
    if (tool_name == "run_terminal_command") {
        return "Filo wants to run a terminal command on your machine.";
    }
    if (tool_name == "delete_file") {
        return "Filo wants to permanently delete a path from your workspace.";
    }
    if (tool_name == "move_file") {
        return "Filo wants to move or rename files in your workspace.";
    }
    if (tool_name == "write_file" || tool_name == "apply_patch"
            || tool_name == "replace" || tool_name == "replace_in_file") {
        return "Filo wants to modify files in your workspace.";
    }
    return "Filo wants your approval before running a potentially sensitive action.";
}

std::string format_diff_line_number(const std::optional<int>& line_number, std::size_t width) {
    if (!line_number) {
        return std::string(width, ' ');
    }
    return std::format("{:>{}}", *line_number, width);
}

Element render_permission_diff_line(const DiffLinePreview& line, std::size_t number_width) {
    const auto add_color = Color::RGB(120, 220, 130);
    const auto del_color = Color::RGB(255, 150, 130);
    const auto hunk_color = Color::RGB(145, 196, 255);

    if (line.kind == DiffLineKind::Header) {
        return text(line.content) | color(Color::GrayDark) | dim;
    }

    if (line.kind == DiffLineKind::Hunk) {
        return text(line.content) | color(hunk_color);
    }

    std::string marker = " ";
    Color marker_color = Color::GrayLight;
    Color line_color = Color::GrayLight;
    switch (line.kind) {
        case DiffLineKind::Add:
            marker = "+";
            marker_color = add_color;
            line_color = add_color;
            break;
        case DiffLineKind::Delete:
            marker = "-";
            marker_color = del_color;
            line_color = del_color;
            break;
        case DiffLineKind::Context:
            marker = " ";
            marker_color = Color::GrayDark;
            line_color = Color::GrayLight;
            break;
        case DiffLineKind::Other:
            marker = " ";
            marker_color = Color::GrayDark;
            line_color = Color::GrayDark;
            break;
        case DiffLineKind::Hunk:
        case DiffLineKind::Header:
            break;
    }

    const std::string old_col = format_diff_line_number(line.old_line, number_width);
    const std::string new_col = format_diff_line_number(line.new_line, number_width);

    return hbox({
        text(std::format("{} {} ", old_col, new_col)) | color(Color::GrayDark),
        text(marker + " ") | color(marker_color) | ftxui::bold,
        text(line.content.empty() ? " " : line.content) | color(line_color) | xflex
    });
}

Element render_permission_diff_preview(const ToolDiffPreview& preview) {
    if (preview.empty()) {
        return emptyElement();
    }

    const std::size_t number_width = diff_line_number_width(preview);

    std::vector<Element> rows;
    rows.reserve(preview.lines.size() + 1);
    for (const auto& line : preview.lines) {
        rows.push_back(render_permission_diff_line(line, number_width));
    }

    if (preview.hidden_line_count > 0) {
        rows.push_back(
            text(std::format("… {} more diff lines", preview.hidden_line_count))
                | color(Color::GrayDark)
                | dim);
    }

    const int body_max_height = static_cast<int>(
        kPermissionDiffPreviewMaxLines + (preview.title.empty() ? 2 : 1));

    auto body = vbox(std::move(rows))
        | yframe
        | size(HEIGHT, LESS_THAN, body_max_height);

    Elements panel_rows = {
        hbox({ text("Diff Preview") | dim, text(" (changes to be applied)") | color(Color::GrayDark) }),
        separator(),
    };

    if (!preview.title.empty()) {
        panel_rows.push_back(
            paragraph(std::format("File: {}", preview.title))
                | color(ColorYellowBright)
                | ftxui::bold
                | xflex);
        panel_rows.push_back(separator());
    }

    panel_rows.push_back(std::move(body));

    return vbox(std::move(panel_rows)) | UiBorder(ColorYellowDark);
}

Element make_panel_shell(std::string_view tag,
                         std::string_view helper,
                         Element body,
                         int min_body_height,
                         Color accent,
                         Color tag_text = Color::Black) {
    return vbox({
        hbox({
            text(std::format(" {} ", tag)) | ftxui::bold | color(tag_text) | bgcolor(accent),
            filler(),
            text(std::string(helper)) | color(Color::GrayDark)
        }),
        separator(),
        body | size(HEIGHT, GREATER_THAN, min_body_height)
    }) | UiBorder(accent);
}

Element make_prompt_line(Element input_line) {
    return hbox({
        text(" > ") | ftxui::bold | color(ColorYellowBright),
        input_line | xflex
    }) | xflex;
}

Element make_prompt_box(Element input_line,
                        Color accent = ColorYellowDark) {
    return make_prompt_line(std::move(input_line))
        | UiBorder(accent)
        | xflex;
}

Element make_selection_row(std::string_view label,
                           std::string_view description,
                           bool selected,
                           Color accent) {
    auto row = vbox({
        text(std::string(label)) | ftxui::bold,
        text(std::string(description)) | dim
    });

    if (selected) {
        return row | bgcolor(accent) | color(Color::Black);
    }
    return row | color(Color::White);
}

Element make_command_selection_row(const CommandSuggestion& suggestion,
                                   bool selected) {
    const Color meta_color = selected ? Color::Black : Color::GrayDark;
    const Color badge_color = selected ? Color::Black : ColorYellowDark;
    const std::string description = suggestion.description.empty()
        ? "No description available."
        : suggestion.description;

    Elements title_chunks;
    title_chunks.push_back(text(suggestion.display_name) | ftxui::bold);
    if (!suggestion.aliases_label.empty()) {
        title_chunks.push_back(
            text("  aliases: " + suggestion.aliases_label)
                | color(meta_color)
                | dim);
    }

    auto row = vbox({
        hbox({
            hbox(std::move(title_chunks)) | xflex,
            suggestion.accepts_arguments
                ? (text("[args]") | color(badge_color) | ftxui::bold)
                : emptyElement()
        }),
        text(description) | color(meta_color) | dim
    });

    if (selected) {
        return row | bgcolor(ColorYellowDark) | color(Color::Black);
    }
    return row | color(Color::White);
}

Element render_search_snippet(std::string_view snippet,
                              std::string_view query,
                              bool selected) {
    const Color base_color = selected ? Color::White : Color::GrayLight;
    const Color match_fg = selected ? Color::Black : Color::Black;
    const Color match_bg = selected ? ColorYellowBright : ColorYellowDark;

    const auto match = find_case_insensitive(snippet, query);
    if (!match.has_value() || query.empty()) {
        return paragraph(std::string(snippet)) | color(base_color) | xflex;
    }

    const std::size_t start = *match;
    const std::size_t length = std::min(query.size(), snippet.size() - start);
    const std::string before = std::string(snippet.substr(0, start));
    const std::string found = std::string(snippet.substr(start, length));
    const std::string after = std::string(snippet.substr(start + length));

    return hbox({
        paragraph(before) | color(base_color),
        text(found) | ftxui::bold | color(match_fg) | bgcolor(match_bg),
        paragraph(after) | color(base_color) | xflex,
    });
}

// Shared picker panel: renders a scrollable suggestion list above the prompt box.
// Uses manual viewport slicing — FTXUI's yframe+focus+size(LESS_THAN) pattern
// does not scroll because size() caps requirement_.min_y before yframe can
// compute a virtual dimension larger than its box.
//
// Viewport policy: keep the selected item at kPickerScrollOffset rows from the
// top of the visible window once scrolling starts.  This causes the view to
// begin moving after kPickerScrollOffset downward key-presses, which is visually
// immediate enough that users notice the list is longer than one screen.
Element render_picker_panel(std::string_view tag,
                            std::vector<Element> rows,
                            Element input_line,
                            std::string_view input_text,
                            int selected_index,
                            int max_visible_items = kPickerMaxDisplayRows,
                            std::string_view helper_prefix = "↑↓ navigate  ↵/Tab complete") {
    const int total = static_cast<int>(rows.size());
    const int window = std::max(1, max_visible_items);
    // How far from the top of the visible window the selected item sits.
    // With smaller windows, ease this down so tall rows still feel balanced.
    const int kScrollOffset = std::clamp(window / 2, 0, 3);

    // view_start: first item index in the viewport.
    // clamp keeps us within [0, total-window] so we never show empty rows.
    int view_start = std::clamp(selected_index - kScrollOffset, 0,
                                std::max(0, total - window));
    const int view_end = std::min(total, view_start + window);
    const int above = view_start;
    const int below = total - view_end;

    std::vector<Element> visible;
    visible.reserve(static_cast<std::size_t>(view_end - view_start + 2));
    if (above > 0) {
        visible.push_back(hbox({
            text(" ↑ ") | color(ColorYellowDark),
            text(std::format("{} more above", above)) | color(Color::GrayDark) | dim | xflex
        }));
    }
    for (int i = view_start; i < view_end; ++i) {
        visible.push_back(std::move(rows[static_cast<std::size_t>(i)]));
    }
    if (below > 0) {
        visible.push_back(hbox({
            text(" ↓ ") | color(ColorYellowDark),
            text(std::format("{} more below", below)) | color(Color::GrayDark) | dim | xflex
        }));
    }

    const std::string helper = total > 0
        ? std::format("{}  {} of {}", helper_prefix, selected_index + 1, total)
        : std::string(helper_prefix);

    return make_panel_shell(
        tag,
        helper,
        vbox({
            vbox(std::move(visible)),
            separator(),
            make_prompt_box(std::move(input_line))
        }),
        prompt_body_height(input_text, 2),
        ColorYellowDark);
}

} // namespace

Element render_startup_banner_panel(std::string_view provider_name,
                                    std::string_view model_name,
                                    int mcp_server_count,
                                    std::string_view provider_setup_hint) {
    static constexpr std::array<std::string_view, 6> kBannerLogoLines = {
        "  ███████╗██╗██╗      ██████╗",
        "  ██╔════╝██║██║     ██╔═══██╗",
        "  █████╗  ██║██║     ██║   ██║",
        "  ██╔══╝  ██║██║     ██║   ██║",
        "  ██║     ██║███████╗╚██████╔╝",
        "  ╚═╝     ╚═╝╚══════╝ ╚═════╝",
    };

    Elements rows;
    rows.reserve(kBannerLogoLines.size() + 6);

    for (const auto line : kBannerLogoLines) {
        rows.push_back(text(std::string(line)) | color(ColorYellowBright) | ftxui::bold);
    }

    rows.push_back(text(""));
    rows.push_back(
        paragraph(std::format(
            "Filo AI Agent  —  provider: {}  —  model: {}  —  MCP servers: {}",
            provider_name,
            model_name.empty() ? "<provider default>" : std::string(model_name),
            mcp_server_count))
            | color(Color::White)
            | xflex);

    const auto hint_lines = split_lines(std::string(provider_setup_hint));
    for (const auto& line : hint_lines) {
        if (line.empty()) {
            continue;
        }
        rows.push_back(paragraph(line) | color(Color::GrayLight) | xflex);
    }

    return vbox(std::move(rows)) | UiBorder(ColorYellowDark);
}

Element render_default_prompt_panel(Element input_line,
                                    std::string_view) {
    return vbox({
        separator(),
        make_prompt_box(std::move(input_line), ColorYellowBright) | xflex
    });
}

Element render_command_prompt_panel(const std::vector<CommandSuggestion>& suggestions,
                                    int selected_index,
                                    Element input_line,
                                    std::string_view input_text) {
    std::vector<Element> rows;
    rows.reserve(suggestions.size());
    for (std::size_t i = 0; i < suggestions.size(); ++i) {
        rows.push_back(make_command_selection_row(
            suggestions[i],
            static_cast<int>(i) == selected_index));
    }

    return render_picker_panel("COMMANDS",
                               std::move(rows),
                               std::move(input_line),
                               input_text,
                               selected_index,
                               kCommandPickerMaxDisplayRows,
                               "↑↓ navigate  ↵ run  Tab complete");
}

Element render_mention_prompt_panel(const std::vector<MentionSuggestion>& suggestions,
                                    int selected_index,
                                    Element input_line,
                                    std::string_view input_text) {
    std::vector<Element> rows;
    rows.reserve(suggestions.size());
    // Compact single-line format: "[d] path/"  or  "    path/file.cpp"
    for (std::size_t i = 0; i < suggestions.size(); ++i) {
        const auto& s = suggestions[i];
        const bool sel = static_cast<int>(i) == selected_index;

        // Type badge: "d" for directory, space for file — keeps paths aligned.
        const std::string badge = s.is_directory ? " d " : "   ";

        auto row = hbox({
            text(badge)
                | color(s.is_directory
                    ? (sel ? Color::Black : ColorYellowBright)
                    : Color::GrayDark),
            text(s.display_path)
                | color(sel ? Color::Black : Color::White)
                | xflex,
        });

        rows.push_back(sel ? (row | bgcolor(ColorYellowDark))
                           : row);
    }

    return render_picker_panel("FILES", std::move(rows), std::move(input_line),
                               input_text, selected_index);
}

Element render_permission_prompt_panel(std::string_view tool_name,
                                       std::string_view args_preview,
                                       const ToolDiffPreview& diff_preview,
                                       std::string_view allow_label,
                                       int selected_index) {
    const auto option_once = make_selection_row(
        "[1] Yes",
        "Run this tool call now.",
        selected_index == 0,
        ColorWarn);
    const auto option_always = make_selection_row(
        std::format("[2] Yes, don't ask again for {}", allow_label),
        "Auto-approve matching calls for the rest of this session.",
        selected_index == 1,
        ColorWarn);
    const auto option_yolo = make_selection_row(
        "[3] Yes, enable YOLO mode",
        "Auto-approve ALL sensitive tools for this session.",
        selected_index == 2,
        ColorWarn);
    const auto option_reject = make_selection_row(
        "[4] No, suggest something",
        "Reject this call and tell Filo what to do instead.",
        selected_index == 3,
        ColorWarn);

    auto preview_box = vbox({
        hbox({ text("Tool") | dim, filler(), text(std::string(tool_name)) | ftxui::bold | color(ColorYellowBright) }),
        separator(),
        render_permission_fields(tool_name, args_preview)
    }) | UiBorder(ColorYellowDark);

    std::vector<Element> children = {
        hbox({
            text("\xe2\x9a\xa0 WARNING ") | ftxui::bold | color(ColorWarn),
            filler(),
            text("Up/Down: select  Enter: confirm  1/Y  2/A  3  4/N")
                | color(Color::GrayDark)
        }),
        separator(),
        text(permission_intent_message(tool_name))
            | color(Color::White),
        filler(),
        preview_box,
    };

    if (!diff_preview.empty()) {
        children.push_back(filler());
        children.push_back(render_permission_diff_preview(diff_preview));
    }

    children.push_back(filler());
    children.push_back(option_once);
    children.push_back(option_always);
    children.push_back(option_yolo);
    children.push_back(option_reject);
    children.push_back(filler());
    children.push_back(text("Esc / 4 / N rejects.  Ctrl+Y toggles YOLO globally.") | dim);

    return vbox(std::move(children))
        | UiBorder(ColorWarn)
        | size(HEIGHT, GREATER_THAN, 14);
}

Element render_model_selection_panel(int selected_index,
                                     std::string_view manual_description,
                                     std::string_view router_description,
                                     std::string_view auto_description,
                                     std::string_view router_policy,
                                     bool router_available,
                                     bool local_model_available) {
    const auto manual_option = make_selection_row(
        "[1] Manual",
        manual_description,
        selected_index == 0,
        ColorYellowDark);

    const std::string router_label = router_available
        ? std::format("[2] Router ({})", router_policy.empty() ? "policy not set" : router_policy)
        : "[2] Router (unavailable)";

    const auto router_option = make_selection_row(
        router_label,
        router_description,
        selected_index == 1,
        router_available ? ColorYellowDark : Color::GrayDark);

    const std::string auto_label = router_available
        ? "[3] Auto"
        : "[3] Auto (unavailable — needs router)";

    const auto auto_option = make_selection_row(
        auto_label,
        auto_description,
        selected_index == 2,
        router_available ? ColorYellowDark : Color::GrayDark);

    return vbox({
        hbox({
            text(" SELECT MODEL ") | ftxui::bold | color(Color::Black) | bgcolor(ColorYellowBright),
            filler(),
            text("Up/Down: select  Enter: confirm  1/2/3: quick choose")
                | color(Color::GrayDark)
        }),
        separator(),
        text("Choose how Filo routes model calls for this session.") | color(Color::White),
        filler(),
        manual_option,
        router_option,
        auto_option,
        filler(),
        text("Tip: /model manual, /model router, /model auto, or /model <provider>.") | dim,
        local_model_available
            ? (text("L: browse local GGUF files   Esc: close") | dim)
            : (text("Esc closes this panel.") | dim),
    }) | UiBorder(ColorYellowBright) | size(HEIGHT, GREATER_THAN, 14);
}

Element render_provider_selection_panel(const std::vector<std::string>& providers,
                                        int selected_index) {
    std::vector<Element> rows;
    rows.reserve(providers.size());
    for (std::size_t i = 0; i < providers.size(); ++i) {
        const bool sel = (static_cast<int>(i) == selected_index);
        auto row = hbox({
            text(std::format(" [{}] ", i + 1)) | color(ColorYellowBright),
            text(providers[i]) | ftxui::bold
        });
        rows.push_back(sel ? (row | bgcolor(ColorYellowDark) | color(Color::Black))
                           : (row | color(Color::White)));
    }

    return vbox({
        hbox({
            text(" AUTH PROVIDER ") | ftxui::bold | color(Color::Black) | bgcolor(ColorYellowBright),
            filler(),
            text("Up/Down: select  Enter: confirm  Esc: cancel") | color(Color::GrayDark)
        }),
        separator(),
        text("Select an authentication provider:") | color(Color::White),
        separator(),
        vbox(std::move(rows)),
        filler(),
        text("Esc closes this panel.") | dim
    }) | UiBorder(ColorYellowBright)
      | size(HEIGHT, GREATER_THAN, std::max(8, static_cast<int>(providers.size()) + 6));
}

Element render_settings_panel(std::string_view scope_label,
                              std::string_view scope_path,
                              const std::vector<SettingsPanelRow>& rows,
                              int selected_index,
                              std::string_view status_message) {
    Elements rendered_rows;
    rendered_rows.reserve(rows.size());

    for (std::size_t i = 0; i < rows.size(); ++i) {
        const bool selected = static_cast<int>(i) == selected_index;
        const auto& row = rows[i];
        std::string label = std::format(" [{}] {}: {}", i + 1, row.label, row.value);
        std::string description = row.description;
        if (row.inherited) {
            description += " Current scope inherits this value.";
        }

        rendered_rows.push_back(
            make_selection_row(label,
                               description,
                               selected,
                               ColorYellowDark));
    }

    Element status = status_message.empty()
        ? text("Tab switches scope. Left/Right changes values. Backspace resets. Esc closes.") | dim
        : text(std::string(status_message)) | color(ColorYellowBright);

    return vbox({
        hbox({
            text(" SETTINGS ") | ftxui::bold | color(Color::Black) | bgcolor(ColorYellowBright),
            text("  " + std::string(scope_label)) | ftxui::bold | color(ColorYellowBright),
            filler(),
            text("Tab: scope  Left/Right: change  Backspace: reset  Esc: close")
                | color(Color::GrayDark)
        }),
        separator(),
        text(std::string(scope_path)) | color(Color::GrayDark),
        separator(),
        vbox(std::move(rendered_rows)),
        filler(),
        status
    }) | UiBorder(ColorYellowBright)
      | size(HEIGHT, GREATER_THAN, std::max(10, static_cast<int>(rows.size()) * 2 + 6));
}

Element render_local_model_picker_panel(std::string_view current_dir,
                                        const std::vector<LocalModelEntry>& entries,
                                        int selected_index) {
    constexpr int kMaxVisible = 12;
    const int total = static_cast<int>(entries.size());

    std::vector<Element> visible_rows;
    if (total == 0) {
        visible_rows.push_back(text("  No .gguf files or subdirectories found here.") | dim);
    } else {
        const int scroll_offset = std::clamp(kMaxVisible / 2, 0, 3);
        const int view_start = std::clamp(selected_index - scroll_offset, 0,
                                          std::max(0, total - kMaxVisible));
        const int view_end = std::min(total, view_start + kMaxVisible);
        const int above = view_start;
        const int below = total - view_end;

        if (above > 0) {
            visible_rows.push_back(hbox({
                text(" \xe2\x86\x91 ") | color(ColorYellowDark),
                text(std::format("{} more above", above)) | color(Color::GrayDark) | dim | xflex
            }));
        }
        for (int i = view_start; i < view_end; ++i) {
            const auto& e = entries[static_cast<std::size_t>(i)];
            const bool sel = (i == selected_index);
            const std::string badge = e.is_directory ? " d " : "   ";
            auto row = hbox({
                text(badge) | color(e.is_directory
                    ? (sel ? Color::Black : ColorYellowBright)
                    : Color::GrayDark),
                text(e.name) | color(sel ? Color::Black : Color::White) | xflex,
            });
            visible_rows.push_back(sel ? (row | bgcolor(ColorYellowDark)) : row);
        }
        if (below > 0) {
            visible_rows.push_back(hbox({
                text(" \xe2\x86\x93 ") | color(ColorYellowDark),
                text(std::format("{} more below", below)) | color(Color::GrayDark) | dim | xflex
            }));
        }
    }

    return vbox({
        hbox({
            text(" LOCAL MODEL ") | ftxui::bold | color(Color::Black) | bgcolor(ColorYellowBright),
            filler(),
            text("\xe2\x86\x91\xe2\x86\x93  Enter: open/select  Esc: go up / close") | color(Color::GrayDark)
        }),
        separator(),
        text(std::format("  {}", current_dir)) | color(Color::GrayLight) | dim,
        separator(),
        vbox(std::move(visible_rows)),
        filler(),
        text("d = directory   Select a .gguf file to use as the local model.") | dim,
    }) | UiBorder(ColorYellowBright)
      | size(HEIGHT, GREATER_THAN, 12);
}

Element render_session_picker_panel(const std::vector<core::session::SessionInfo>& sessions,
                                           int selected_index) {
    if (sessions.empty()) {
        return vbox({
            hbox({
                text(" SESSIONS ") | ftxui::bold | color(Color::Black) | bgcolor(ColorYellowBright),
                filler(),
            }),
            separator(),
            text("  No saved sessions found.") | dim,
            filler(),
        }) | UiBorder(ColorYellowBright)
           | size(HEIGHT, EQUAL, 10);
    }

    const int max_visible = 8;
    const int start_idx = std::max(0, std::min(selected_index - max_visible / 2, 
                                              static_cast<int>(sessions.size()) - max_visible));
    const int end_idx = std::min(start_idx + max_visible, static_cast<int>(sessions.size()));

    Elements visible_rows;
    for (int i = start_idx; i < end_idx; ++i) {
        const auto& s = sessions[static_cast<std::size_t>(i)];
        bool is_selected = (i == selected_index);

        std::string ts = s.last_active_at.empty() ? s.created_at : s.last_active_at;
        if (ts.size() >= 16 && ts[10] == 'T') ts[10] = ' ';
        ts = ts.substr(0, 16);

        auto row = hbox({
            text(is_selected ? " \xe2\x96\xb6 " : "   ") | color(ColorYellowBright),
            text(std::format(" {} ", s.session_id)) | ftxui::bold | color(is_selected ? ColorYellowBright : Color::White),
            text(std::format(" {}  ", ts)) | color(is_selected ? Color::White : Color::GrayDark),
            text(std::format("{}/{:18} ", s.provider, s.model)) | color(is_selected ? Color::White : Color::GrayLight),
            text(std::format(" {:3d} turns ", s.turn_count)) | color(is_selected ? Color::White : Color::GrayDark),
            text(std::format(" [{}] ", s.mode)) | color(is_selected ? ColorYellowBright : Color::GrayDark),
            filler(),
        });

        if (is_selected) {
            row = row | bgcolor(Color::GrayDark);
        }
        visible_rows.push_back(row);
    }

    if (start_idx > 0) {
        visible_rows.insert(visible_rows.begin(), 
            hbox({text(" \xe2\x86\x91 ") | color(ColorYellowDark), text("more above") | dim}));
    }
    if (end_idx < static_cast<int>(sessions.size())) {
        visible_rows.push_back(
            hbox({text(" \xe2\x86\x93 ") | color(ColorYellowDark), text("more below") | dim}));
    }

    return vbox({
        hbox({
            text(" SESSIONS ") | ftxui::bold | color(Color::Black) | bgcolor(ColorYellowBright),
            filler(),
            text("\xe2\x86\x91\xe2\x86\x93  Enter: resume  Backspace/Del: delete  Esc: close") | color(Color::GrayDark)
        }),
        separator(),
        vbox(std::move(visible_rows)),
        filler(),
    }) | UiBorder(ColorYellowBright)
      | size(HEIGHT, GREATER_THAN, 12);
}

Element render_conversation_search_panel(std::string_view query,
                                         const std::vector<ConversationSearchHit>& hits,
                                         int selected_index) {
    constexpr int kMaxVisible = 8;
    const int total = static_cast<int>(hits.size());
    const int clamped_selected = total > 0
        ? std::clamp(selected_index, 0, total - 1)
        : 0;
    const int start_idx = std::max(0, std::min(clamped_selected - kMaxVisible / 2,
                                               std::max(0, total - kMaxVisible)));
    const int end_idx = std::min(start_idx + kMaxVisible, total);

    Elements rows;
    if (query.empty()) {
        rows.push_back(text("Type to search message text, tool output, and status lines.") | dim);
    } else if (hits.empty()) {
        rows.push_back(text(std::format("No matches for '{}'.", query)) | color(Color::GrayLight));
    } else {
        for (int i = start_idx; i < end_idx; ++i) {
            const auto& hit = hits[static_cast<std::size_t>(i)];
            const bool selected = (i == clamped_selected);

            auto row = vbox({
                hbox({
                    text(selected ? " > " : "   ") | color(ColorYellowBright),
                    text(std::format("[{}] ", hit.message_index + 1)) | color(ColorYellowDark),
                    text(hit.role) | ftxui::bold | color(selected ? ColorYellowBright : Color::White),
                }),
                hbox({
                    text("   "),
                    render_search_snippet(hit.snippet, query, selected) | xflex
                }),
            });

            if (selected) {
                row = row | bgcolor(Color::GrayDark);
            }
            rows.push_back(std::move(row));
        }

        if (start_idx > 0) {
            rows.insert(rows.begin(),
                hbox({text(" ↑ ") | color(ColorYellowDark), text("more above") | dim}));
        }
        if (end_idx < total) {
            rows.push_back(
                hbox({text(" ↓ ") | color(ColorYellowDark), text("more below") | dim}));
        }
    }

    return vbox({
        hbox({
            text(" SEARCH ") | ftxui::bold | color(Color::Black) | bgcolor(ColorYellowBright),
            filler(),
            text("Ctrl+F/Esc: close  Enter: jump  Backspace: edit  ↑↓: navigate")
                | color(Color::GrayDark),
        }),
        separator(),
        hbox({
            text(" Query: ") | color(ColorYellowBright),
            text(std::string(query.empty() ? "<empty>" : query))
                | color(Color::White)
                | ftxui::bold
                | xflex
        }),
        separator(),
        vbox(std::move(rows)),
    }) | UiBorder(ColorYellowBright)
      | size(HEIGHT, GREATER_THAN, 12);
}

// ---------------------------------------------------------------------------
// Question Dialog (AskUserQuestion tool) - kimi-cli style
// ---------------------------------------------------------------------------

namespace {

// Cyan color matching kimi-cli's question panel
inline const auto ColorQuestionCyan = ftxui::Color::RGB(0, 180, 220);
inline const auto ColorQuestionYellow = ftxui::Color::RGB(255, 200, 80);

Element render_question_tabs(const QuestionDialogState& state) {
    if (state.questions.size() <= 1) {
        return emptyElement();
    }
    
    Elements tabs;
    for (size_t i = 0; i < state.questions.size(); ++i) {
        const auto& q = state.questions[i];
        std::string label = q.header.empty() ? std::format("Q{}", i + 1) : q.header;
        
        std::string icon;
        ftxui::Color style;
        if (static_cast<int>(i) == state.current_question_index) {
            icon = "\xe2\x97\x8f";  // ●
            style = ColorQuestionCyan;
        } else if (i < state.answers.size()) {
            icon = "\xe2\x9c\x93";  // ✓
            style = Color::Green;
        } else {
            icon = "\xe2\x97\x8b";  // ○
            style = Color::GrayDark;
        }
        
        if (i > 0) {
            tabs.push_back(text("  ") | color(Color::GrayDark));
        }
        tabs.push_back(text(std::format("{} {}", icon, label)) | color(style));
    }
    
    return hbox(std::move(tabs));
}

Element render_question_option(const QuestionDialogOption& opt,
                                int index,
                                int selected_index,
                                bool multi_select,
                                const std::vector<int>& multi_selected,
                                bool show_other_input,
                                const std::string& other_input) {
    const int num = index + 1;
    const bool is_selected = (index == selected_index);
    const bool is_other_opt = (opt.label == "Other");
    
    Elements lines;
    
    // Main option line
    Element option_line;
    if (multi_select) {
        // Checkbox style for multi-select
        bool checked = std::find(multi_selected.begin(), multi_selected.end(), index) != multi_selected.end();
        std::string checkbox = checked ? "[\xe2\x9c\x93]" : "[ ]";  // [✓] or [ ]
        
        if (is_selected) {
            option_line = text(std::format("{} {}", checkbox, opt.label)) | color(ColorQuestionCyan);
        } else {
            option_line = text(std::format("{} {}", checkbox, opt.label)) | color(Color::GrayLight);
        }
    } else {
        // Radio style for single-select
        if (is_selected) {
            if (is_other_opt && show_other_input) {
                // Show inline input for Other option
                std::string display = other_input.empty() ? "" : other_input;
                option_line = text(std::format("\xe2\x86\x92 [{}] {}: {}\xe2\x96\x88", num, opt.label, display)) | color(ColorQuestionCyan);
            } else {
                option_line = text(std::format("\xe2\x86\x92 [{}] {}", num, opt.label)) | color(ColorQuestionCyan);
            }
        } else {
            option_line = text(std::format("  [{}] {}", num, opt.label)) | color(Color::GrayLight);
        }
    }
    lines.push_back(option_line);
    
    // Description (if any)
    if (!opt.description.empty() && !(is_other_opt && show_other_input)) {
        lines.push_back(text(std::format("      {}", opt.description)) | dim);
    }
    
    return vbox(std::move(lines));
}

} // namespace

Element render_question_dialog_panel(const QuestionDialogState& state) {
    if (state.questions.empty() || state.current_question_index >= static_cast<int>(state.questions.size())) {
        return emptyElement();
    }
    
    const auto& current_q = state.questions[state.current_question_index];
    Elements children;
    
    // Tab navigation (if multiple questions)
    auto tabs = render_question_tabs(state);
    if (tabs != emptyElement()) {
        children.push_back(std::move(tabs));
        children.push_back(text(""));  // Empty line
    }
    
    // Question text with ? prefix in yellow
    children.push_back(text(std::format("? {}", current_q.question)) | color(ColorQuestionYellow));
    
    // Multi-select hint
    if (current_q.multi_select) {
        children.push_back(text("  (SPACE to toggle, ENTER to submit)") | dim);
    }
    children.push_back(text(""));  // Empty line
    
    // Body content hint (if present)
    if (!current_q.body.empty()) {
        children.push_back(text("  \xe2\x96\xb6 Press ctrl-e to view full content") | color(ColorQuestionCyan));
        children.push_back(text(""));
    }
    
    // Options
    for (size_t i = 0; i < current_q.options.size(); ++i) {
        children.push_back(render_question_option(
            current_q.options[i],
            static_cast<int>(i),
            state.selected_option,
            current_q.multi_select,
            state.multi_selected,
            state.show_other_input,
            state.other_input_text
        ));
    }
    
    // Other input hint
    if (state.show_other_input) {
        children.push_back(text(""));
        children.push_back(text("  Type your answer, then press Enter to submit.") | dim);
    } else if (state.questions.size() > 1) {
        children.push_back(text(""));
        children.push_back(text("  \xe2\x97\x84/\xe2\x96\xba switch  \xe2\x96\xb2/\xe2\x96\xbc select  Enter: submit  Esc: exit") | dim);
    }
    
    // Build the panel with cyan border and "? QUESTION" title
    auto content = vbox(std::move(children));
    
    return vbox({
        hbox({
            text(" ? QUESTION ") | ftxui::bold | color(ColorQuestionCyan),
            filler(),
            text("Up/Down: select  Enter: confirm  1-5: quick select  Esc: exit") | color(Color::GrayDark)
        }),
        separator(),
        content,
        filler(),
    }) | UiBorder(ColorQuestionCyan)
      | size(HEIGHT, GREATER_THAN, 12);
}

} // namespace tui
