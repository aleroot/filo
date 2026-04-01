#include "Conversation.hpp"
#include "Constants.hpp"
#include "MarkdownRenderer.hpp"
#include "StringUtils.hpp"
#include "TuiTheme.hpp"
#include "core/permissions/PermissionSystem.hpp"

#include <ftxui/dom/elements.hpp>
#include <simdjson.h>
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <format>
#include <limits>
#include <optional>
#include <random>

using namespace ftxui;

namespace tui {

namespace {

// ============================================================================
// String Utilities
// ============================================================================

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
    for (unsigned char ch : text) {
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

std::string truncate_preview(std::string_view text, std::size_t max_len = kToolPreviewMaxLen) {
    std::string cleaned = collapse_whitespace(text);
    if (cleaned.size() <= max_len) {
        return cleaned;
    }
    if (max_len <= 1) {
        return cleaned.substr(0, max_len);
    }
    return cleaned.substr(0, max_len - 1) + "...";
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

std::optional<std::string> first_string_field(const simdjson::dom::object& object) {
    for (const auto field : object) {
        std::string_view value;
        if (field.value.get(value) == simdjson::SUCCESS) {
            return std::string(value);
        }
    }
    return std::nullopt;
}

std::optional<int> extract_int_field(const simdjson::dom::object& object, std::string_view key) {
    int64_t signed_value = 0;
    if (object[key].get(signed_value) == simdjson::SUCCESS) {
        if (signed_value > std::numeric_limits<int>::max()) {
            return std::numeric_limits<int>::max();
        }
        if (signed_value < std::numeric_limits<int>::min()) {
            return std::numeric_limits<int>::min();
        }
        return static_cast<int>(signed_value);
    }

    uint64_t unsigned_value = 0;
    if (object[key].get(unsigned_value) == simdjson::SUCCESS) {
        if (unsigned_value > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            return std::numeric_limits<int>::max();
        }
        return static_cast<int>(unsigned_value);
    }

    return std::nullopt;
}

bool has_output_truncation_marker(std::string_view output) {
    return output.find("[OUTPUT TRUNCATED AT") != std::string_view::npos;
}

std::string extract_patch_preview(std::string_view patch) {
    const auto patch_text = trim_copy(patch);
    if (patch_text.empty()) {
        return "patch";
    }
    const std::string_view patch_view{patch_text};
    
    constexpr std::array prefixes = {
        std::string_view{"*** Update File: "},
        std::string_view{"*** Add File: "},
        std::string_view{"*** Delete File: "},
        std::string_view{"*** Move to: "},
        std::string_view{"--- "},
        std::string_view{"+++ "}
    };
    
    std::size_t start = 0;
    while (start < patch_view.size()) {
        const std::size_t end = patch_view.find('\n', start);
        const std::string_view line = patch_view.substr(
            start, end == std::string_view::npos ? std::string_view::npos : end - start);
        
        for (const auto prefix : prefixes) {
            if (line.starts_with(prefix)) {
                return truncate_preview(line.substr(prefix.size()));
            }
        }
        
        if (!trim_copy(line).empty()) {
            return truncate_preview(line);
        }
        
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    
    return "patch";
}

// ============================================================================
// Message ID Generation
// ============================================================================

std::string generate_message_id() {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    static std::uniform_int_distribution<> dis(0, 15);
    
    std::string id;
    id.reserve(16);
    for (int i = 0; i < 16; ++i) {
        id += "0123456789abcdef"[dis(gen)];
    }
    return id;
}

// ============================================================================
// Rendering Helpers
// ============================================================================

Element render_text_lines_preserving_newlines(std::string_view content, Color text_color) {
    std::vector<std::string> lines = split_lines(std::string(content));
    if (lines.empty()) {
        lines.emplace_back();
    }

    std::vector<Element> rows;
    rows.reserve(lines.size());
    for (const auto& line : lines) {
        if (line.empty()) {
            rows.push_back(ftxui::text(""));
            continue;
        }

        std::size_t indent_len = 0;
        while (indent_len < line.size()
               && (line[indent_len] == ' ' || line[indent_len] == '\t')) {
            ++indent_len;
        }

        if (indent_len > 0 && indent_len < line.size()) {
            rows.push_back(
                hbox({
                    ftxui::text(line.substr(0, indent_len)) | ftxui::color(text_color),
                    paragraph(line.substr(indent_len)) | ftxui::color(text_color) | xflex
                }) | xflex);
        } else {
            rows.push_back(paragraph(line) | ftxui::color(text_color) | xflex);
        }
    }

    return vbox(std::move(rows)) | xflex;
}

struct LimitedTextResult {
    std::string text;
    std::size_t hidden_lines = 0;
};

LimitedTextResult clamp_tool_result_lines(std::string_view content, std::size_t max_lines) {
    const std::vector<std::string> lines = split_lines(std::string(content));
    if (lines.size() <= max_lines || max_lines == 0) {
        return {
            .text = std::string(content),
            .hidden_lines = 0,
        };
    }

    std::string clamped;
    std::size_t used_lines = 0;
    for (const auto& line : lines) {
        if (used_lines >= max_lines) {
            break;
        }
        if (!clamped.empty()) {
            clamped.push_back('\n');
        }
        clamped += line;
        ++used_lines;
    }

    return {
        .text = std::move(clamped),
        .hidden_lines = lines.size() - max_lines,
    };
}

Element render_lightbulb_prefix(bool show) {
    if (!show) {
        return ftxui::text("    ");
    }
    return hbox({
        ftxui::text("💡") | ftxui::color(ColorYellowBright),
        ftxui::text("  "),
    });
}

Element user_message_bubble(std::string_view content, std::string_view timestamp) {
    auto bubble = render_text_lines_preserving_newlines(content, Color::White)
                | UiBorder(ColorYellowBright);

    if (timestamp.empty()) {
        return bubble;
    }
    return vbox({
        hbox({ filler(), ftxui::text(std::string(timestamp)) | ftxui::color(Color::GrayDark) }),
        std::move(bubble)
    });
}

// ============================================================================
// Gemini CLI Style Status Message Rendering
// ============================================================================

Element render_info_badge(std::string_view icon, Color badge_color) {
    return ftxui::text(std::string(icon)) | ftxui::color(badge_color) | bold;
}

Element render_status_text(std::string_view text_content, Color text_color, std::size_t indent_width = 3) {
    std::vector<std::string> lines = split_lines(std::string(text_content));
    if (lines.empty()) {
        return ftxui::text("") | ftxui::color(text_color);
    }

    std::vector<Element> rows;
    rows.reserve(lines.size());
    
    for (std::size_t i = 0; i < lines.size(); ++i) {
        const auto& line = lines[i];
        if (line.empty()) {
            rows.push_back(ftxui::text(""));
            continue;
        }
        
        if (i == 0) {
            rows.push_back(paragraph(std::string(line)) | ftxui::color(text_color) | xflex);
        } else {
            rows.push_back(
                hbox({
                    ftxui::text(std::string(indent_width, ' ')),
                    paragraph(std::string(line)) | ftxui::color(text_color) | xflex
                }));
        }
    }
    
    return vbox(std::move(rows));
}

// ============================================================================
// Tool Group Rendering (Gemini CLI Style)
// ============================================================================

Element render_tool_header(const ToolActivity& tool, 
                           std::size_t tick,
                           bool show_spinner,
                           int /*terminal_width*/,
                           Color /*border_color*/,
                           bool /*is_first*/) {
    const Color status_color = tui::tool_status_color(tool.status);
    std::string icon = std::string(tool.status == ToolActivity::Status::Executing && show_spinner
                                   ? tui::tool_status_spinner(tick)
                                   : tui::tool_status_icon(tool.status));
    
    Element status_el = ftxui::text(std::format(" {} ", icon)) | ftxui::color(status_color) | bold;
    Element name_el = ftxui::text(tool.name) | bold | ftxui::color(ColorYellowBright);
    
    std::vector<Element> header_items;
    header_items.push_back(std::move(status_el));
    header_items.push_back(ftxui::text(" "));
    header_items.push_back(std::move(name_el));

    if (tool.auto_approved) {
        header_items.push_back(ftxui::text("  "));
        header_items.push_back(
            ftxui::text("auto-approved") | ftxui::color(ColorYellowDark) | dim);
    }
    
    if (!tool.description.empty()) {
        header_items.push_back(ftxui::text("  "));
        header_items.push_back(
            ftxui::text(tool.description) | ftxui::color(Color::GrayDark) | xflex);
    }
    
    if (tool.status != ToolActivity::Status::Pending && 
        tool.status != ToolActivity::Status::Executing) {
        header_items.push_back(filler());
        std::string status_label = std::string(tui::tool_status_label(tool.status));
        if (tool.name == "run_terminal_command" && tool.result.exit_code.has_value()) {
            status_label += std::format(" · exit {}", *tool.result.exit_code);
        }
        header_items.push_back(ftxui::text(std::move(status_label)) | ftxui::color(status_color));
    }
    
    return hbox(std::move(header_items)) | xflex;
}

Element render_tool_progress(const ToolActivity& tool) {
    if (!tool.progress.has_value() || !tool.progress_total.has_value()) {
        return emptyElement();
    }
    
    const int progress = *tool.progress;
    const int total = *tool.progress_total;
    const float ratio = total > 0 ? static_cast<float>(progress) / total : 0.0f;
    const int filled = static_cast<int>(ratio * 20);
    const int empty = 20 - filled;
    
    std::string bar = std::format("[{}{}] {}/{} {}",
        std::string(filled, '#'),
        std::string(empty, '-'),
        progress,
        total,
        tool.progress_message);
    
    return ftxui::text(bar) | ftxui::color(ColorYellowDark);
}

Element render_tool_result(const ToolActivity& tool,
                           const ConversationRenderOptions& options,
                           int /*available_height*/,
                           int /*terminal_width*/) {
    if (tool.result.empty()) {
        return emptyElement();
    }
    
    const bool is_error = tool.status == ToolActivity::Status::Failed ||
                          tool.status == ToolActivity::Status::Denied;
    const Color text_color = is_error ? ColorToolFail : Color::GrayLight;

    const bool is_terminal_output = tool.name == "run_terminal_command";
    std::vector<Element> rows;
    if (is_terminal_output) {
        std::string label = "Terminal output";
        if (tool.result.exit_code.has_value()) {
            label += std::format(" (exit {})", *tool.result.exit_code);
        }
        rows.push_back(ftxui::text(std::move(label)) | ftxui::color(ColorYellowDark) | bold);
    }

    const std::size_t preview_lines = options.tool_result_preview_max_lines == 0
        ? kToolResultPreviewMaxLines
        : options.tool_result_preview_max_lines;
    const LimitedTextResult display_result = options.expand_tool_results
        ? LimitedTextResult{
            .text = tool.result.summary,
            .hidden_lines = 0,
        }
        : clamp_tool_result_lines(tool.result.summary, preview_lines);
    rows.push_back(render_text_lines_preserving_newlines(display_result.text, text_color));

    if (display_result.hidden_lines > 0) {
        rows.push_back(
            ftxui::text(std::format(
                "... {} more lines hidden to keep the conversation readable.",
                display_result.hidden_lines))
            | ftxui::color(Color::GrayDark)
            | dim);
    }

    if (tool.result.truncated) {
        rows.push_back(
            ftxui::text("Output was truncated after reaching the terminal output limit.")
            | ftxui::color(ColorWarn)
            | dim);
    }

    Element content = vbox(std::move(rows)) | xflex;
    if (is_terminal_output) {
        content = content | UiBorder(Color::GrayDark);
    }
    return content;
}

Element render_tool_diff_preview(const ToolDiffPreview& preview) {
    if (preview.empty()) {
        return emptyElement();
    }
    
    std::vector<Element> rows;
    if (!preview.title.empty()) {
        rows.push_back(ftxui::text(std::format("Diff: {}", preview.title)) | 
                      ftxui::color(ColorYellowDark) | bold);
    }
    
    for (const auto& line : preview.lines) {
        Color line_color = Color::GrayLight;
        std::string prefix = "  ";
        
        switch (line.kind) {
            case DiffLineKind::Add:
                line_color = Color::RGB(120, 220, 130);
                prefix = "+ ";
                break;
            case DiffLineKind::Delete:
                line_color = Color::RGB(255, 150, 130);
                prefix = "- ";
                break;
            case DiffLineKind::Hunk:
                line_color = Color::RGB(145, 196, 255);
                prefix = "@@ ";
                break;
            case DiffLineKind::Header:
                line_color = Color::GrayDark;
                prefix = "  ";
                break;
            default:
                break;
        }
        
        rows.push_back(
            hbox({
                ftxui::text(prefix) | ftxui::color(line_color),
                ftxui::text(line.content) | ftxui::color(line_color) | xflex
            }));
    }
    
    if (preview.hidden_line_count > 0) {
        rows.push_back(
            ftxui::text(std::format("... {} more lines", preview.hidden_line_count)) | 
            dim | ftxui::color(Color::GrayDark));
    }
    
    return vbox(std::move(rows)) | border;
}

Element render_tool_item(const ToolActivity& tool,
                         std::size_t tick,
                         const ConversationRenderOptions& options,
                         int terminal_width,
                         Color border_color,
                         bool is_first,
                         bool /*is_last*/) {
    std::vector<Element> tool_elements;
    
    tool_elements.push_back(
        render_tool_header(tool, tick, options.show_spinner, terminal_width, 
                          border_color, is_first));
    
    if (tool.status == ToolActivity::Status::Executing && tool.progress.has_value()) {
        tool_elements.push_back(render_tool_progress(tool));
    }
    
    if (!tool.result.empty()) {
        tool_elements.push_back(separator() | ftxui::color(Color::GrayDark));
        tool_elements.push_back(render_tool_result(tool, options, 10, terminal_width));
    }
    
    if (!tool.diff_preview.empty()) {
        tool_elements.push_back(render_tool_diff_preview(tool.diff_preview));
    }
    
    return vbox(std::move(tool_elements));
}

Element render_tool_group_container(const UiMessage& msg,
                                    std::size_t tick,
                                    const ConversationRenderOptions& options,
                                    int terminal_width) {
    if (msg.tools.empty()) {
        return emptyElement();
    }
    
    const int content_width = terminal_width - 4;
    const Color border_color = Color::GrayDark;
    
    std::vector<Element> content_elements;
    
    for (std::size_t i = 0; i < msg.tools.size(); ++i) {
        const auto& tool = msg.tools[i];
        const bool is_first = (i == 0);
        const bool is_last = (i == msg.tools.size() - 1);
        
        content_elements.push_back(
            render_tool_item(tool, tick, options, content_width, 
                           border_color, is_first, is_last));
        
        if (!is_last) {
            content_elements.push_back(separator() | ftxui::color(Color::GrayDark) | dim);
        }
    }
    
    Element content = vbox(std::move(content_elements));
    return content | border | ftxui::color(border_color);
}

} // anonymous namespace

// ============================================================================
// Public API Implementation
// ============================================================================

auto current_time_str() -> std::string {
    auto now = std::chrono::system_clock::now();
    return std::format("{:%H:%M:%S}", std::chrono::floor<std::chrono::seconds>(now));
}

// ============================================================================
// Message Factory Functions
// ============================================================================

UiMessage make_user_message(std::string text, std::string timestamp) {
    UiMessage msg;
    msg.type = MessageType::User;
    msg.id = generate_message_id();
    msg.text = std::move(text);
    msg.timestamp = std::move(timestamp);
    return msg;
}

UiMessage make_assistant_message(std::string text, std::string timestamp, bool pending) {
    UiMessage msg;
    msg.type = MessageType::Assistant;
    msg.id = generate_message_id();
    msg.text = std::move(text);
    msg.timestamp = std::move(timestamp);
    msg.pending = pending;
    msg.thinking = pending;
    return msg;
}

UiMessage make_info_message(std::string text, std::optional<std::string> secondary_text) {
    UiMessage msg;
    msg.type = MessageType::Info;
    msg.id = generate_message_id();
    msg.text = std::move(text);
    if (secondary_text) {
        msg.secondary_text = std::move(*secondary_text);
    }
    msg.margin_top = 1;
    return msg;
}

UiMessage make_warning_message(std::string text) {
    UiMessage msg;
    msg.type = MessageType::Warning;
    msg.id = generate_message_id();
    msg.text = std::move(text);
    msg.margin_top = 1;
    return msg;
}

UiMessage make_error_message(std::string text) {
    UiMessage msg;
    msg.type = MessageType::Error;
    msg.id = generate_message_id();
    msg.text = std::move(text);
    msg.margin_top = 1;
    return msg;
}

UiMessage make_tool_group_message(std::vector<ToolActivity> tools,
                                   bool border_top,
                                   bool border_bottom) {
    UiMessage msg;
    msg.type = MessageType::ToolGroup;
    msg.id = generate_message_id();
    msg.tools = std::move(tools);
    msg.tool_group_border_top = border_top;
    msg.tool_group_border_bottom = border_bottom;
    return msg;
}

UiMessage make_system_message(std::string text) {
    UiMessage msg;
    msg.type = MessageType::System;
    msg.id = generate_message_id();
    msg.text = std::move(text);
    return msg;
}

ToolActivity make_tool_activity(std::string id,
                                 std::string name,
                                 std::string args,
                                 std::string description) {
    ToolActivity tool;
    tool.id = std::move(id);
    tool.name = std::move(name);
    tool.args = std::move(args);
    tool.description = std::move(description);
    tool.status = ToolActivity::Status::Pending;
    return tool;
}

void apply_tool_result(ToolActivity& tool, std::string_view result_payload) {
    tool.result.clear();
    auto set_result_summary = [&](std::string summary) {
        tool.result.summary = std::move(summary);
        tool.result.truncated = has_output_truncation_marker(tool.result.summary);
    };

    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(result_payload).get(document) != simdjson::SUCCESS) {
        tool.status = ToolActivity::Status::Succeeded;
        set_result_summary(std::string(result_payload));
        return;
    }

    simdjson::dom::object object;
    if (document.get(object) != simdjson::SUCCESS) {
        tool.status = ToolActivity::Status::Succeeded;
        set_result_summary(std::string(result_payload));
        return;
    }

    if (const auto error = extract_string_field(object, {"error"})) {
        tool.status = ToolActivity::Status::Failed;
        set_result_summary(*error);
        return;
    }

    if (tool.name == "run_terminal_command") {
        if (const auto exit_code = extract_int_field(object, "exit_code")) {
            tool.result.exit_code = *exit_code;
            tool.status = (*exit_code == 0)
                ? ToolActivity::Status::Succeeded
                : ToolActivity::Status::Failed;
        } else {
            tool.status = ToolActivity::Status::Succeeded;
        }

        if (const auto output = extract_string_field(object, {"output"})) {
            set_result_summary(*output);
        } else if (tool.result.exit_code.has_value()) {
            set_result_summary((*tool.result.exit_code == 0)
                ? "Command completed with no output."
                : std::format("Command exited with status {}.", *tool.result.exit_code));
        } else {
            set_result_summary("Done");
        }
        return;
    }

    if (const auto output = extract_string_field(object, {"output"})) {
        tool.status = ToolActivity::Status::Succeeded;
        set_result_summary(output->empty() ? std::string("Done") : *output);
        return;
    }

    if (const auto content = extract_string_field(object, {"content"})) {
        tool.status = ToolActivity::Status::Succeeded;
        set_result_summary(content->empty() ? std::string("Done") : *content);
        return;
    }

    if (const auto matches = extract_string_field(object, {"matches"})) {
        tool.status = ToolActivity::Status::Succeeded;
        set_result_summary(matches->empty() ? "no matches" : *matches);
        return;
    }

    bool success = false;
    if (object["success"].get(success) == simdjson::SUCCESS) {
        tool.status = success ? ToolActivity::Status::Succeeded : ToolActivity::Status::Failed;
        set_result_summary(success ? "Done" : "Tool reported failure.");
        return;
    }

    tool.status = ToolActivity::Status::Succeeded;
    set_result_summary(std::string(result_payload));
}

// ============================================================================
// Rendering Functions
// ============================================================================

Element render_user_message(const UiMessage& msg, const ConversationRenderOptions& options) {
    const std::string_view timestamp = options.show_timestamps
        ? std::string_view{msg.timestamp}
        : std::string_view{};
    return vbox({
        user_message_bubble(msg.text, timestamp),
        ftxui::text("")
    });
}

Element render_assistant_message(const UiMessage& msg,
                                 std::size_t tick,
                                 const ConversationRenderOptions& options) {
    std::vector<Element> elements;

    bool has_executing_tools = false;
    bool has_completed_tools = false;
    for (const auto& tool : msg.tools) {
        if (tool.status == ToolActivity::Status::Executing) {
            has_executing_tools = true;
        } else if (tool.status == ToolActivity::Status::Succeeded ||
                   tool.status == ToolActivity::Status::Failed) {
            has_completed_tools = true;
        }
    }
    
    bool show_thinking = msg.thinking ||
                         (msg.pending && has_completed_tools && !has_executing_tools);

    // Show waiting state when empty and not thinking
    if (msg.text.empty() && msg.tools.empty() && !msg.thinking) {
        const auto label = msg.pending ? "Waiting for the model..." : "No response.";
        const bool show = msg.pending || msg.show_lightbulb;
        elements.push_back(
            hbox({
                render_lightbulb_prefix(show),
                ftxui::text(label) | ftxui::color(Color::GrayLight) | dim
            }));
    }

    // Render tools first (completed work)
    if (!msg.tools.empty()) {
        UiMessage tool_group = msg;
        tool_group.type = MessageType::ToolGroup;
        elements.push_back(render_tool_group(tool_group, tick, options));
    }

    // Render text response (middle)
    if (!msg.text.empty()) {
        if (!msg.tools.empty()) {
            elements.push_back(ftxui::text(""));
        }
        elements.push_back(render_markdown(msg.text, Color::White));
    }

    // Render thinking indicator at the BOTTOM (current activity)
    // This ensures users see what's happening NOW when looking at the bottom
    if (show_thinking) {
        std::string label;
        if (has_completed_tools && !has_executing_tools && !msg.thinking) {
            label = options.show_spinner
                ? std::string("Analyzing") + std::string(thinking_pulse_frame(tick))
                : std::string("Analyzing...");
        } else {
            label = options.show_spinner
                ? std::string("Thinking") + std::string(thinking_pulse_frame(tick))
                : std::string("Thinking...");
        }
        elements.push_back(
            hbox({
                render_lightbulb_prefix(true),
                ftxui::text(label) | ftxui::color(ColorYellowDark) | dim
            }));
    }

    return vbox(std::move(elements));
}

Element render_info_message(const UiMessage& msg) {
    const std::string_view icon = msg.icon.empty() ? "ℹ " : std::string_view(msg.icon);
    const Color msg_color = msg.custom_color.value_or(ColorYellowDark);
    
    std::vector<Element> lines;
    lines.push_back(
        hbox({
            render_info_badge(icon, msg_color),
            ftxui::text(" "),
            render_status_text(msg.text, msg_color)
        }));
    
    if (!msg.secondary_text.empty()) {
        lines.push_back(
            hbox({
                ftxui::text(std::string(icon.size(), ' ')),
                ftxui::text(" "),
                ftxui::text(msg.secondary_text) | ftxui::color(Color::GrayDark)
            }));
    }
    
    Element result = vbox(std::move(lines));
    
    if (msg.margin_top > 0) {
        result = vbox({
            ftxui::text(std::string(msg.margin_top, '\n')),
            result
        });
    }
    
    return result;
}

Element render_warning_message(const UiMessage& msg) {
    const std::string_view warn_icon = "⚠ ";
    const Color warn_color = ColorWarn;
    
    Element result = hbox({
        render_info_badge(warn_icon, warn_color),
        ftxui::text(" "),
        render_status_text(msg.text, warn_color)
    });
    
    if (msg.margin_top > 0) {
        result = vbox({
            ftxui::text(std::string(msg.margin_top, '\n')),
            result
        });
    }
    
    return result;
}

Element render_error_message(const UiMessage& msg) {
    const std::string_view err_icon = "✗ ";
    const Color err_color = ColorToolFail;
    
    Element result = hbox({
        render_info_badge(err_icon, err_color),
        ftxui::text(" "),
        render_status_text(msg.text, err_color)
    });
    
    if (msg.margin_top > 0) {
        result = vbox({
            ftxui::text(std::string(msg.margin_top, '\n')),
            result
        });
    }
    
    return result;
}

Element render_tool_group(const UiMessage& msg,
                          std::size_t tick,
                          const ConversationRenderOptions& options) {
    return render_tool_group_container(msg, tick, options, 100);
}

Element render_system_message(const UiMessage& msg) {
    std::vector<Element> lines;
    for (const auto& line : split_lines(msg.text)) {
        lines.push_back(ftxui::text(line) | ftxui::color(ColorYellowDark));
    }
    return vbox(std::move(lines));
}

// ============================================================================
// Main Render Function
// ============================================================================

Decorator scroll_position_relative(float x, float y) {
    class Impl : public ftxui::Node {
    public:
        Impl(Element child, float x, float y)
            : ftxui::Node(ftxui::unpack(std::move(child))), x_(x), y_(y) {}

        void ComputeRequirement() override {
            children_[0]->ComputeRequirement();
            requirement_ = children_[0]->requirement();
            requirement_.focused.enabled = false;
            requirement_.focused.box.x_min = int(float(requirement_.min_x) * x_);
            requirement_.focused.box.y_min = int(float(requirement_.min_y) * y_);
            requirement_.focused.box.x_max = int(float(requirement_.min_x) * x_);
            requirement_.focused.box.y_max = int(float(requirement_.min_y) * y_);
        }

        void SetBox(ftxui::Box box) override {
            ftxui::Node::SetBox(box);
            children_[0]->SetBox(box);
        }

    private:
        const float x_;
        const float y_;
    };

    return [x, y](Element child) {
        return std::make_shared<Impl>(std::move(child), x, y);
    };
}

Element render_history_panel(const std::vector<UiMessage>& messages,
                             std::size_t tick,
                             ConversationRenderOptions options) {
    std::vector<Element> msg_elements;
    msg_elements.reserve(messages.size() * 2);

    for (const auto& msg : messages) {
        switch (msg.type) {
            case MessageType::User:
                msg_elements.push_back(render_user_message(msg, options));
                break;
            case MessageType::Assistant:
                msg_elements.push_back(render_assistant_message(msg, tick, options));
                msg_elements.push_back(ftxui::text(""));
                break;
            case MessageType::Info:
                msg_elements.push_back(render_info_message(msg));
                break;
            case MessageType::Warning:
                msg_elements.push_back(render_warning_message(msg));
                break;
            case MessageType::Error:
                msg_elements.push_back(render_error_message(msg));
                break;
            case MessageType::ToolGroup:
                msg_elements.push_back(render_tool_group(msg, tick, options));
                msg_elements.push_back(ftxui::text(""));
                break;
            case MessageType::System:
                msg_elements.push_back(render_system_message(msg));
                break;
        }
    }

    return vbox(std::move(msg_elements))
        | scroll_position_relative(0, options.scroll_pos)
        | vscroll_indicator
        | yframe
        | yflex;
}

// ============================================================================
// Tool Status Helpers
// ============================================================================

Color tool_status_color(ToolActivity::Status status) {
    switch (status) {
        case ToolActivity::Status::Pending:    return ColorToolPending;
        case ToolActivity::Status::Executing:  return ColorYellowBright;
        case ToolActivity::Status::Succeeded:  return ColorToolDone;
        case ToolActivity::Status::Failed:     return ColorToolFail;
        case ToolActivity::Status::Denied:     return Color::GrayLight;
        case ToolActivity::Status::Cancelled:  return Color::GrayLight;
    }
    return Color::White;
}

std::string_view tool_status_icon(ToolActivity::Status status) {
    switch (status) {
        case ToolActivity::Status::Pending:    return "○";
        case ToolActivity::Status::Executing:  return "◐";
        case ToolActivity::Status::Succeeded:  return "✓";
        case ToolActivity::Status::Failed:     return "✗";
        case ToolActivity::Status::Denied:     return "⊘";
        case ToolActivity::Status::Cancelled:  return "⊘";
    }
    return "○";
}

std::string_view tool_status_spinner(std::size_t tick) {
    static constexpr std::array<std::string_view, 4> frames = {
        "○", "◔", "◑", "◕"
    };
    return frames[tick % frames.size()];
}

std::string_view tool_status_label(ToolActivity::Status status) {
    switch (status) {
        case ToolActivity::Status::Pending:    return "Pending";
        case ToolActivity::Status::Executing:  return "Running";
        case ToolActivity::Status::Succeeded:  return "Done";
        case ToolActivity::Status::Failed:     return "Failed";
        case ToolActivity::Status::Denied:     return "Denied";
        case ToolActivity::Status::Cancelled:  return "Cancelled";
    }
    return "";
}

// ============================================================================
// Animation Frames
// ============================================================================

std::string_view thinking_pulse_frame(std::size_t tick) {
    static constexpr std::array<std::string_view, 6> frames = {
        "   ", ".  ", ".. ", "...", " ..", "  ."
    };
    return frames[tick % frames.size()];
}

std::string_view spinner_frame(std::size_t tick) {
    static constexpr std::array<std::string_view, 4> frames = {
        "·", "•", "●", "•"
    };
    return frames[tick % frames.size()];
}

// ============================================================================
// Tool Summary Functions
// ============================================================================

std::string summarize_tool_arguments(std::string_view tool_name, std::string_view tool_args) {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(tool_args).get(document) != simdjson::SUCCESS) {
        return truncate_preview(tool_args);
    }

    simdjson::dom::object object;
    if (document.get(object) != simdjson::SUCCESS) {
        return truncate_preview(tool_args);
    }

    if (tool_name == "move_file") {
        const auto source = extract_string_field(object, {"source_path", "from_path"});
        const auto destination = extract_string_field(object, {"destination_path", "to_path"});
        if (source && destination) {
            return truncate_preview(*source + " -> " + *destination);
        }
    }

    if (tool_name == "grep_search") {
        const auto pattern = extract_string_field(object, {"pattern"});
        const auto dir = extract_string_field(object, {"dir_path"});
        if (pattern && dir) {
            return truncate_preview(*pattern + " in " + *dir);
        }
    }

    if (tool_name == "apply_patch") {
        if (const auto patch = extract_string_field(object, {"patch"})) {
            return extract_patch_preview(*patch);
        }
    }

    if (tool_name == "run_terminal_command") {
        const auto command = extract_string_field(object, {"command"});
        const auto working_dir = extract_string_field(object, {"working_dir"});
        if (command && working_dir) {
            return std::format("cwd: {} | cmd: {}", *working_dir, *command);
        }
        if (command) {
            return std::format("cmd: {}", *command);
        }
    }

    if (const auto preview = extract_string_field(object, {
            "command",
            "file_path",
            "path",
            "dir_path",
            "destination_path",
            "source_path",
            "pattern",
            "query",
            "url",
            "name"
        })) {
        return truncate_preview(*preview);
    }

    if (const auto preview = first_string_field(object)) {
        return truncate_preview(*preview);
    }

    return {};
}

std::string format_tool_description(std::string_view tool_name, std::string_view tool_args) {
    return summarize_tool_arguments(tool_name, tool_args);
}

// ============================================================================
// Tool Lookup
// ============================================================================

ToolActivity* find_tool_activity(UiMessage& message, std::string_view tool_id) {
    for (auto& tool : message.tools) {
        if (tool.id == tool_id) {
            return &tool;
        }
    }
    return nullptr;
}

const ToolActivity* find_tool_activity(const UiMessage& message, std::string_view tool_id) {
    for (const auto& tool : message.tools) {
        if (tool.id == tool_id) {
            return &tool;
        }
    }
    return nullptr;
}

// ============================================================================
// Allow-list Helpers
// ============================================================================

std::string make_allow_key(std::string_view tool_name, std::string_view tool_args) {
    return core::permissions::make_allow_key(tool_name, tool_args);
}

std::string make_allow_label(std::string_view tool_name, std::string_view tool_args) {
    return core::permissions::make_allow_label(tool_name, tool_args);
}

// ============================================================================
// summarize_tool_result (for test compatibility)
// ============================================================================

ToolResultSummary summarize_tool_result(std::string_view result) {
    ToolResultSummary summary;

    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(result).get(document) != simdjson::SUCCESS) {
        if (result.find("\"error\"") != std::string_view::npos) {
            summary.state = ToolResultSummary::State::Failed;
            summary.preview = truncate_preview(result);
        }
        return summary;
    }

    simdjson::dom::object object;
    if (document.get(object) != simdjson::SUCCESS) {
        return summary;
    }

    if (const auto error = extract_string_field(object, {"error"})) {
        summary.state = error->find("denied") != std::string_view::npos
            ? ToolResultSummary::State::Denied
            : ToolResultSummary::State::Failed;
        summary.preview = truncate_preview(*error);
        return summary;
    }

    if (const auto matches = extract_string_field(object, {"matches"})) {
        summary.preview = matches->empty() ? "no matches" : truncate_preview(*matches);
        return summary;
    }

    if (const auto output = extract_string_field(object, {"output"})) {
        summary.preview = std::string(*output);
        return summary;
    }

    if (extract_string_field(object, {"content"})) {
        summary.preview = "content loaded";
        return summary;
    }

    if (const auto value = extract_string_field(object, {"time", "message", "result"})) {
        summary.preview = truncate_preview(*value);
        return summary;
    }

    bool success = false;
    if (object["success"].get(success) == simdjson::SUCCESS && success) {
        summary.preview = "done";
    }

    return summary;
}

} // namespace tui
