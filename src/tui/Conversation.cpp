#include "Conversation.hpp"
#include "Constants.hpp"
#include "MarkdownRenderer.hpp"
#include "StringUtils.hpp"
#include "TuiTheme.hpp"
#include "core/permissions/PermissionSystem.hpp"
#include "core/tools/ToolNames.hpp"
#include "core/utils/JsonUtils.hpp"
#include "core/utils/StringUtils.hpp"

#include <ftxui/dom/elements.hpp>
#include <ftxui/dom/node.hpp>
#include <ftxui/dom/selection.hpp>
#include <ftxui/screen/screen.hpp>
#include <ftxui/screen/string.hpp>
#include <simdjson.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <ctime>
#include <format>
#include <memory>
#include <optional>
#include <random>

using namespace ftxui;

namespace tui {
namespace {

// A small reactive leaf for fixed-size animation fields and elapsed labels.
// The surrounding transcript DOM remains immutable and cacheable; only this
// leaf reads live state during FTXUI's normal layout/render pass.
//
// Selection handling mirrors ftxui::text(): FTXUI only calls Select() while a
// selection is active, so the per-frame state is reset in ComputeRequirement()
// (which always runs) to avoid leaking a stale highlight into the next frame.
class LiveText final : public ftxui::Node {
public:
    explicit LiveText(std::function<std::string()> value)
        : value_(std::move(value)) {}

    void ComputeRequirement() override {
        refresh();
        sel_start_ = -1;  // Reset each frame; Select() repopulates when active.
        requirement_ = {};
        requirement_.min_x = static_cast<int>(glyphs_.size());
        requirement_.min_y = 1;
    }

    void Select(ftxui::Selection& selection) override {
        if (ftxui::Box::Intersection(selection.GetBox(), box_).IsEmpty()) {
            return;
        }
        // LiveText is always a single line, so the selection on this node is a
        // single contiguous horizontal range. Saturate it against our row and
        // record the exact cells so Render() highlights and the clipboard copy
        // stay per-cell, identical to ftxui::text().
        const ftxui::Box row_box{box_.x_min, box_.x_max, box_.y_min, box_.y_min};
        if (ftxui::Box::Intersection(selection.GetBox(), row_box).IsEmpty()) {
            return;
        }
        const ftxui::Selection row_sel = selection.SaturateHorizontal(row_box);
        sel_start_ = row_sel.GetBox().x_min;
        sel_end_ = row_sel.GetBox().x_max;

        std::string part;
        int x = box_.x_min;
        for (const auto& glyph : glyphs_) {
            if (x > box_.x_max) {
                break;
            }
            if (sel_start_ <= x && x <= sel_end_) {
                part += glyph;
            }
            ++x;
        }
        selection.AddPart(part, box_.y_min, sel_start_, sel_end_);
    }

    void Render(ftxui::Screen& screen) override {
        // Use the value captured during ComputeRequirement() so glyph count
        // and assigned geometry stay coherent for the entire FTXUI frame.
        int x = box_.x_min;
        for (const auto& glyph : glyphs_) {
            if (x > box_.x_max) {
                break;
            }
            auto& cell = screen.CellAt(x, box_.y_min);
            cell.character = glyph;
            if (sel_start_ != -1 && x >= sel_start_ && x <= sel_end_) {
                screen.GetSelectionStyle()(cell);
            }
            ++x;
        }
    }

private:
    void refresh() {
        std::string next = value_();
        if (next == text_) {
            return;
        }
        text_ = std::move(next);
        glyphs_ = ftxui::Utf8ToGlyphs(text_);
    }

    std::function<std::string()> value_;
    std::string text_;
    std::vector<std::string> glyphs_;
    int sel_start_ = -1;
    int sel_end_ = -1;
};

ftxui::Element live_text(std::function<std::string()> value) {
    return std::make_shared<LiveText>(std::move(value));
}

ftxui::Element pulse_text(std::string prefix,
                          std::size_t fallback_tick,
                          const ConversationRenderOptions& options) {
    const auto* tick = options.animation_tick;
    return live_text([
        prefix = std::move(prefix), tick, fallback_tick
    ]() {
        const auto frame = tick != nullptr
            ? tick->load(std::memory_order_relaxed)
            : fallback_tick;
        return prefix + std::string(thinking_pulse_frame(frame));
    });
}

ftxui::Element status_spinner_text(std::size_t fallback_tick,
                                   const ConversationRenderOptions& options) {
    const auto* tick = options.animation_tick;
    return live_text([tick, fallback_tick]() {
        const auto frame = tick != nullptr
            ? tick->load(std::memory_order_relaxed)
            : fallback_tick;
        return std::string(tool_status_spinner(frame));
    });
}

} // namespace

bool remove_latest_ui_turn(std::vector<UiMessage>& messages) {
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (it->type == MessageType::User || it->type == MessageType::ShellCommand) {
            messages.erase(it.base() - 1, messages.end());
            return true;
        }
    }
    return false;
}

bool truncate_ui_before_user_turn(
    std::vector<UiMessage>& messages,
    std::size_t user_ordinal) {

    std::size_t current = 0;
    const auto target = std::ranges::find_if(messages, [&](const UiMessage& message) {
        if (message.type != MessageType::User) return false;
        return current++ == user_ordinal;
    });
    if (target == messages.end()) return false;
    messages.erase(target, messages.end());
    return true;
}

namespace {

std::string truncate_preview(std::string_view text, std::size_t max_len = kToolPreviewMaxLen) {
    std::string cleaned = core::utils::str::collapse_ascii_whitespace_copy(text);
    if (cleaned.size() <= max_len) {
        return cleaned;
    }
    if (max_len <= 1) {
        return cleaned.substr(0, max_len);
    }
    return cleaned.substr(0, max_len - 1) + "...";
}

bool has_output_truncation_marker(std::string_view output) {
    return output.find("[OUTPUT TRUNCATED AT") != std::string_view::npos;
}

std::string extract_patch_preview(std::string_view patch) {
    const auto patch_text = core::utils::str::trim_ascii_copy(patch);
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
        
        if (!core::utils::str::trim_ascii_copy(line).empty()) {
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
    return ftxui::text(std::string(icon)) | ftxui::color(badge_color) | ftxui::bold;
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
                           const ConversationRenderOptions& options,
                           int /*terminal_width*/,
                           Color /*border_color*/,
                           bool /*is_first*/) {
    const Color status_color = tui::tool_status_color(tool.status);
    Element icon_el = tool.status == ToolActivity::Status::Executing && options.show_spinner
        ? status_spinner_text(tick, options)
        : ftxui::text(std::string(tui::tool_status_icon(tool.status)));
    Element status_el = hbox({ftxui::text(" "), std::move(icon_el), ftxui::text(" ")})
                      | ftxui::color(status_color) | ftxui::bold;
    Element name_el = ftxui::text(tool.name) | ftxui::bold | ftxui::color(ColorYellowBright);
    
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
        if (core::tools::names::is_terminal_tool(tool.name) && tool.result.exit_code.has_value()) {
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
    const Color text_color = is_error ? static_cast<Color>(ColorToolFail) : Color{Color::GrayLight};

    const bool is_terminal_output = core::tools::names::is_terminal_tool(tool.name);
    std::vector<Element> rows;
    if (is_terminal_output) {
        std::string label = "Terminal output";
        if (tool.result.exit_code.has_value()) {
            label += std::format(" (exit {})", *tool.result.exit_code);
        }
        rows.push_back(ftxui::text(std::move(label)) | ftxui::color(ColorYellowDark) | ftxui::bold);
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
                      ftxui::color(ColorYellowDark) | ftxui::bold);
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

std::string subagent_status_label(const ToolActivity::SubagentActivity& subagent) {
    if (subagent.status == ToolActivity::Status::Executing) {
        std::string label = "running";
        if (subagent.steps > 0) {
            label += std::format(" · {} {}", subagent.steps, subagent.steps == 1 ? "step" : "steps");
        }
        if (subagent.tool_calls > 0) {
            label += std::format(" · {} {}", subagent.tool_calls, subagent.tool_calls == 1 ? "tool" : "tools");
        }
        return label;
    }
    if (subagent.status == ToolActivity::Status::Succeeded) {
        return subagent.tool_calls > 0
            ? std::format("completed · {} {}", subagent.tool_calls, subagent.tool_calls == 1 ? "tool" : "tools")
            : "completed";
    }
    if (subagent.status == ToolActivity::Status::Cancelled) {
        return "cancelled";
    }
    return "failed";
}

Element render_subagent_activity(const ToolActivity::SubagentActivity& subagent,
                                 std::size_t tick,
                                 const ConversationRenderOptions& options) {
    const bool running = subagent.status == ToolActivity::Status::Executing;
    const Color status_color = tool_status_color(subagent.status);
    Element icon = running && options.show_spinner
        ? status_spinner_text(tick, options)
        : ftxui::text(std::string(tool_status_icon(subagent.status)));

    std::vector<Element> header;
    header.push_back(ftxui::text("  "));
    header.push_back(std::move(icon) | ftxui::color(status_color) | ftxui::bold);
    header.push_back(ftxui::text(" @"));
    header.push_back(ftxui::text(subagent.worker_name.empty() ? "subagent" : subagent.worker_name)
                     | ftxui::bold
                     | ftxui::color(ColorYellowBright));
    if (!subagent.description.empty()) {
        header.push_back(ftxui::text("  "));
        header.push_back(ftxui::text(truncate_preview(subagent.description, 64))
                         | ftxui::color(Color::GrayLight)
                         | xflex);
    } else {
        header.push_back(filler());
    }
    header.push_back(ftxui::text("  "));
    header.push_back(ftxui::text(subagent_status_label(subagent)) | ftxui::color(status_color));

    std::vector<Element> rows;
    rows.push_back(hbox(std::move(header)) | xflex);

    if (!subagent.provider.empty() || !subagent.model.empty()) {
        const std::string runtime = subagent.provider.empty()
            ? subagent.model
            : (subagent.model.empty()
                ? subagent.provider
                : std::format("{} · {}", subagent.provider, subagent.model));
        rows.push_back(
            hbox({
                ftxui::text("    "),
                ftxui::text(runtime) | ftxui::color(Color::GrayDark) | dim,
            }));
    }

    const std::size_t max_tools = options.expand_tool_results ? subagent.recent_tools.size() : std::min<std::size_t>(subagent.recent_tools.size(), 4);
    const std::size_t hidden_tools = subagent.recent_tools.size() > max_tools
        ? subagent.recent_tools.size() - max_tools
        : 0;
    for (std::size_t i = subagent.recent_tools.size() - max_tools; i < subagent.recent_tools.size(); ++i) {
        const auto& tool = subagent.recent_tools[i];
        rows.push_back(
            hbox({
                ftxui::text("    "),
                ftxui::text(std::string(tool_status_icon(tool.status))) | ftxui::color(tool_status_color(tool.status)),
                ftxui::text(" "),
                ftxui::text(tool.name) | ftxui::color(ColorYellowDark),
                ftxui::text(tool.description.empty() ? "" : "  "),
                ftxui::text(tool.description) | ftxui::color(Color::GrayDark) | xflex,
            }) | xflex);
    }
    if (hidden_tools > 0) {
        rows.push_back(
            hbox({
                ftxui::text("    "),
                ftxui::text(std::format("{} earlier tool {} hidden", hidden_tools, hidden_tools == 1 ? "call" : "calls"))
                | ftxui::color(Color::GrayDark)
                | dim,
            }));
    }

    if (!subagent.summary.empty() && subagent.status != ToolActivity::Status::Executing) {
        rows.push_back(
            hbox({
                ftxui::text("    "),
                paragraph(truncate_preview(subagent.summary, options.expand_tool_results ? 1200 : 180))
                    | ftxui::color(Color::GrayLight)
                    | xflex,
            }) | xflex);
    } else if (running && !subagent.latest_text.empty() && options.expand_tool_results) {
        rows.push_back(
            hbox({
                ftxui::text("    "),
                paragraph(truncate_preview(subagent.latest_text, 320))
                    | ftxui::color(Color::GrayDark)
                    | xflex,
            }) | xflex);
    }

    return vbox(std::move(rows));
}

Element render_subagent_group(const ToolActivity& tool,
                              std::size_t tick,
                              const ConversationRenderOptions& options) {
    if (tool.subagents.empty()) {
        return emptyElement();
    }

    std::vector<Element> rows;
    rows.push_back(ftxui::text(tool.subagents.size() == 1 ? "Subagent" : "Subagents")
                   | ftxui::color(ColorYellowDark)
                   | ftxui::bold);
    for (const auto& subagent : tool.subagents) {
        rows.push_back(render_subagent_activity(subagent, tick, options));
    }
    return vbox(std::move(rows));
}

struct AssistantActivityState {
    bool has_executing_tools = false;
    bool has_completed_tools = false;
    bool show_thinking = false;
};

AssistantActivityState compute_assistant_activity_state(const UiMessage& msg) {
    AssistantActivityState state;
    for (const auto& tool : msg.tools) {
        if (tool.status == ToolActivity::Status::Executing) {
            state.has_executing_tools = true;
        } else if (tool.status == ToolActivity::Status::Succeeded ||
                   tool.status == ToolActivity::Status::Failed) {
            state.has_completed_tools = true;
        }
        for (const auto& subagent : tool.subagents) {
            if (subagent.status == ToolActivity::Status::Executing) {
                state.has_executing_tools = true;
            } else if (subagent.status == ToolActivity::Status::Succeeded ||
                       subagent.status == ToolActivity::Status::Failed ||
                       subagent.status == ToolActivity::Status::Cancelled) {
                state.has_completed_tools = true;
            }
        }
    }

    state.show_thinking = msg.thinking ||
        (msg.pending && state.has_completed_tools && !state.has_executing_tools);
    return state;
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
        render_tool_header(tool, tick, options, terminal_width,
                          border_color, is_first));
    
    if (tool.status == ToolActivity::Status::Executing && tool.progress.has_value()) {
        tool_elements.push_back(render_tool_progress(tool));
    }

    if (!tool.subagents.empty()) {
        tool_elements.push_back(separator() | ftxui::color(Color::GrayDark));
        tool_elements.push_back(render_subagent_group(tool, tick, options));
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

auto local_time_str(std::chrono::system_clock::time_point time) -> std::string {
    const auto tt = std::chrono::system_clock::to_time_t(time);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif
    return std::format("{:02d}:{:02d}:{:02d}", tm.tm_hour, tm.tm_min, tm.tm_sec);
}

auto current_time_str() -> std::string {
    return local_time_str(std::chrono::system_clock::now());
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

UiMessage make_shell_command_message(std::string command,
                                     std::string timestamp,
                                     bool pending) {
    UiMessage msg;
    msg.type = MessageType::ShellCommand;
    msg.id = generate_message_id();
    msg.text = std::move(command);
    msg.timestamp = std::move(timestamp);
    msg.pending = pending;
    msg.finalized = !pending;
    return msg;
}

UiMessage make_assistant_message(std::string text, std::string timestamp, bool pending) {
    UiMessage msg;
    msg.type = MessageType::Assistant;
    msg.id = generate_message_id();
    msg.text = std::move(text);
    msg.timestamp = std::move(timestamp);
    msg.pending = pending;
    // A message created in a non-pending state is already complete and must
    // never be reverted to pending by a late/out-of-band callback.
    msg.finalized = !pending;
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

UiMessage make_system_disclosure_message(std::string summary,
                                         std::string details) {
    UiMessage msg = make_system_message(std::move(summary));
    msg.disclosure_text = std::move(details);
    return msg;
}

void append_ui_message(std::vector<UiMessage>& messages, UiMessage message) {
    if (!messages.empty()) {
        auto& previous = messages.back();
        const bool same_system_summary =
            previous.type == MessageType::System
            && message.type == MessageType::System
            && previous.text == message.text;
        if (same_system_summary) {
            const std::size_t previous_count = std::max<std::size_t>(previous.repeat_count, 1);
            const std::size_t incoming_count = std::max<std::size_t>(message.repeat_count, 1);
            previous.repeat_count = previous_count + incoming_count;
            if (!message.secondary_text.empty()) {
                previous.secondary_text = std::move(message.secondary_text);
            }
            if (!message.disclosure_text.empty()) {
                // Keep the latest details for repeated events.
                previous.disclosure_text = std::move(message.disclosure_text);
            }
            return;
        }
    }

    message.repeat_count = std::max<std::size_t>(message.repeat_count, 1);
    messages.push_back(std::move(message));
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

    if (const auto error = core::utils::json::first_string_field(object, {"error"})) {
        tool.status = ToolActivity::Status::Failed;
        set_result_summary(*error);
        return;
    }

    if (core::tools::names::is_terminal_tool(tool.name)) {
        if (const auto exit_code = core::utils::json::optional_int_field_clamped(object, "exit_code")) {
            tool.result.exit_code = *exit_code;
            tool.status = (*exit_code == 0)
                ? ToolActivity::Status::Succeeded
                : ToolActivity::Status::Failed;
        } else {
            tool.status = ToolActivity::Status::Succeeded;
        }

        if (const auto output = core::utils::json::first_string_field(object, {"output"})) {
            set_result_summary(*output);
            if (output->find("[INTERRUPTED:") != std::string::npos) {
                tool.status = ToolActivity::Status::Cancelled;
            }
        } else if (tool.result.exit_code.has_value()) {
            set_result_summary((*tool.result.exit_code == 0)
                ? "Command completed with no output."
                : std::format("Command exited with status {}.", *tool.result.exit_code));
        } else {
            set_result_summary("Done");
        }
        return;
    }

    if (const auto output = core::utils::json::first_string_field(object, {"output"})) {
        tool.status = ToolActivity::Status::Succeeded;
        set_result_summary(output->empty() ? std::string("Done") : *output);
        return;
    }

    if (const auto content = core::utils::json::first_string_field(object, {"content"})) {
        tool.status = ToolActivity::Status::Succeeded;
        set_result_summary(content->empty() ? std::string("Done") : *content);
        return;
    }

    if (const auto matches = core::utils::json::first_string_field(object, {"matches"})) {
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

Element render_shell_command_message(const UiMessage& msg,
                                     std::size_t tick,
                                     const ConversationRenderOptions& options) {
    std::vector<Element> rows;
    if (options.show_timestamps && !msg.timestamp.empty()) {
        rows.push_back(
            hbox({ filler(), ftxui::text(msg.timestamp) | ftxui::color(Color::GrayDark) }));
    }

    rows.push_back(
        hbox({
            ftxui::text("$ ") | ftxui::color(ColorYellowBright) | ftxui::bold,
            paragraph(msg.text) | ftxui::color(Color::White) | xflex,
        }));

    const ftxui::Color output_color = msg.stopped
        ? static_cast<ftxui::Color>(ColorToolFail)
        : ftxui::Color::GrayLight;
    const ftxui::Color border_color = msg.stopped
        ? static_cast<ftxui::Color>(ColorToolFail)
        : static_cast<ftxui::Color>(ColorYellowDark);

    if (msg.pending) {
        Element label = options.show_spinner
            ? pulse_text("Running", tick, options)
            : ftxui::text("Running...");
        rows.push_back(
            hbox({
                ftxui::text("  "),
                std::move(label) | ftxui::color(ColorYellowDark) | dim,
            }));
    } else if (!msg.secondary_text.empty()) {
        rows.push_back(
            hbox({
                ftxui::text("  "),
                render_text_lines_preserving_newlines(
                    msg.secondary_text,
                    output_color)
                    | xflex,
            }));
    } else {
        rows.push_back(
            hbox({
                ftxui::text("  "),
                ftxui::text("Command completed with no output.")
                    | ftxui::color(Color::GrayDark)
                    | dim,
            }));
    }

    return vbox(std::move(rows)) | UiBorder(border_color);
}

Element render_assistant_message(const UiMessage& msg,
                                 std::size_t tick,
                                 const ConversationRenderOptions& options) {
    std::vector<Element> elements;
    const auto activity = compute_assistant_activity_state(msg);

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
    if (activity.show_thinking) {
        Element label;
        if (activity.has_completed_tools && !activity.has_executing_tools && !msg.thinking) {
            label = options.show_spinner
                ? pulse_text("Analyzing", tick, options)
                : ftxui::text("Analyzing...");
        } else {
            label = options.show_spinner
                ? pulse_text("Thinking", tick, options)
                : ftxui::text("Thinking...");
        }
        Elements indicator_row = {
            render_lightbulb_prefix(true),
            std::move(label) | ftxui::color(ColorYellowDark) | dim
        };
        if (options.activity_elapsed) {
            const auto elapsed = options.activity_elapsed;
            const auto message_id = msg.id;
            indicator_row.push_back(
                live_text([elapsed, message_id]() {
                    const auto value = elapsed(message_id);
                    return value.empty() ? std::string{} : std::format(" ({})", value);
                })
                | ftxui::color(Color::GrayDark)
                | dim);
        } else if (!msg.activity_elapsed.empty()) {
            indicator_row.push_back(
                ftxui::text(std::format(" ({})", msg.activity_elapsed))
                | ftxui::color(Color::GrayDark)
                | dim);
        }
        elements.push_back(
            hbox(std::move(indicator_row)));
    }

    // Show stopped indicator if generation was interrupted
    if (msg.stopped) {
        elements.push_back(
            hbox({
                ftxui::text("⏹ ") | ftxui::color(Color::Red),
                ftxui::text("Stopped") | ftxui::color(Color::Red) | dim
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

Element render_system_message(const UiMessage& msg,
                              const ConversationRenderOptions& options) {
    if (!msg.disclosure_text.empty()) {
        std::vector<Element> lines;
        bool expanded = options.expand_system_details;
        if (options.system_disclosure_expanded != nullptr) {
            if (const auto it = options.system_disclosure_expanded->find(msg.id);
                it != options.system_disclosure_expanded->end()) {
                expanded = expanded || it->second;
            }
        }

        std::string summary = std::string(expanded ? "▼ " : "▶ ") + msg.text;
        if (msg.repeat_count > 1) {
            summary += std::format("  (x{})", msg.repeat_count);
        }
        if (!expanded) {
            summary += "  (click or Ctrl+O for details)";
        }

        Element summary_el = ftxui::text(std::move(summary)) | ftxui::color(ColorYellowDark);
        if (options.system_disclosure_hitboxes != nullptr) {
            auto& box = (*options.system_disclosure_hitboxes)[msg.id];
            summary_el = std::move(summary_el) | reflect(box);
        }
        lines.push_back(std::move(summary_el));

        if (expanded) {
            if (msg.repeat_count > 1) {
                lines.push_back(
                    hbox({
                        ftxui::text("  "),
                        paragraph(std::format(
                            "Collapsed {} repeated events. Showing latest details.",
                            msg.repeat_count))
                        | ftxui::color(Color::GrayDark)
                        | dim
                        | xflex,
                    }) | xflex);
            }
            for (const auto& line : split_lines(msg.disclosure_text)) {
                if (line.empty()) {
                    lines.push_back(ftxui::text(""));
                    continue;
                }
                lines.push_back(
                    hbox({
                        ftxui::text("  "),
                        paragraph(line) | ftxui::color(Color::GrayDark) | xflex,
                    }) | xflex);
            }
        }
        return vbox(std::move(lines));
    }

    const std::string body = msg.repeat_count > 1
        ? std::format("{}  (x{})", msg.text, msg.repeat_count)
        : msg.text;

    std::vector<Element> lines;
    for (const auto& line : split_lines(body)) {
        lines.push_back(ftxui::text(line) | ftxui::color(ColorYellowDark));
    }
    return vbox(std::move(lines));
}

// ============================================================================
// Main Render Function
// ============================================================================

Decorator scroll_position_relative(float x,
                                   float y,
                                   std::shared_ptr<ConversationScrollAnchor> scroll_anchor = {}) {
    class Impl : public ftxui::Node {
    public:
        Impl(Element child,
             float x,
             float y,
             std::shared_ptr<ConversationScrollAnchor> scroll_anchor)
            : ftxui::Node(ftxui::unpack(std::move(child))),
              x_(x),
              y_(y),
              scroll_anchor_(scroll_anchor) {}

        void ComputeRequirement() override {
            children_[0]->ComputeRequirement();
            requirement_ = children_[0]->requirement();
            requirement_.focused.enabled = false;
            requirement_.focused.box.x_min = int(float(requirement_.min_x) * x_);
            if (scroll_anchor_ != nullptr) {
                const int content_height = std::max(requirement_.min_y, 1);
                scroll_anchor_->content_height = content_height;
                if (scroll_anchor_->follow_bottom) {
                    scroll_anchor_->focus_y = content_height;
                }
                // FTXUI's flexbox deliberately reports provisional heights
                // during its iterative layout pass. Never write a clamp based
                // on one of those transient heights back into a held persistent
                // anchor: a later iteration can restore the full height, but
                // the lost focus position cannot be recovered.
                requirement_.focused.box.y_min = std::clamp(
                    scroll_anchor_->focus_y,
                    0,
                    content_height);
            } else {
                requirement_.focused.box.y_min = int(float(requirement_.min_y) * y_);
            }
            requirement_.focused.box.x_max = int(float(requirement_.min_x) * x_);
            requirement_.focused.box.y_max = requirement_.focused.box.y_min;
        }

        void SetBox(ftxui::Box box) override {
            ftxui::Node::SetBox(box);
            children_[0]->SetBox(box);
        }

    private:
        const float x_;
        const float y_;
        const std::shared_ptr<ConversationScrollAnchor> scroll_anchor_;
    };

    return [x, y, scroll_anchor](Element child) {
        return std::make_shared<Impl>(std::move(child), x, y, scroll_anchor);
    };
}

Element apply_scroll_viewport(Element content,
                              float scroll_pos,
                              std::shared_ptr<ConversationScrollAnchor> scroll_anchor) {
    return std::move(content)
        | scroll_position_relative(0, scroll_pos, scroll_anchor)
        | vscroll_indicator
        | yframe
        | yflex;
}

Element render_history_content(const std::vector<UiMessage>& messages,
                               std::size_t tick,
                               ConversationRenderOptions options) {
    std::vector<Element> msg_elements;
    msg_elements.reserve(messages.size() * 2);

    for (const auto& msg : messages) {
        switch (msg.type) {
            case MessageType::User:
                msg_elements.push_back(render_user_message(msg, options));
                break;
            case MessageType::ShellCommand:
                msg_elements.push_back(render_shell_command_message(msg, tick, options));
                msg_elements.push_back(ftxui::text(""));
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
                msg_elements.push_back(render_system_message(msg, options));
                break;
        }
    }

    return vbox(std::move(msg_elements));
}

Element render_history_panel(const std::vector<UiMessage>& messages,
                             std::size_t tick,
                             ConversationRenderOptions options) {
    return apply_scroll_viewport(
        render_history_content(messages, tick, options),
        options.scroll_pos,
        options.scroll_anchor);
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

bool message_uses_animation(const UiMessage& message, bool show_spinner) {
    if (!show_spinner) {
        return false;
    }

    for (const auto& tool : message.tools) {
        if (tool.status == ToolActivity::Status::Executing) {
            return true;
        }
        for (const auto& subagent : tool.subagents) {
            if (subagent.status == ToolActivity::Status::Executing) {
                return true;
            }
        }
    }

    if (message.type == MessageType::Assistant) {
        return compute_assistant_activity_state(message).show_thinking;
    }

    if (message.type == MessageType::ShellCommand) {
        return message.pending;
    }

    return false;
}

bool conversation_uses_animation(const std::vector<UiMessage>& messages, bool show_spinner) {
    return std::any_of(messages.begin(), messages.end(), [show_spinner](const UiMessage& message) {
        return message_uses_animation(message, show_spinner);
    });
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

    if (tool_name == "task") {
        const auto worker = core::utils::json::first_string_field(object, {"subagent_type", "worker", "agent"});
        const auto description = core::utils::json::first_string_field(object, {"description", "title", "task"});
        if (worker && description) {
            return truncate_preview("@" + *worker + " · " + *description);
        }
        if (description) {
            return truncate_preview(*description);
        }
        if (worker) {
            return truncate_preview("@" + *worker);
        }
    }

    if (tool_name == core::tools::names::kMoveFile) {
        const auto source = core::utils::json::first_string_field(object, {"source_path", "from_path"});
        const auto destination = core::utils::json::first_string_field(object, {"destination_path", "to_path"});
        if (source && destination) {
            return truncate_preview(*source + " -> " + *destination);
        }
    }

    if (tool_name == core::tools::names::kGrepSearch) {
        const auto pattern = core::utils::json::first_string_field(object, {"pattern"});
        const auto dir = core::utils::json::first_string_field(object, {"path"});
        if (pattern && dir) {
            return truncate_preview(*pattern + " in " + *dir);
        }
    }

    if (tool_name == core::tools::names::kApplyPatch) {
        if (const auto patch = core::utils::json::first_string_field(object, {"patch"})) {
            return extract_patch_preview(*patch);
        }
    }

    if (core::tools::names::is_terminal_tool(tool_name)) {
        const auto command = core::utils::json::first_string_field(object, {"command"});
        const auto working_dir = core::utils::json::first_string_field(object, {"working_dir"});
        if (command && working_dir) {
            return std::format("cwd: {} | cmd: {}", *working_dir, *command);
        }
        if (command) {
            return std::format("cmd: {}", *command);
        }
    }

    if (const auto preview = core::utils::json::first_string_field(object, {
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

    if (const auto preview = core::utils::json::first_string_field(object)) {
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

ToolActivity::SubagentActivity* find_subagent_activity(ToolActivity& tool,
                                                       std::string_view subagent_id) {
    for (auto& subagent : tool.subagents) {
        if (subagent.id == subagent_id) {
            return &subagent;
        }
    }
    return nullptr;
}

const ToolActivity::SubagentActivity* find_subagent_activity(const ToolActivity& tool,
                                                             std::string_view subagent_id) {
    for (const auto& subagent : tool.subagents) {
        if (subagent.id == subagent_id) {
            return &subagent;
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

    if (const auto error = core::utils::json::first_string_field(object, {"error"})) {
        summary.state = error->find("denied") != std::string_view::npos
            ? ToolResultSummary::State::Denied
            : ToolResultSummary::State::Failed;
        summary.preview = truncate_preview(*error);
        return summary;
    }

    if (const auto matches = core::utils::json::first_string_field(object, {"matches"})) {
        summary.preview = matches->empty() ? "no matches" : truncate_preview(*matches);
        return summary;
    }

    if (const auto output = core::utils::json::first_string_field(object, {"output"})) {
        summary.preview = std::string(*output);
        return summary;
    }

    if (core::utils::json::first_string_field(object, {"content"})) {
        summary.preview = "content loaded";
        return summary;
    }

    if (const auto value = core::utils::json::first_string_field(object, {"time", "message", "result"})) {
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
