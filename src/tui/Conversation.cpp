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
#include <cstdint>
#include <ctime>
#include <format>
#include <memory>
#include <optional>
#include <random>
#include <span>

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

// A single-glyph leaf whose glyph is fixed but whose foreground color is
// recomputed every frame. Used for the "breathing" lightbulb: the bulb stays
// on-screen (no jarring on/off flicker) and instead smoothly glows brighter and
// dimmer in sync with the thinking dots. Kept deliberately tiny — no selection
// handling — since it is a decorative indicator, never selectable text.
class LiveColorText final : public ftxui::Node {
public:
    LiveColorText(std::string glyph, std::function<ftxui::Color()> color)
        : glyph_(std::move(glyph)), color_(std::move(color)) {
        glyphs_ = ftxui::Utf8ToGlyphs(glyph_);
    }

    void ComputeRequirement() override {
        requirement_ = {};
        requirement_.min_x = static_cast<int>(glyphs_.size());
        requirement_.min_y = 1;
    }

    void Render(ftxui::Screen& screen) override {
        const ftxui::Color fg = color_();
        int x = box_.x_min;
        for (const auto& glyph : glyphs_) {
            if (x > box_.x_max) {
                break;
            }
            auto& cell = screen.CellAt(x, box_.y_min);
            cell.character = glyph;
            cell.foreground_color = fg;
            ++x;
        }
    }

private:
    std::string glyph_;
    std::vector<std::string> glyphs_;
    std::function<ftxui::Color()> color_;
};

ftxui::Element live_color_text(std::string glyph, std::function<ftxui::Color()> color) {
    return std::make_shared<LiveColorText>(std::move(glyph), std::move(color));
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

// A lightbulb prefix that gently "breathes" while an activity phase is live.
// Rather than a hard on/off blink (which reads as a glitch and fights the
// rhythm of the thinking dots), the bulb stays on-screen and its color is
// smoothly interpolated between a dim amber and a bright glow over the SAME
// 6-frame period as thinking_pulse_frame(). Peak brightness lines up with the
// fullest dots, so the bulb and the "..." pulse feel like one coherent effect.
// Trailing spacing matches render_lightbulb_prefix so layout is identical.
ftxui::Element render_live_lightbulb_prefix(const ConversationRenderOptions& options,
                                            std::size_t fallback_tick) {
    const auto* tick = options.animation_tick;
    auto glow = live_color_text("💡", [tick, fallback_tick]() -> ftxui::Color {
        const auto frame = tick != nullptr
            ? tick->load(std::memory_order_relaxed)
            : fallback_tick;
        // Triangle wave over the 6-frame dots cycle: phase 0 (no dots) is
        // dimmest, phase 3 (full "...") is brightest, then eases back down.
        constexpr std::size_t kPeriod = 6;
        const std::size_t phase = frame % kPeriod;
        const std::size_t half = kPeriod / 2;  // 3
        const std::size_t up = phase <= half ? phase : (kPeriod - phase);
        const float t = static_cast<float>(up) / static_cast<float>(half);  // 0..1
        // Interpolate dim amber -> bright glow. Keeps the bulb clearly visible
        // at its dimmest so it never appears to vanish.
        const auto lerp = [t](unsigned char lo, unsigned char hi) -> int {
            return static_cast<int>(lo + (hi - lo) * t);
        };
        return ftxui::Color::RGB(
            static_cast<uint8_t>(lerp(150, 255)),
            static_cast<uint8_t>(lerp(110, 246)),
            static_cast<uint8_t>(lerp(20, 90)));
    });
    return hbox({
        std::move(glow),
        ftxui::text("  "),
    });
}

// Generic agent progress is intentionally distinct from model reasoning. The
// neutral spinner says the turn is still active without implying that there is
// hidden thought text to reveal.
ftxui::Element render_working_prefix(bool active,
                                     const ConversationRenderOptions& options,
                                     std::size_t fallback_tick) {
    if (!active) {
        return ftxui::text("  ");
    }
    Element glyph = options.show_spinner
        ? status_spinner_text(fallback_tick, options)
        : ftxui::text("○");
    return hbox({
        std::move(glyph) | ftxui::color(Color::GrayDark) | dim,
        ftxui::text(" "),
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

std::string latest_completed_assistant_source(
    const std::vector<UiMessage>& messages) {
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (it->type == MessageType::Assistant && !it->pending && !it->text.empty()) {
            return it->assistant_source_text.empty()
                ? it->text
                : it->assistant_source_text;
        }
    }
    return {};
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

UiMessage* LiveAssistantTimeline::current(std::vector<UiMessage>& messages) {
    if (current_message_id_.empty()) {
        return nullptr;
    }
    const auto it = std::ranges::find_if(messages, [&](const UiMessage& message) {
        return message.type == MessageType::Assistant
            && message.id == current_message_id_;
    });
    return it == messages.end() ? nullptr : &*it;
}

void LiveAssistantTimeline::finalize_step(
    UiMessage& message,
    std::string reasoning_elapsed,
    bool stopped) {
    message.pending = false;
    message.finalized = true;
    message.thinking = false;
    message.stopped = stopped;
    message.reasoning_active = false;
    if (!message.reasoning_text.empty() && message.reasoning_elapsed.empty()) {
        message.reasoning_elapsed = std::move(reasoning_elapsed);
        if (message.reasoning_elapsed.empty()) {
            message.reasoning_elapsed = "0s";
        }
    }
    if (!message.text.empty() || !message.tools.empty()) {
        message.show_activity_status = true;
    }
}

UiMessage* LiveAssistantTimeline::begin_step(
    std::vector<UiMessage>& messages,
    std::string previous_reasoning_elapsed) {
    UiMessage* active = current(messages);
    if (active == nullptr || active->finalized) {
        current_message_id_.clear();
        return nullptr;
    }

    if (!step_started_) {
        step_started_ = true;
        active->pending = true;
        active->thinking = true;
        return active;
    }

    finalize_step(*active, std::move(previous_reasoning_elapsed), false);
    append_ui_message(messages, make_assistant_message("", "", true));
    current_message_id_ = messages.back().id;
    return &messages.back();
}

void LiveAssistantTimeline::finish(
    std::vector<UiMessage>& messages,
    std::string reasoning_elapsed,
    bool stopped) {
    UiMessage* active = current(messages);
    current_message_id_.clear();
    if (active != nullptr) {
        finalize_step(*active, std::move(reasoning_elapsed), stopped);
    }
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

// ============================================================================
// Tool Presentation
// ============================================================================

/// Acknowledgement `apply_tool_result` substitutes when a tool reports success
/// without a message. Renderers key off it to avoid showing a bare "Done" next
/// to a diff that already says everything.
inline constexpr std::string_view kDoneSummary = "Done";

/// Web-search snippets are prose; anything longer just pushes the next result
/// off screen.
inline constexpr std::size_t kWebSnippetPreviewChars = 240;

/// Ceiling on the JSON a finished tool may retain for its structured renderer.
/// Transcripts live for the whole session, so a runaway payload must not be
/// held twice; past this size the tool falls back to plain-text rendering.
inline constexpr std::size_t kMaxRetainedRawPayloadBytes = 512 * 1024;

/// Semantic grouping that selects a tool's specialized result renderer.
enum class ToolPresentationKind {
    Read,
    Write,
    Edit,
    Shell,
    Grep,
    Files,
    Directory,
    WebSearch,
    WebFetch,
    Todo,
    Task,
    Generic,
};

/// How a tool is surfaced in the transcript: which renderer handles its result,
/// and the human-facing label shown instead of the raw tool name.
struct ToolPresentation {
    ToolPresentationKind kind = ToolPresentationKind::Generic;
    std::string_view     label;
};

/// Single source of truth for tool presentation. Each tool appears exactly once,
/// so its renderer and its label cannot drift apart.
ToolPresentation tool_presentation(std::string_view name) {
    using namespace core::tools::names;
    using Kind = ToolPresentationKind;
    if (name == kReadFile)                          return {Kind::Read,      "Read"};
    if (name == kWriteFile)                         return {Kind::Write,     "Write"};
    if (name == kApplyPatch)                        return {Kind::Edit,      "Patch"};
    if (name == kSearchReplace || is_replace_tool(name)) return {Kind::Edit, "Edit"};
    if (name == kDeleteFile)                        return {Kind::Edit,      "Delete"};
    if (name == kMoveFile)                          return {Kind::Edit,      "Move"};
    if (name == kCreateDirectory)                   return {Kind::Edit,      "Create directory"};
    if (is_terminal_tool(name))                     return {Kind::Shell,     "Shell"};
    if (name == kPython)                            return {Kind::Shell,     "Python"};
    if (name == kGrepSearch)                        return {Kind::Grep,      "Search"};
    if (name == kFileSearch)                        return {Kind::Files,     "Find files"};
    if (name == kListDirectory)                     return {Kind::Directory, "List"};
    if (name == kWebSearch)                         return {Kind::WebSearch, "Web search"};
    if (name == kFetchUrl)                          return {Kind::WebFetch,  "Fetch"};
    if (name == kWriteTodos)                        return {Kind::Todo,      "Plan"};
    if (is_subagent_tool(name))                     return {Kind::Task,      "Agent"};
    if (name == kMemory)                            return {Kind::Generic,   "Memory"};
    if (name == kActivateSkill)                     return {Kind::Generic,   "Skill"};
    if (name == kAskUserQuestion)                   return {Kind::Generic,   "Question"};
    if (name == kGetCurrentTime)                    return {Kind::Generic,   "Time"};
    return {Kind::Generic, name};
}

/// True for tools whose renderer reads JSON fields that `apply_tool_result`
/// discards when it collapses a payload down to a display string. Those tools
/// keep the original JSON in `Result::raw_payload`; see `apply_tool_result`.
bool tool_result_needs_raw_payload(std::string_view name) {
    switch (tool_presentation(name).kind) {
        case ToolPresentationKind::Grep:
        case ToolPresentationKind::Files:
        case ToolPresentationKind::Directory:
        case ToolPresentationKind::WebSearch:
        case ToolPresentationKind::WebFetch:
        case ToolPresentationKind::Todo:
            return true;
        case ToolPresentationKind::Read:
        case ToolPresentationKind::Write:
        case ToolPresentationKind::Edit:
        case ToolPresentationKind::Shell:
        case ToolPresentationKind::Task:
        case ToolPresentationKind::Generic:
            return false;
    }
    return false;
}

/// The JSON a structured renderer should parse: the retained original when the
/// summary no longer contains it, otherwise the summary itself.
std::string_view tool_result_payload(const ToolActivity& tool) {
    return tool.result.raw_payload.empty()
        ? std::string_view{tool.result.summary}
        : std::string_view{tool.result.raw_payload};
}

std::string join_with(const std::vector<std::string>& parts, std::string_view separator) {
    std::string joined;
    for (const auto& part : parts) {
        if (!joined.empty()) joined += separator;
        joined += part;
    }
    return joined;
}

// ============================================================================
// Tool Result Parsing
// ============================================================================

/// Parses `payload` as a JSON object and hands it to `fn`. Returns false when
/// the payload is absent or not an object, which is the normal case for tools
/// that report plain-text errors.
template <typename Fn>
bool with_json_object(std::string_view payload, Fn&& fn) {
    if (payload.empty()) return false;
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(payload).get(document) != simdjson::SUCCESS) return false;
    simdjson::dom::object object;
    if (document.get(object) != simdjson::SUCCESS) return false;
    fn(object);
    return true;
}

struct SearchResultRow {
    std::string  path;
    std::int64_t line = 0;
    std::string  text;
};

struct PathResultRow {
    std::string_view marker;  // Points at a string literal.
    std::string      path;
};

struct WebResultRow {
    std::string title;
    std::string url;
    std::string snippet;
};

struct TodoRow {
    std::string content;
    std::string status;
};

struct FetchMetadata {
    std::string  url;
    std::string  content_type;
    std::string  title;
    std::int64_t status_code = 0;
    bool         truncated = false;
    bool         parsed = false;
};

bool parse_search_rows(std::string_view payload, std::vector<SearchResultRow>& out) {
    bool found = false;
    const bool parsed = with_json_object(payload, [&](simdjson::dom::object object) {
        simdjson::dom::array matches;
        if (object["matches"].get_array().get(matches) != simdjson::SUCCESS) return;
        found = true;
        out.reserve(matches.size());
        for (const auto item : matches) {
            simdjson::dom::object match;
            if (item.get_object().get(match) != simdjson::SUCCESS) continue;
            const auto path = core::utils::json::first_string_field(match, {"path"});
            if (!path) continue;
            const auto text = core::utils::json::first_string_field(match, {"text"});
            std::int64_t line = 0;
            (void)match["line"].get(line);
            out.push_back({
                .path = *path,
                .line = line,
                .text = text.value_or(std::string{}),
            });
        }
    });
    return parsed && found;
}

bool parse_path_rows(std::string_view payload, std::vector<PathResultRow>& out) {
    bool found = false;
    const bool parsed = with_json_object(payload, [&](simdjson::dom::object object) {
        simdjson::dom::array files;
        if (object["files"].get_array().get(files) == simdjson::SUCCESS) {
            found = true;
            out.reserve(files.size());
            for (const auto item : files) {
                std::string_view path;
                if (item.get(path) == simdjson::SUCCESS) {
                    out.push_back({.marker = "·", .path = std::string(path)});
                }
            }
            return;
        }

        simdjson::dom::array entries;
        if (object["entries"].get_array().get(entries) != simdjson::SUCCESS) return;
        found = true;
        out.reserve(entries.size());
        for (const auto item : entries) {
            simdjson::dom::object entry;
            if (item.get_object().get(entry) != simdjson::SUCCESS) continue;
            const auto name = core::utils::json::first_string_field(entry, {"name"});
            if (!name) continue;
            const auto type = core::utils::json::first_string_field(entry, {"type"});
            out.push_back({
                .marker = type && *type == "dir" ? "▸" : "·",
                .path = *name,
            });
        }
    });
    return parsed && found;
}

bool parse_web_rows(std::string_view payload, std::vector<WebResultRow>& out) {
    bool found = false;
    const bool parsed = with_json_object(payload, [&](simdjson::dom::object object) {
        simdjson::dom::array results;
        if (object["results"].get_array().get(results) != simdjson::SUCCESS) return;
        found = true;
        out.reserve(results.size());
        for (const auto item : results) {
            simdjson::dom::object result;
            if (item.get_object().get(result) != simdjson::SUCCESS) continue;
            out.push_back({
                .title = core::utils::json::first_string_field(result, {"title"})
                             .value_or(std::string{"Untitled"}),
                .url = core::utils::json::first_string_field(result, {"url"})
                           .value_or(std::string{}),
                .snippet = core::utils::json::first_string_field(result, {"snippet", "content"})
                               .value_or(std::string{}),
            });
        }
    });
    return parsed && found;
}

bool parse_todo_rows(std::string_view payload, std::vector<TodoRow>& out) {
    bool found = false;
    const bool parsed = with_json_object(payload, [&](simdjson::dom::object object) {
        simdjson::dom::array array;
        if (object["todos"].get_array().get(array) != simdjson::SUCCESS) return;
        found = true;
        out.reserve(array.size());
        for (const auto item : array) {
            simdjson::dom::object todo;
            if (item.get_object().get(todo) != simdjson::SUCCESS) continue;
            const auto content = core::utils::json::first_string_field(todo, {"content"});
            if (!content) continue;
            const auto status = core::utils::json::first_string_field(todo, {"status"});
            out.push_back({
                .content = *content,
                .status = status.value_or(std::string{"pending"}),
            });
        }
    });
    return parsed && found;
}

bool parse_fetch_metadata(std::string_view payload, FetchMetadata& out) {
    out.parsed = with_json_object(payload, [&](simdjson::dom::object object) {
        out.url = core::utils::json::first_string_field(object, {"url"})
                      .value_or(std::string{});
        out.content_type = core::utils::json::first_string_field(object, {"content_type"})
                               .value_or(std::string{});
        out.title = core::utils::json::first_string_field(object, {"title"})
                        .value_or(std::string{});
        (void)object["status_code"].get(out.status_code);
        (void)object["truncated"].get(out.truncated);
    });
    return out.parsed;
}

/// First line requested by a `read_file` call, so the transcript's gutter
/// matches the file's real line numbers. Defaults to 1.
std::int64_t read_start_line(std::string_view tool_args) {
    std::int64_t start_line = 1;
    with_json_object(tool_args, [&](simdjson::dom::object object) {
        for (const auto field : {"offset_line", "start_line", "offset"}) {
            std::int64_t value = 0;
            if (object[field].get(value) == simdjson::SUCCESS && value > 0) {
                start_line = value;
                return;
            }
        }
    });
    return start_line;
}

/// Everything a tool's header and body need, parsed exactly once per frame.
/// The header previously parsed the payload to compute a count and the body
/// parsed it again to build rows — two full simdjson passes per tool per frame.
struct ToolResultView {
    ToolPresentation             presentation;
    std::vector<SearchResultRow> search_rows;
    std::vector<PathResultRow>   path_rows;
    std::vector<WebResultRow>    web_rows;
    std::vector<TodoRow>         todo_rows;
    FetchMetadata                fetch;
    std::size_t                  read_line_count = 0;
    std::int64_t                 read_start_line = 1;
    std::size_t                  diff_additions = 0;
    std::size_t                  diff_deletions = 0;
    /// True when the payload parsed into the shape this tool's renderer expects.
    bool                         payload_parsed = false;
};

ToolResultView build_tool_result_view(const ToolActivity& tool) {
    ToolResultView view;
    view.presentation = tool_presentation(tool.name);
    const std::string_view payload = tool_result_payload(tool);

    switch (view.presentation.kind) {
        case ToolPresentationKind::Read:
            view.read_line_count = visible_line_count(tool.result.summary);
            view.read_start_line = read_start_line(tool.args);
            break;
        case ToolPresentationKind::Write:
        case ToolPresentationKind::Edit:
            // Read the totals off the model, never off what is drawn: a card
            // showing 10 of 26 diff lines must still report the whole change.
            view.diff_additions = tool.diff_preview.added_count;
            view.diff_deletions = tool.diff_preview.deleted_count;
            break;
        case ToolPresentationKind::Grep:
            view.payload_parsed = parse_search_rows(payload, view.search_rows);
            break;
        case ToolPresentationKind::Files:
        case ToolPresentationKind::Directory:
            view.payload_parsed = parse_path_rows(payload, view.path_rows);
            break;
        case ToolPresentationKind::WebSearch:
            view.payload_parsed = parse_web_rows(payload, view.web_rows);
            break;
        case ToolPresentationKind::WebFetch:
            view.payload_parsed = parse_fetch_metadata(payload, view.fetch);
            break;
        case ToolPresentationKind::Todo:
            // The result echoes the persisted plan; the arguments are the only
            // source while the call is still pending.
            view.payload_parsed = parse_todo_rows(payload, view.todo_rows)
                || parse_todo_rows(tool.args, view.todo_rows);
            break;
        case ToolPresentationKind::Shell:
        case ToolPresentationKind::Task:
        case ToolPresentationKind::Generic:
            break;
    }
    return view;
}

// ============================================================================
// Tool Header Metric
// ============================================================================

std::string count_label(std::size_t count,
                        std::string_view singular,
                        std::string_view plural) {
    return std::format("{} {}", count, count == 1 ? singular : plural);
}

/// Compact detail for the right edge of a tool header, e.g. "12 matches".
/// Returns empty when there is nothing meaningful to report — every metric is
/// only trustworthy for a successful call, so failures deliberately fall through
/// to the status label in `tool_result_metric`.
std::string tool_metric_detail(const ToolActivity& tool, const ToolResultView& view) {
    const bool succeeded = tool.status == ToolActivity::Status::Succeeded;
    const bool structured = succeeded && view.payload_parsed;

    switch (view.presentation.kind) {
        case ToolPresentationKind::Read:
            return succeeded ? count_label(view.read_line_count, "line", "lines")
                             : std::string{};
        case ToolPresentationKind::Write:
        case ToolPresentationKind::Edit:
            if (succeeded && (view.diff_additions != 0 || view.diff_deletions != 0)) {
                return std::format("+{} -{}", view.diff_additions, view.diff_deletions);
            }
            return {};
        case ToolPresentationKind::Shell:
            // A cancelled command still carries the exit status of the process we
            // killed, so reporting it would claim a clean exit for an abort.
            if ((succeeded || tool.status == ToolActivity::Status::Failed)
                && tool.result.exit_code.has_value()) {
                return std::format("exit {}", *tool.result.exit_code);
            }
            return {};
        case ToolPresentationKind::Grep:
            return structured ? count_label(view.search_rows.size(), "match", "matches")
                              : std::string{};
        case ToolPresentationKind::Files:
            return structured ? count_label(view.path_rows.size(), "file", "files")
                              : std::string{};
        case ToolPresentationKind::Directory:
            return structured ? count_label(view.path_rows.size(), "entry", "entries")
                              : std::string{};
        case ToolPresentationKind::WebSearch:
            return structured ? count_label(view.web_rows.size(), "result", "results")
                              : std::string{};
        case ToolPresentationKind::WebFetch:
            return view.fetch.parsed && view.fetch.status_code != 0
                ? std::to_string(view.fetch.status_code)
                : std::string{};
        case ToolPresentationKind::Todo:
            return succeeded && !view.todo_rows.empty()
                ? count_label(view.todo_rows.size(), "item", "items")
                : std::string{};
        case ToolPresentationKind::Task:
            return tool.subagents.empty()
                ? std::string{}
                : count_label(tool.subagents.size(), "agent", "agents");
        case ToolPresentationKind::Generic:
            return {};
    }
    return {};
}

/// Header metric with a guaranteed non-empty result for finished calls: a
/// failed, denied or cancelled tool always shows its status even when no
/// domain-specific metric could be computed.
std::string tool_result_metric(const ToolActivity& tool, const ToolResultView& view) {
    if (tool.status == ToolActivity::Status::Pending
        || tool.status == ToolActivity::Status::Executing) {
        return {};
    }
    std::string detail = tool_metric_detail(tool, view);
    return detail.empty() ? std::string(tool_status_label(tool.status)) : std::move(detail);
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

std::string format_message_time_label(std::string_view timestamp,
                                      std::string_view elapsed) {
    if (!timestamp.empty() && !elapsed.empty()) {
        return std::format("{} · {}", timestamp, elapsed);
    }
    if (!timestamp.empty()) {
        return std::string(timestamp);
    }
    if (!elapsed.empty()) {
        return std::string(elapsed);
    }
    return {};
}

Element user_message_bubble(std::string_view content,
                            std::string_view timestamp,
                            std::string_view elapsed) {
    auto body = render_text_lines_preserving_newlines(content, Color::White);
    auto time_label = format_message_time_label(timestamp, elapsed);
    if (time_label.empty()) {
        return body | UiBorder(ColorYellowBright);
    }
    // Keep the clock/elapsed row inside the yellow question box so start time
    // and completed-turn duration share the same chrome as the prompt itself.
    return vbox({
        hbox({ filler(), ftxui::text(std::move(time_label)) | ftxui::color(Color::GrayDark) }),
        std::move(body),
    }) | UiBorder(ColorYellowBright);
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

std::size_t specialized_preview_limit(const ConversationRenderOptions& options,
                                      std::size_t compact_floor = 20) {
    if (options.expand_tool_results) return 0;
    return std::max(options.tool_result_preview_max_lines, compact_floor);
}

Element placeholder_text(std::string_view label) {
    return ftxui::text(std::string(label)) | ftxui::color(Color::GrayDark) | dim;
}

/// Shared skeleton for every list-shaped tool renderer: clamp to the compact
/// preview limit, emit rows, and footer the elided remainder. `emit_row` appends
/// one *or more* elements per item, so renderers that interleave group headings
/// (grep) or multi-line entries (web search) use the same path as flat lists.
template <typename EmitRow>
Element render_limited_rows(std::size_t total,
                            const ConversationRenderOptions& options,
                            std::string_view overflow_noun,
                            EmitRow&& emit_row) {
    const std::size_t limit = specialized_preview_limit(options);
    const std::size_t shown = limit == 0 ? total : std::min(total, limit);

    std::vector<Element> rows;
    rows.reserve(shown + 1);
    for (std::size_t i = 0; i < shown; ++i) {
        emit_row(rows, i);
    }
    if (shown < total) {
        rows.push_back(
            ftxui::text(std::format("  … {} more {}", total - shown, overflow_noun))
            | ftxui::color(Color::GrayDark)
            | dim);
    }
    return vbox(std::move(rows)) | xflex;
}

Element render_numbered_tool_text(std::string_view content,
                                  std::int64_t start_line,
                                  const ConversationRenderOptions& options) {
    // Views into `content`; the caller owns the buffer for the duration of the
    // call, and only the lines actually shown are copied into elements.
    auto lines = split_lines_view(content);
    while (!lines.empty() && lines.back().empty()) lines.pop_back();
    if (lines.empty()) {
        return placeholder_text("No content");
    }

    const std::size_t limit = specialized_preview_limit(options);
    const std::size_t shown = limit == 0 ? lines.size() : std::min(lines.size(), limit);
    const std::int64_t last_line = start_line + static_cast<std::int64_t>(shown) - 1;
    const std::size_t number_width = std::format("{}", last_line).size();

    return render_limited_rows(
        lines.size(), options, "lines",
        [&](std::vector<Element>& rows, std::size_t i) {
            rows.push_back(
                hbox({
                    ftxui::text(std::format(
                        "{:>{}}", start_line + static_cast<std::int64_t>(i), number_width))
                        | ftxui::color(Color::GrayDark),
                    ftxui::text(" │ ") | ftxui::color(ColorYellowDark) | dim,
                    paragraph(std::string(lines[i])) | ftxui::color(Color::GrayLight) | xflex,
                }) | xflex);
        });
}

Element render_search_results(const ToolResultView& view,
                              const ConversationRenderOptions& options) {
    if (view.search_rows.empty()) {
        return placeholder_text("No matches");
    }

    std::string current_path;
    return render_limited_rows(
        view.search_rows.size(), options, "matches",
        [&](std::vector<Element>& rows, std::size_t i) {
            const auto& match = view.search_rows[i];
            if (match.path != current_path) {
                current_path = match.path;
                rows.push_back(
                    ftxui::text(current_path)
                    | ftxui::color(ColorYellowDark)
                    | ftxui::bold);
            }
            rows.push_back(
                hbox({
                    ftxui::text(std::format("{:>5}", match.line))
                        | ftxui::color(Color::GrayDark),
                    ftxui::text(" │ ") | ftxui::color(ColorYellowDark) | dim,
                    paragraph(match.text) | ftxui::color(Color::GrayLight) | xflex,
                }) | xflex);
        });
}

Element render_path_results(const ToolResultView& view,
                            const ConversationRenderOptions& options) {
    if (view.path_rows.empty()) {
        return placeholder_text("No results");
    }
    return render_limited_rows(
        view.path_rows.size(), options, "entries",
        [&](std::vector<Element>& rows, std::size_t i) {
            rows.push_back(
                hbox({
                    ftxui::text(std::format("{} ", view.path_rows[i].marker))
                        | ftxui::color(ColorYellowDark),
                    ftxui::text(view.path_rows[i].path)
                        | ftxui::color(Color::GrayLight)
                        | xflex,
                }) | xflex);
        });
}

Element render_web_results(const ToolResultView& view,
                           const ConversationRenderOptions& options) {
    if (view.web_rows.empty()) {
        return placeholder_text("No results");
    }
    return render_limited_rows(
        view.web_rows.size(), options, "results",
        [&](std::vector<Element>& rows, std::size_t i) {
            const auto& result = view.web_rows[i];
            rows.push_back(
                hbox({
                    ftxui::text(std::format("{}  ", i + 1)) | ftxui::color(ColorYellowDark),
                    paragraph(result.title)
                        | ftxui::color(Color::GrayLight)
                        | ftxui::bold
                        | xflex,
                }) | xflex);
            if (!result.url.empty()) {
                rows.push_back(
                    hbox({
                        ftxui::text("   "),
                        paragraph(result.url) | ftxui::color(Color::GrayDark) | dim | xflex,
                    }) | xflex);
            }
            if (!result.snippet.empty()) {
                rows.push_back(
                    hbox({
                        ftxui::text("   "),
                        paragraph(truncate_preview(result.snippet, kWebSnippetPreviewChars))
                            | ftxui::color(Color::GrayLight)
                            | xflex,
                    }) | xflex);
            }
        });
}

Element render_todo_results(const ToolResultView& view) {
    if (view.todo_rows.empty()) {
        return placeholder_text("Plan updated");
    }

    std::vector<Element> rows;
    rows.reserve(view.todo_rows.size());
    for (const auto& todo : view.todo_rows) {
        const bool done = todo.status == "completed";
        const bool active = todo.status == "in_progress";
        const std::string_view marker = done ? "✓" : active ? "●" : "○";
        const Color marker_color = done
            ? static_cast<Color>(ColorToolDone)
            : active ? static_cast<Color>(ColorYellowBright)
                     : static_cast<Color>(ColorToolPending);
        Element content = paragraph(todo.content)
            | ftxui::color(done ? Color::GrayDark : Color::GrayLight)
            | xflex;
        if (done) content = std::move(content) | dim;
        rows.push_back(
            hbox({
                ftxui::text(std::format("{} ", marker)) | ftxui::color(marker_color),
                std::move(content),
            }) | xflex);
    }
    return vbox(std::move(rows)) | xflex;
}

Element render_fetch_result(const ToolActivity& tool,
                            const ToolResultView& view,
                            const ConversationRenderOptions& options) {
    if (!view.fetch.parsed) {
        return render_text_lines_preserving_newlines(tool.result.summary, Color::GrayLight);
    }

    std::vector<std::string> metadata;
    if (view.fetch.status_code != 0) {
        metadata.push_back(std::to_string(view.fetch.status_code));
    }
    if (!view.fetch.content_type.empty()) metadata.push_back(view.fetch.content_type);
    if (view.fetch.truncated) metadata.emplace_back("truncated");

    std::vector<Element> rows;
    if (!view.fetch.title.empty()) {
        rows.push_back(
            paragraph(view.fetch.title) | ftxui::color(Color::GrayLight) | ftxui::bold | xflex);
    }
    if (!metadata.empty()) {
        rows.push_back(
            ftxui::text(join_with(metadata, " · "))
            | ftxui::color(ColorYellowDark)
            | dim);
    }
    // The description already shows the URL for most fetches; repeating it is noise.
    if (!view.fetch.url.empty() && view.fetch.url != tool.description) {
        rows.push_back(paragraph(view.fetch.url) | ftxui::color(Color::GrayDark) | dim | xflex);
    }

    const auto display = options.expand_tool_results
        ? LimitedTextResult{.text = tool.result.summary, .hidden_lines = 0}
        : clamp_tool_result_lines(tool.result.summary, specialized_preview_limit(options));
    if (!display.text.empty()) {
        rows.push_back(separator() | ftxui::color(Color::GrayDark) | dim);
        rows.push_back(render_text_lines_preserving_newlines(display.text, Color::GrayLight));
    }
    if (display.hidden_lines > 0) {
        rows.push_back(
            ftxui::text(std::format("… {} more lines", display.hidden_lines))
            | ftxui::color(Color::GrayDark)
            | dim);
    }
    return vbox(std::move(rows)) | xflex;
}

Element render_tool_header(const ToolActivity& tool,
                           const ToolResultView& view,
                           const std::string& disclosure_key,
                           std::size_t tick,
                           const ConversationRenderOptions& options,
                           int /*terminal_width*/,
                           Color /*border_color*/,
                           bool /*is_first*/,
                           bool expandable,
                           bool expanded) {
    const Color status_color = tui::tool_status_color(tool.status);
    Element icon_el = tool.status == ToolActivity::Status::Executing && options.show_spinner
        ? status_spinner_text(tick, options)
        : ftxui::text(std::string(tui::tool_status_icon(tool.status)));
    Element status_el = hbox({ftxui::text(" "), std::move(icon_el), ftxui::text(" ")})
                      | ftxui::color(status_color) | ftxui::bold;

    // Chevron plus label form the click target. Reflecting the whole header row
    // instead would swallow clicks across the full terminal width and break
    // FTXUI's text selection on every tool line.
    Element toggle_el = hbox({
        expandable
            ? ftxui::text(expanded ? "▼ " : "▶ ") | ftxui::color(ColorYellowDark) | dim
            : ftxui::text("  "),
        ftxui::text(std::string(view.presentation.label))
            | ftxui::bold
            | ftxui::color(ColorYellowBright),
    });
    if (expandable && options.system_disclosure_hitboxes != nullptr) {
        auto& box = (*options.system_disclosure_hitboxes)[disclosure_key];
        toggle_el = std::move(toggle_el) | reflect(box);
    }

    std::vector<Element> header_items;
    header_items.push_back(std::move(status_el));
    header_items.push_back(std::move(toggle_el));

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

    // Right-edge label. A collapsed card is the only thing standing between the
    // reader and the change, so a diff that chose to stay closed must say how
    // big it is — otherwise "collapsed by default" reads as "silently hidden".
    std::vector<std::string> trailing;
    if (tool.status != ToolActivity::Status::Pending &&
        tool.status != ToolActivity::Status::Executing) {
        trailing.push_back(tool_result_metric(tool, view));
    }
    if (!expanded && tool.diff_preview.total_line_count > kToolDiffAutoExpandMaxLines) {
        trailing.push_back(std::format("{} diff lines", tool.diff_preview.total_line_count));
    }
    std::erase_if(trailing, [](const std::string& part) { return part.empty(); });

    if (!trailing.empty()) {
        header_items.push_back(filler());
        header_items.push_back(
            ftxui::text(join_with(trailing, " · "))
            | ftxui::color(status_color)
            | dim);
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
                           const ToolResultView& view,
                           const ConversationRenderOptions& options,
                           int /*terminal_width*/) {
    if (tool.result.empty()) {
        return emptyElement();
    }

    const bool is_error = tool.status == ToolActivity::Status::Failed ||
                          tool.status == ToolActivity::Status::Denied;
    const Color text_color = is_error ? static_cast<Color>(ColorToolFail) : Color{Color::GrayLight};

    // Specialized renderers assume a well-formed success payload. Anything else
    // — failed, denied, cancelled mid-write — falls back to plain text so a
    // partial or non-JSON body is shown verbatim instead of as "No results".
    if (tool.status == ToolActivity::Status::Succeeded) {
        switch (view.presentation.kind) {
            case ToolPresentationKind::Read:
                return render_numbered_tool_text(
                    tool.result.summary, view.read_start_line, options);
            case ToolPresentationKind::Grep:
                return render_search_results(view, options);
            case ToolPresentationKind::Files:
            case ToolPresentationKind::Directory:
                return render_path_results(view, options);
            case ToolPresentationKind::WebSearch:
                return render_web_results(view, options);
            case ToolPresentationKind::WebFetch:
                return render_fetch_result(tool, view, options);
            case ToolPresentationKind::Todo:
                return render_todo_results(view);
            case ToolPresentationKind::Write:
            case ToolPresentationKind::Edit:
            case ToolPresentationKind::Shell:
            case ToolPresentationKind::Task:
            case ToolPresentationKind::Generic:
                break;
        }
    }

    const bool is_terminal_output = view.presentation.kind == ToolPresentationKind::Shell
        && core::tools::names::is_terminal_tool(tool.name);
    std::vector<Element> rows;
    if (is_terminal_output) {
        rows.push_back(ftxui::text("Output") | ftxui::color(ColorYellowDark) | ftxui::bold);
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

/// A successful edit can return transport/storage metadata rather than a
/// human-facing acknowledgement (for example, the truncation envelope used for
/// oversized tool results). The diff remains the useful result; keep this
/// machine-oriented JSON available without letting it dominate an auto-opened
/// edit card.
bool edit_result_is_raw_json(const ToolActivity& tool,
                             const ToolResultView& view) {
    if (tool.status != ToolActivity::Status::Succeeded
        || tool.diff_preview.empty()
        || tool.result.summary.empty()
        || tool.result.summary == kDoneSummary
        || (view.presentation.kind != ToolPresentationKind::Write
            && view.presentation.kind != ToolPresentationKind::Edit)) {
        return false;
    }

    return with_json_object(tool.result.summary, [](simdjson::dom::object) {});
}

Element render_raw_tool_result_disclosure(
    const ToolActivity& tool,
    const ToolResultView& view,
    std::string_view disclosure_key,
    const ConversationRenderOptions& options,
    int terminal_width) {
    bool expanded = options.expand_tool_results;
    if (!expanded && options.system_disclosure_expanded != nullptr) {
        if (const auto it =
                options.system_disclosure_expanded->find(std::string(disclosure_key));
            it != options.system_disclosure_expanded->end()) {
            expanded = it->second;
        }
    }

    Element toggle = hbox({
        ftxui::text(expanded ? "▼ " : "▶ ") | ftxui::color(Color::GrayDark) | dim,
        ftxui::text("Raw result") | ftxui::color(Color::GrayLight),
        filler(),
        ftxui::text("JSON") | ftxui::color(Color::GrayDark) | dim,
    }) | xflex;
    if (options.system_disclosure_hitboxes != nullptr) {
        auto& box =
            (*options.system_disclosure_hitboxes)[std::string(disclosure_key)];
        toggle = std::move(toggle) | reflect(box);
    }

    std::vector<Element> rows;
    rows.push_back(std::move(toggle));
    if (expanded) {
        rows.push_back(separator() | ftxui::color(Color::GrayDark) | dim);
        rows.push_back(render_tool_result(tool, view, options, terminal_width));
    }
    return vbox(std::move(rows)) | UiBorder(Color::GrayDark);
}

/// Draws a diff at the budget the current disclosure state allows.
/// `max_lines == 0` means "draw everything the model kept".
Element render_tool_diff_preview(const ToolDiffPreview& preview, std::size_t max_lines) {
    if (preview.empty()) {
        return emptyElement();
    }

    const auto& all_lines = preview.lines();
    const std::size_t shown = max_lines == 0
        ? all_lines.size()
        : std::min(all_lines.size(), max_lines);
    const std::size_t hidden = preview.total_line_count - shown;

    std::vector<Element> rows;
    if (!preview.title.empty()) {
        rows.push_back(ftxui::text(std::format("Diff: {}", preview.title)) | 
                      ftxui::color(ColorYellowDark) | ftxui::bold);
    }

    for (const auto& line : std::span{all_lines}.first(shown)) {
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
    
    if (hidden > 0) {
        // Be explicit about which kind of "more" this is: a display budget the
        // card chose, or a change so large the model never kept the rest. The
        // second case must not imply that another click would reveal it.
        const std::string note = shown < all_lines.size()
            ? std::format("... {} more lines — showing the first {} of {}",
                          hidden, shown, preview.total_line_count)
            : std::format("... {} more lines — diff too large to display in full", hidden);
        rows.push_back(ftxui::text(note) | dim | ftxui::color(Color::GrayDark));
    }
    
    return vbox(std::move(rows)) | UiBorder(Color::GrayDark);
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
                         std::size_t index_in_message,
                         std::size_t tick,
                         const ConversationRenderOptions& options,
                         int terminal_width,
                         Color border_color,
                         bool is_first,
                         bool /*is_last*/) {
    // Parsed once and threaded through the header and the body: the header needs
    // a count and the body needs the rows, and both used to parse independently.
    const ToolResultView view = build_tool_result_view(tool);

    const bool expandable = tool_has_disclosure_body(tool);
    const auto disclosure_key = tool_disclosure_key(tool, index_in_message);
    // The default is a pure function of the tool so that the renderer and the
    // mouse handler agree without the renderer having to publish state.
    bool expanded = tool_disclosure_defaults_expanded(tool);
    if (options.system_disclosure_expanded != nullptr) {
        if (const auto it = options.system_disclosure_expanded->find(disclosure_key);
            it != options.system_disclosure_expanded->end()) {
            expanded = it->second;
        }
    }
    if (options.expand_tool_results) expanded = true;
    expanded = expanded && expandable;

    std::vector<Element> tool_elements;
    tool_elements.push_back(
        render_tool_header(tool, view, disclosure_key, tick, options, terminal_width,
                           border_color, is_first, expandable, expanded));

    if (!expanded) {
        return vbox(std::move(tool_elements));
    }

    std::vector<Element> body;
    const auto add_section = [&body](Element element) {
        if (!body.empty()) {
            body.push_back(separator() | ftxui::color(Color::GrayDark));
        }
        body.push_back(std::move(element));
    };

    if (tool.status == ToolActivity::Status::Executing && tool.progress.has_value()) {
        body.push_back(render_tool_progress(tool));
    }

    if (!tool.subagents.empty()) {
        add_section(render_subagent_group(tool, tick, options));
    }

    const bool raw_edit_result = edit_result_is_raw_json(tool, view);

    // File-modification tools report a bare acknowledgement; the diff already
    // says everything the acknowledgement would, so skip the redundant line.
    const bool diff_supersedes_summary =
        !tool.diff_preview.empty()
        && tool.status == ToolActivity::Status::Succeeded
        && (tool.result.summary == kDoneSummary || tool.result.summary.empty());

    // A failed/cancelled edit describes a proposed change, not one that landed.
    // Lead with the failure so the diff cannot be mistaken for applied work.
    const bool result_precedes_diff =
        !tool.result.empty()
        && !diff_supersedes_summary
        && !raw_edit_result
        && tool.status != ToolActivity::Status::Succeeded;
    if (result_precedes_diff) {
        add_section(render_tool_result(tool, view, options, terminal_width));
    }

    if (raw_edit_result) {
        add_section(render_raw_tool_result_disclosure(
            tool,
            view,
            tool_raw_result_disclosure_key(tool, index_in_message),
            options,
            terminal_width));
    }

    // The change is the primary result of an edit. Draw it before any ancillary
    // human-facing response text. Raw machine metadata has its own compact,
    // collapsed row immediately above the diff.
    if (!tool.diff_preview.empty()) {
        // Expanding is an explicit request to read the change, so the diff is
        // drawn in full; the budget only bounds how tall one card can get.
        // Ctrl+O ("show me everything") lifts even that.
        add_section(render_tool_diff_preview(
            tool.diff_preview,
            options.expand_tool_results ? 0 : options.tool_diff_expanded_max_lines));
    }

    if (!tool.result.empty()
        && !diff_supersedes_summary
        && !raw_edit_result
        && !result_precedes_diff) {
        add_section(render_tool_result(tool, view, options, terminal_width));
    }

    if (!body.empty()) {
        tool_elements.push_back(
            hbox({
                ftxui::text("   │ ") | ftxui::color(ColorYellowDark) | dim,
                vbox(std::move(body)) | xflex,
            }) | xflex);
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
            render_tool_item(tool, i, tick, options, content_width,
                             border_color, is_first, is_last));

        if (!is_last) {
            content_elements.push_back(separator() | ftxui::color(Color::GrayDark) | dim);
        }
    }
    
    Element content = vbox(std::move(content_elements));
    return content | UiBorder(border_color);
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

void stamp_user_turn_elapsed(std::vector<UiMessage>& messages,
                             std::string_view message_id,
                             std::string elapsed) {
    if (message_id.empty() || elapsed.empty()) {
        return;
    }
    for (auto& msg : messages) {
        if (msg.type == MessageType::User && msg.id == message_id) {
            msg.activity_elapsed = std::move(elapsed);
            return;
        }
    }
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
    msg.assistant_source_text = msg.text;
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
                                 std::string description,
                                 bool build_diff_preview) {
    ToolActivity tool;
    tool.id = std::move(id);
    tool.name = std::move(name);
    tool.args = std::move(args);
    tool.description = std::move(description);
    if (build_diff_preview) {
        tool.diff_preview = build_tool_diff_preview(tool.name, tool.args);
    }
    tool.status = ToolActivity::Status::Pending;
    return tool;
}

namespace {

/// Derives `tool.status` and `tool.result.summary` from a raw tool payload.
/// Deliberately collapses structured results down to the single string the
/// generic renderer shows; `apply_tool_result` preserves the original JSON for
/// the tools that need more than that.
void apply_tool_result_summary(ToolActivity& tool, std::string_view result_payload) {
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

} // namespace

void apply_tool_result(ToolActivity& tool, std::string_view result_payload) {
    tool.result.clear();
    apply_tool_result_summary(tool, result_payload);

    // Structured renderers (grep, file/dir listings, web, plan, fetch) need
    // fields the summary drops. Retaining the payload only when the summary no
    // longer contains it keeps exactly one copy in transcript memory, and frees
    // those renderers from depending on which branch above happened to match.
    // Oversized payloads are skipped: the renderer degrades to plain text rather
    // than doubling the memory held by a long-lived transcript.
    if (tool_result_needs_raw_payload(tool.name)
        && tool.result.summary != result_payload
        && result_payload.size() <= kMaxRetainedRawPayloadBytes) {
        tool.result.raw_payload = std::string(result_payload);
    }
}

// ============================================================================
// Tool Disclosure
// ============================================================================

bool tool_has_disclosure_body(const ToolActivity& tool) {
    return !tool.result.empty()
        || !tool.diff_preview.empty()
        || !tool.subagents.empty()
        || tool.progress.has_value();
}

bool tool_disclosure_defaults_expanded(const ToolActivity& tool) {
    switch (tool.status) {
        case ToolActivity::Status::Failed:
        case ToolActivity::Status::Denied:
        case ToolActivity::Status::Cancelled:
            // Anything that went wrong is worth reading without a click.
            return true;
        case ToolActivity::Status::Executing:
            // Live progress is only useful while it is still moving.
            return !tool.subagents.empty() || tool.progress.has_value();
        case ToolActivity::Status::Pending:
        case ToolActivity::Status::Succeeded:
            break;
    }
    // A pending or successful edit still shows its diff: it is the change the
    // user is being asked to trust, not incidental output. Long diffs are the
    // exception — auto-opening a 300-line rewrite buries every neighbouring
    // card, so those start collapsed and advertise their size in the header.
    return !tool.diff_preview.empty()
        && tool.diff_preview.total_line_count <= kToolDiffAutoExpandMaxLines;
}

std::string tool_disclosure_key(const ToolActivity& tool, std::size_t index_in_message) {
    if (!tool.id.empty()) {
        return "tool:" + tool.id;
    }
    return std::format("tool:{}:{}:{}", index_in_message, tool.name, tool.description);
}

std::string tool_raw_result_disclosure_key(const ToolActivity& tool,
                                           std::size_t index_in_message) {
    return tool_disclosure_key(tool, index_in_message) + ":raw-result";
}

// ============================================================================
// Rendering Functions
// ============================================================================

Element render_user_message(const UiMessage& msg, const ConversationRenderOptions& options) {
    const std::string_view timestamp = options.show_timestamps
        ? std::string_view{msg.timestamp}
        : std::string_view{};
    return vbox({
        user_message_bubble(msg.text, timestamp, msg.activity_elapsed),
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

namespace {

// Sub-key under which an assistant message's activity-box expand state and
// hitbox are tracked. Distinct from the message id so a card can carry both an
// activity box and other disclosures without collisions.
[[nodiscard]] std::string reasoning_disclosure_key(std::string_view message_id) {
    return std::string(message_id) + ":reasoning";
}

// Present-continuous verb for the live phase ("Thinking" / "Analyzing").
[[nodiscard]] std::string_view activity_present_label(UiMessage::ActivityKind kind) {
    return kind == UiMessage::ActivityKind::Analyzing ? "Analyzing" : "Thinking";
}

// Past-tense verb for the finished trace ("Thought" / "Analyzed").
[[nodiscard]] std::string_view activity_past_label(UiMessage::ActivityKind kind) {
    return kind == UiMessage::ActivityKind::Analyzing ? "Analyzed" : "Thought";
}

// A reasoning disclosure only exists when there is actual provider-supplied
// reasoning content to reveal.
[[nodiscard]] bool activity_is_expandable(const UiMessage& msg) {
    return !msg.reasoning_text.empty();
}

// Generic activity is represented by the neutral working indicator. The
// lightbulb disclosure is reserved exclusively for real reasoning content.
[[nodiscard]] bool has_activity_disclosure(const UiMessage& msg,
                                           const ConversationRenderOptions& options) {
    if (!options.show_reasoning) {
        return false;
    }
    return !msg.reasoning_text.empty();
}

// Renders the provider-supplied reasoning disclosure for an assistant turn.
//
// States, one code path:
//   * Live: a collapsed disclosure with a pulsing "Thinking…" header + timer.
//   * Finished with text: collapsed "▶ Thought for Ns", expandable via click or
//     Ctrl+O to reveal the chain of thought.
//
// Reuses the existing system-disclosure state map + hitbox machinery (keyed by
// reasoning_disclosure_key) so toggle persistence, cache invalidation and mouse
// handling all come for free.
[[nodiscard]] Element render_reasoning_disclosure(
    const UiMessage& msg,
    std::size_t tick,
    const ConversationRenderOptions& options) {
    const std::string key = reasoning_disclosure_key(msg.id);
    const bool expandable = activity_is_expandable(msg);
    // The disclosure is live while the turn is still running; its header
    // animates and shows the present-tense label.
    const bool is_live = msg.pending && !msg.finalized;
    bool expanded = options.expand_system_details;
    if (options.system_disclosure_expanded != nullptr) {
        if (const auto it = options.system_disclosure_expanded->find(key);
            it != options.system_disclosure_expanded->end()) {
            expanded = expanded || it->second;
        }
    }
    expanded = expanded && expandable;

    std::vector<Element> lines;

    // --- Summary / header line -------------------------------------------------
    Elements header;
    // The lightbulb is reserved for genuine provider reasoning. While the
    // disclosure is live it gently breathes; once finished it is static.
    header.push_back(is_live
        ? render_live_lightbulb_prefix(options, tick)
        : render_lightbulb_prefix(true));
    // A disclosure always has content, so it always gets an interactive chevron.
    if (expandable) {
        header.push_back(
            ftxui::text(expanded ? "▼ " : "▶ ") | ftxui::color(ColorYellowDark));
    }

    if (is_live) {
        const std::string present(activity_present_label(msg.reasoning_kind));
        header.push_back(
            (options.show_spinner ? pulse_text(present, tick, options)
                                  : ftxui::text(present + "..."))
            | ftxui::color(ColorYellowDark) | dim);
        if (options.activity_elapsed) {
            const auto elapsed = options.activity_elapsed;
            const auto message_id = msg.id;
            header.push_back(
                live_text([elapsed, message_id]() {
                    const auto value = elapsed(message_id);
                    return value.empty() ? std::string{} : std::format(" ({})", value);
                })
                | ftxui::color(Color::GrayDark) | dim);
        } else if (!msg.activity_elapsed.empty()) {
            header.push_back(
                ftxui::text(std::format(" ({})", msg.activity_elapsed))
                | ftxui::color(Color::GrayDark) | dim);
        }
    } else {
        const std::string past(activity_past_label(msg.reasoning_kind));
        std::string summary = msg.reasoning_elapsed.empty() ? past : std::format("{} for {}", past, msg.reasoning_elapsed);
        header.push_back(ftxui::text(std::move(summary)) | ftxui::color(ColorYellowDark) | dim);
    }

    Element header_el = hbox(std::move(header));
    // Register a click hitbox for the disclosure header.
    if (expandable && options.system_disclosure_hitboxes != nullptr) {
        auto& box = (*options.system_disclosure_hitboxes)[key];
        header_el = std::move(header_el) | reflect(box);
    }
    lines.push_back(std::move(header_el));

    // --- Body ------------------------------------------------------------------
    if (expanded) {
        // While streaming, clamp to the tail so the box does not grow without
        // bound; when finished/expanded, show the full text.
        std::string body = msg.reasoning_text;
        if (msg.reasoning_active
            && options.reasoning_stream_preview_max_lines > 0) {
            const auto all = split_lines(body);
            if (all.size() > options.reasoning_stream_preview_max_lines) {
                const std::size_t start =
                    all.size() - options.reasoning_stream_preview_max_lines;
                std::string clamped;
                for (std::size_t i = start; i < all.size(); ++i) {
                    if (!clamped.empty()) {
                        clamped.push_back('\n');
                    }
                    clamped += all[i];
                }
                body = std::move(clamped);
            }
        }
        for (const auto& line : split_lines(body)) {
            if (line.empty()) {
                lines.push_back(ftxui::text(""));
                continue;
            }
            lines.push_back(
                hbox({
                    ftxui::text("  "),
                    paragraph(line) | ftxui::color(Color::GrayDark) | dim | xflex,
                }) | xflex);
        }
    }

    return vbox(std::move(lines));
}

} // namespace

Element render_assistant_message(const UiMessage& msg,
                                 std::size_t tick,
                                 const ConversationRenderOptions& options) {
    std::vector<Element> elements;
    const auto activity = compute_assistant_activity_state(msg);

    // Show waiting state when empty and not thinking
    if (msg.text.empty() && msg.tools.empty() && !msg.thinking) {
        const auto label = msg.pending ? "Waiting for the model..." : "No response.";
        const bool show = msg.pending || msg.show_activity_status;
        elements.push_back(
            hbox({
                render_working_prefix(show, options, tick),
                ftxui::text(label) | ftxui::color(Color::GrayLight) | dim
            }));
    }

    // Render real provider reasoning before the visible response. Generic model
    // activity stays in the neutral working indicator below.
    const bool has_reasoning = has_activity_disclosure(msg, options);
    // A live reasoning disclosure owns the activity label, avoiding duplication.
    const bool disclosure_owns_live_indicator =
        has_reasoning && msg.pending && !msg.finalized;
    if (has_reasoning) {
        elements.push_back(render_reasoning_disclosure(msg, tick, options));
    }

    // Provider text introduces the work that follows, so keep it before this
    // step's tool cards. This ordering also makes live and resumed transcripts
    // identical at provider-step boundaries.
    if (!msg.text.empty()) {
        if (has_reasoning) {
            elements.push_back(ftxui::text(""));
        }
        elements.push_back(render_markdown(msg.text, Color::White));
    }

    // Tool calls belong to this provider step and follow its narration.
    if (!msg.tools.empty()) {
        if (has_reasoning || !msg.text.empty()) {
            elements.push_back(ftxui::text(""));
        }
        UiMessage tool_group = msg;
        tool_group.type = MessageType::ToolGroup;
        elements.push_back(render_tool_group(tool_group, tick, options));
    }

    // Render thinking indicator at the BOTTOM (current activity)
    // This ensures users see what's happening NOW when looking at the bottom.
    // Suppressed while the disclosure above is already streaming a live label
    // (avoids a double indicator).
    if (activity.show_thinking && !disclosure_owns_live_indicator) {
        Element label;
        label = options.show_spinner
            ? pulse_text("Working", tick, options)
            : ftxui::text("Working...");
        Elements indicator_row = {
            render_working_prefix(true, options, tick),
            std::move(label) | ftxui::color(Color::GrayLight) | dim
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

    // Bottom-left bookend for the completed turn: same gray clock/elapsed
    // language as the yellow question box, opposite corner so a long answer
    // still shows when the turn finished without scrolling back up.
    if (msg.finalized) {
        const std::string_view timestamp = options.show_timestamps
            ? std::string_view{msg.timestamp}
            : std::string_view{};
        auto time_label = format_message_time_label(timestamp, msg.activity_elapsed);
        if (!time_label.empty()) {
            elements.push_back(
                hbox({
                    ftxui::text(std::move(time_label)) | ftxui::color(Color::GrayDark),
                    filler(),
                }));
        }
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
    msg_elements.reserve(messages.size());

    for (const auto& msg : messages) {
        msg_elements.push_back(render_history_message(msg, tick, options));
    }

    return vbox(std::move(msg_elements));
}

Element render_history_message(const UiMessage& message,
                               std::size_t tick,
                               const ConversationRenderOptions& options) {
    Element card;
    bool trailing_space = false;
    switch (message.type) {
        case MessageType::User:
            card = render_user_message(message, options);
            break;
        case MessageType::ShellCommand:
            card = render_shell_command_message(message, tick, options);
            trailing_space = true;
            break;
        case MessageType::Assistant:
            card = render_assistant_message(message, tick, options);
            trailing_space = true;
            break;
        case MessageType::Info:
            card = render_info_message(message);
            break;
        case MessageType::Warning:
            card = render_warning_message(message);
            break;
        case MessageType::Error:
            card = render_error_message(message);
            break;
        case MessageType::ToolGroup:
            card = render_tool_group(message, tick, options);
            trailing_space = true;
            break;
        case MessageType::System:
            card = render_system_message(message, options);
            break;
    }
    if (!trailing_space) {
        return card;
    }
    return vbox({std::move(card), ftxui::text("")});
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
