#include "HistoryComponent.hpp"

#include "Constants.hpp"

#include <algorithm>

namespace tui {
namespace {

int estimate_wrapped_line_count(std::string_view text) {
    if (text.empty()) {
        return 1;
    }

    int total_lines = 0;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto newline = text.find('\n', start);
        const std::size_t len = newline == std::string_view::npos
            ? text.size() - start
            : newline - start;
        const int wrapped = std::max(
            1,
            static_cast<int>((len + static_cast<std::size_t>(kEstimatedLineWidthChars) - 1)
                             / static_cast<std::size_t>(kEstimatedLineWidthChars)));
        total_lines += wrapped;

        if (newline == std::string_view::npos) {
            break;
        }
        start = newline + 1;
    }

    return std::max(total_lines, 1);
}

int estimate_tool_line_cost(const ToolActivity& tool) {
    int lines = 1; // Header row

    if (!tool.description.empty()) {
        lines += estimate_wrapped_line_count(tool.description);
    }
    if (!tool.result.empty()) {
        lines += 1; // Separator / label row
        lines += estimate_wrapped_line_count(tool.result.summary);
        if (tool.result.truncated) {
            lines += 1;
        }
    }
    if (!tool.diff_preview.empty()) {
        lines += 1; // Diff header
        lines += static_cast<int>(tool.diff_preview.lines.size());
        if (tool.diff_preview.hidden_line_count > 0) {
            lines += 1;
        }
    }

    return std::max(lines, 1);
}

int estimate_message_line_cost(
    const UiMessage& msg,
    const std::unordered_map<std::string, bool>& disclosure_expanded,
    bool expand_all_system_details) {
    int lines = std::max(0, msg.margin_top);

    switch (msg.type) {
        case MessageType::User: {
            if (!msg.timestamp.empty()) {
                lines += 1;
            }
            lines += estimate_wrapped_line_count(msg.text);
            lines += 1; // Spacer after user bubble
            break;
        }
        case MessageType::Assistant: {
            bool has_executing_tools = false;
            bool has_completed_tools = false;
            for (const auto& tool : msg.tools) {
                if (tool.status == ToolActivity::Status::Executing) {
                    has_executing_tools = true;
                } else if (tool.status == ToolActivity::Status::Succeeded
                           || tool.status == ToolActivity::Status::Failed) {
                    has_completed_tools = true;
                }
            }
            const bool show_thinking = msg.thinking
                || (msg.pending && has_completed_tools && !has_executing_tools);

            if (msg.text.empty() && msg.tools.empty() && !msg.thinking) {
                lines += 1;
            }

            if (!msg.tools.empty()) {
                lines += 2; // Tool-group border top/bottom
                for (std::size_t i = 0; i < msg.tools.size(); ++i) {
                    lines += estimate_tool_line_cost(msg.tools[i]);
                    if (i + 1 < msg.tools.size()) {
                        lines += 1; // Separator between tool rows
                    }
                }
            }

            if (!msg.text.empty()) {
                if (!msg.tools.empty()) {
                    lines += 1; // Spacer between tools and markdown response
                }
                lines += estimate_wrapped_line_count(msg.text);
            }

            if (show_thinking) {
                lines += 1;
            }

            lines += 1; // Spacer after assistant message
            break;
        }
        case MessageType::Info:
        case MessageType::Warning:
        case MessageType::Error:
        case MessageType::System: {
            lines += estimate_wrapped_line_count(msg.text);
            if (!msg.secondary_text.empty()) {
                lines += estimate_wrapped_line_count(msg.secondary_text);
            }
            if (!msg.disclosure_text.empty()) {
                bool expanded = expand_all_system_details;
                if (const auto it = disclosure_expanded.find(msg.id);
                    it != disclosure_expanded.end()) {
                    expanded = expanded || it->second;
                }
                if (expanded) {
                    lines += estimate_wrapped_line_count(msg.disclosure_text);
                }
            }
            break;
        }
        case MessageType::ToolGroup: {
            lines += 2; // Tool-group border top/bottom
            for (std::size_t i = 0; i < msg.tools.size(); ++i) {
                lines += estimate_tool_line_cost(msg.tools[i]);
                if (i + 1 < msg.tools.size()) {
                    lines += 1; // Separator between tool rows
                }
            }
            lines += 1; // Spacer after tool group message
            break;
        }
    }

    return std::max(lines, 1);
}

int estimate_history_line_cost(
    const std::vector<UiMessage>& messages,
    const std::unordered_map<std::string, bool>& disclosure_expanded,
    bool expand_all_system_details) {
    int total = 0;
    for (const auto& msg : messages) {
        total += estimate_message_line_cost(
            msg,
            disclosure_expanded,
            expand_all_system_details);
    }
    return std::max(total, 1);
}

} // namespace

HistoryComponent::HistoryComponent(
    std::function<std::vector<UiMessage>()> get_messages,
    const std::atomic<size_t>& animation_tick,
    std::function<ConversationRenderOptions()> get_options)
    : get_messages_(std::move(get_messages)),
      animation_tick_(animation_tick),
      get_options_(std::move(get_options)) {}

ftxui::Element HistoryComponent::OnRender() {
    // Auto-scroll to bottom if new messages appeared.
    const auto messages = get_messages_();
    auto options = get_options_();
    options.scroll_pos = scroll_pos_;
    options.system_disclosure_expanded = &disclosure_expanded_;
    disclosure_hitboxes_.clear();
    options.system_disclosure_hitboxes = &disclosure_hitboxes_;

    const std::size_t disclosure_hash = [&]() -> std::size_t {
        std::size_t seed = 0;
        for (const auto& [message_id, expanded] : disclosure_expanded_) {
            if (!expanded) {
                continue;
            }
            seed ^= std::hash<std::string>{}(message_id)
                + static_cast<std::size_t>(0x9e3779b97f4a7c15ULL)
                + (seed << 6)
                + (seed >> 2);
        }
        return seed;
    }();

    std::size_t layout_fingerprint = history_layout_fingerprint(messages);
    layout_fingerprint = combine_hash(
        layout_fingerprint,
        static_cast<std::size_t>(options.expand_system_details));
    layout_fingerprint = combine_hash(layout_fingerprint, disclosure_hash);
    if (layout_fingerprint != last_layout_fingerprint_) {
        estimated_content_lines_ = estimate_history_line_cost(
            messages,
            disclosure_expanded_,
            options.expand_system_details);
        last_layout_fingerprint_ = layout_fingerprint;
    }
    if (messages.size() > last_message_count_) {
        scroll_pos_ = 1.0f;
        last_message_count_ = messages.size();
    }

    return render_history_panel(
        messages,
        animation_tick_.load(std::memory_order_relaxed),
        options);
}

bool HistoryComponent::Focusable() const {
    return true;
}

bool HistoryComponent::OnEvent(ftxui::Event event) {
    if (event.is_mouse()) {
        return OnMouseEvent(event);
    }

    if (Focused()) {
        if (event == ftxui::Event::ArrowUp) {
            ScrollUp(arrow_step_ratio());
            return true;
        }
        if (event == ftxui::Event::ArrowDown) {
            ScrollDown(arrow_step_ratio());
            return true;
        }
        if (event == ftxui::Event::PageUp) {
            ScrollUp(page_step_ratio());
            return true;
        }
        if (event == ftxui::Event::PageDown) {
            ScrollDown(page_step_ratio());
            return true;
        }
        if (event == ftxui::Event::Home) {
            scroll_pos_ = 0.0f;
            return true;
        }
        if (event == ftxui::Event::End) {
            scroll_pos_ = 1.0f;
            return true;
        }
        // Escape or Enter returns focus to the input.
        if (event == ftxui::Event::Escape || event == ftxui::Event::Return) {
            return false;
        }
    }
    return false;
}

void HistoryComponent::ScrollUp(float amount) {
    scroll_pos_ = std::max(0.0f, scroll_pos_ - amount);
}

void HistoryComponent::ScrollDown(float amount) {
    scroll_pos_ = std::min(1.0f, scroll_pos_ + amount);
}

void HistoryComponent::ScrollPageUp() {
    ScrollUp(page_step_ratio());
}

void HistoryComponent::ScrollPageDown() {
    ScrollDown(page_step_ratio());
}

void HistoryComponent::JumpToMessage(std::size_t message_index, std::size_t message_count) {
    if (message_count <= 1) {
        scroll_pos_ = 1.0f;
        return;
    }
    const float ratio = static_cast<float>(message_index)
        / static_cast<float>(message_count - 1);
    scroll_pos_ = std::clamp(ratio, 0.0f, 1.0f);
}

bool HistoryComponent::HandleWheel(ftxui::Event event) {
    if (!event.is_mouse()) {
        return false;
    }
    if (event.mouse().button == ftxui::Mouse::WheelUp) {
        ScrollUp(wheel_step_ratio());
        return true;
    }
    if (event.mouse().button == ftxui::Mouse::WheelDown) {
        ScrollDown(wheel_step_ratio());
        return true;
    }
    return false;
}

bool HistoryComponent::OnMouseEvent(ftxui::Event event) {
    if (event.mouse().button == ftxui::Mouse::Left
        && event.mouse().motion == ftxui::Mouse::Pressed) {
        for (const auto& [message_id, box] : disclosure_hitboxes_) {
            if (box.Contain(event.mouse().x, event.mouse().y)) {
                disclosure_expanded_[message_id] = !disclosure_expanded_[message_id];
                last_layout_fingerprint_ = 0;  // force line-cost refresh next render
                return true;
            }
        }
    }

    // Only handle wheel events; never steal focus from clicks.
    if (event.mouse().button == ftxui::Mouse::WheelUp) {
        ScrollUp(wheel_step_ratio());
        return true;
    }
    if (event.mouse().button == ftxui::Mouse::WheelDown) {
        ScrollDown(wheel_step_ratio());
        return true;
    }
    return false;
}

std::size_t HistoryComponent::combine_hash(std::size_t seed, std::size_t value) {
    // 64-bit mix constant (works fine on 32-bit std::size_t as well).
    constexpr std::size_t kMix = static_cast<std::size_t>(0x9e3779b97f4a7c15ULL);
    seed ^= value + kMix + (seed << 6) + (seed >> 2);
    return seed;
}

std::size_t HistoryComponent::history_layout_fingerprint(
    const std::vector<UiMessage>& messages) {
    std::size_t seed = messages.size();
    for (const auto& msg : messages) {
        seed = combine_hash(seed, static_cast<std::size_t>(msg.type));
        seed = combine_hash(seed, msg.text.size());
        seed = combine_hash(seed, msg.secondary_text.size());
        seed = combine_hash(seed, msg.disclosure_text.size());
        seed = combine_hash(seed, msg.timestamp.size());
        seed = combine_hash(seed, static_cast<std::size_t>(msg.margin_top));
        seed = combine_hash(seed, static_cast<std::size_t>(msg.margin_bottom));
        seed = combine_hash(seed, static_cast<std::size_t>(msg.pending));
        seed = combine_hash(seed, static_cast<std::size_t>(msg.thinking));
        seed = combine_hash(seed, static_cast<std::size_t>(msg.show_lightbulb));
        seed = combine_hash(seed, msg.tools.size());
        for (const auto& tool : msg.tools) {
            seed = combine_hash(seed, tool.name.size());
            seed = combine_hash(seed, tool.description.size());
            seed = combine_hash(seed, tool.result.summary.size());
            seed = combine_hash(seed, static_cast<std::size_t>(tool.result.truncated));
            seed = combine_hash(seed, static_cast<std::size_t>(tool.auto_approved));
            seed = combine_hash(seed, static_cast<std::size_t>(tool.status));
            if (tool.result.exit_code.has_value()) {
                seed = combine_hash(seed, static_cast<std::size_t>(*tool.result.exit_code));
            }
            seed = combine_hash(seed, tool.diff_preview.lines.size());
            seed = combine_hash(
                seed,
                static_cast<std::size_t>(tool.diff_preview.hidden_line_count));
        }
    }
    return seed;
}

float HistoryComponent::line_based_ratio(float lines, float min_ratio, float max_ratio) const {
    const float denominator = std::max(1.0f, static_cast<float>(estimated_content_lines_));
    return std::clamp(lines / denominator, min_ratio, max_ratio);
}

float HistoryComponent::wheel_step_ratio() const {
    // Aim for ~3 visual lines per wheel notch, regardless of transcript size.
    return line_based_ratio(3.0f, 0.0015f, 0.04f);
}

float HistoryComponent::arrow_step_ratio() const {
    // Keyboard arrows should be more precise than wheel scrolling.
    return line_based_ratio(2.0f, 0.001f, 0.03f);
}

float HistoryComponent::page_step_ratio() const {
    // Page navigation targets a viewport-sized jump without skipping huge chunks.
    return line_based_ratio(18.0f, 0.03f, 0.30f);
}

} // namespace tui
