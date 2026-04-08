#pragma once

#include "Conversation.hpp"

#include <atomic>
#include <functional>
#include <unordered_map>
#include <vector>

#include <ftxui/component/component.hpp>
#include <ftxui/component/event.hpp>
#include <ftxui/screen/box.hpp>

namespace tui {

class HistoryComponent : public ftxui::ComponentBase {
public:
    HistoryComponent(
        std::function<std::vector<UiMessage>()> get_messages,
        const std::atomic<size_t>& animation_tick,
        std::function<ConversationRenderOptions()> get_options);

    ftxui::Element OnRender() override;
    bool Focusable() const override;
    bool OnEvent(ftxui::Event event) override;

    void ScrollUp(float amount);
    void ScrollDown(float amount);
    void ScrollPageUp();
    void ScrollPageDown();
    void JumpToMessage(std::size_t message_index, std::size_t message_count);

    // Helper for external controllers (e.g. input forwarding wheel events).
    bool HandleWheel(ftxui::Event event);

private:
    bool OnMouseEvent(ftxui::Event event);
    static std::size_t combine_hash(std::size_t seed, std::size_t value);
    static std::size_t history_layout_fingerprint(const std::vector<UiMessage>& messages);
    float line_based_ratio(float lines, float min_ratio, float max_ratio) const;
    float wheel_step_ratio() const;
    float arrow_step_ratio() const;
    float page_step_ratio() const;

    std::function<std::vector<UiMessage>()> get_messages_;
    const std::atomic<size_t>& animation_tick_;
    std::function<ConversationRenderOptions()> get_options_;

    float scroll_pos_ = 1.0f;
    size_t last_message_count_ = 0;
    int estimated_content_lines_ = 1;
    std::size_t last_layout_fingerprint_ = 0;
    std::unordered_map<std::string, bool> disclosure_expanded_;
    std::unordered_map<std::string, ftxui::Box> disclosure_hitboxes_;
};

} // namespace tui
