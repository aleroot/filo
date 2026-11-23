#include "SettingsInput.hpp"

#include "core/config/ConfigManager.hpp"
#include "core/context/ContextWindowTracker.hpp"

#include <format>

namespace tui {

void SettingsTextEntry::begin() {
    active_ = true;
    text_.clear();
}

SettingsTextEntry::Outcome SettingsTextEntry::handle(const ftxui::Event& event) {
    if (!active_) return Outcome::Cancelled;
    if (event == ftxui::Event::Escape) {
        active_ = false;
        return Outcome::Cancelled;
    }
    if (event == ftxui::Event::Return) {
        active_ = false;
        return Outcome::Submitted;
    }
    if (event == ftxui::Event::Backspace) {
        // Drop one UTF-8 character: its continuation bytes, then its lead.
        while (!text_.empty() && (static_cast<unsigned char>(text_.back()) & 0xC0) == 0x80) {
            text_.pop_back();
        }
        if (!text_.empty()) text_.pop_back();
    } else if (event.is_character()) {
        text_ += event.character();
    }
    return Outcome::Editing;
}

const std::vector<std::string_view>& auto_compact_presets() {
    static const std::vector<std::string_view> presets{
        "auto", "25%", "40%", "50%", "60%", "90%", "0"};
    return presets;
}

std::string describe_auto_compact_setting(std::string_view value) {
    const auto canonical = core::config::normalize_auto_compact_setting(value);
    if (!canonical) return std::string(value);
    if (*canonical == "auto") {
        return std::format(
            "Default ({:.0f}% of window)",
            core::context::CompactionTriggerPolicy::kDefaultContextUsageRatio * 100.0);
    }
    if (*canonical == "0") return "Disabled";
    if (canonical->ends_with('%')) return std::format("{} of window", *canonical);
    const auto tokens = std::stoll(*canonical);
    if (tokens % 1'000'000 == 0) return std::format("{}M tokens", tokens / 1'000'000);
    if (tokens % 1'000 == 0) return std::format("{}k tokens", tokens / 1'000);
    return std::format("{} tokens", tokens);
}

SettingsCustomInput auto_compact_custom_input() {
    return SettingsCustomInput{
        .hint = "35%, 400k, 400000, auto or off",
        .parse = [](std::string_view text) {
            return core::config::normalize_auto_compact_setting(text);
        },
        .describe = [](std::string_view value) { return describe_auto_compact_setting(value); },
    };
}

} // namespace tui
