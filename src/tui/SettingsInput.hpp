#pragma once

#include "core/config/ConfigManager.hpp"

#include <ftxui/component/event.hpp>

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tui {

/// "User" or "Workspace", as /settings names a scope.
[[nodiscard]] std::string settings_scope_label(core::config::SettingsScope scope);

/// A /settings value that can also be typed, not only picked from the choices.
struct SettingsCustomInput {
    /// Example input, shown while typing and when the text is refused.
    std::string hint;
    /// The value to save for the typed text, or nullopt when it is not valid.
    std::function<std::optional<std::string>(std::string_view)> parse;
    /// Label for a stored value that is not one of the choices.
    std::function<std::string(std::string_view)> describe;
};

/// The line a /settings row is typed into. While active it owns every key
/// until Enter submits or Esc cancels.
class SettingsTextEntry {
public:
    enum class Outcome { Editing, Submitted, Cancelled };

    void begin();
    void cancel() noexcept { active_ = false; }
    [[nodiscard]] bool active() const noexcept { return active_; }
    /// The text typed so far; after Submitted, the submitted text.
    [[nodiscard]] const std::string& text() const noexcept { return text_; }

    Outcome handle(const ftxui::Event& event);

private:
    bool active_ = false;
    std::string text_;
};

/// The auto-compaction presets /settings cycles through, as stored values.
[[nodiscard]] const std::vector<std::string_view>& auto_compact_presets();

/// How /settings shows a stored auto-compaction value ("40% of window",
/// "400k tokens", ...). A value the setting does not accept is shown as is.
[[nodiscard]] std::string describe_auto_compact_setting(std::string_view value);

/// Typing a custom auto-compaction share or token count.
[[nodiscard]] SettingsCustomInput auto_compact_custom_input();

} // namespace tui
