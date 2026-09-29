#pragma once

#include <format>
#include <string>
#include <string_view>

namespace core::commands {

/// Success text for a model switch. Starts with this so the transcript can
/// show what is now in use instead of a generic "Switched to …" acknowledgement.
inline constexpr std::string_view kModelInUsePrefix = "Using ";

/// True for a model switch that took effect. "Switched" remains accepted so
/// older results and tests keep their success path.
[[nodiscard]] inline bool is_model_switch_success(std::string_view message) {
    return message.starts_with("Switched") || message.starts_with(kModelInUsePrefix);
}

/// Transcript mark for a command result: an "ℹ" indication for a "Using …"
/// line, "✓" for any other success, "✗" for a failure.
[[nodiscard]] inline std::string_view result_mark(bool success, std::string_view message) {
    if (!success) {
        return "✗  ";
    }
    return message.starts_with(kModelInUsePrefix) ? "ℹ  " : "✓  ";
}

/// Transcript mark for a model-switch result.
[[nodiscard]] inline std::string_view model_switch_mark(std::string_view message) {
    return result_mark(is_model_switch_success(message), message);
}

/// "1M context", "200k context", or empty when the window is unknown.
/// Binary megatoken sizes (1<<20, 2<<20, 4<<20) are shown as 1M/2M/4M.
[[nodiscard]] inline std::string format_context_size_label(int context_tokens) {
    if (context_tokens <= 0) {
        return {};
    }
    if (context_tokens % 1'000'000 == 0) {
        return std::format("{}M context", context_tokens / 1'000'000);
    }
    if (context_tokens % (1 << 20) == 0) {
        const int mib = context_tokens / (1 << 20);
        if (mib == 1 || mib == 2 || mib == 4) {
            return std::format("{}M context", mib);
        }
    }
    if (context_tokens % 1000 == 0) {
        return std::format("{}k context", context_tokens / 1000);
    }
    if (context_tokens % 1024 == 0) {
        return std::format("{}k context", context_tokens / 1024);
    }
    return std::format("{} context", context_tokens);
}

/// First line of the model-switch indication, without the saved-source suffix.
/// Effort is omitted when unset or "auto"; wire value "none" is shown as "off".
[[nodiscard]] inline std::string format_model_in_use_notice(
    std::string_view label,
    int context_tokens,
    std::string_view effort_level) {
    std::string notice{kModelInUsePrefix};
    notice += label;
    if (const std::string context = format_context_size_label(context_tokens);
        !context.empty()) {
        notice += " · ";
        notice += context;
    }
    if (!effort_level.empty() && effort_level != "auto") {
        notice += " · ";
        notice += effort_level == "none" ? std::string_view{"off"} : effort_level;
    }
    return notice;
}

/// Pins the source of a successful switch onto the first line:
/// "from <file> · /model" when it was saved, otherwise just "· /model".
/// Lines after the first (setup hints) stay put. Other messages are unchanged.
[[nodiscard]] inline std::string append_model_in_use_source(
    std::string notice,
    bool persisted,
    std::string_view saved_source = "model_defaults.json") {
    if (!notice.starts_with(kModelInUsePrefix)) {
        return notice;
    }
    const std::string source = persisted
        ? std::format(" · from {} · /model", saved_source)
        : std::string{" · /model"};
    const auto newline = notice.find('\n');
    if (newline == std::string::npos) {
        notice += source;
        return notice;
    }
    notice.insert(newline, source);
    return notice;
}

} // namespace core::commands
