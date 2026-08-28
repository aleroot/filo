#include "ActivityTimer.hpp"

#include <format>

namespace tui {

void ActivityTimerRegistry::start(std::string_view operation_id) {
    start_at(operation_id, Clock::now());
}

void ActivityTimerRegistry::start_at(std::string_view operation_id, Clock::time_point start_time) {
    if (operation_id.empty()) {
        return;
    }
    std::lock_guard lock(mutex_);
    starts_[std::string(operation_id)] = start_time;
}

void ActivityTimerRegistry::stop(std::string_view operation_id) {
    if (operation_id.empty()) {
        return;
    }
    std::lock_guard lock(mutex_);
    starts_.erase(std::string(operation_id));
}

void ActivityTimerRegistry::clear() {
    std::lock_guard lock(mutex_);
    starts_.clear();
}

std::optional<std::chrono::seconds> ActivityTimerRegistry::elapsed(
    std::string_view operation_id,
    Clock::time_point now) const {
    if (operation_id.empty()) {
        return std::nullopt;
    }

    std::lock_guard lock(mutex_);
    const auto it = starts_.find(std::string(operation_id));
    if (it == starts_.end()) {
        return std::nullopt;
    }

    auto delta = std::chrono::duration_cast<std::chrono::seconds>(now - it->second);
    if (delta < std::chrono::seconds::zero()) {
        delta = std::chrono::seconds::zero();
    }
    return delta;
}

std::string format_elapsed_compact(std::chrono::seconds elapsed,
                                   ElapsedFormat format) {
    if (elapsed < std::chrono::seconds::zero()) {
        elapsed = std::chrono::seconds::zero();
    }

    const auto total_seconds = elapsed.count();
    const auto total_hours = total_seconds / 3600;
    const auto days = total_hours / 24;
    const auto hours = total_hours % 24;
    const auto minutes = (total_seconds % 3600) / 60;
    const auto seconds = total_seconds % 60;

    if (format == ElapsedFormat::humanized) {
        if (days > 0) {
            return hours == 0
                ? std::format("{}d", days)
                : std::format("{}d {}h", days, hours);
        }
        if (total_hours > 0) {
            return minutes == 0
                ? std::format("{}h", total_hours)
                : std::format("{}h {}m", total_hours, minutes);
        }
        if (minutes > 0) {
            return seconds == 0
                ? std::format("{}m", minutes)
                : std::format("{}m {:02}s", minutes, seconds);
        }
        return std::format("{}s", seconds);
    }

    if (days > 0) {
        return std::format(
            "{}d {:02}h {:02}m {:02}s", days, hours, minutes, seconds);
    }
    if (total_hours > 0) {
        return std::format("{}h {:02}m {:02}s", total_hours, minutes, seconds);
    }
    if (minutes > 0) {
        return std::format("{}m {:02}s", minutes, seconds);
    }
    return std::format("{}s", seconds);
}

} // namespace tui
