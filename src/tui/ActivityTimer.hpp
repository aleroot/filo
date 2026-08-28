#pragma once

#include <chrono>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

namespace tui {

/**
 * @brief Thread-safe registry of monotonic elapsed timers keyed by operation ID.
 */
class ActivityTimerRegistry {
public:
    using Clock = std::chrono::steady_clock;

    void start(std::string_view operation_id);
    void start_at(std::string_view operation_id, Clock::time_point start_time);
    void stop(std::string_view operation_id);
    void clear();

    [[nodiscard]] std::optional<std::chrono::seconds> elapsed(
        std::string_view operation_id,
        Clock::time_point now = Clock::now()) const;

private:
    mutable std::mutex                                        mutex_;
    std::unordered_map<std::string, Clock::time_point>        starts_;
};

enum class ElapsedFormat {
    /// Timer-style output retains seconds and zero-pads subordinate units.
    precise,
    /// Relative-time output keeps at most the two most useful units.
    humanized,
};

[[nodiscard]] std::string format_elapsed_compact(
    std::chrono::seconds elapsed,
    ElapsedFormat format = ElapsedFormat::precise);

} // namespace tui
