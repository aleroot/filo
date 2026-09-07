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
    /// Human-readable output keeps at most the two most significant units.
    /// Below the hour seconds still carry information ("6m 16s"); from the
    /// first hour on they are dropped in favour of the coarser unit ("1h",
    /// "4h 13m", "2d 1h"). Used for both durations and "... ago" ages.
    humanized,
};

[[nodiscard]] std::string format_elapsed_compact(
    std::chrono::seconds elapsed,
    ElapsedFormat format = ElapsedFormat::precise);

} // namespace tui
