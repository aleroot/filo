#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <string>
#include <string_view>

namespace core::utils::time {

/**
 * @brief Parse an ISO 8601 string, numeric timestamp (epoch seconds/milliseconds),
 *        or relative duration ("6s", "20ms", "2m30s") into unix epoch seconds.
 */
[[nodiscard]] inline int64_t parse_timestamp_or_duration(std::string_view sv, int64_t now_seconds = 0) noexcept {
    if (sv.empty()) return 0;
    if (now_seconds == 0) {
        now_seconds = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    const std::string s(sv);

    // 1. Check for ISO 8601 string format (e.g. 2025-01-15T12:00:30...)
    int year = 0, month = 0, day = 0, hour = 0, min = 0, sec = 0;
    if (std::sscanf(s.c_str(), "%d-%d-%dT%d:%d:%d", &year, &month, &day, &hour, &min, &sec) >= 6) {
        std::tm t{};
        t.tm_year = year - 1900;
        t.tm_mon  = month - 1;
        t.tm_mday = day;
        t.tm_hour = hour;
        t.tm_min  = min;
        t.tm_sec  = sec;
#if defined(_WIN32)
        int64_t epoch = static_cast<int64_t>(_mkgmtime(&t));
#else
        int64_t epoch = static_cast<int64_t>(timegm(&t));
#endif
        if (epoch < 0) return 0;

        // Check for timezone offset (+HH:MM or -HH:MM)
        const auto plus_pos = s.find_last_of('+');
        const auto minus_pos = s.find_last_of('-');
        std::size_t tz_pos = std::string::npos;
        int tz_sign = 1;
        if (plus_pos != std::string::npos && plus_pos > 10) {
            tz_pos = plus_pos;
            tz_sign = -1; // offset +02:00 means UTC = local - 2h
        } else if (minus_pos != std::string::npos && minus_pos > 10) {
            tz_pos = minus_pos;
            tz_sign = 1; // offset -05:00 means UTC = local + 5h
        }
        if (tz_pos != std::string::npos) {
            int tz_h = 0, tz_m = 0;
            if (std::sscanf(s.c_str() + tz_pos + 1, "%d:%d", &tz_h, &tz_m) >= 1) {
                epoch += tz_sign * (tz_h * 3600 + tz_m * 60);
            }
        }
        return epoch;
    }

    // 2. Check for duration format (e.g. "6s", "20ms", "1m30s", "500ms")
    if (sv.ends_with("ms") || sv.ends_with("MS")) {
        char* end = nullptr;
        const double ms = std::strtod(s.c_str(), &end);
        if (end != s.c_str() && ms >= 0.0) {
            return now_seconds + static_cast<int64_t>(ms / 1000.0 + 0.999);
        }
    }
    if (sv.ends_with("s") || sv.ends_with("S")) {
        const auto m_pos = s.find('m');
        if (m_pos != std::string::npos && m_pos < s.size() - 1) {
            int m = 0;
            double sec_val = 0;
            if (std::sscanf(s.c_str(), "%dm%lfs", &m, &sec_val) >= 2) {
                return now_seconds + m * 60 + static_cast<int64_t>(sec_val + 0.999);
            }
        }
        char* end = nullptr;
        const double sec_val = std::strtod(s.c_str(), &end);
        if (end != s.c_str() && sec_val >= 0.0) {
            return now_seconds + static_cast<int64_t>(sec_val + 0.999);
        }
    }
    if (sv.ends_with("m") || sv.ends_with("M")) {
        char* end = nullptr;
        const double min_val = std::strtod(s.c_str(), &end);
        if (end != s.c_str() && min_val >= 0.0) {
            return now_seconds + static_cast<int64_t>(min_val * 60.0 + 0.999);
        }
    }
    if (sv.ends_with("h") || sv.ends_with("H")) {
        char* end = nullptr;
        const double h_val = std::strtod(s.c_str(), &end);
        if (end != s.c_str() && h_val >= 0.0) {
            return now_seconds + static_cast<int64_t>(h_val * 3600.0 + 0.999);
        }
    }

    // 3. Fallback: numeric timestamp (epoch seconds or milliseconds)
    char* end = nullptr;
    const int64_t val = std::strtoll(s.c_str(), &end, 10);
    if (end != s.c_str() && val > 0) {
        if (val > 100'000'000'000LL) {
            return val / 1000LL;
        }
        return val;
    }

    return 0;
}

} // namespace core::utils::time
