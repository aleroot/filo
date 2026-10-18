#pragma once

#include "../../utils/AsciiUtils.hpp"
#include "../../utils/StringUtils.hpp"

#include <cpr/cpr.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <charconv>
#include <string>
#include <string_view>
#include <utility>
#include <chrono>
#include <cmath>
#include <limits>
#include <curl/curl.h>

namespace core::llm::transport {

[[nodiscard]] inline std::optional<int> parse_status_line(std::string_view line) {
    if (!line.starts_with("HTTP/")) return std::nullopt;
    const auto space = line.find(' ');
    if (space == std::string_view::npos) return std::nullopt;
    line = core::utils::str::trim_ascii_view(line.substr(space));
    int status = 0;
    const auto parsed = std::from_chars(line.data(), line.data() + line.size(), status);
    if (parsed.ec != std::errc{} || status < 100 || status > 599) return std::nullopt;
    return status;
}

[[nodiscard]] inline std::optional<std::pair<std::string, std::string>>
parse_header_line(std::string_view line) {
    line = core::utils::str::trim_ascii_view(line);
    if (line.empty()) return std::nullopt;

    const auto colon = line.find(':');
    if (colon == std::string_view::npos) return std::nullopt;

    auto name = core::utils::str::trim_ascii_copy(line.substr(0, colon));
    auto value = core::utils::str::trim_ascii_copy(line.substr(colon + 1));
    if (name.empty()) return std::nullopt;

    return std::pair{std::move(name), std::move(value)};
}

[[nodiscard]] inline std::optional<std::string> find_header(
    const cpr::Header& headers,
    std::string_view name) {
    for (const auto& [key, value] : headers) {
        if (core::utils::ascii::iequals(key, name)) return value;
    }
    return std::nullopt;
}

// Round fractional server delays up, and honor both delta-seconds and the
// HTTP-date form. A server deadline is a lower bound on the next attempt.
[[nodiscard]] inline int32_t retry_after_seconds(const cpr::Header& headers) {
    const auto seconds = find_header(headers, "retry-after");
    const auto millis = find_header(headers, "retry-after-ms");
    double delay = 0;
    for (const auto& [value, divisor] : {
             std::pair{seconds, 1.0}, std::pair{millis, 1000.0}}) {
        if (!value) continue;
        try {
            std::size_t consumed = 0;
            const double parsed = std::stod(*value, &consumed);
            if (consumed == value->size() && std::isfinite(parsed)) {
                delay = std::max(delay, parsed / divisor);
                continue;
            }
        } catch (...) {}
        if (divisor == 1.0) {
            const auto date = curl_getdate(value->c_str(), nullptr);
            if (date >= 0) {
                const double now = std::chrono::duration<double>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                delay = std::max(delay, static_cast<double>(date) - now);
            }
        }
    }
    return static_cast<int32_t>(std::min(
        std::ceil(std::max(0.0, delay)),
        static_cast<double>(std::numeric_limits<int32_t>::max())));
}

[[nodiscard]] inline bool retry_vetoed(const cpr::Header& headers) {
    const auto value = find_header(headers, "x-should-retry");
    return value && core::utils::ascii::iequals(
        core::utils::str::trim_ascii_view(*value), "false");
}

} // namespace core::llm::transport
