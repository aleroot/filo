#pragma once

#include <simdjson.h>

#include <algorithm>
#include <expected>
#include <optional>
#include <string>

namespace core::agent {

// Both delegation interfaces use the same per-run budget contract. Configured
// profile defaults are sanitized separately; malformed call arguments fail.
struct DelegationBudget {
    static constexpr int kMinSteps = 1;
    static constexpr int kMaxSteps = 64;

    [[nodiscard]] static int sanitize(int value, int fallback) noexcept {
        return value < kMinSteps ? fallback : std::min(value, kMaxSteps);
    }

    [[nodiscard]] static std::expected<std::optional<int>, std::string> parse(
        simdjson::dom::object object) {
        simdjson::dom::element value;
        if (object["max_steps"].get(value) != simdjson::SUCCESS || value.is_null()) {
            return std::optional<int>{};
        }
        int64_t steps = 0;
        if (value.get(steps) != simdjson::SUCCESS || steps < kMinSteps || steps > kMaxSteps) {
            return std::unexpected("max_steps must be an integer between 1 and 64.");
        }
        return std::optional<int>{static_cast<int>(steps)};
    }
};

} // namespace core::agent
