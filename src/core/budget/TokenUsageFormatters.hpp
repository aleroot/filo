#pragma once

#include "../llm/Models.hpp"

#include <array>
#include <concepts>
#include <cmath>
#include <cstdint>
#include <format>
#include <string>
#include <string_view>

namespace core::budget::formatters {

template <typename Formatter>
concept TokenCountFormatter = requires(const Formatter& formatter, std::int64_t value) {
    { formatter.format(value) } -> std::convertible_to<std::string>;
};

template <typename Formatter>
concept UsageCostFormatting = requires(const Formatter& formatter, double usd) {
    { formatter.format(usd) } -> std::convertible_to<std::string>;
    { formatter.format_fixed_4(usd) } -> std::convertible_to<std::string>;
};

class CompactTokenCountFormatter final {
public:
    [[nodiscard]] std::string format(std::int64_t value) const {
        if (value > -1000 && value < 1000) {
            return std::to_string(value);
        }

        const bool negative = value < 0;
        const long double magnitude = negative
            ? -static_cast<long double>(value)
            : static_cast<long double>(value);

        std::size_t unit = 0;
        while (unit + 1 < kUnits_.size() && magnitude >= kUnits_[unit + 1].divisor) {
            ++unit;
        }

        long double rounded = 0.0L;
        while (true) {
            const long double scaled = magnitude / kUnits_[unit].divisor;
            rounded = std::round(scaled * 10.0L) / 10.0L;
            if (rounded < 1000.0L || unit + 1 >= kUnits_.size()) {
                break;
            }
            ++unit;
        }

        std::string number = std::format("{:.1f}", static_cast<double>(rounded));
        if (number.ends_with(".0")) {
            number.resize(number.size() - 2);
        }

        return std::format("{}{}{}", negative ? "-" : "", number, kUnits_[unit].suffix);
    }

private:
    struct Unit {
        long double divisor;
        std::string_view suffix;
    };

    inline static constexpr std::array<Unit, 5> kUnits_ = {{
        {1.0L, ""},
        {1'000.0L, "k"},
        {1'000'000.0L, "M"},
        {1'000'000'000.0L, "B"},
        {1'000'000'000'000.0L, "T"},
    }};
};

class UsageCostFormatter final {
public:
    [[nodiscard]] std::string format(double usd) const {
        if (usd < 0.0001) return "$0";
        if (usd < 0.01) return std::format("${:.4f}", usd);
        if (usd < 1.0) return std::format("${:.3f}", usd);
        return std::format("${:.2f}", usd);
    }

    [[nodiscard]] std::string format_fixed_4(double usd) const {
        return std::format("${:.4f}", usd);
    }
};

static_assert(TokenCountFormatter<CompactTokenCountFormatter>);
static_assert(UsageCostFormatting<UsageCostFormatter>);

template <TokenCountFormatter TokenFormatter = CompactTokenCountFormatter,
          UsageCostFormatting CostFormatter = UsageCostFormatter>
class TokenUsageStatusFormatter final {
public:
    explicit TokenUsageStatusFormatter(
        TokenFormatter token_formatter = TokenFormatter{},
        CostFormatter cost_formatter = CostFormatter{})
        : token_formatter_(token_formatter),
          cost_formatter_(cost_formatter) {}

    [[nodiscard]] std::string format(const core::llm::TokenUsage& usage,
                                     double cost_usd = 0.0) const {
        if (!usage.has_data()) return {};
        if (cost_usd > 0.0) {
            return std::format("↑{} ↓{}  {}",
                token_formatter_.format(usage.prompt_tokens),
                token_formatter_.format(usage.completion_tokens),
                cost_formatter_.format_fixed_4(cost_usd));
        }
        return std::format("↑{} ↓{}",
            token_formatter_.format(usage.prompt_tokens),
            token_formatter_.format(usage.completion_tokens));
    }

private:
    TokenFormatter token_formatter_;
    CostFormatter cost_formatter_;
};

} // namespace core::budget::formatters
