#pragma once

#include <cstdint>

namespace core::llm {

enum class ReasoningCapability : std::uint16_t {
    Effort = 1U << 0,
    MaxEffort = 1U << 1,
    XHighEffort = 1U << 2,
    Disable = 1U << 3,
    Required = 1U << 4,
    FixedMax = 1U << 5,
};

class ReasoningCapabilities {
public:
    constexpr ReasoningCapabilities() noexcept = default;
    constexpr explicit ReasoningCapabilities(ReasoningCapability capability) noexcept
        : features_(static_cast<Storage>(capability)) {}

    [[nodiscard]] constexpr bool supports(ReasoningCapability capability) const noexcept {
        return (features_ & static_cast<Storage>(capability)) != 0;
    }

    [[nodiscard]] constexpr bool supports_effort() const noexcept {
        return supports(ReasoningCapability::Effort);
    }

private:
    using Storage = std::uint16_t;

    constexpr explicit ReasoningCapabilities(Storage features) noexcept
        : features_(features) {}

    Storage features_ = 0;

    friend constexpr ReasoningCapabilities operator|(
        ReasoningCapability lhs,
        ReasoningCapability rhs) noexcept;
    friend constexpr ReasoningCapabilities operator|(
        ReasoningCapabilities lhs,
        ReasoningCapability rhs) noexcept;
};

[[nodiscard]] constexpr ReasoningCapabilities operator|(
    ReasoningCapability lhs,
    ReasoningCapability rhs) noexcept {
    return ReasoningCapabilities{
        static_cast<ReasoningCapabilities::Storage>(
            static_cast<ReasoningCapabilities::Storage>(lhs)
            | static_cast<ReasoningCapabilities::Storage>(rhs))};
}

[[nodiscard]] constexpr ReasoningCapabilities operator|(
    ReasoningCapabilities lhs,
    ReasoningCapability rhs) noexcept {
    return ReasoningCapabilities{
        static_cast<ReasoningCapabilities::Storage>(
            lhs.features_ | static_cast<ReasoningCapabilities::Storage>(rhs))};
}

} // namespace core::llm
