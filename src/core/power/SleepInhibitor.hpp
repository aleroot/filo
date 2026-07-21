#pragma once

#include <memory>
#include <string_view>

namespace core::power {

/**
 * A scoped operating-system assertion that prevents idle system sleep.
 * Releasing the last owner releases the assertion.
 */
class SleepInhibitionLease {
public:
    virtual ~SleepInhibitionLease() = default;

    [[nodiscard]] virtual bool is_active() const noexcept = 0;
};

/** Platform-neutral source of scoped sleep-inhibition leases. */
class SleepInhibitor {
public:
    virtual ~SleepInhibitor() = default;

    [[nodiscard]] virtual std::shared_ptr<SleepInhibitionLease>
    inhibit(std::string_view reason) = 0;
};

/** Creates the native inhibitor for the current platform. */
[[nodiscard]] std::shared_ptr<SleepInhibitor> make_sleep_inhibitor();

} // namespace core::power
