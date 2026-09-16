#pragma once

#include "LandrunPolicy.hpp"

#include <string>
#include <string_view>

namespace core::landrun {

struct LandrunProbe {
    bool available{false};
    std::string backend;
    std::string detail;
};

struct LandrunResult {
    bool success{false};
    std::string detail;
};

/** Applies an irreversible policy in the fresh landrun helper process. */
class ILandrunDriver {
public:
    virtual ~ILandrunDriver() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual LandrunProbe probe() const = 0;

    /**
     * Whether this backend can subtract `protected_*_paths` from a granted
     * root. Allow-list-only mechanisms (Landlock) cannot and must fail closed
     * in apply(); callers use this to decide whether a subtractive policy is
     * even worth composing on this platform.
     */
    [[nodiscard]] virtual bool supports_protected_paths() const noexcept = 0;

    [[nodiscard]] virtual LandrunResult apply(const LandrunPolicy& policy) const = 0;
};

} // namespace core::landrun
