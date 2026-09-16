#pragma once

#include "../LandrunDriver.hpp"

namespace core::landrun {

class LinuxLandlockDriver final : public ILandrunDriver {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] LandrunProbe probe() const override;
    /// Landlock rules only grant; there is no way to carve a file out of an
    /// allowed hierarchy, so a subtractive policy is refused in apply().
    [[nodiscard]] bool supports_protected_paths() const noexcept override { return false; }
    [[nodiscard]] LandrunResult apply(const LandrunPolicy& policy) const override;
};

} // namespace core::landrun
