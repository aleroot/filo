#pragma once

#include "../LandrunDriver.hpp"

namespace core::landrun {

class LinuxLandlockDriver final : public ILandrunDriver {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] LandrunProbe probe() const override;
    [[nodiscard]] LandrunResult apply(const LandrunPolicy& policy) const override;
};

} // namespace core::landrun
