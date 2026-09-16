#pragma once

#include "../LandrunDriver.hpp"

namespace core::landrun {

class MacOSSeatbeltDriver final : public ILandrunDriver {
public:
    [[nodiscard]] std::string_view name() const noexcept override;
    [[nodiscard]] LandrunProbe probe() const override;
    /// SBPL is last-match-wins, so `(deny file-read* (subpath ...))` after a
    /// broad allow subtracts exactly one hierarchy.
    [[nodiscard]] bool supports_protected_paths() const noexcept override { return true; }
    [[nodiscard]] LandrunResult apply(const LandrunPolicy& policy) const override;
};

} // namespace core::landrun
