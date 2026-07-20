#pragma once

#include "LandrunDriver.hpp"

#include <memory>

namespace core::landrun {

[[nodiscard]] std::unique_ptr<ILandrunDriver> make_landrun_driver();

} // namespace core::landrun
