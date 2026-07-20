#pragma once

#include "LandrunDriver.hpp"

namespace core::landrun {

/** Exercises serialization, helper startup, policy enforcement, and exec in a disposable child. */
[[nodiscard]] LandrunResult verify_landrun_readiness(
    const LandrunPolicy& policy);

} // namespace core::landrun
