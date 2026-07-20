#pragma once

#include "LandrunDriver.hpp"

namespace core::landrun {

/** Closes every inherited descriptor except stdin/stdout/stderr in the fresh helper. */
[[nodiscard]] LandrunResult sanitize_inherited_descriptors();

} // namespace core::landrun
