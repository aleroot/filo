#pragma once

#include "LandrunPolicy.hpp"

#include <string>
#include <vector>

namespace core::landrun {

/** Builds the child environment. Secure mode is allowlist-only. */
[[nodiscard]] std::vector<std::string> build_landrun_environment(
    const LandrunPolicy& policy,
    char* const inherited_environment[]);

} // namespace core::landrun
