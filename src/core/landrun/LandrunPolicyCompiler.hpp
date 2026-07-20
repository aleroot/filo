#pragma once

#include "LandrunPolicy.hpp"
#include "../workspace/SessionWorkspace.hpp"

namespace core::landrun {

class LandrunPolicyCompiler {
public:
    [[nodiscard]] static LandrunPolicy compile(
        const core::workspace::SessionWorkspace& workspace,
        LandrunMode mode);
};

} // namespace core::landrun
