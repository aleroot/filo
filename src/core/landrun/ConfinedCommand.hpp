#pragma once

#include "LandrunPolicy.hpp"

#include <chrono>
#include <functional>
#include <string>
#include <string_view>

namespace core::landrun {

struct ConfinedCommandResult {
    int exit_code = -1;
    bool timed_out = false;
    std::string error;
};

// Streams raw stdout/stderr bytes without truncation. The callback must retain
// only the data it needs. A confined, network-denying policy is mandatory;
// setup failures never fall back to executing on the host.
[[nodiscard]] ConfinedCommandResult run_confined_command(
    const LandrunPolicy& policy,
    std::string_view command,
    const std::function<void(std::string_view)>& consume,
    std::chrono::milliseconds timeout = std::chrono::seconds(30));

} // namespace core::landrun
