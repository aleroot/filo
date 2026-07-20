#pragma once

#include "../LandrunDriver.hpp"

namespace core::landrun {

[[nodiscard]] LandrunResult apply_linux_seccomp_hardening(bool deny_network);

} // namespace core::landrun
