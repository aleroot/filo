#pragma once

#include "LandrunMode.hpp"

#include <string>

namespace core::landrun {

[[nodiscard]] std::string landrun_status_label(LandrunMode mode);
[[nodiscard]] std::string landrun_status_detail(LandrunMode mode);
[[nodiscard]] std::string landrun_status_label();
[[nodiscard]] std::string landrun_status_detail();

} // namespace core::landrun
