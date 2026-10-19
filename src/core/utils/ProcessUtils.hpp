#pragma once

#include <cerrno>
#include <signal.h>

namespace core::utils::process {

// Permission errors and invalid identifiers cannot establish quiescence.
[[nodiscard]] inline bool group_may_exist(int group) noexcept {
    return group <= 1 || ::killpg(group, 0) == 0 || errno != ESRCH;
}

} // namespace core::utils::process
