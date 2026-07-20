#pragma once

namespace core::landrun {

inline constexpr int kLandrunHelperFailure = 125;

[[nodiscard]] bool is_landrun_helper_invocation(int argc, char* const argv[]) noexcept;
[[nodiscard]] int run_landrun_helper(int argc, char* const argv[]);

} // namespace core::landrun
