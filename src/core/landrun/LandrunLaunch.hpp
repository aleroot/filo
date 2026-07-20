#pragma once

#include "LandrunPolicy.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace core::landrun {

struct LandrunLaunch {
    std::string executable;
    std::vector<std::string> arguments;
};

/** Builds exact argv arrays; no shell parsing or policy text crosses the boundary. */
[[nodiscard]] LandrunLaunch prepare_landrun_launch(
    const LandrunPolicy& policy,
    const std::filesystem::path& target,
    std::vector<std::string> target_arguments);

[[nodiscard]] LandrunLaunch prepare_shell_launch(const LandrunPolicy& policy);

} // namespace core::landrun
