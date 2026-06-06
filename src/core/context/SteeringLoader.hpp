#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace core::context {

struct SteeringLoadResult {
    std::string block;
    std::vector<std::string> source_labels;
};

[[nodiscard]] SteeringLoadResult load_project_steering_context(
    const std::filesystem::path& project_root);

[[nodiscard]] std::string load_project_steering_block(const std::filesystem::path& project_root);

} // namespace core::context
