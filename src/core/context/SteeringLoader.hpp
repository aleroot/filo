#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace core::context {

struct SteeringFile {
    std::filesystem::path path;
    std::string label;
    std::string content;
};

struct SteeringLoadResult {
    std::string block;
    std::vector<std::string> source_labels;
    std::vector<SteeringFile> files;
};

[[nodiscard]] SteeringLoadResult load_project_steering_context(
    const std::filesystem::path& project_root);

[[nodiscard]] std::string load_project_steering_block(const std::filesystem::path& project_root);

} // namespace core::context
