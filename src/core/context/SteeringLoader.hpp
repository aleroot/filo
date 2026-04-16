#pragma once

#include <filesystem>
#include <string>

namespace core::context {

[[nodiscard]] std::string load_project_steering_block(const std::filesystem::path& project_root);

} // namespace core::context
