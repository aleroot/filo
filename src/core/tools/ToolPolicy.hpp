#pragma once

#include "../context/SessionContext.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::tools::policy {

[[nodiscard]] std::string canonical_tool_name(std::string_view name);

[[nodiscard]] bool is_tool_allowed(std::string_view tool_name,
                                   const std::vector<std::string>& allowed_tools);

[[nodiscard]] std::optional<std::string> enforce_path_policy(
    std::string_view tool_name,
    const std::filesystem::path& resolved_path,
    const core::context::SessionContext& context);

[[nodiscard]] std::optional<std::string> enforce_command_policy(
    std::string_view tool_name,
    std::string_view command);

[[nodiscard]] std::optional<std::string> enforce_url_policy(
    std::string_view tool_name,
    std::string_view text);

} // namespace core::tools::policy
