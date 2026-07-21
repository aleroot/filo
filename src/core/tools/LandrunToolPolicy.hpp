#pragma once

#include "ToolNames.hpp"
#include "../landrun/LandrunMode.hpp"

#include <string_view>

namespace core::tools {

/**
 * Bridges product sandbox capabilities to in-process tool capabilities.
 *
 * Child-process tools remain available because Landrun confines the process
 * tree itself. Native workspace mutation tools require an explicit workspace
 * write capability because they execute inside Filo's process.
 */
[[nodiscard]] constexpr bool landrun_allows_tool(
    core::landrun::LandrunMode mode,
    std::string_view tool_name) noexcept
{
    return core::landrun::landrun_workspace_writable(mode)
        || !names::is_file_modification_tool(tool_name);
}

} // namespace core::tools
