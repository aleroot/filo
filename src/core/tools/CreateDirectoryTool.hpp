#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Tool that creates a directory (and any missing parents).
 *
 * Exposed to MCP clients as @c create_directory.
 *
 * ### Behaviour
 * - Calls @c std::filesystem::create_directories, so intermediate parent
 *   directories are created as needed.
 * - Silently succeeds if the directory already exists.
 *
 * ### Result JSON
 * @code{.json}
 * // Success
 * {"success": true, "path": "/abs/path/to/dir"}
 *
 * // Failure
 * {"error": "<message>"}
 * @endcode
 *
 * @note @c idempotentHint is @c true — calling this multiple times with the
 *       same path is safe and has no additional effect.
 */
class CreateDirectoryTool : public Tool {
public:
    ToolDefinition get_definition() const override;
    std::string execute(const std::string& json_args, const core::context::SessionContext& context) override;
};

} // namespace core::tools
