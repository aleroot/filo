#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Tool that moves or renames a file or directory.
 *
 * Exposed to MCP clients as @c move_file.
 *
 * ### Behaviour
 * - Destination parent directories are created automatically.
 * - Uses @c std::filesystem::rename first; falls back to copy + remove when
 *   source and destination are on different filesystems.
 *
 * ### Result JSON
 * @code{.json}
 * // Success
 * {"success": true, "from": "/old/path", "to": "/new/path"}
 *
 * // Failure
 * {"error": "<message>"}
 * @endcode
 *
 * @note @c destructiveHint is @c true — the source path is removed.
 */
class MoveFileTool : public Tool {
public:
    ToolDefinition get_definition() const override;
    std::string execute(const std::string& json_args, const core::context::SessionContext& context) override;
};

} // namespace core::tools
