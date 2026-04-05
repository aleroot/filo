#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Tool that permanently deletes a file or empty directory.
 *
 * Exposed to MCP clients as @c delete_file.
 *
 * ### Result JSON
 * @code{.json}
 * // Success
 * {"success": true, "deleted": "/abs/path/to/file.txt"}
 *
 * // Failure
 * {"error": "<message>"}
 * @endcode
 *
 * @warning This action cannot be undone.  @c destructiveHint is @c true.
 */
class DeleteFileTool : public Tool {
public:
    ToolDefinition get_definition() const override;
    std::string execute(const std::string& json_args, const core::context::SessionContext& context) override;
};

} // namespace core::tools
