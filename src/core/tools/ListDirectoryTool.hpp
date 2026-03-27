#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Tool that lists the immediate children of a directory.
 *
 * Exposed to MCP clients as @c list_directory.
 *
 * ### Result JSON
 * @code{.json}
 * // Success
 * {"entries": [{"type": "file", "name": "main.cpp"}, {"type": "dir", "name": "include"}]}
 *
 * // Failure
 * {"error": "<message>"}
 * @endcode
 *
 * @note @c readOnlyHint and @c idempotentHint are both @c true.
 */
class ListDirectoryTool : public Tool {
public:
    ToolDefinition get_definition() const override;
    std::string    execute(const std::string& json_args) override;
};

} // namespace core::tools
