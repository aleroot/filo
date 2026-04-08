#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Tool that searches for files by glob pattern, recursively.
 *
 * Exposed to MCP clients as @c file_search.
 *
 * ### Behaviour
 * - Pure C++ — no external tools required, works everywhere.
 * - Supports @c * (any chars) and @c ? (single char) wildcards.
 * - Patterns with path separators match against relative paths from the search root.
 * - Plain directory patterns (for example @c modules/core) return files under that directory.
 * - Skips @c .git, @c node_modules, @c build, and similar directories.
 * - Returns up to 100 matching file paths.
 *
 * ### Result JSON
 * @code{.json}
 * // Success
 * {"files": ["src/core/mcp/McpDispatcher.cpp", "src/core/mcp/McpDispatcher.hpp"]}
 *
 * // Failure
 * {"error": "<message>"}
 * @endcode
 *
 * @note @c readOnlyHint and @c idempotentHint are both @c true.
 */
class FileSearchTool : public Tool {
public:
    ToolDefinition get_definition() const override;
    std::string execute(const std::string& json_args, const core::context::SessionContext& context) override;
};

} // namespace core::tools
