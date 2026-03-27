#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Tool that reads a file's contents, with optional line-range slicing.
 *
 * Exposed to MCP clients as @c read_file.
 *
 * ### Behaviour
 * - With no optional arguments, reads the whole file (fast path: single
 *   @c rdbuf() copy).
 * - @c offset_line and @c limit_lines enable sliced reads without loading
 *   the entire file, which matters for large source files.
 * - Files larger than 1 MB (whole read) or 512 KB (sliced read) are
 *   automatically truncated and a @c [TRUNCATED] marker is appended.
 *
 * ### Result JSON
 * @code{.json}
 * // Success
 * {"content": "<file text>"}
 *
 * // Failure
 * {"error": "<message>"}
 * @endcode
 *
 * @note @c readOnlyHint and @c idempotentHint are both @c true — this tool
 *       never modifies any state.
 */
class ReadFileTool : public Tool {
public:
    ToolDefinition get_definition() const override;
    std::string    execute(const std::string& json_args) override;
};

} // namespace core::tools
