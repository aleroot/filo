#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Tool that replaces the first (and only) occurrence of an exact string in a file.
 *
 * Exposed to MCP clients as @c replace.
 *
 * ### Behaviour
 * - Searches for @c old_string using a plain @c std::string::find (no regex).
 * - If @c old_string appears more than once the call fails with an error
 *   instructing the caller to include more context to disambiguate.
 * - Returns the 1-based line number of the replacement so Lampo can
 *   scroll the editor to the right location.
 *
 * ### Result JSON
 * @code{.json}
 * // Success
 * {
 *   "success": true,
 *   "file_path": "/abs/path/to/file.cpp",
 *   "replaced_at_line": 42
 * }
 *
 * // Failure
 * {"error": "<message>"}
 * @endcode
 *
 * @note @c destructiveHint is @c true.  This tool is @em not idempotent —
 *       a second call with the same @c old_string will fail because the text
 *       was already replaced.
 */
class ReplaceTool : public Tool {
public:
    ToolDefinition get_definition() const override;
    std::string execute(const std::string& json_args, const core::context::SessionContext& context) override;
};

} // namespace core::tools
