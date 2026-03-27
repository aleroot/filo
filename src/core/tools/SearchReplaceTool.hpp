#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Tool that applies one or more SEARCH/REPLACE blocks to a file.
 *
 * Exposed to MCP clients as @c search_replace.
 *
 * ### Block format
 * The same format used by Claude Code, Cursor, and other AI editors:
 * @code
 * <<<<<<< SEARCH
 * [exact text to find]
 * =======
 * [replacement text]
 * >>>>>>> REPLACE
 * @endcode
 * Multiple blocks may be concatenated in a single @c content argument.
 * Blocks are applied sequentially in order.
 *
 * ### Behaviour
 * - Replacement is always exact (byte-for-byte).  Fuzzy matching (LCS-based,
 *   difflib-compatible similarity ratio) is used @em only to generate helpful
 *   error diagnostics when a search text is not found.
 * - Warns (but does not fail) if a search text appears more than once; only
 *   the first occurrence is replaced in that case.
 * - The file is only written if at least one block was successfully applied.
 *
 * ### Result JSON
 * @code{.json}
 * // Success
 * {
 *   "success": true,
 *   "blocks_applied": 2,
 *   "lines_changed": -1,
 *   "warnings": ["Block 2: search text appeared twice — first occurrence replaced."]
 * }
 *
 * // Failure (one or more blocks not found)
 * {"error": "Block 1: Search text not found in src/foo.cpp.\nClosest match (91.3% similarity) at lines 42–44:\n…"}
 * @endcode
 *
 * @note @c destructiveHint is @c true — modifies file content on disk.
 */
class SearchReplaceTool : public Tool {
public:
    ToolDefinition get_definition() const override;
    std::string    execute(const std::string& json_args) override;
};

} // namespace core::tools
