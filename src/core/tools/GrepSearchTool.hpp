#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Tool that searches file contents with an ECMAScript regex, recursively.
 *
 * Exposed to MCP clients as @c grep_search.
 *
 * ### Behaviour
 * - Pure C++ (@c std::regex) — no external tools required, works everywhere.
 * - Skips binary files (heuristic: null byte in first 512 B).
 * - Skips common noise directories: @c .git, @c node_modules, @c build, etc.
 * - Optional @c include_pattern accepts glob filters by filename or relative path.
 * - Plain directory values in @c include_pattern include files beneath that directory.
 * - Returns up to 100 matches across all files.
 *
 * ### Result JSON
 * @code{.json}
 * // Success
 * {"matches": [{"path": "src/foo.cpp", "line": 42, "text": "  auto result = foo();"}]}
 *
 * // Failure
 * {"error": "<message>"}
 * @endcode
 *
 * @note @c readOnlyHint and @c idempotentHint are both @c true.
 */
class GrepSearchTool : public Tool {
public:
    ToolDefinition get_definition() const override;
    std::string execute(const std::string& json_args, const core::context::SessionContext& context) override;
};

} // namespace core::tools
