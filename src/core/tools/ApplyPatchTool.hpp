#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Tool that applies a unified diff patch to the filesystem.
 *
 * Exposed to MCP clients as @c apply_patch.
 *
 * ### Behaviour
 * - Delegates to the system @c patch binary (looked up once at startup at
 *   @c /usr/bin/patch or @c /usr/local/bin/patch).
 * - The patch content is written to a collision-safe temp file in @c /tmp
 *   (name includes PID + monotonic counter), then @c patch -p1 is run.
 * - @c working_dir sets the directory in which @c patch is invoked
 *   (@c -d flag), defaulting to filo's CWD.
 * - stdout + stderr from @c patch are captured and returned in @c output,
 *   which lists the patched file paths on success (useful for Lampo's diff
 *   UI to know which files changed).
 *
 * ### Result JSON
 * @code{.json}
 * // Success
 * {"success": true, "output": "patching file src/foo.cpp\n"}
 *
 * // Failure
 * {"error": "Patch failed", "output": "<patch stderr>"}
 * @endcode
 *
 * @note @c destructiveHint is @c true — modifies files on disk.
 */
class ApplyPatchTool : public Tool {
public:
    ToolDefinition get_definition() const override;
    std::string execute(
        const std::string& json_args,
        const core::context::SessionContext& context) override;
};

} // namespace core::tools
