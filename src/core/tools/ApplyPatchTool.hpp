#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Applies a patch for every provider from the same tool.
 *
 * The patch text selects the format. An envelope that starts with
 * `*** Begin Patch` is applied in-process by matching context lines. Anything
 * else is a unified diff: hunk counts are checked first, then the system
 * `patch -p1 -f -i` applies it. No provider has its own patch path.
 * `working_dir` selects the directory and defaults to the session workspace.
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
