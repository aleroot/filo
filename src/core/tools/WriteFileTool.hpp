#pragma once

#include "Tool.hpp"

namespace core::tools {

/**
 * @brief Tool that writes (or creates) a file with the given content.
 *
 * Exposed to MCP clients as @c write_file.
 *
 * ### Behaviour
 * - Parent directories are created automatically (@c create_directories).
 * - The file is opened in binary+truncate mode, so content is written verbatim.
 * - The previous file content (up to 512 KB) is read @em before overwriting
 *   and returned in @c previous_content so clients such as Lampo can compute
 *   and display a diff without a separate round-trip.
 *
 * ### Result JSON
 * @code{.json}
 * // Success — new file
 * {
 *   "success": true,
 *   "file_path": "/abs/path/to/file.txt",
 *   "created": true,
 *   "bytes_written": 1234,
 *   "previous_content": ""
 * }
 *
 * // Success — overwrite
 * {
 *   "success": true,
 *   "file_path": "/abs/path/to/file.txt",
 *   "created": false,
 *   "bytes_written": 1234,
 *   "previous_content": "old content here…"
 * }
 *
 * // Failure
 * {"error": "<message>"}
 * @endcode
 *
 * @note @c destructiveHint is @c true — the tool overwrites existing files
 *       without confirmation.  @c idempotentHint is also @c true because
 *       writing the same bytes a second time produces the same observable state.
 */
class WriteFileTool : public Tool {
public:
    ToolDefinition get_definition() const override;
    std::string    execute(const std::string& json_args) override;
};

} // namespace core::tools
