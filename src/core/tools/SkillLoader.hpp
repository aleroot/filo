#pragma once

#include "SkillManifest.hpp"
#include "ToolManager.hpp"
#include <filesystem>
#include <optional>
#include <vector>

namespace core::tools {

/**
 * @brief Discovers and registers user-defined **Tool** skills from the filesystem.
 *
 * SkillLoader handles @c SkillType::Tool skills only — Python scripts that the
 * LLM can call as tools.  @c SkillType::Prompt skills (markdown templates
 * invoked as slash commands) are handled by @ref core::commands::SkillCommandLoader.
 *
 * Both loaders share the same layered directory layout and call
 * @ref parse_manifest(), which is the single source of truth for SKILL.md parsing.
 *
 * ## Directory layout
 *
 * Two locations are scanned in order.  Because @c ToolManager::register_tool
 * silently overwrites on name collision, project-local skills take precedence:
 *
 *  1. @c ~/.config/filo/skills/  — global, available in every project
 *  2. @c ./.filo/skills/         — project-local, overrides globals of the same name
 *
 * Each immediate subdirectory of a skills root is one skill candidate.
 * A valid Tool-skill candidate contains:
 *  - A @c SKILL.md file with YAML frontmatter (see @ref SkillManifest)
 *  - A Python script whose filename matches the @c entry_point field
 *
 * Errors (bad frontmatter, missing script, Python load failure) are logged and
 * skipped; the remaining skills still load.
 *
 * ## Skill types
 *
 * @c parse_manifest() auto-detects the skill type:
 * - @c entry_point present → @c SkillType::Tool  (handled here)
 * - @c entry_point absent  → @c SkillType::Prompt (handled by SkillCommandLoader)
 * - @c type: prompt in frontmatter → always @c SkillType::Prompt
 *
 * ## Python entry-point contract (Tool skills)
 *
 * The script named by @c entry_point must export two callables:
 *
 * @code{.py}
 * import json
 *
 * def get_schema() -> dict:
 *     """Return a dict matching PythonTool's expected schema format."""
 *     return {
 *         "name": "weather_lookup",        # must match SKILL.md `name`
 *         "description": "Fetches weather for a given city.",
 *         "parameters": [
 *             {
 *                 "name": "city",
 *                 "type": "string",
 *                 "description": "City name to query.",
 *                 "required": True,
 *             }
 *         ],
 *     }
 *
 * def execute(json_args: str) -> str:
 *     """Execute the tool; json_args is a JSON object string."""
 *     args = json.loads(json_args)
 *     city = args["city"]
 *     # … do work …
 *     return json.dumps({"output": f"Weather in {city}: sunny"})
 * @endcode
 *
 * On failure @c execute() should return @c {"error": "…"} (the MCP error contract).
 *
 * @note The Python module name is inferred from the @c entry_point stem
 *       (i.e. @c weather.py → module @c weather).  Two skills sharing a stem
 *       name will collide in Python's @c sys.modules; use unique filenames.
 *
 * ## Typical usage
 *
 * Call once after built-in tool registration and before the agent starts:
 *
 * @code{.cpp}
 * // In MainApp.cpp, after registering built-in tools:
 * #ifdef FILO_ENABLE_PYTHON
 *     core::tools::SkillLoader::discover_and_register(tool_manager);
 * #endif
 * @endcode
 *
 * @note This class is compiled only when @c FILO_ENABLE_PYTHON is defined.
 *
 * @see SkillManifest, PythonTool, core::commands::SkillCommandLoader
 */
class SkillLoader {
public:
    /**
     * @brief Scans all standard skill directories and registers valid skills.
     *
     * Iterates @ref default_search_paths() and calls @ref load_from_directory()
     * for each path.  Non-fatal errors are logged at @c warn / @c error level
     * and do not abort the scan.
     *
     * @param tool_manager  The @c ToolManager singleton to register skills into.
     * @return              Total number of skills successfully registered.
     */
    static int discover_and_register(ToolManager& tool_manager);

    /**
     * @brief Returns the ordered list of skill directories to scan.
     *
     * Order:
     *  1. @c $HOME/.config/filo/skills/  (global)
     *  2. @c $PWD/.filo/skills/          (project-local)
     *
     * Directories that do not exist are silently skipped by
     * @ref load_from_directory().
     *
     * @return Vector of candidate paths (may not exist on disk).
     */
    static std::vector<std::filesystem::path> default_search_paths();

    /**
     * @brief Parses the @c SKILL.md inside @p skill_dir and returns a manifest.
     *
     * Reads the YAML frontmatter block (between the leading @c --- delimiters)
     * and extracts @c name, @c description, @c entry_point, and @c enabled.
     * Returns @c std::nullopt when:
     *  - @c SKILL.md does not exist or cannot be opened
     *  - No opening @c --- delimiter is found
     *  - Any required field (@c name, @c description, @c entry_point) is absent
     *
     * @param skill_dir  Absolute or relative path to the skill's directory.
     * @return           Populated @c SkillManifest on success, @c std::nullopt otherwise.
     */
    static std::optional<SkillManifest>
    parse_manifest(const std::filesystem::path& skill_dir);

    /**
     * @brief Scans a single skills root directory and registers valid skills.
     *
     * Iterates every immediate subdirectory of @p root, parses its manifest,
     * constructs a @c PythonTool from the entry point, and registers it with
     * @p tool_manager.  Non-existent or non-directory @p root paths return 0.
     *
     * @param root          Base skills directory to scan (e.g. @c ~/.config/filo/skills/).
     * @param tool_manager  Registry to receive the loaded skills.
     * @return              Number of skills successfully registered from @p root.
     */
    static int load_from_directory(const std::filesystem::path& root,
                                   ToolManager& tool_manager);
};

} // namespace core::tools
