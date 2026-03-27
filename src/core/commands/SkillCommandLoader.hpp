#pragma once

#include "CommandExecutor.hpp"
#include <filesystem>
#include <vector>

namespace core::commands {

/**
 * @brief Discovers and registers user-defined **Prompt** skills as slash commands.
 *
 * SkillCommandLoader is the command-layer counterpart of @ref core::tools::SkillLoader.
 * It scans the same layered search paths but only processes @c SkillType::Prompt
 * manifests — SKILL.md files that have no @c entry_point field.
 *
 * ## Layered precedence (lowest → highest)
 *
 *  1. @c ~/.config/filo/skills/  — global, available in every project
 *  2. @c ./.filo/skills/         — project-local, overrides globals of the same name
 *
 * Because @c CommandExecutor::register_command() appends to the registry and
 * @c try_execute() iterates in registration order, the last registration wins on
 * name collisions.  Loading global skills before project-local skills therefore
 * gives project-level overrides higher precedence.
 *
 * ## Skill format
 *
 * A Prompt skill is a subdirectory containing a @c SKILL.md with no @c entry_point:
 *
 * @code{.yaml}
 * ---
 * name: summarise
 * description: Summarise the provided text concisely.
 * ---
 *
 * Please summarise the following in one paragraph:
 *
 * $ARGUMENTS
 * @endcode
 *
 * Invoking @c /summarise <text> sends the expanded body to the agent as a user turn.
 *
 * ## Typical usage
 *
 * Call once after @c CommandExecutor construction and before @c describe_commands():
 *
 * @code{.cpp}
 * core::commands::CommandExecutor cmd_executor;
 * core::commands::SkillCommandLoader::discover_and_register(cmd_executor);
 * const auto command_index = cmd_executor.describe_commands(); // includes skills
 * @endcode
 *
 * @see core::tools::SkillLoader, core::commands::SkillCommand, core::tools::SkillManifest
 */
class SkillCommandLoader {
public:
    /**
     * @brief Scans all standard skill directories and registers valid Prompt skills.
     *
     * Reuses @ref core::tools::SkillLoader::default_search_paths() for path discovery.
     * Non-fatal errors are logged and do not abort the scan.
     *
     * @param executor  The @c CommandExecutor to register skill commands into.
     * @return          Total number of Prompt skills successfully registered.
     */
    static int discover_and_register(CommandExecutor& executor);

    /**
     * @brief Scans a single skills root directory and registers valid Prompt skills.
     *
     * Iterates every immediate subdirectory of @p root, parses its @c SKILL.md, and
     * registers a @c SkillCommand for each enabled @c SkillType::Prompt manifest.
     * Non-existent or non-directory @p root paths return 0 without error.
     *
     * @param root      Base skills directory to scan (e.g. @c ~/.config/filo/skills/).
     * @param executor  Registry to receive the loaded skill commands.
     * @return          Number of Prompt skills successfully registered from @p root.
     */
    static int load_from_directory(const std::filesystem::path& root,
                                   CommandExecutor& executor);
};

} // namespace core::commands
