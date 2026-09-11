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
 * Roots come from @ref core::tools::SkillRegistry::default_search_roots(), which
 * covers **every** workspace root, not just the primary. For a single-root
 * workspace the layering is:
 *
 *  1. @c ~/.claude/skills/             — Claude-compatible global fallback
 *  2. @c ~/.agents/skills/             — Agent Skills-compatible global fallback
 *  3. @c <primary>/.claude/skills/     — Claude-compatible project fallback
 *  4. @c <primary>/.agents/skills/     — Agent Skills-compatible project fallback
 *  5. @c ~/.config/filo/skills/        — Filo-native global skills
 *  6. @c <primary>/.filo/skills/       — Filo-native project skills
 *
 * Additional workspace roots (granted with @c -w or @c /workspace add) are scanned
 * *before* all of the above, so the primary workspace and the user's own
 * directories always win a name collision against a secondary workspace. Prompt
 * skills are plain markdown — no code runs — so unlike Python Tool skills they are
 * accepted from every granted root.
 *
 * @c CommandExecutor::register_command() replaces any command already registered
 * under the same name or alias, so the last registration wins outright: scanning
 * in precedence order is all that is needed for the primary to override.
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
 * core::commands::SkillCommandLoader::discover_and_register(
 *     cmd_executor, core::workspace::Workspace::get_instance().ordered_roots());
 * const auto command_index = cmd_executor.describe_commands(); // includes skills
 * @endcode
 *
 * @note Registration is a **startup** step: `/workspace change` re-points the
 *       workspace but does not re-run this loader, so commands from the previous
 *       workspace stay registered until restart. Fixing that needs a way to
 *       unregister a command — `register_command()` can replace one by name, but
 *       nothing removes names that no longer exist. Instruction-skill activation
 *       and the `[Agent Skills]` catalog have no such gap: they resolve the live
 *       session workspace on every turn.
 *
 * @see core::tools::SkillLoader, core::commands::SkillCommand, core::tools::SkillManifest
 */
class SkillCommandLoader {
public:
    /**
     * @brief Scans the workspace's skill directories and registers valid Prompt skills.
     *
     * Reuses @ref core::tools::SkillRegistry::default_search_paths() for path
     * discovery, so this loader and the skill catalog can never disagree about
     * which roots exist or which one wins a collision. Non-fatal errors are logged
     * and do not abort the scan.
     *
     * @param executor       The @c CommandExecutor to register skill commands into.
     * @param workspace_roots Roots from @ref core::workspace::ordered_roots():
     *                       index 0 is the primary, the rest are additional.
     * @return               Total number of Prompt skills successfully registered.
     */
    static int discover_and_register(
        CommandExecutor& executor,
        const std::vector<std::filesystem::path>& workspace_roots);

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
