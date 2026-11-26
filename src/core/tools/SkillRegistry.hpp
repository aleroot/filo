#pragma once

#include "SkillManifest.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::workspace {
class Workspace;
}

namespace core::tools {

enum class SkillScope {
    User,
    Project,
};

/// Which workspace a Project-scope skills root belongs to.
enum class SkillWorkspaceOrigin {
    Primary,     ///< The workspace Filo is working in (`workspace_roots[0]`).
    Additional,  ///< A root granted with `-w` or `/workspace add`.
};

struct SkillSearchRoot {
    /// Skills directory to scan, e.g. `<root>/.filo/skills`.
    std::filesystem::path path;
    SkillScope scope = SkillScope::Project;
    std::string label;
    SkillWorkspaceOrigin origin = SkillWorkspaceOrigin::Primary;
    /// Workspace root @ref path lives under; empty for user-scope roots.
    std::filesystem::path workspace_root;
    /// Set when @ref path is `<plugin>/skills` of an installed plugin bundle
    /// (see @ref SkillRegistry::plugin_install_dirs): the canonical plugin root.
    std::filesystem::path plugin_root = {};
    /// True for Filo's own `skills` directories (`~/.config/filo/skills`,
    /// `<root>/.filo/skills`), whose sibling `agents` directory holds agent
    /// definitions.
    bool filo_native = false;

    /**
     * True when executable (Python) skills may be registered from this root.
     *
     * The user's own directories and the primary workspace are trusted with
     * code; an additional workspace root is not. Granting a root with `-w`
     * grants *read* access, and silently turning that into in-process code
     * execution would be a privilege the user never asked for. Instructions and
     * prompt templates are data, so those are accepted from every root.
     */
    [[nodiscard]] bool permits_tool_skills() const noexcept {
        return scope == SkillScope::User || origin == SkillWorkspaceOrigin::Primary;
    }
};

/// @ref default_search_roots() partitioned by @ref SkillSearchRoot::permits_tool_skills().
struct SkillRootTrustSplit {
    std::vector<SkillSearchRoot> trusted;     ///< May contribute executable skills.
    std::vector<SkillSearchRoot> restricted;  ///< Instructions and prompt skills only.
};

/**
 * @brief Discovery of Agent Skills across a whole workspace.
 *
 * Every entry point takes the workspace roots produced by
 * @ref core::workspace::ordered_roots() — index 0 is the primary, the rest are
 * additional roots in grant order. There is deliberately no ambient default:
 * skill roots used to fall back to `std::filesystem::current_path()`, which only
 * matched the primary because `main()` happens to chdir there, and which could
 * never see an additional root at all.
 */
class SkillRegistry {
public:
    /**
     * Layered skill roots for a workspace, in scan order.
     *
     * Scan order *is* precedence: a later root overwrites an earlier one on a
     * name collision (see @ref discover_all). Additional workspace roots come
     * first, so both the user's own directories and the primary workspace beat a
     * secondary workspace's skill of the same name — one rule to remember:
     * *the primary and the user win over any additional root*.
     *
     * For a single-root workspace the sequence is exactly the historical one:
     * `~/.claude`, `~/.agents`, `<primary>/.claude`, `<primary>/.agents`,
     * `~/.config/filo`, `<primary>/.filo` — with the user directories omitted
     * under landrun, as before.
     */
    [[nodiscard]] static std::vector<SkillSearchRoot>
    default_search_roots(const std::vector<std::filesystem::path>& workspace_roots);

    [[nodiscard]] static std::vector<std::filesystem::path>
    default_search_paths(const std::vector<std::filesystem::path>& workspace_roots);

    /**
     * The search roots partitioned by the executable-skill trust rule.
     *
     * The rule itself is @ref SkillSearchRoot::permits_tool_skills(); this split
     * exists so it is applied in exactly one place, and so it stays testable in
     * builds without embedded Python, where @ref core::tools::SkillLoader — the
     * only consumer that registers code — compiles to a stub.
     */
    [[nodiscard]] static SkillRootTrustSplit
    split_tool_skill_roots(const std::vector<std::filesystem::path>& workspace_roots);

    [[nodiscard]] static std::optional<SkillManifest>
    parse_manifest(const std::filesystem::path& skill_dir);

    /// @ref parse_manifest plus the provenance of the root it was found under
    /// (workspace, additional-workspace flag, plugin root).
    [[nodiscard]] static std::optional<SkillManifest>
    parse_manifest_in(const SkillSearchRoot& root, const std::filesystem::path& skill_dir);

    /// True when @p dir carries a plugin manifest: `plugin.json`,
    /// `.claude-plugin/plugin.json` or `.codex-plugin/plugin.json`.
    [[nodiscard]] static bool is_plugin_root(const std::filesystem::path& dir);

    /// The plugin a skill directory belongs to: `<plugin>/skills/<skill>` where
    /// `<plugin>` passes @ref is_plugin_root. Symlinks are resolved first, so a
    /// skill linked into `.agents/skills` from a plugin checkout still finds
    /// its bundle.
    [[nodiscard]] static std::optional<std::filesystem::path>
    detect_plugin_root(const std::filesystem::path& skill_dir);

    /// Directories whose subdirectories are installed plugin bundles, in scan
    /// order: additional workspaces' `.filo/plugins`, `~/.config/filo/plugins`
    /// (omitted under landrun), then `<primary>/.filo/plugins`.
    [[nodiscard]] static std::vector<SkillSearchRoot>
    plugin_install_dirs(const std::vector<std::filesystem::path>& workspace_roots);

    /// Every plugin root the workspace can see — installed bundles and the
    /// bundles of discovered skills — canonical and de-duplicated.
    [[nodiscard]] static std::vector<std::filesystem::path>
    plugin_roots(const std::vector<std::filesystem::path>& workspace_roots);

    /// Directories the session may read as skill content: every discovered
    /// skill directory and plugin root plus the plugin install directories.
    /// Empty under landrun, whose sandbox does not grant them to the shell.
    [[nodiscard]] static std::vector<std::filesystem::path>
    package_read_roots(const std::vector<std::filesystem::path>& workspace_roots);

    /// Makes @ref package_read_roots of @p workspace's roots its read-only
    /// extension roots; call again whenever those roots change.
    static void grant_package_read_roots(core::workspace::Workspace& workspace);

    /// Where an invoked skill lives, for the prompt: its directory, its plugin
    /// root when it has one, and how relative paths resolve.
    [[nodiscard]] static std::string invocation_context(const SkillManifest& skill);

    /// @p prompt (a Prompt skill's body, arguments expanded) followed by
    /// @ref invocation_context; unchanged when either part would be empty.
    [[nodiscard]] static std::string with_invocation_context(std::string prompt,
                                                             const SkillManifest& skill);

    /// Every enabled skill in the workspace, collisions resolved by scan order.
    [[nodiscard]] static std::vector<SkillManifest>
    discover_all(const std::vector<std::filesystem::path>& workspace_roots);

    /// Prompt-type skills the model may load, i.e. what `activate_skill` and
    /// the catalog expose. Skills with `disable-model-invocation: true` are
    /// left out: they stay available as slash commands only.
    [[nodiscard]] static std::vector<SkillManifest>
    discover_instruction_skills(const std::vector<std::filesystem::path>& workspace_roots);

    [[nodiscard]] static std::optional<SkillManifest>
    find_instruction_skill(
        std::string_view name,
        const std::vector<std::filesystem::path>& workspace_roots);

    [[nodiscard]] static std::string
    build_catalog_prompt(const std::vector<std::filesystem::path>& workspace_roots);

    [[nodiscard]] static std::vector<std::filesystem::path>
    list_relative_resources(const SkillManifest& skill, std::size_t max_entries = 200);
};

} // namespace core::tools
