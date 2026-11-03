#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace core::agent {

/**
 * @brief A subagent defined in a Markdown file: YAML frontmatter, then the
 * persona prompt as the body.
 *
 * Definitions use Filo's own vocabulary — the same settings a
 * `subagents.<name>` config entry has:
 *
 * | Key                       | Meaning                                                    |
 * |---------------------------|------------------------------------------------------------|
 * | `name`, `description`     | Required. The name is the `subagent_type` to request       |
 * | `tools`                   | Allow list of Filo tool names; `task` grants delegation    |
 * | `read-only`               | Only tools whose annotations declare them read-only        |
 * | `isolated`                | Run without project instructions, skill catalog, memories  |
 * | `max-steps`               | Default model-step budget                                  |
 * | `provider`, `model`       | Where the worker runs (default: the parent's)              |
 * | `enabled`                 | `false` skips the file                                     |
 *
 * Other keys (a host's own metadata such as `license` or `color`) are ignored.
 *
 * Where a file comes from decides how much of it is honoured:
 *
 *  - The user's own `~/.config/filo/agents/` and the primary workspace's
 *    `.filo/agents/`: every key.
 *  - An additional workspace root's `.filo/agents/`: granting a root with
 *    `-w` grants reading it, not running what it defines with more than that
 *    (the same rule as @ref core::tools::SkillSearchRoot::permits_tool_skills),
 *    so it is held to the plugin rules below, without the bundle note.
 *  - A plugin bundle's `agents/` directory, the portable layout that hosts
 *    load as named subagents. A bundle is third-party content written for many
 *    hosts, so it contributes its name, description, persona and step budget,
 *    and may narrow its tools with `tools`. It cannot grant itself anything:
 *    the worker is read-only, isolated, cannot delegate and runs on the
 *    parent's provider and model. The user lifts any of that for one persona
 *    in `subagents.<name>`, which always wins over the file.
 */
struct AgentDefinition {
    std::string name;
    std::string description;
    /// The persona, followed by where it lives when it ships in a plugin.
    std::string prompt;
    /// Filo tool names, `task` excluded; nullopt = no allow list.
    std::optional<std::vector<std::string>> tools;
    bool allow_task_tool = false;
    bool read_only = false;
    bool isolated = false;
    std::optional<int> max_steps;
    std::string provider;
    std::string model;
    std::filesystem::path source;
    /// The bundle the file ships in; empty for the user's own definitions.
    std::filesystem::path plugin_root;
};

struct AgentDirectory {
    std::filesystem::path path;
    std::filesystem::path plugin_root;  ///< Empty for the user's own directories.
    /// False for an additional workspace root, whose personas are restricted
    /// like a plugin's.
    bool trusted = true;
};

class AgentDefinitionRegistry {
public:
    /// Parses one definition; nullopt (with a logged reason) when it is
    /// malformed or disabled. @p plugin_root marks a plugin's persona;
    /// @p trusted false restricts it the same way without being a plugin.
    [[nodiscard]] static std::optional<AgentDefinition>
    parse_file(const std::filesystem::path& file,
               const std::filesystem::path& plugin_root = {},
               bool trusted = true);

    /// Agent directories, lowest precedence first: each plugin's `agents/`
    /// (see @ref core::tools::SkillRegistry::plugin_roots), then the user's
    /// `agents/` directories beside Filo's own skill directories.
    [[nodiscard]] static std::vector<AgentDirectory>
    search_dirs(const std::vector<std::filesystem::path>& workspace_roots);

    /// Every definition; a later directory replaces an earlier one's name.
    [[nodiscard]] static std::vector<AgentDefinition>
    discover(const std::vector<std::filesystem::path>& workspace_roots);
};

} // namespace core::agent
