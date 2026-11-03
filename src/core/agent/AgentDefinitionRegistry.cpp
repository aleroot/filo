#include "AgentDefinitionRegistry.hpp"

#include "../logging/Logger.hpp"
#include "../tools/Frontmatter.hpp"
#include "../tools/SkillRegistry.hpp"
#include "../tools/ToolNames.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/PathUtils.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <string_view>
#include <system_error>
#include <utility>

namespace fs = std::filesystem;

namespace core::agent {

namespace {

namespace frontmatter = core::tools::frontmatter;

/// Letters, digits, `-`, `_` and `.`, starting with a letter or digit: what
/// `subagent_type` and `subagents.<name>` can name unambiguously.
[[nodiscard]] bool is_valid_name(std::string_view name) {
    const auto alnum = [](char ch) {
        return core::utils::ascii::is_alnum(static_cast<unsigned char>(ch));
    };
    if (name.empty() || !alnum(name.front())) return false;
    return std::ranges::all_of(name, [&](char ch) {
        return alnum(ch) || ch == '-' || ch == '_' || ch == '.';
    });
}

} // namespace

std::optional<AgentDefinition>
AgentDefinitionRegistry::parse_file(const fs::path& file,
                                    const fs::path& plugin_root,
                                    bool trusted) {
    const auto document = frontmatter::parse_file(file);
    if (!document) return std::nullopt;
    const frontmatter::FieldReader reader(*document, file, "Agent definition");

    if (reader.flag({"enabled"}) == false) return std::nullopt;

    AgentDefinition definition;
    definition.source = core::utils::path::normalized(file);
    definition.plugin_root = plugin_root;
    // Matched lowercased, as the orchestrator normalises profile names.
    definition.name = core::utils::str::to_lower_ascii_copy(reader.text({"name"}).value_or(""));
    definition.description = reader.text({"description"}).value_or("");
    if (!is_valid_name(definition.name)) {
        reader.warn("'name' must be letters, digits, '-', '_' or '.'; skipped");
        return std::nullopt;
    }
    if (definition.description.empty()) {
        reader.warn("'description' is required; skipped");
        return std::nullopt;
    }
    if (document->body.empty()) {
        reader.warn("the persona body is empty; skipped");
        return std::nullopt;
    }

    if (auto tools = reader.list({"tools"})) {
        std::vector<std::string> names;
        for (auto& name : *tools) {
            if (name == core::tools::names::kTask) {
                definition.allow_task_tool = true;
            } else if (!std::ranges::contains(names, name)) {
                names.push_back(std::move(name));
            }
        }
        definition.tools = std::move(names);
    }
    definition.max_steps = reader.positive_integer({"max-steps", "max_steps"});

    const bool from_plugin = !plugin_root.empty();
    if (from_plugin || !trusted) {
        // Third-party content, or a root granted for reading only: the
        // persona may narrow what it gets, never widen it. Everything it
        // cannot decide is pinned here.
        if (definition.allow_task_tool) {
            reader.warn("a plugin or additional-workspace persona cannot grant itself 'task'; "
                        "enable it with subagents.<name>.allow_task_tool");
            definition.allow_task_tool = false;
        }
        if (reader.has_any({"read-only", "read_only", "isolated", "provider", "model"})) {
            core::logging::debug("Agent definition '{}': read-only, isolated, provider and "
                                 "model are set by Filo for plugin and additional-workspace "
                                 "personas",
                                 file.string());
        }
        definition.read_only = true;
        definition.isolated = true;
        definition.prompt = document->body;
        if (from_plugin) {
            definition.prompt += "\n\nThis persona ships in the plugin bundle at "
                + plugin_root.string() + " (definition: " + definition.source.string()
                + "). Relative paths in it, such as agents/ or resources/, resolve against "
                  "that directory.";
        }
        return definition;
    }

    definition.read_only = reader.flag({"read-only", "read_only"}).value_or(false);
    definition.isolated = reader.flag({"isolated"}).value_or(false);
    definition.provider = reader.text({"provider"}).value_or("");
    definition.model = reader.text({"model"}).value_or("");
    definition.prompt = document->body;
    return definition;
}

std::vector<AgentDirectory>
AgentDefinitionRegistry::search_dirs(const std::vector<fs::path>& workspace_roots) {
    using core::tools::SkillRegistry;
    std::vector<AgentDirectory> dirs;
    for (const auto& plugin : SkillRegistry::plugin_roots(workspace_roots)) {
        dirs.push_back(AgentDirectory{.path = plugin / "agents", .plugin_root = plugin});
    }
    for (const auto& root : SkillRegistry::default_search_roots(workspace_roots)) {
        if (root.filo_native) {
            dirs.push_back(AgentDirectory{
                .path = root.path.parent_path() / "agents",
                .plugin_root = {},
                .trusted = root.permits_tool_skills(),
            });
        }
    }
    return dirs;
}

std::vector<AgentDefinition>
AgentDefinitionRegistry::discover(const std::vector<fs::path>& workspace_roots) {
    std::vector<AgentDefinition> definitions;
    for (const auto& dir : search_dirs(workspace_roots)) {
        std::error_code ec;
        std::vector<fs::path> files;
        for (const auto& entry : fs::directory_iterator(dir.path, ec)) {
            std::error_code type_ec;
            if (entry.path().extension() == ".md" && entry.is_regular_file(type_ec)) {
                files.push_back(entry.path());
            }
        }
        std::ranges::sort(files);

        for (const auto& file : files) {
            auto definition = parse_file(file, dir.plugin_root, dir.trusted);
            if (!definition) continue;
            const auto existing = std::ranges::find(definitions, definition->name, &AgentDefinition::name);
            if (existing == definitions.end()) {
                definitions.push_back(std::move(*definition));
                continue;
            }
            if (existing->plugin_root == definition->plugin_root) {
                core::logging::warn("Agent definitions '{}' and '{}' share the name '{}'; using the latter",
                                    existing->source.string(), definition->source.string(),
                                    definition->name);
            }
            *existing = std::move(*definition);
        }
    }
    return definitions;
}

} // namespace core::agent
