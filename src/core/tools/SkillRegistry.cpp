#include "SkillRegistry.hpp"
#include "Frontmatter.hpp"
#include "../landrun/LandrunSettings.hpp"
#include "../logging/Logger.hpp"
#include "../utils/PathUtils.hpp"
#include "../workspace/Workspace.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <iterator>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace fs = std::filesystem;

namespace core::tools {

namespace {

namespace path = core::utils::path;

[[nodiscard]] bool is_resource_root(std::string_view name) {
    return name == "scripts" || name == "references" || name == "assets";
}

[[nodiscard]] bool safe_relative_resource(const fs::path& path) {
    if (path.empty() || path.is_absolute()) return false;
    for (const auto& part : path) {
        if (part == "..") return false;
    }
    return true;
}

/// A project-local skills directory and the label it is reported under.
using DirLabel = std::pair<std::string_view, std::string_view>;

/// Compatibility directories first, Filo-native last: within one workspace root,
/// a native `.filo/skills` entry always outranks its compatibility aliases.
constexpr std::array<DirLabel, 3> kProjectDirs{{
    {".claude", "project-claude"},
    {".agents", "project-agent"},
    {".filo", "project-filo"},
}};

/// The primary workspace scans these two, then the user-native root, then its own
/// `.filo` — the historical interleaving, preserved exactly.
constexpr std::span<const DirLabel> kCompatProjectDirs{kProjectDirs.data(), 2};
constexpr std::span<const DirLabel> kNativeProjectDirs{kProjectDirs.data() + 2, 1};

[[nodiscard]] SkillSearchRoot user_skill_root(const fs::path& home,
                                             std::string_view relative_dir,
                                             std::string_view label,
                                             bool filo_native = false) {
    return SkillSearchRoot{
        .path = home / fs::path(relative_dir) / "skills",
        .scope = SkillScope::User,
        .label = std::string(label),
        // User roots belong to no workspace; the origin only feeds
        // permits_tool_skills(), which short-circuits on SkillScope::User.
        .origin = SkillWorkspaceOrigin::Primary,
        .workspace_root = {},
        .plugin_root = {},
        .filo_native = filo_native,
    };
}

/// $HOME for the user-level roots, or empty when they must not be scanned:
/// landrun does not grant the sandboxed shell the user's home directories.
[[nodiscard]] fs::path user_home() {
    if (core::landrun::LandrunSettings::instance().enabled()) return {};
    return path::home_directory();
}

/// `<root>/.filo/plugins` of workspace root number @p index; 0 is the primary.
[[nodiscard]] SkillSearchRoot project_plugin_dir(const fs::path& workspace_root, std::size_t index) {
    const bool primary = index == 0;
    return SkillSearchRoot{
        .path = workspace_root / ".filo" / "plugins",
        .scope = SkillScope::Project,
        .label = primary ? std::string("project-plugin") : std::format("ws{}-project-plugin", index),
        .origin = primary ? SkillWorkspaceOrigin::Primary : SkillWorkspaceOrigin::Additional,
        .workspace_root = workspace_root,
        .plugin_root = {},
        .filo_native = false,
    };
}

/// `~/.config/filo/plugins`.
[[nodiscard]] SkillSearchRoot user_plugin_dir(const fs::path& home) {
    return SkillSearchRoot{
        .path = home / ".config" / "filo" / "plugins",
        .scope = SkillScope::User,
        .label = "user-plugin",
        .origin = SkillWorkspaceOrigin::Primary,
        .workspace_root = {},
        .plugin_root = {},
        .filo_native = false,
    };
}

/// One skills root per plugin bundle under @p install_dir, ordered by
/// directory name so precedence between two plugins is deterministic. A
/// subdirectory counts as a bundle when it has a plugin manifest or a
/// `skills`/`agents` directory; hidden entries are ignored.
[[nodiscard]] std::vector<SkillSearchRoot> plugin_skill_roots(const SkillSearchRoot& install_dir) {
    std::vector<SkillSearchRoot> out;
    std::error_code ec;
    if (!fs::is_directory(install_dir.path, ec)) return out;

    std::vector<fs::path> bundles;
    for (const auto& entry : fs::directory_iterator(install_dir.path, ec)) {
        if (ec) break;
        if (!entry.is_directory(ec)) continue;
        const auto& dir = entry.path();
        if (dir.filename().string().starts_with('.')) continue;
        if (!SkillRegistry::is_plugin_root(dir)
            && !fs::is_directory(dir / "skills", ec)
            && !fs::is_directory(dir / "agents", ec)) {
            continue;
        }
        bundles.push_back(dir);
    }
    std::ranges::sort(bundles);

    for (const auto& dir : bundles) {
        auto root = install_dir;
        root.plugin_root = path::normalized(dir);
        root.path = root.plugin_root / "skills";
        root.label = std::format("{}:{}", install_dir.label, dir.filename().string());
        out.push_back(std::move(root));
    }
    return out;
}

/// A plugin root must never widen read access to the home directory or above
/// it, whatever manifest happens to sit there.
[[nodiscard]] bool acceptable_plugin_root(const fs::path& candidate) {
    if (candidate.empty() || candidate == candidate.root_path()) return false;
    const auto home = path::home_directory();
    return home.empty() || !path::relative_within(path::normalized(home), candidate);
}

void append_unique(std::vector<fs::path>& out, const fs::path& path) {
    if (!path.empty() && std::ranges::find(out, path) == out.end()) out.push_back(path);
}

} // namespace

std::vector<SkillSearchRoot>
SkillRegistry::plugin_install_dirs(const std::vector<fs::path>& workspace_roots) {
    std::vector<SkillSearchRoot> dirs;
    for (std::size_t i = workspace_roots.size(); i-- > 1;) {
        if (!workspace_roots[i].empty()) dirs.push_back(project_plugin_dir(workspace_roots[i], i));
    }
    if (const auto home = user_home(); !home.empty()) dirs.push_back(user_plugin_dir(home));
    if (!workspace_roots.empty() && !workspace_roots.front().empty()) {
        dirs.push_back(project_plugin_dir(workspace_roots.front(), 0));
    }
    return dirs;
}

std::vector<SkillSearchRoot>
SkillRegistry::default_search_roots(const std::vector<fs::path>& workspace_roots) {
    std::vector<SkillSearchRoot> roots;

    const fs::path home_path = user_home();

    // Each level's installed plugin bundles rank just below that level's own
    // native skills, in the order plugin_install_dirs() lists them.
    auto append_plugins = [&](const SkillSearchRoot& install_dir) {
        std::ranges::move(plugin_skill_roots(install_dir), std::back_inserter(roots));
    };

    // An empty workspace root is skipped rather than scanned as a relative path:
    // `fs::is_directory(".filo/skills")` would silently resolve against the
    // process cwd, which is exactly the ambient coupling these APIs exist to
    // remove.
    auto append_project_dirs = [&](const fs::path& workspace_root,
                                   SkillWorkspaceOrigin origin,
                                   std::string_view label_prefix,
                                   std::span<const DirLabel> dirs) {
        if (workspace_root.empty()) {
            return;
        }
        for (const auto& [dir, label] : dirs) {
            roots.push_back(SkillSearchRoot{
                .path = workspace_root / fs::path(dir) / "skills",
                .scope = SkillScope::Project,
                .label = std::format("{}{}", label_prefix, label),
                .origin = origin,
                .workspace_root = workspace_root,
                .plugin_root = {},
                .filo_native = dir == ".filo",
            });
        }
    };

    // Additional workspace roots are scanned first because precedence here is
    // "last scanned wins": that makes the primary — and the user's own
    // directories — outrank any secondary workspace. They are walked in reverse
    // grant order so that workspace order and precedence order agree, i.e. the
    // first `-w` after the primary beats the second, exactly as steering's
    // fallback chain prefers the nearer root.
    for (std::size_t i = workspace_roots.size(); i-- > 1;) {
        if (!workspace_roots[i].empty()) append_plugins(project_plugin_dir(workspace_roots[i], i));
        append_project_dirs(workspace_roots[i],
                            SkillWorkspaceOrigin::Additional,
                            std::format("ws{}-", i),
                            kProjectDirs);
    }

    if (!home_path.empty()) {
        roots.push_back(user_skill_root(home_path, ".claude", "user-claude"));
        roots.push_back(user_skill_root(home_path, ".agents", "user-agent"));
    }

    // The primary's directories stay split around the user-native root so that a
    // single-root workspace keeps the historical precedence byte for byte:
    // ~/.config/filo/skills outranks <primary>/.claude|.agents but not
    // <primary>/.filo.
    const fs::path primary = workspace_roots.empty() ? fs::path{} : workspace_roots.front();
    append_project_dirs(primary, SkillWorkspaceOrigin::Primary, {}, kCompatProjectDirs);

    if (!home_path.empty()) {
        append_plugins(user_plugin_dir(home_path));
        roots.push_back(user_skill_root(home_path, ".config/filo", "user-filo", true));
    }

    if (!primary.empty()) append_plugins(project_plugin_dir(primary, 0));
    append_project_dirs(primary, SkillWorkspaceOrigin::Primary, {}, kNativeProjectDirs);
    return roots;
}

std::vector<fs::path>
SkillRegistry::default_search_paths(const std::vector<fs::path>& workspace_roots) {
    std::vector<fs::path> paths;
    for (const auto& root : default_search_roots(workspace_roots)) {
        paths.push_back(root.path);
    }
    return paths;
}

SkillRootTrustSplit
SkillRegistry::split_tool_skill_roots(const std::vector<fs::path>& workspace_roots) {
    SkillRootTrustSplit split;
    for (auto& root : default_search_roots(workspace_roots)) {
        auto& bucket = root.permits_tool_skills() ? split.trusted : split.restricted;
        bucket.push_back(std::move(root));
    }
    return split;
}

bool SkillRegistry::is_plugin_root(const fs::path& dir) {
    std::error_code ec;
    return fs::is_regular_file(dir / "plugin.json", ec)
        || fs::is_regular_file(dir / ".claude-plugin" / "plugin.json", ec)
        || fs::is_regular_file(dir / ".codex-plugin" / "plugin.json", ec);
}

std::optional<fs::path> SkillRegistry::detect_plugin_root(const fs::path& skill_dir) {
    const auto skills_dir = path::normalized(skill_dir).parent_path();
    if (skills_dir.filename() != "skills") return std::nullopt;
    const auto candidate = skills_dir.parent_path();
    if (!acceptable_plugin_root(candidate) || !is_plugin_root(candidate)) return std::nullopt;
    return candidate;
}

std::optional<SkillManifest>
SkillRegistry::parse_manifest(const fs::path& skill_dir) {
    const fs::path manifest_path = skill_dir / "SKILL.md";
    if (!fs::exists(manifest_path)) return std::nullopt;

    const auto parsed = frontmatter::parse_file(manifest_path);
    if (!parsed.has_value()) return std::nullopt;

    SkillManifest manifest;
    manifest.enabled = true;
    manifest.skill_dir = fs::exists(skill_dir) ? fs::weakly_canonical(skill_dir) : skill_dir;
    manifest.manifest_path = fs::exists(manifest_path)
        ? fs::weakly_canonical(manifest_path)
        : manifest_path;

    const frontmatter::FieldReader fields(*parsed, manifest_path, "Skill");
    if (auto value = fields.text({"name"})) manifest.name = std::move(*value);
    if (auto value = fields.text({"description"})) manifest.description = std::move(*value);
    if (auto value = fields.text({"entry_point", "entry-point"})) {
        manifest.entry_point = std::move(*value);
    }
    manifest.enabled = fields.flag({"enabled"}).value_or(manifest.enabled);
    manifest.user_invocable =
        fields.flag({"user_invocable", "user-invocable"}).value_or(manifest.user_invocable);
    manifest.model_invocable =
        !fields.flag({"disable-model-invocation", "disable_model_invocation"}).value_or(false);
    if (auto value = fields.text({"model"})) manifest.model_hint = std::move(*value);
    if (auto value = fields.text({"license"})) manifest.license = std::move(*value);
    if (auto value = fields.text({"compatibility"})) manifest.compatibility = std::move(*value);

    bool explicit_type = false;
    if (const auto value = fields.text({"type"})) {
        const auto type = core::utils::str::to_lower_ascii_copy(*value);
        manifest.type = type == "prompt" || type == "instruction" || type == "agent"
            ? SkillType::Prompt
            : SkillType::Tool;
        explicit_type = true;
    }

    if (auto value = fields.list({"allowed-tools", "allowed_tools"})) {
        manifest.allowed_tools = std::move(*value);
    }

    if (manifest.name.empty()) {
        core::logging::warn("SkillRegistry: '{}' missing required 'name' field",
                            manifest_path.string());
        return std::nullopt;
    }
    if (manifest.description.empty()) {
        core::logging::warn("SkillRegistry: skill '{}' missing required 'description' field",
                            manifest.name);
        return std::nullopt;
    }

    if (!explicit_type) {
        manifest.type = manifest.entry_point.empty() ? SkillType::Prompt : SkillType::Tool;
    }
    if (manifest.type == SkillType::Tool && manifest.entry_point.empty()) {
        core::logging::warn("SkillRegistry: Tool skill '{}' missing required 'entry_point' field",
                            manifest.name);
        return std::nullopt;
    }

    if (manifest.type == SkillType::Prompt) {
        manifest.body = parsed->body;
    }
    manifest.plugin_root = detect_plugin_root(manifest.skill_dir).value_or(fs::path{});
    return manifest;
}

std::optional<SkillManifest>
SkillRegistry::parse_manifest_in(const SkillSearchRoot& root, const fs::path& skill_dir) {
    auto manifest = parse_manifest(skill_dir);
    if (!manifest.has_value()) return manifest;
    // Provenance travels with the manifest so consumers can tell a secondary
    // workspace's skill from the primary's without re-deriving which root it
    // came from.
    manifest->workspace_root = root.workspace_root;
    manifest->from_additional_workspace = root.origin == SkillWorkspaceOrigin::Additional;
    if (!root.plugin_root.empty()) manifest->plugin_root = root.plugin_root;
    return manifest;
}

std::vector<SkillManifest>
SkillRegistry::discover_all(const std::vector<fs::path>& workspace_roots) {
    std::vector<SkillManifest> ordered;
    std::unordered_map<std::string, std::size_t> by_name;

    for (const auto& root : default_search_roots(workspace_roots)) {
        std::error_code ec;
        if (!fs::is_directory(root.path, ec)) continue;

        for (const auto& entry : fs::directory_iterator(root.path, ec)) {
            if (ec) break;
            if (!entry.is_directory(ec)) continue;

            auto manifest = parse_manifest_in(root, entry.path());
            if (!manifest.has_value() || !manifest->enabled) continue;

            if (auto it = by_name.find(manifest->name); it != by_name.end()) {
                ordered[it->second] = std::move(*manifest);
            } else {
                by_name[manifest->name] = ordered.size();
                ordered.push_back(std::move(*manifest));
            }
        }
    }

    return ordered;
}

std::vector<fs::path>
SkillRegistry::plugin_roots(const std::vector<fs::path>& workspace_roots) {
    std::vector<fs::path> roots;
    for (const auto& install_dir : plugin_install_dirs(workspace_roots)) {
        for (const auto& bundle : plugin_skill_roots(install_dir)) {
            append_unique(roots, bundle.plugin_root);
        }
    }
    for (const auto& skill : discover_all(workspace_roots)) {
        append_unique(roots, skill.plugin_root);
    }
    return roots;
}

std::vector<fs::path>
SkillRegistry::package_read_roots(const std::vector<fs::path>& workspace_roots) {
    std::vector<fs::path> roots;
    if (core::landrun::LandrunSettings::instance().enabled()) return roots;

    std::error_code ec;
    for (const auto& install_dir : plugin_install_dirs(workspace_roots)) {
        if (fs::is_directory(install_dir.path, ec)) {
            append_unique(roots, path::normalized(install_dir.path));
        }
    }
    for (const auto& root : default_search_roots(workspace_roots)) {
        if (fs::is_directory(root.path, ec)) append_unique(roots, path::normalized(root.path));
        append_unique(roots, root.plugin_root);
    }
    for (const auto& skill : discover_all(workspace_roots)) {
        append_unique(roots, skill.skill_dir);
        append_unique(roots, skill.plugin_root);
    }

    // A root inside another root grants nothing more; drop it so the list
    // (also shown to MCP clients) names each bundle once.
    const auto nested = [&](const fs::path& candidate) {
        return std::ranges::any_of(roots, [&](const fs::path& other) {
            return other != candidate && path::relative_within(candidate, other).has_value();
        });
    };
    std::vector<fs::path> outermost;
    for (const auto& root : roots) {
        if (!nested(root)) outermost.push_back(root);
    }
    return outermost;
}

void SkillRegistry::grant_package_read_roots(core::workspace::Workspace& workspace) {
    workspace.set_extension_roots(package_read_roots(workspace.ordered_roots()));
}

std::string SkillRegistry::with_invocation_context(std::string prompt, const SkillManifest& skill) {
    if (prompt.empty() || skill.skill_dir.empty()) return prompt;
    return prompt + "\n\n" + invocation_context(skill);
}

std::string SkillRegistry::invocation_context(const SkillManifest& skill) {
    std::string out = "Skill directory: " + skill.skill_dir.string() + "\n";
    if (skill.plugin_root.empty()) {
        out += "Relative paths in this skill are relative to the skill directory.\n";
        return out;
    }
    out += "Plugin root: " + skill.plugin_root.string() + "\n";
    out += "Relative paths in this skill are relative to the skill directory; paths that are "
           "not there (for example agents/, resources/ or skills/<name>/) resolve against the "
           "plugin root. Read bundled files from these absolute locations, never from "
           "similarly named files in the project.\n";
    return out;
}

std::vector<SkillManifest>
SkillRegistry::discover_instruction_skills(const std::vector<fs::path>& workspace_roots) {
    auto skills = discover_all(workspace_roots);
    std::erase_if(skills, [](const SkillManifest& skill) {
        return !skill.is_agent_instruction_skill() || !skill.model_invocable;
    });
    std::ranges::sort(skills, {}, &SkillManifest::name);
    return skills;
}

std::optional<SkillManifest>
SkillRegistry::find_instruction_skill(std::string_view name,
                                     const std::vector<fs::path>& workspace_roots) {
    for (auto& skill : discover_instruction_skills(workspace_roots)) {
        if (skill.name == name) return skill;
    }
    return std::nullopt;
}

std::string SkillRegistry::build_catalog_prompt(const std::vector<fs::path>& workspace_roots) {
    const auto skills = discover_instruction_skills(workspace_roots);
    if (skills.empty()) return {};

    std::ostringstream out;
    out << "\n\n[Agent Skills]\n";
    out << "The following Agent Skills provide specialized instructions. "
           "When a task matches a skill description, call `activate_skill` with the skill name "
           "before proceeding. If activated instructions reference relative files, call "
           "`activate_skill` again with `resource_path` set to that relative path.\n";
    out << "<available_skills>\n";
    for (const auto& skill : skills) {
        out << "  <skill>\n"
            << "    <name>" << core::utils::str::escape_xml(skill.name) << "</name>\n"
            << "    <description>" << core::utils::str::escape_xml(skill.description) << "</description>\n";
        // Only annotated when the skill belongs to another workspace: a
        // single-root session's catalog stays byte-identical, so the cached
        // prompt prefix is not disturbed and no redundant path is emitted.
        if (skill.from_additional_workspace && !skill.workspace_root.empty()) {
            out << "    <workspace>" << core::utils::str::escape_xml(skill.workspace_root.string())
                << "</workspace>\n";
        }
        out << "  </skill>\n";
    }
    out << "</available_skills>";
    return out.str();
}

std::vector<fs::path>
SkillRegistry::list_relative_resources(const SkillManifest& skill, std::size_t max_entries) {
    std::vector<fs::path> resources;
    std::error_code ec;
    if (!fs::is_directory(skill.skill_dir, ec)) return resources;

    for (const auto& root : fs::directory_iterator(skill.skill_dir, ec)) {
        if (ec) break;
        if (!root.is_directory(ec) || !is_resource_root(root.path().filename().string())) {
            continue;
        }
        for (const auto& entry : fs::recursive_directory_iterator(root.path(), ec)) {
            if (ec) break;
            if (!entry.is_regular_file(ec)) continue;
            auto relative = fs::relative(entry.path(), skill.skill_dir, ec);
            if (ec || !safe_relative_resource(relative)) continue;
            resources.push_back(relative);
            if (resources.size() >= max_entries) return resources;
        }
    }
    std::ranges::sort(resources);
    return resources;
}

} // namespace core::tools
