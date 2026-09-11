#include "SkillLoader.hpp"
#include "SkillRegistry.hpp"
#ifdef FILO_ENABLE_PYTHON
#include "PythonTool.hpp"
#endif
#include "../logging/Logger.hpp"
#include "../landrun/LandrunSettings.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace core::tools {

// ---------------------------------------------------------------------------
// SkillLoader
// ---------------------------------------------------------------------------

std::optional<SkillManifest>
SkillLoader::parse_manifest(const fs::path& skill_dir) {
    return SkillRegistry::parse_manifest(skill_dir);
}

#ifdef FILO_ENABLE_PYTHON

namespace {

/// True when @p skills_dir holds at least one executable (Tool) skill candidate.
/// Consulted only to decide whether a refused additional-workspace root is worth
/// warning about, so granting a repo that ships no Python stays silent.
[[nodiscard]] bool has_tool_skill_candidates(const fs::path& skills_dir) {
    std::error_code ec;
    if (!fs::is_directory(skills_dir, ec) || ec) {
        return false;
    }
    for (const auto& entry : fs::directory_iterator(skills_dir, ec)) {
        if (ec) break;
        if (!entry.is_directory(ec)) continue;
        const auto manifest = SkillRegistry::parse_manifest(entry.path());
        if (manifest.has_value() && manifest->enabled
            && manifest->type == SkillType::Tool) {
            return true;
        }
    }
    return false;
}

} // namespace

int SkillLoader::load_from_directory(const fs::path& root, ToolManager& tool_manager) {
    if (!core::landrun::LandrunSettings::instance().permits(
            core::landrun::LandrunCapability::in_process_untrusted_code)) {
        core::logging::warn(
            "landrun secure mode: refusing to import executable Python skill code");
        return 0;
    }
    if (!fs::exists(root) || !fs::is_directory(root)) return 0;

    int count = 0;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_directory()) continue;

        auto maybe_manifest = parse_manifest(entry.path());
        if (!maybe_manifest) continue;

        const auto& manifest = *maybe_manifest;

        if (!manifest.enabled) {
            core::logging::info("SkillLoader: skipping disabled skill '{}'", manifest.name);
            continue;
        }

        // Prompt skills are registered as slash commands by SkillCommandLoader, not here.
        if (manifest.type == SkillType::Prompt) {
            core::logging::info("SkillLoader: skipping Prompt skill '{}' "
                                "(registered by SkillCommandLoader)", manifest.name);
            continue;
        }

        const fs::path script_path = manifest.skill_dir / manifest.entry_point;
        if (!fs::exists(script_path)) {
            core::logging::warn("SkillLoader: entry_point '{}' not found for skill '{}'",
                                script_path.string(), manifest.name);
            continue;
        }

        // Python module name is the entry_point stem (filename without extension).
        const std::string module_name = script_path.stem().string();

        try {
            auto tool = std::make_shared<PythonTool>(script_path.string(), module_name);
            tool_manager.register_tool(std::move(tool));
            core::logging::info("SkillLoader: registered Tool skill '{}'", manifest.name);
            ++count;
        } catch (const std::exception& e) {
            core::logging::error("SkillLoader: failed to load skill '{}': {}",
                                 manifest.name, e.what());
        }
    }
    return count;
}

int SkillLoader::discover_and_register(ToolManager& tool_manager,
                                       const std::vector<fs::path>& workspace_roots) {
    if (!core::landrun::LandrunSettings::instance().permits(
            core::landrun::LandrunCapability::in_process_untrusted_code)) {
        return 0;
    }

    // Trust boundary: the user's own directories and the primary workspace may
    // contribute executable skills; a merely-granted additional root may not. Its
    // Prompt and instruction skills are still discovered by their own loaders.
    const auto trust = SkillRegistry::split_tool_skill_roots(workspace_roots);
    for (const auto& root : trust.restricted) {
        if (has_tool_skill_candidates(root.path)) {
            core::logging::warn(
                "SkillLoader: ignoring executable skill(s) under '{}' - an additional "
                "workspace root contributes instructions and prompt skills, not code",
                root.path.string());
        }
    }

    int total = 0;
    for (const auto& root : trust.trusted) {
        total += load_from_directory(root.path, tool_manager);
    }
    if (total > 0) {
        core::logging::info("SkillLoader: {} Tool skill(s) loaded", total);
    }
    return total;
}

#else // FILO_ENABLE_PYTHON

int SkillLoader::load_from_directory(const fs::path&, ToolManager&) { return 0; }
int SkillLoader::discover_and_register(ToolManager&, const std::vector<fs::path>&) { return 0; }

#endif // FILO_ENABLE_PYTHON

} // namespace core::tools
