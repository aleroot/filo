#include "SkillLoader.hpp"
#include "SkillRegistry.hpp"
#ifdef FILO_ENABLE_PYTHON
#include "PythonTool.hpp"
#endif
#include "../logging/Logger.hpp"
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace core::tools {

// ---------------------------------------------------------------------------
// SkillLoader
// ---------------------------------------------------------------------------

std::vector<fs::path> SkillLoader::default_search_paths() {
    return SkillRegistry::default_search_paths();
}

std::optional<SkillManifest>
SkillLoader::parse_manifest(const fs::path& skill_dir) {
    return SkillRegistry::parse_manifest(skill_dir);
}

#ifdef FILO_ENABLE_PYTHON

int SkillLoader::load_from_directory(const fs::path& root, ToolManager& tool_manager) {
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

int SkillLoader::discover_and_register(ToolManager& tool_manager) {
    int total = 0;
    for (const auto& root : default_search_paths()) {
        total += load_from_directory(root, tool_manager);
    }
    if (total > 0) {
        core::logging::info("SkillLoader: {} Tool skill(s) loaded", total);
    }
    return total;
}

#else // FILO_ENABLE_PYTHON

int SkillLoader::load_from_directory(const fs::path&, ToolManager&) { return 0; }
int SkillLoader::discover_and_register(ToolManager&)                { return 0; }

#endif // FILO_ENABLE_PYTHON

} // namespace core::tools
