#include "SkillCommandLoader.hpp"
#include "SkillCommand.hpp"
#include "../tools/SkillLoader.hpp"
#include "../logging/Logger.hpp"
#include <filesystem>
#include <memory>

namespace fs = std::filesystem;

namespace core::commands {

int SkillCommandLoader::load_from_directory(const fs::path& root,
                                            CommandExecutor& executor) {
    if (!fs::exists(root) || !fs::is_directory(root)) return 0;

    int count = 0;
    for (const auto& entry : fs::directory_iterator(root)) {
        if (!entry.is_directory()) continue;

        auto maybe = core::tools::SkillLoader::parse_manifest(entry.path());
        if (!maybe) continue;

        const auto& m = *maybe;

        if (!m.enabled) {
            core::logging::info("SkillCommandLoader: skipping disabled skill '{}'", m.name);
            continue;
        }

        // Only handle Prompt-type skills; Tool skills are registered by SkillLoader.
        if (m.type != core::tools::SkillType::Prompt) continue;

        executor.register_command(std::make_unique<SkillCommand>(m));
        core::logging::info("SkillCommandLoader: registered '/{}'", m.name);
        ++count;
    }
    return count;
}

int SkillCommandLoader::discover_and_register(CommandExecutor& executor) {
    int total = 0;
    // Reuse SkillLoader's canonical search-path ordering (global → project-local).
    for (const auto& root : core::tools::SkillLoader::default_search_paths()) {
        total += load_from_directory(root, executor);
    }
    if (total > 0) {
        core::logging::info("SkillCommandLoader: {} prompt skill(s) loaded", total);
    }
    return total;
}

} // namespace core::commands
