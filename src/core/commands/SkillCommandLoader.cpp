#include "SkillCommandLoader.hpp"
#include "SkillCommand.hpp"
#include "../tools/SkillRegistry.hpp"
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

        auto maybe = core::tools::SkillRegistry::parse_manifest(entry.path());
        if (!maybe) continue;

        const auto& m = *maybe;

        if (!m.enabled) {
            core::logging::info("SkillCommandLoader: skipping disabled skill '{}'", m.name);
            continue;
        }

        // Only handle Prompt-type skills; Tool skills are registered by SkillLoader.
        if (m.type != core::tools::SkillType::Prompt) continue;
        if (!m.user_invocable) {
            core::logging::info("SkillCommandLoader: skipping non-user-invocable Prompt skill '{}'",
                                m.name);
            continue;
        }

        executor.register_command(std::make_unique<SkillCommand>(m));
        core::logging::info("SkillCommandLoader: registered '/{}'", m.name);
        ++count;
    }
    return count;
}

int SkillCommandLoader::discover_and_register(
    CommandExecutor& executor,
    const std::vector<fs::path>& workspace_roots) {
    int total = 0;
    // The registry owns the canonical ordering — additional workspace roots,
    // then the user's directories, then the primary — so prompt skills and the
    // Agent Skills catalog can never disagree about who wins a name collision.
    // register_command() replaces on collision, hence the last root scanned wins.
    for (const auto& root : core::tools::SkillRegistry::default_search_paths(workspace_roots)) {
        total += load_from_directory(root, executor);
    }
    if (total > 0) {
        core::logging::info("SkillCommandLoader: {} prompt skill(s) loaded", total);
    }
    return total;
}

} // namespace core::commands
