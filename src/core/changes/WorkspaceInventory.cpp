#include "WorkspaceInventory.hpp"
#include "../workspace/AgentIgnore.hpp"

#include <algorithm>
#include <set>

namespace core::changes {
WorkspaceInventory workspace_inventory(
    std::span<const std::filesystem::path> roots, std::size_t limit,
    const std::function<bool(const std::filesystem::path&)>& observable) {
    namespace fs = std::filesystem;
    WorkspaceInventory result;
    const auto ignored = core::workspace::AgentIgnoreMatcher::load_for_roots(
        {roots.begin(), roots.end()}, {".gitignore", ".ignore", ".agentignore"});
    std::set<fs::path> seen;
    std::size_t visited = 0;
    const auto private_directory = [](const fs::path& name) {
        return name == ".git" || name == ".hg" || name == ".svn" || name == ".filo";
    };
    const auto excluded = [&](const fs::path& path) {
        if (std::ranges::any_of(path, private_directory) || (observable && !observable(path))) return true;
        return ignored.is_ignored(path, core::workspace::AgentIgnorePathKind::File);
    };
    const auto add = [&](const fs::path& path) {
        std::error_code ec;
        if (!fs::is_regular_file(fs::symlink_status(path, ec)) || excluded(path)) return;
        if (!seen.insert(path).second) return;
        if (result.paths.size() < limit) result.paths.push_back(path);
        else result.complete = false;
    };
    for (const auto& root : roots) {
        std::error_code ec;
        if (fs::is_regular_file(fs::symlink_status(root, ec))) { add(root); continue; }
        for (fs::recursive_directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
            if (++visited > std::max<std::size_t>(1024, limit * 8)) { result.complete = false; break; }
            const auto status = it->symlink_status(ec);
            if (ec) break;
            if (fs::is_symlink(status)) { it.disable_recursion_pending(); continue; }
            if (fs::is_directory(status)) {
                if (it.depth() >= 32) result.complete = false;
                if (it.depth() >= 32 || private_directory(it->path().filename()) ||
                    ignored.can_prune_directory(it->path())) it.disable_recursion_pending();
            } else if (fs::is_regular_file(status)) add(it->path());
            if (result.paths.size() >= limit && !result.complete) break;
        }
        if (ec) result.complete = false;
    }
    std::ranges::sort(result.paths);
    return result;
}
}
