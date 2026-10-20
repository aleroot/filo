#pragma once
#include <filesystem>
#include <functional>
#include <span>
#include <vector>

namespace core::changes {
struct WorkspaceInventory {
    std::vector<std::filesystem::path> paths;
    bool complete = true;
};
// Bounded native filesystem inventory; no source-control dependency. Paths
// are canonical, so callers need not normalize them again.
[[nodiscard]] WorkspaceInventory workspace_inventory(
    std::span<const std::filesystem::path> roots, std::size_t limit,
    const std::function<bool(const std::filesystem::path&)>& observable = {});
}
