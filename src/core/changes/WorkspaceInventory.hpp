#pragma once
#include <filesystem>
#include <functional>
#include <span>
#include <vector>

namespace core::utils { class MountTable; }

namespace core::changes {
struct WorkspaceInventory {
    std::vector<std::filesystem::path> paths;
    /// Every local file was listed. Remote filesystems never count against it.
    bool complete = true;
    /// Roots and mount points left unwalked because they are on a remote
    /// filesystem: a share is not read file by file around every command,
    /// and one that stops answering would block the walk in the kernel.
    std::vector<std::filesystem::path> remote;
    /// Whether a file of the roots may be missing from @ref paths.
    [[nodiscard]] bool partial() const noexcept { return !complete || !remote.empty(); }
};
// Bounded native filesystem inventory; no source-control dependency. Paths
// are canonical, so callers need not normalize them again.
[[nodiscard]] WorkspaceInventory workspace_inventory(
    std::span<const std::filesystem::path> roots, std::size_t limit,
    const std::function<bool(const std::filesystem::path&)>& observable = {});
/// As above, against @p mounts rather than the machine's mount table.
[[nodiscard]] WorkspaceInventory workspace_inventory(
    std::span<const std::filesystem::path> roots, std::size_t limit,
    const std::function<bool(const std::filesystem::path&)>& observable,
    const core::utils::MountTable& mounts);
}
