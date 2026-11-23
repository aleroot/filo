#pragma once

#include <filesystem>
#include <utility>
#include <vector>

namespace core::utils {

/// The filesystems mounted on this machine, and which of them are remote
/// (SMB, NFS, AFP, WebDAV, sshfs…). Read from the kernel's mount table
/// without contacting any filesystem: a stalled network share answers
/// nothing, and a `statfs` or `realpath` through it blocks the caller in the
/// kernel, beyond the reach of any cancellation.
class MountTable final {
public:
    struct Mount {
        std::filesystem::path point;
        bool remote = false;
    };

    MountTable() = default;
    explicit MountTable(std::vector<Mount> mounts);

    /// The mounts as the kernel lists them now: `getfsstat(MNT_NOWAIT)` on
    /// BSDs and macOS, `/proc/self/mountinfo` on Linux. Empty, so that every
    /// path counts as local, where neither is available.
    [[nodiscard]] static MountTable current();

    /// Whether @p path, absolute and lexically normal, lies on a remote
    /// filesystem: whether the deepest mount containing it is remote.
    /// Lexical only, so a symlink into a share is not seen through.
    [[nodiscard]] bool remote(const std::filesystem::path& path) const;

    /// Whether @p directory is itself the mount point of a remote filesystem,
    /// so a walk of a local tree can stop at it.
    [[nodiscard]] bool remote_mount_point(const std::filesystem::path& directory) const;

private:
    std::vector<Mount> mounts_;
    /// Remote mount points, sorted: a walk asks once per directory entry, and
    /// a container host can list thousands of mounts.
    std::vector<std::filesystem::path> remote_points_;
};

} // namespace core::utils
