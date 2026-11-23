#include "MountTable.hpp"
#include "PathUtils.hpp"

#include <algorithm>
#include <cstddef>

#if defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
#include <sys/param.h>
#include <sys/mount.h>
#define FILO_MOUNT_TABLE_GETFSSTAT 1
#elif defined(__linux__)
#include <array>
#include <fstream>
#include <string>
#include <string_view>
#endif

namespace core::utils {
namespace {

#ifdef FILO_MOUNT_TABLE_GETFSSTAT
std::vector<MountTable::Mount> read_mounts() {
    std::vector<MountTable::Mount> mounts;
    // MNT_NOWAIT returns what the kernel holds without asking any filesystem
    // for fresh statistics, so a stalled server cannot block the call.
    for (int attempt = 0; attempt < 4; ++attempt) {
        const int count = ::getfsstat(nullptr, 0, MNT_NOWAIT);
        if (count <= 0) return mounts;
        std::vector<struct statfs> buffer(static_cast<std::size_t>(count) + 8);
        const int listed = ::getfsstat(buffer.data(),
            static_cast<int>(buffer.size() * sizeof(struct statfs)), MNT_NOWAIT);
        if (listed < 0) return mounts;
        // A full buffer may have missed a mount made in between: ask again.
        if (static_cast<std::size_t>(listed) == buffer.size()) continue;
        mounts.reserve(static_cast<std::size_t>(listed));
        for (int i = 0; i < listed; ++i) {
            mounts.push_back({.point = std::filesystem::path(buffer[i].f_mntonname).lexically_normal(),
                              .remote = (buffer[i].f_flags & MNT_LOCAL) == 0});
        }
        return mounts;
    }
    return mounts;
}
#elif defined(__linux__)
// Linux has no MNT_LOCAL: the remote filesystems are named by type.
bool remote_type(std::string_view type) {
    static constexpr std::array<std::string_view, 21> kRemote{
        "nfs", "nfs4", "cifs", "smb3", "smbfs", "ncpfs", "afs", "9p", "ceph", "glusterfs",
        "lustre", "gpfs", "beegfs", "davfs", "fuse.sshfs", "fuse.rclone", "fuse.s3fs",
        "fuse.gcsfuse", "fuse.glusterfs", "fuse.smbnetfs", "fuse.gvfsd-fuse"};
    for (const auto remote : kRemote) {
        if (type == remote) return true;
    }
    return false;
}

// mountinfo escapes space, tab, newline and backslash as three octal digits.
std::string unescape(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == '\\' && i + 3 < text.size()) {
            const auto digit = [&](std::size_t at) { return text[at] >= '0' && text[at] <= '7'; };
            if (digit(i + 1) && digit(i + 2) && digit(i + 3)) {
                out.push_back(static_cast<char>((text[i + 1] - '0') * 64 + (text[i + 2] - '0') * 8
                                                + (text[i + 3] - '0')));
                i += 3;
                continue;
            }
        }
        out.push_back(text[i]);
    }
    return out;
}

std::vector<MountTable::Mount> read_mounts() {
    std::vector<MountTable::Mount> mounts;
    // The kernel renders this file from its own mount list; reading it
    // touches none of the filesystems it names.
    std::ifstream in("/proc/self/mountinfo");
    for (std::string line; std::getline(in, line);) {
        // id parent major:minor root mount-point options [optional…] - type source super
        std::string_view rest(line);
        std::string_view fields[5];
        bool ok = true;
        for (auto& field : fields) {
            const auto space = rest.find(' ');
            if (space == std::string_view::npos) { ok = false; break; }
            field = rest.substr(0, space);
            rest.remove_prefix(space + 1);
        }
        const auto separator = rest.find(" - ");
        if (!ok || separator == std::string_view::npos) continue;
        auto type = rest.substr(separator + 3);
        type = type.substr(0, type.find(' '));
        mounts.push_back({.point = std::filesystem::path(unescape(fields[4])).lexically_normal(),
                          .remote = remote_type(type)});
    }
    return mounts;
}
#else
std::vector<MountTable::Mount> read_mounts() { return {}; }
#endif

} // namespace

MountTable::MountTable(std::vector<Mount> mounts) : mounts_(std::move(mounts)) {
    for (const auto& mount : mounts_) {
        if (mount.remote) remote_points_.push_back(mount.point);
    }
    std::ranges::sort(remote_points_, path::NativeLess{});
}

MountTable MountTable::current() { return MountTable(read_mounts()); }

bool MountTable::remote(const std::filesystem::path& path) const {
    const Mount* deepest = nullptr;
    for (const auto& mount : mounts_) {
        if (!path::normalized_within(path, mount.point)) continue;
        if (!deepest || mount.point.native().size() > deepest->point.native().size()) deepest = &mount;
    }
    return deepest && deepest->remote;
}

bool MountTable::remote_mount_point(const std::filesystem::path& directory) const {
    return std::ranges::binary_search(remote_points_, directory, path::NativeLess{});
}

} // namespace core::utils
