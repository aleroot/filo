#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>

struct stat;

namespace core::utils {

/// What one `lstat` says about a path: whether it is there, what it is, and an
/// identity of the bytes a regular file holds.
///
/// Like git's index, the identity covers the device, inode, size, mode and
/// both the modification and the status-change time at nanosecond resolution.
/// So it moves with every write, including one that puts the old mtime back:
/// no process can set ctime. Equal identities are taken to mean equal bytes,
/// which is what lets a file be compared without being read.
struct FileStamp {
    enum class Kind : std::uint8_t { Missing, Regular, Other };

    /// How long after its last change a file's identity is trusted. A write
    /// within the same timestamp tick as the observation could leave every
    /// field as it was — git's "racy" entries — and kernels stamp files from
    /// a coarse clock. A file this fresh is simply read again next time.
    static constexpr std::chrono::seconds kSettleTime{2};

    Kind kind = Kind::Missing;
    unsigned mode = 0;           ///< Permission bits, setuid/setgid/sticky included.
    std::uint64_t size = 0;
    std::uint64_t identity = 0;  ///< Regular files only, and never 0 for one.
    std::int64_t changed_ns = 0; ///< ctime, nanoseconds since the epoch.

    /// The final component is not followed: a symlink is `Other`.
    [[nodiscard]] static FileStamp of(const std::filesystem::path& path) noexcept;
    [[nodiscard]] static FileStamp from(const struct ::stat& status) noexcept;

    [[nodiscard]] bool exists() const noexcept { return kind != Kind::Missing; }
    [[nodiscard]] bool regular() const noexcept { return kind == Kind::Regular; }
    /// Identity safe to compare later, for a stamp taken at `observed`: 0 when
    /// the file changed too recently for its identity to witness its bytes.
    [[nodiscard]] std::uint64_t settled_identity(
        std::chrono::system_clock::time_point observed) const noexcept {
        const auto changed = std::chrono::system_clock::time_point{
            std::chrono::duration_cast<std::chrono::system_clock::duration>(
                std::chrono::nanoseconds{changed_ns})};
        return regular() && changed + kSettleTime < observed ? identity : 0;
    }
    bool operator==(const FileStamp&) const = default;
};

} // namespace core::utils
