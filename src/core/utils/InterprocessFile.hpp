#pragma once

#include <cstddef>
#include <filesystem>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>

namespace core::utils {

// Advisory lock backed by flock(2). A process-local mutex is held as well,
// because flock semantics for separate descriptors in one process vary across
// supported POSIX platforms.
class InterprocessFileLock {
public:
    InterprocessFileLock() = default;
    ~InterprocessFileLock();

    InterprocessFileLock(const InterprocessFileLock&) = delete;
    InterprocessFileLock& operator=(const InterprocessFileLock&) = delete;
    InterprocessFileLock(InterprocessFileLock&& other) noexcept;
    InterprocessFileLock& operator=(InterprocessFileLock&& other) noexcept;

    [[nodiscard]] static std::optional<InterprocessFileLock> acquire(
        const std::filesystem::path& lock_path,
        std::string* error = nullptr);

    // Returns nullopt with an empty error when another process owns the lock.
    [[nodiscard]] static std::optional<InterprocessFileLock> try_acquire(
        const std::filesystem::path& lock_path,
        std::string* error = nullptr);

    [[nodiscard]] bool owns_lock() const noexcept { return fd_ >= 0; }

private:
    InterprocessFileLock(int fd,
                         std::shared_ptr<std::mutex> process_mutex,
                         std::unique_lock<std::mutex> process_lock) noexcept;

    [[nodiscard]] static std::optional<InterprocessFileLock> acquire_impl(
        const std::filesystem::path& lock_path,
        bool wait,
        std::string* error);

    void release() noexcept;

    int fd_ = -1;
    std::shared_ptr<std::mutex> process_mutex_;
    std::unique_lock<std::mutex> process_lock_;
};

[[nodiscard]] std::filesystem::path lock_path_for(
    const std::filesystem::path& target);

// Read/write for the owner only (0600): the mode for files holding secrets.
inline constexpr std::filesystem::perms kPrivateFilePermissions =
    std::filesystem::perms::owner_read | std::filesystem::perms::owner_write;

// Crash-safe replacement: unique private temp file, fsync, atomic rename, then
// best-effort directory fsync. Callers that perform read/modify/write must hold
// an InterprocessFileLock across the complete transaction.
//
// The file keeps the target's current permissions (0600 when it is new) unless
// `permissions` is given; then the replacement is created with exactly those,
// so a secret is never readable under a wider mode, not even briefly.
[[nodiscard]] bool atomic_write_file(const std::filesystem::path& target,
                                     std::string_view content,
                                     std::string* error = nullptr,
                                     std::optional<std::filesystem::perms> permissions = std::nullopt);

struct RegularFileContents {
    std::string bytes;
    std::optional<std::filesystem::perms> permissions; // nullopt means absent.
};

// Reads one regular file without following the final symlink or blocking on a
// special file. Refuses oversized files and identity/metadata changes during
// the read. Absence is a successful result, distinct from an unreadable file.
[[nodiscard]] std::expected<RegularFileContents, std::string> read_regular_file(
    const std::filesystem::path& path, std::size_t max_bytes);

// Whole-file read; nullopt when the file cannot be opened or read.
[[nodiscard]] std::optional<std::string> read_file(const std::filesystem::path& path);

} // namespace core::utils
