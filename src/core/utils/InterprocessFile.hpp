#pragma once

#include <filesystem>
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

// Crash-safe replacement: unique private temp file, fsync, atomic rename, then
// best-effort directory fsync. Callers that perform read/modify/write must hold
// an InterprocessFileLock across the complete transaction.
[[nodiscard]] bool atomic_write_file(const std::filesystem::path& target,
                                     std::string_view content,
                                     std::string* error = nullptr);

} // namespace core::utils
