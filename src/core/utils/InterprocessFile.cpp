#include "InterprocessFile.hpp"

#include "Uuid.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <format>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_map>

namespace core::utils {
namespace {

std::string normalized_key(const std::filesystem::path& path) {
    std::error_code ec;
    const auto absolute = path.is_absolute()
        ? path
        : std::filesystem::current_path(ec) / path;
    return (ec ? path : absolute).lexically_normal().string();
}

std::shared_ptr<std::mutex> process_mutex_for(const std::filesystem::path& path) {
    static std::mutex registry_mutex;
    static std::unordered_map<std::string, std::weak_ptr<std::mutex>> registry;

    std::lock_guard lock(registry_mutex);
    const auto key = normalized_key(path);
    if (auto existing = registry[key].lock()) {
        return existing;
    }
    auto created = std::make_shared<std::mutex>();
    registry[key] = created;
    return created;
}

} // namespace

std::optional<InterprocessFileLock> InterprocessFileLock::acquire_impl(
    const std::filesystem::path& lock_path,
    bool wait,
    std::string* error) {
    if (error) error->clear();

    std::error_code ec;
    if (const auto parent = lock_path.parent_path(); !parent.empty()) {
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            if (error) {
                *error = std::format("cannot create lock directory '{}': {}",
                                     parent.string(), ec.message());
            }
            return std::nullopt;
        }
    }

    auto process_mutex = process_mutex_for(lock_path);
    std::unique_lock process_lock(*process_mutex, std::defer_lock);
    if (wait) {
        process_lock.lock();
    } else if (!process_lock.try_lock()) {
        return std::nullopt;
    }

    const std::string native = lock_path.string();
    const int fd = ::open(native.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
    if (fd < 0) {
        if (error) {
            *error = std::format("cannot open lock file '{}': {}",
                                 native, std::strerror(errno));
        }
        return std::nullopt;
    }

    const int operation = LOCK_EX | (wait ? 0 : LOCK_NB);
    int result;
    do {
        result = ::flock(fd, operation);
    } while (result != 0 && errno == EINTR && wait);

    if (result != 0) {
        const int saved_errno = errno;
        (void)::close(fd);
        if (!wait && (saved_errno == EWOULDBLOCK || saved_errno == EAGAIN)) {
            return std::nullopt;
        }
        if (error) {
            *error = std::format("cannot lock '{}': {}",
                                 native, std::strerror(saved_errno));
        }
        return std::nullopt;
    }

    return InterprocessFileLock(fd, std::move(process_mutex), std::move(process_lock));
}

namespace {

bool write_all(int fd, std::string_view content, std::string& error) {
    std::size_t written = 0;
    while (written < content.size()) {
        const ssize_t result = ::write(
            fd, content.data() + written, content.size() - written);
        if (result < 0) {
            if (errno == EINTR) continue;
            error = std::strerror(errno);
            return false;
        }
        written += static_cast<std::size_t>(result);
    }
    return true;
}

} // namespace

InterprocessFileLock::InterprocessFileLock(
    int fd,
    std::shared_ptr<std::mutex> process_mutex,
    std::unique_lock<std::mutex> process_lock) noexcept
    : fd_(fd),
      process_mutex_(std::move(process_mutex)),
      process_lock_(std::move(process_lock)) {}

InterprocessFileLock::~InterprocessFileLock() {
    release();
}

InterprocessFileLock::InterprocessFileLock(InterprocessFileLock&& other) noexcept
    : fd_(other.fd_),
      process_mutex_(std::move(other.process_mutex_)),
      process_lock_(std::move(other.process_lock_)) {
    other.fd_ = -1;
}

InterprocessFileLock& InterprocessFileLock::operator=(
    InterprocessFileLock&& other) noexcept {
    if (this == &other) return *this;
    release();
    fd_ = other.fd_;
    process_mutex_ = std::move(other.process_mutex_);
    process_lock_ = std::move(other.process_lock_);
    other.fd_ = -1;
    return *this;
}

void InterprocessFileLock::release() noexcept {
    if (fd_ >= 0) {
        (void)::flock(fd_, LOCK_UN);
        (void)::close(fd_);
        fd_ = -1;
    }
    if (process_lock_.owns_lock()) process_lock_.unlock();
    process_mutex_.reset();
}

std::optional<InterprocessFileLock> InterprocessFileLock::acquire(
    const std::filesystem::path& lock_path,
    std::string* error) {
    return acquire_impl(lock_path, true, error);
}

std::optional<InterprocessFileLock> InterprocessFileLock::try_acquire(
    const std::filesystem::path& lock_path,
    std::string* error) {
    return acquire_impl(lock_path, false, error);
}

std::filesystem::path lock_path_for(const std::filesystem::path& target) {
    return std::filesystem::path(target.string() + ".lock");
}

bool atomic_write_file(const std::filesystem::path& target,
                       std::string_view content,
                       std::string* error) {
    if (error) error->clear();

    std::error_code ec;
    const auto parent = target.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            if (error) {
                *error = std::format("cannot create directory '{}': {}",
                                     parent.string(), ec.message());
            }
            return false;
        }
    }

    mode_t mode = 0600;
    struct stat target_stat {};
    if (::stat(target.c_str(), &target_stat) == 0) {
        mode = target_stat.st_mode & 0777;
    }

    const auto temp = std::filesystem::path(std::format(
        "{}.tmp.{}.{}", target.string(), static_cast<long long>(::getpid()),
        random_uuid_v4()));
    const std::string temp_native = temp.string();
    const int fd = ::open(
        temp_native.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, mode);
    if (fd < 0) {
        if (error) {
            *error = std::format("cannot create temporary file '{}': {}",
                                 temp_native, std::strerror(errno));
        }
        return false;
    }

    bool ok = true;
    std::string write_error;
    if (!write_all(fd, content, write_error)) {
        ok = false;
    } else if (::fsync(fd) != 0) {
        write_error = std::strerror(errno);
        ok = false;
    }
    if (::close(fd) != 0 && ok) {
        write_error = std::strerror(errno);
        ok = false;
    }

    if (ok && ::rename(temp_native.c_str(), target.c_str()) != 0) {
        write_error = std::strerror(errno);
        ok = false;
    }

    if (!ok) {
        (void)::unlink(temp_native.c_str());
        if (error) {
            *error = std::format("cannot replace '{}': {}",
                                 target.string(), write_error);
        }
        return false;
    }

    if (!parent.empty()) {
        const int dir_fd = ::open(parent.c_str(), O_RDONLY | O_CLOEXEC);
        if (dir_fd >= 0) {
            (void)::fsync(dir_fd);
            (void)::close(dir_fd);
        }
    }
    return true;
}

} // namespace core::utils
