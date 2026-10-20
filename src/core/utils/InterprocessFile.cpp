#include "InterprocessFile.hpp"

#include "Uuid.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <format>
#include <fstream>
#include <iterator>
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
                       std::string* error,
                       std::optional<std::filesystem::perms> permissions) {
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
    if (permissions) {
        mode = static_cast<mode_t>(*permissions & std::filesystem::perms::mask);
    } else if (::stat(target.c_str(), &target_stat) == 0) {
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

std::optional<std::string> read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return std::nullopt;
    std::string content(std::istreambuf_iterator<char>(in), {});
    if (in.bad()) return std::nullopt;
    return content;
}

RegularFileReader::RegularFileReader(std::filesystem::path path, int fd) noexcept
    : path_(std::move(path)), fd_(fd) {}

RegularFileReader::RegularFileReader(RegularFileReader&& other) noexcept
    : path_(std::move(other.path_)), fd_(std::exchange(other.fd_, -1)), stamp_(other.stamp_),
      remaining_(other.remaining_), buffer_(std::move(other.buffer_)) {}

RegularFileReader::~RegularFileReader() {
    if (fd_ >= 0) ::close(fd_);
}

std::expected<RegularFileReader, std::string> RegularFileReader::open(
    const std::filesystem::path& path, std::size_t max_bytes) {
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
    if (fd < 0) {
        if (errno == ENOENT || errno == ENOTDIR) return RegularFileReader(path, -1);
        return std::unexpected("Cannot read file: " + path.string());
    }
    RegularFileReader reader(path, fd);
    struct stat status{};
    if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) || status.st_size < 0 ||
        static_cast<std::uintmax_t>(status.st_size) > max_bytes)
        return std::unexpected("Path is not a supported regular file or exceeds the read limit: " + path.string());
    reader.stamp_ = FileStamp::from(status);
    reader.remaining_ = reader.stamp_.size;
    return reader;
}

std::expected<std::span<const char>, std::string> RegularFileReader::next() {
    if (remaining_ == 0) return std::span<const char>{};
    if (!buffer_) buffer_ = std::make_unique_for_overwrite<char[]>(kChunkBytes);
    const auto want = static_cast<std::size_t>(std::min<std::uint64_t>(remaining_, kChunkBytes));
    std::size_t got = 0;
    while (got < want) {
        const auto n = ::read(fd_, buffer_.get() + got, want - got);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return std::unexpected("File changed while reading: " + path_.string());
        got += static_cast<std::size_t>(n);
    }
    remaining_ -= got;
    return std::span<const char>{buffer_.get(), got};
}

std::expected<void, std::string> RegularFileReader::finish() const {
    if (!stamp_.exists()) return {};
    char extra;
    if (remaining_ != 0 || ::read(fd_, &extra, 1) != 0)
        return std::unexpected("File grew while reading: " + path_.string());
    // Identity covers device, inode, size, mode, mtime and ctime: any write or
    // replacement since the open moved it.
    if (!unchanged())
        return std::unexpected("File changed while reading: " + path_.string());
    return {};
}

std::expected<RegularFileContents, std::string> read_regular_file(
    const std::filesystem::path& path, std::size_t max_bytes) {
    auto reader = RegularFileReader::open(path, max_bytes);
    if (!reader) return std::unexpected(reader.error());
    RegularFileContents result;
    if (!reader->stamp().exists()) return result;
    result.bytes.reserve(static_cast<std::size_t>(reader->stamp().size));
    for (;;) {
        const auto chunk = reader->next();
        if (!chunk) return std::unexpected(chunk.error());
        if (chunk->empty()) break;
        result.bytes.append(chunk->data(), chunk->size());
    }
    if (auto done = reader->finish(); !done) return std::unexpected(done.error());
    result.permissions = static_cast<std::filesystem::perms>(reader->stamp().mode & 0777);
    result.stamp = reader->stamp();
    return result;
}

} // namespace core::utils
