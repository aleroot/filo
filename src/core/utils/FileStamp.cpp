#include "FileStamp.hpp"

#include <array>
#include <bit>
#include <span>

#include <sys/stat.h>

namespace core::utils {

FileStamp FileStamp::of(const std::filesystem::path& path) noexcept {
    struct ::stat status{};
    if (::lstat(path.c_str(), &status) != 0) return {};
    return from(status);
}

FileStamp FileStamp::from(const struct ::stat& status) noexcept {
    if (!S_ISREG(status.st_mode)) return {.kind = Kind::Other};
#if defined(__APPLE__)
    const auto& modified = status.st_mtimespec;
    const auto& changed = status.st_ctimespec;
#else
    const auto& modified = status.st_mtim;
    const auto& changed = status.st_ctim;
#endif
    const auto mode = static_cast<unsigned>(status.st_mode & 07777);
    const auto size = static_cast<std::uint64_t>(status.st_size);
    const std::array fields{
        static_cast<std::uint64_t>(status.st_dev), static_cast<std::uint64_t>(status.st_ino),
        size, static_cast<std::uint64_t>(mode),
        static_cast<std::uint64_t>(modified.tv_sec), static_cast<std::uint64_t>(modified.tv_nsec),
        static_cast<std::uint64_t>(changed.tv_sec), static_cast<std::uint64_t>(changed.tv_nsec)};
    std::uint64_t identity = 14695981039346656037ull;  // FNV-1a
    for (const auto byte : std::as_bytes(std::span{fields})) {
        identity = (identity ^ std::to_integer<std::uint64_t>(byte)) * 1099511628211ull;
    }
    return {.kind = Kind::Regular, .mode = mode, .size = size, .identity = identity | 1,
            .changed_ns = static_cast<std::int64_t>(changed.tv_sec) * 1'000'000'000 + changed.tv_nsec};
}

} // namespace core::utils
