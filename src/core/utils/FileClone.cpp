#include "FileClone.hpp"

#include "UniqueFd.hpp"

#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/clonefile.h>
#elif defined(__linux__)
#include <linux/fs.h>
#include <sys/ioctl.h>
#endif

namespace core::utils {

std::expected<void, std::error_code> clone_file(const std::filesystem::path& source,
                                                const std::filesystem::path& target) noexcept {
    const auto failed = [] { return std::unexpected(std::error_code(errno, std::generic_category())); };
#if defined(__APPLE__)
    if (::clonefile(source.c_str(), target.c_str(), CLONE_NOFOLLOW | CLONE_NOOWNERCOPY) != 0) return failed();
    return {};
#elif defined(__linux__)
    const UniqueFd in(::open(source.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
    if (!in) return failed();
    UniqueFd out(::open(target.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600));
    if (!out) return failed();
    if (::ioctl(out.get(), FICLONE, in.get()) != 0) {
        const std::error_code error(errno, std::generic_category());
        out.reset();
        ::unlink(target.c_str());
        return std::unexpected(error);
    }
    return {};
#else
    static_cast<void>(source);
    static_cast<void>(target);
    return std::unexpected(std::make_error_code(std::errc::not_supported));
#endif
}

bool clone_unsupported(std::error_code error) noexcept {
    if (error.category() != std::generic_category()) return false;
    switch (error.value()) {
        case EXDEV: case ENOTSUP: case EINVAL: case ENOSYS: case ENOTTY:
#if EOPNOTSUPP != ENOTSUP
        case EOPNOTSUPP:
#endif
            return true;
        default:
            return false;
    }
}

} // namespace core::utils
