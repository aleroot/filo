#include "LandrunDescriptorSanitizer.hpp"

#if defined(__linux__) || defined(__APPLE__)
#include <charconv>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <format>
#include <string_view>
#include <system_error>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>
#endif

#if defined(__linux__)
#include <linux/close_range.h>
#endif

namespace core::landrun {

#if defined(__linux__) || defined(__APPLE__)
namespace {

[[nodiscard]] LandrunResult close_from_descriptor_directory(
    const char* descriptor_directory)
{
    DIR* directory = ::opendir(descriptor_directory);
    if (!directory) {
        return {.success = false,
                .detail = std::format("cannot enumerate inherited descriptors: {}",
                                      std::strerror(errno))};
    }

    const int directory_fd = ::dirfd(directory);
    std::vector<int> descriptors;
    errno = 0;
    while (const dirent* entry = ::readdir(directory)) {
        int descriptor = -1;
        const std::string_view name(entry->d_name);
        const auto [end, error] = std::from_chars(
            name.data(), name.data() + name.size(), descriptor);
        if (error == std::errc{} && end == name.data() + name.size()
            && descriptor > STDERR_FILENO && descriptor != directory_fd) {
            descriptors.push_back(descriptor);
        }
    }
    const int enumeration_error = errno;
    ::closedir(directory);
    for (const int descriptor : descriptors) {
        while (::close(descriptor) != 0 && errno == EINTR) {}
    }
    if (enumeration_error != 0) {
        return {.success = false,
                .detail = std::format("cannot enumerate inherited descriptors: {}",
                                      std::strerror(enumeration_error))};
    }
    return {.success = true};
}

} // namespace
#endif

LandrunResult sanitize_inherited_descriptors() {
#if defined(__linux__)
#ifdef __NR_close_range
    if (::syscall(__NR_close_range,
                  static_cast<unsigned int>(STDERR_FILENO + 1),
                  ~0U,
                  CLOSE_RANGE_UNSHARE) == 0) {
        return {.success = true};
    }
    if (errno != ENOSYS && errno != EINVAL) {
        return {.success = false,
                .detail = std::format("cannot close inherited descriptors: {}",
                                      std::strerror(errno))};
    }
#endif
    return close_from_descriptor_directory("/proc/self/fd");
#elif defined(__APPLE__)
    return close_from_descriptor_directory("/dev/fd");
#else
    // Unsupported platforms never enter secure mode.
    return {.success = true};
#endif
}

} // namespace core::landrun
