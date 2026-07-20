#include "LinuxLandlockDriver.hpp"
#include "LinuxSeccompHardening.hpp"

#if defined(__linux__)
#include <linux/landlock.h>
#include <sys/prctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <fcntl.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <format>
#include <utility>
#endif

namespace core::landrun {

std::string_view LinuxLandlockDriver::name() const noexcept {
    return "linux-landlock";
}

#if defined(__linux__)
namespace {

// ABI 3 mediates truncate; ABI 6 additionally scopes signals and abstract
// Unix sockets so a child cannot target Filo or unrelated user processes.
constexpr int kMinimumSafeAbi = 6;

[[nodiscard]] int landlock_abi() {
    return static_cast<int>(::syscall(
        __NR_landlock_create_ruleset, nullptr, 0,
        LANDLOCK_CREATE_RULESET_VERSION));
}

[[nodiscard]] std::uint64_t handled_rights_for_abi(int abi) {
    std::uint64_t rights =
        LANDLOCK_ACCESS_FS_EXECUTE |
        LANDLOCK_ACCESS_FS_WRITE_FILE |
        LANDLOCK_ACCESS_FS_READ_FILE |
        LANDLOCK_ACCESS_FS_READ_DIR |
        LANDLOCK_ACCESS_FS_REMOVE_DIR |
        LANDLOCK_ACCESS_FS_REMOVE_FILE |
        LANDLOCK_ACCESS_FS_MAKE_CHAR |
        LANDLOCK_ACCESS_FS_MAKE_DIR |
        LANDLOCK_ACCESS_FS_MAKE_REG |
        LANDLOCK_ACCESS_FS_MAKE_SOCK |
        LANDLOCK_ACCESS_FS_MAKE_FIFO |
        LANDLOCK_ACCESS_FS_MAKE_BLOCK |
        LANDLOCK_ACCESS_FS_MAKE_SYM;
#ifdef LANDLOCK_ACCESS_FS_REFER
    if (abi >= 2) rights |= LANDLOCK_ACCESS_FS_REFER;
#endif
#ifdef LANDLOCK_ACCESS_FS_TRUNCATE
    if (abi >= 3) rights |= LANDLOCK_ACCESS_FS_TRUNCATE;
#endif
    return rights;
}

[[nodiscard]] std::uint64_t read_only_rights() {
    return LANDLOCK_ACCESS_FS_EXECUTE |
           LANDLOCK_ACCESS_FS_READ_FILE |
           LANDLOCK_ACCESS_FS_READ_DIR;
}

[[nodiscard]] bool add_path_rule(int ruleset_fd,
                                 const std::filesystem::path& path,
                                 std::uint64_t allowed,
                                 std::string& error)
{
    const int path_fd = ::open(path.c_str(), O_PATH | O_CLOEXEC);
    if (path_fd < 0) {
        error = std::format("cannot open policy root '{}': {}",
                            path.string(), std::strerror(errno));
        return false;
    }
    landlock_path_beneath_attr rule{
        .allowed_access = allowed,
        .parent_fd = path_fd,
    };
    const int result = static_cast<int>(::syscall(
        __NR_landlock_add_rule, ruleset_fd, LANDLOCK_RULE_PATH_BENEATH,
        &rule, 0));
    const int saved_errno = errno;
    ::close(path_fd);
    if (result != 0) {
        error = std::format("cannot add policy root '{}': {}",
                            path.string(), std::strerror(saved_errno));
        return false;
    }
    return true;
}

} // namespace
#endif

LandrunProbe LinuxLandlockDriver::probe() const {
#if !defined(__linux__)
    return {.available = false,
            .backend = std::string(name()),
            .detail = "not running on Linux"};
#else
#if !defined(LANDLOCK_SCOPE_SIGNAL) || !defined(LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET)
    return {.available = false,
            .backend = std::string(name()),
            .detail = "build headers lack Landlock ABI 6 process scopes"};
#else
    errno = 0;
    const int abi = landlock_abi();
    if (abi < 0) {
        return {.available = false,
                .backend = std::string(name()),
                .detail = std::format("Landlock unavailable: {}", std::strerror(errno))};
    }
    if (abi < kMinimumSafeAbi) {
        return {.available = false,
                .backend = std::string(name()),
                .detail = std::format(
                    "Landlock ABI {} is below the secure minimum ABI 6", abi)};
    }
    return {.available = true,
            .backend = std::string(name()),
            .detail = std::format(
                "Landlock ABI {} + seccomp; policy-root reads and writes confined, "
                "workspace contents readable, unsupported metadata operations remain",
                abi)};
#endif
#endif
}

LandrunResult LinuxLandlockDriver::apply(const LandrunPolicy& policy) const {
#if !defined(__linux__)
    (void)policy;
    return {.success = false, .detail = "Landlock is only available on Linux"};
#else
#if !defined(LANDLOCK_SCOPE_SIGNAL) || !defined(LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET)
    (void)policy;
    return {.success = false,
            .detail = "build headers lack Landlock ABI 6 process scopes"};
#else
    if (!policy.enabled()) return {.success = true};
    if (policy.writable_roots.empty()) {
        return {.success = false,
                .detail = "workspace-write policy has no writable root"};
    }
    if (!policy.protected_read_paths.empty()
        || !policy.protected_write_paths.empty()) {
        return {.success = false,
                .detail = "Landlock cannot subtract protected paths from a granted workspace; "
                          "move sensitive files out of the workspace or use the strong Linux backend"};
    }
    const int abi = landlock_abi();
    if (abi < kMinimumSafeAbi) {
        return {.success = false,
                .detail = abi < 0
                    ? std::format("Landlock unavailable: {}", std::strerror(errno))
                    : std::format("Landlock ABI {} is below required ABI 6", abi)};
    }

    const std::uint64_t handled = handled_rights_for_abi(abi);
    landlock_ruleset_attr ruleset{.handled_access_fs = handled};
#if defined(LANDLOCK_SCOPE_SIGNAL) && defined(LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET)
    ruleset.scoped = LANDLOCK_SCOPE_SIGNAL | LANDLOCK_SCOPE_ABSTRACT_UNIX_SOCKET;
#endif
    const int ruleset_fd = static_cast<int>(::syscall(
        __NR_landlock_create_ruleset, &ruleset, sizeof(ruleset), 0));
    if (ruleset_fd < 0) {
        return {.success = false,
                .detail = std::format("cannot create Landlock ruleset: {}",
                                      std::strerror(errno))};
    }

    std::string error;
    bool ok = true;
    for (const auto& root : policy.readable_roots) {
        if (ok) ok = add_path_rule(ruleset_fd, root, read_only_rights(), error);
    }
    for (const auto& root : policy.writable_roots) {
        if (ok) ok = add_path_rule(ruleset_fd, root, handled, error);
    }
    if (!ok) {
        ::close(ruleset_fd);
        return {.success = false, .detail = std::move(error)};
    }
    if (::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        const auto detail = std::format("PR_SET_NO_NEW_PRIVS failed: {}",
                                        std::strerror(errno));
        ::close(ruleset_fd);
        return {.success = false, .detail = detail};
    }
    if (::syscall(__NR_landlock_restrict_self, ruleset_fd, 0) != 0) {
        const auto detail = std::format("landlock_restrict_self failed: {}",
                                        std::strerror(errno));
        ::close(ruleset_fd);
        return {.success = false, .detail = detail};
    }
    ::close(ruleset_fd);
    return apply_linux_seccomp_hardening(!policy.allow_network);
#endif
#endif
}

} // namespace core::landrun
