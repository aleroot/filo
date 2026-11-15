#include "ClientPlatform.hpp"

#include <cstdlib>
#include <fstream>
#include <string>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/utsname.h>
#elif defined(__linux__)
#include <sys/utsname.h>
#endif

namespace core::utils {
namespace {

/// The architecture the binary was built for, in the spelling os_info uses
/// (codex-rs reports it in the same User-Agent).
[[nodiscard]] constexpr std::string_view build_architecture() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "aarch64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x86_64";
#elif defined(__arm__)
    return "arm";
#elif defined(__i386__) || defined(_M_IX86)
    return "x86";
#else
    return "unknown";
#endif
}

#if defined(__APPLE__)
/// The product version of macOS ("15.6.1"), not the kernel's.
[[nodiscard]] std::string macos_product_version() {
    char buffer[64] = {};
    std::size_t size = sizeof(buffer);
    if (::sysctlbyname("kern.osproductversion", buffer, &size, nullptr, 0) != 0) {
        return {};
    }
    return std::string(buffer, size > 0 ? size - 1 : 0);
}
#endif

#if defined(__linux__)
/// The distribution and its version from /etc/os-release, the way os_info
/// reads it.
[[nodiscard]] std::pair<std::string, std::string> linux_distribution() {
    std::ifstream release("/etc/os-release");
    std::string name;
    std::string version;
    std::string line;
    while (std::getline(release, line)) {
        const auto starts_with = [&](const char* key) {
            const std::string prefix = std::string(key) + "=";
            return line.rfind(prefix, 0) == 0;
        };
        if (name.empty() && starts_with("NAME")) {
            name = line.substr(std::string("NAME=").size());
        } else if (version.empty() && starts_with("VERSION_ID")) {
            version = line.substr(std::string("VERSION_ID=").size());
        }
        if (!name.empty() && !version.empty()) break;
    }
    // The values are quoted in the file.
    const auto unquote = [](std::string value) {
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
            return value.substr(1, value.size() - 2);
        }
        return value;
    };
    return {unquote(name), unquote(version)};
}
#endif

[[nodiscard]] ClientPlatform detect_platform() {
    ClientPlatform platform;
    platform.architecture = std::string(build_architecture());

#if defined(__APPLE__)
    platform.os = "macOS";
    platform.version = macos_product_version();
#elif defined(__linux__)
    auto [name, version] = linux_distribution();
    platform.os = name.empty() ? "Linux" : name;
    platform.version = version;
#endif

#if defined(__APPLE__) || defined(__linux__)
    if (platform.version.empty()) {
        struct utsname system {};
        if (::uname(&system) == 0) {
            platform.version = system.release;
        }
    }
#endif
    if (platform.os.empty()) platform.os = "unknown";
    if (platform.version.empty()) platform.version = "unknown";
    return platform;
}

} // namespace

const ClientPlatform& client_platform() {
    static const ClientPlatform platform = detect_platform();
    return platform;
}

std::string_view terminal_token() {
    static const std::string token = [] {
        const char* program = std::getenv("TERM_PROGRAM");
        if (program && *program) {
            std::string value(program);
            const char* version = std::getenv("TERM_PROGRAM_VERSION");
            if (version && *version) {
                value += '/';
                value += version;
            }
            return value;
        }
        const char* term = std::getenv("TERM");
        if (term && *term) return std::string(term);
        return std::string();
    }();
    return token;
}

} // namespace core::utils
