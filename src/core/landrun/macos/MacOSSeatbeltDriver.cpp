#include "MacOSSeatbeltDriver.hpp"

#if defined(__APPLE__)
#include <dlfcn.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <format>
#include <string>
#include <vector>
#endif

namespace core::landrun {

std::string_view MacOSSeatbeltDriver::name() const noexcept {
    return "macos-seatbelt-native";
}

LandrunProbe MacOSSeatbeltDriver::probe() const {
#if defined(__APPLE__)
    const auto symbol = ::dlsym(RTLD_DEFAULT, "sandbox_init_with_parameters");
    if (!symbol) {
        return {.available = false,
                .backend = std::string(name()),
                .detail = "sandbox_init_with_parameters is unavailable"};
    }
    return {.available = true,
            .backend = std::string(name()),
            .detail = "native Seatbelt SPI; host reads, constrained writes, and child networking denied by default"};
#else
    return {.available = false,
            .backend = std::string(name()),
            .detail = "not running on macOS"};
#endif
}

LandrunResult MacOSSeatbeltDriver::apply(const LandrunPolicy& policy) const {
#if !defined(__APPLE__)
    (void)policy;
    return {.success = false, .detail = "native Seatbelt is only available on macOS"};
#else
    using SandboxInitWithParameters = int (*)(
        const char*, std::uint64_t, const char* const[], char**);
    using SandboxFreeError = void (*)(char*);

    const auto init = reinterpret_cast<SandboxInitWithParameters>(
        ::dlsym(RTLD_DEFAULT, "sandbox_init_with_parameters"));
    const auto free_error = reinterpret_cast<SandboxFreeError>(
        ::dlsym(RTLD_DEFAULT, "sandbox_free_error"));
    if (!init) {
        return {.success = false,
                .detail = "sandbox_init_with_parameters is unavailable"};
    }
    if (!policy.enabled()) return {.success = true};
    if (policy.writable_roots.empty()) {
        return {.success = false,
                .detail = "workspace-write policy has no writable root"};
    }

    std::string profile = R"SBPL(
(version 1)
(deny default)
(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))
(allow file-read*)
(allow file-write* (literal "/dev/null"))
(allow sysctl-read)
(allow sysctl-write (sysctl-name "kern.grade_cputype"))
(allow iokit-open (iokit-registry-entry-class "RootDomainUserClient"))
(allow mach-lookup
  (global-name "com.apple.bsd.dirhelper")
  (global-name "com.apple.system.opendirectoryd.libinfo")
  (global-name "com.apple.system.opendirectoryd.membership")
  (global-name "com.apple.PowerManagement.control"))
(allow ipc-posix-sem)
(allow pseudo-tty)
(allow file-read* file-write* file-ioctl (literal "/dev/ptmx"))
(allow file-read* file-write*
  (require-all (regex #"^/dev/ttys[0-9]+")
               (extension "com.apple.sandbox.pty")))
(allow file-ioctl (regex #"^/dev/ttys[0-9]+"))
(deny file-read*
  (regex #"/\.env($|\.)")
  (regex #"/(\.npmrc|\.pypirc|\.netrc|\.mcp\.json|id_rsa|id_ed25519|id_ecdsa|id_dsa|credentials\.json|service-account\.json|secrets\.json|secrets\.yaml)$")
  (regex #"\.(pem|key|p12|pfx|kdbx|jks|keystore)$")
  (regex #"/\.(ssh|gnupg|aws|azure|kube)(/|$)"))
(deny file-write*
  (regex #"/(\.bashrc|\.bash_profile|\.zshrc|\.zprofile|\.profile|\.gitconfig|\.gitmodules|\.mcp\.json)$")
  (regex #"/\.git/(config|hooks)(/|$)")
  (regex #"/\.(filo|codex)(/|$)"))
)SBPL";
    if (policy.allow_network) {
        profile += "\n(allow network*)\n(allow system-socket)\n";
    }

    std::vector<std::string> names;
    std::vector<std::string> values;
    const auto parameter_count = policy.readable_roots.size()
        + policy.writable_roots.size()
        + policy.protected_read_paths.size()
        + policy.protected_write_paths.size();
    names.reserve(parameter_count);
    values.reserve(parameter_count);
    const auto append_parameter = [&](std::string_view prefix,
                                      const std::filesystem::path& path) {
        names.push_back(std::format("{}_{}", prefix, names.size()));
        values.push_back(path.string());
        return names.back();
    };
    for (const auto& root : policy.readable_roots) {
        const auto name = append_parameter("READABLE_ROOT", root);
        profile += std::format(
            "(allow file-read* (subpath (param \"{}\")))\n", name);
    }
    for (std::size_t i = 0; i < policy.writable_roots.size(); ++i) {
        const auto name = append_parameter("WRITABLE_ROOT", policy.writable_roots[i]);
        profile += std::format(
            "(allow file-write* (subpath (param \"{}\")))\n", name);
    }
    for (const auto& path : policy.protected_read_paths) {
        const auto name = append_parameter("PROTECTED_READ", path);
        profile += std::format(
            "(deny file-read* (subpath (param \"{}\")))\n", name);
    }
    for (const auto& path : policy.protected_write_paths) {
        const auto name = append_parameter("PROTECTED_WRITE", path);
        profile += std::format(
            "(deny file-write* (subpath (param \"{}\")))\n", name);
    }

    std::vector<const char*> parameters;
    parameters.reserve(names.size() * 2 + 1);
    for (std::size_t i = 0; i < names.size(); ++i) {
        parameters.push_back(names[i].c_str());
        parameters.push_back(values[i].c_str());
    }
    parameters.push_back(nullptr);

    char* error_buffer = nullptr;
    errno = 0;
    if (init(profile.c_str(), 0, parameters.data(), &error_buffer) != 0) {
        std::string detail = error_buffer && *error_buffer
            ? std::string(error_buffer)
            : std::strerror(errno);
        if (error_buffer && free_error) free_error(error_buffer);
        return {.success = false,
                .detail = std::format("Seatbelt initialization failed: {}", detail)};
    }
    return {.success = true};
#endif
}

} // namespace core::landrun
