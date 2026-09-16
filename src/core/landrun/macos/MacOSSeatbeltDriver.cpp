#include "MacOSSeatbeltDriver.hpp"

#if defined(__APPLE__)
#include <dlfcn.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <format>
#include <string>
#include <string_view>
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

#if defined(__APPLE__)
namespace {

/**
 * SBPL parameters bound alongside a profile. Paths never enter the profile
 * text: each is passed as a named parameter so no quoting rule can be gamed.
 */
class SeatbeltParameters {
public:
    /// Registers @p path under a fresh `<prefix>_<n>` name and returns that name.
    [[nodiscard]] std::string bind(std::string_view prefix,
                                   const std::filesystem::path& path) {
        names_.push_back(std::format("{}_{}", prefix, names_.size()));
        values_.push_back(path.string());
        return names_.back();
    }

    /// NULL-terminated `name, value, name, value, ...` view for the SPI.
    [[nodiscard]] std::vector<const char*> argv() const {
        std::vector<const char*> parameters;
        parameters.reserve(names_.size() * 2 + 1);
        for (std::size_t i = 0; i < names_.size(); ++i) {
            parameters.push_back(names_[i].c_str());
            parameters.push_back(values_[i].c_str());
        }
        parameters.push_back(nullptr);
        return parameters;
    }

private:
    std::vector<std::string> names_;
    std::vector<std::string> values_;
};

/**
 * The confinement profile: default-deny, host readable, writes limited to the
 * roots the policy grants, networking off unless allowed. Path rules are
 * appended by append_path_rules() so both profiles share one binding scheme.
 */
[[nodiscard]] std::string confinement_profile(bool allow_network) {
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
    if (allow_network) {
        profile += "\n(allow network*)\n(allow system-socket)\n";
    }
    return profile;
}

/**
 * The protection-only profile: the host exactly as the user left it, minus
 * the protected paths. Nothing else is confined, so turning a steering file
 * off never also turns off networking or moves HOME.
 */
[[nodiscard]] std::string protection_profile() {
    return "(version 1)\n(allow default)\n";
}

/// Grants and subtractions, in that order: SBPL is last-match-wins, so the
/// denies must follow every allow they carve into.
void append_path_rules(std::string& profile,
                       SeatbeltParameters& parameters,
                       const LandrunPolicy& policy) {
    const auto rule = [&](std::string_view verb, std::string_view operation,
                          std::string_view prefix, const std::filesystem::path& path) {
        profile += std::format("({} {} (subpath (param \"{}\")))\n",
                               verb, operation, parameters.bind(prefix, path));
    };
    for (const auto& root : policy.readable_roots) {
        rule("allow", "file-read*", "READABLE_ROOT", root);
    }
    for (const auto& root : policy.writable_roots) {
        rule("allow", "file-write*", "WRITABLE_ROOT", root);
    }
    for (const auto& path : policy.protected_read_paths) {
        rule("deny", "file-read*", "PROTECTED_READ", path);
    }
    for (const auto& path : policy.protected_write_paths) {
        rule("deny", "file-write*", "PROTECTED_WRITE", path);
    }
}

} // namespace
#endif

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

    std::string profile = policy.confines()
        ? confinement_profile(policy.allow_network)
        : protection_profile();
    SeatbeltParameters parameters;
    append_path_rules(profile, parameters, policy);
    const auto parameter_argv = parameters.argv();

    char* error_buffer = nullptr;
    errno = 0;
    if (init(profile.c_str(), 0, parameter_argv.data(), &error_buffer) != 0) {
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
