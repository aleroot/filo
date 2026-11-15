#include "ClientPlatform.hpp"

#include "AsciiUtils.hpp"

#include "core/version/Version.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <optional>
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

/// The environment value, when it is set and not only white space.
[[nodiscard]] std::optional<std::string> env_value(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw) return std::nullopt;
    std::string value(raw);
    const bool blank = std::ranges::all_of(value, [](unsigned char ch) {
        return ascii::is_space(ch);
    });
    if (blank) return std::nullopt;
    return value;
}

/// Whether the environment names the variable at all, empty or not.
[[nodiscard]] bool env_present(const char* name) {
    return std::getenv(name) != nullptr;
}

/// The characters a header value may carry (codex-rs terminal-detection
/// sanitize_header_value).
[[nodiscard]] bool is_header_value_char(char ch) noexcept {
    return ascii::is_alnum(static_cast<unsigned char>(ch))
        || ch == '-' || ch == '_' || ch == '.' || ch == '/';
}

[[nodiscard]] std::string sanitize_header_value(std::string value) {
    for (char& ch : value) {
        if (!is_header_value_char(ch)) ch = '_';
    }
    return value;
}

/// A terminal that names itself in the environment, with its version when the
/// environment names one.
struct NamedTerminal {
    std::string name;
    std::string version;
};

/// codex-rs terminal-detection detect_terminal_identity_from_env: a terminal
/// that names itself wins over the TERM capability string. The probes read
/// the environment only: they never run a helper, so an untrusted PATH cannot
/// decide what Filo reports.
[[nodiscard]] NamedTerminal detect_named_terminal() {
    const auto term = env_value("TERM");
    if (env_present("GHOSTTY_RESOURCES_DIR")) return {"Ghostty", {}};
    if (env_present("WEZTERM_VERSION")) {
        return {"WezTerm", env_value("WEZTERM_VERSION").value_or({})};
    }
    if (env_present("ITERM_SESSION_ID") || env_present("ITERM_PROFILE")
        || env_present("ITERM_PROFILE_NAME")) {
        return {"iTerm.app", {}};
    }
    if (env_present("KITTY_WINDOW_ID")
        || (term && term->find("kitty") != std::string::npos)) {
        return {"kitty", {}};
    }
    if (env_present("ALACRITTY_SOCKET")
        || (term && *term == "alacritty")) {
        return {"Alacritty", {}};
    }
    if (env_present("KONSOLE_VERSION")) {
        return {"Konsole", env_value("KONSOLE_VERSION").value_or({})};
    }
    if (env_present("GNOME_TERMINAL_SCREEN")) return {"gnome-terminal", {}};
    if (env_present("VTE_VERSION")) {
        return {"VTE", env_value("VTE_VERSION").value_or({})};
    }
    if (env_present("WT_SESSION")) return {"WindowsTerminal", {}};
    return {};
}

/// codex-rs terminal-detection user_agent_token: the program the terminal
/// sets (with its version), else the TERM capability string, else the name
/// the terminal is known by.
[[nodiscard]] std::string detect_terminal_token() {
    if (const auto program = env_value("TERM_PROGRAM");
        program && !ascii::iequals(*program, "tmux")) {
        const auto version = env_value("TERM_PROGRAM_VERSION");
        return sanitize_header_value(
            version ? *program + "/" + *version : *program);
    }

    const auto named = detect_named_terminal();
    if (!named.name.empty()) {
        return sanitize_header_value(
            named.version.empty() ? named.name : named.name + "/" + named.version);
    }
    if (const auto term = env_value("TERM")) {
        return sanitize_header_value(*term);
    }
    return "unknown";
}

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

std::string terminal_token() {
    return detect_terminal_token();
}

const std::string& client_user_agent() {
    // The shape the Codex CLI reports (login/src/auth/default_client.rs):
    // "{product}/{version} ({os} {os_version}; {architecture})".
    static const std::string value = [] {
        const auto& platform = client_platform();
        std::string agent = std::string(core::version::product_name) + "/"
            + std::string(core::version::value);
        agent += " (" + platform.os + " " + platform.version + "; "
            + platform.architecture + ")";
        return agent;
    }();
    return value;
}

} // namespace core::utils
