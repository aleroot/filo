#pragma once

// ---------------------------------------------------------------------------
// HostTarget — the operating system and CPU architecture this binary was
// built for, and the names toolchains give them.
//
// The same target is spelled differently by different ecosystems: a client
// written in Go reports `darwin`/`arm64` (runtime.GOOS / runtime.GOARCH),
// one written in Rust reports `macos`/`aarch64` (std::env::consts). A client
// identity that imitates one of them must use that client's spelling, so the
// spelling is chosen where the value is formatted:
//
//     const auto target = core::utils::build_host_target();
//     std::format("os_type={:go}; arch={:go}", target.os, target.arch);
//     // -> "os_type=darwin; arch=arm64" on Apple silicon
//
// The naming is mandatory: `{}` or an unknown spec is a compile-time error
// for a constant format string, so no call site can pick one by accident.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <cstdint>
#include <format>
#include <string_view>

namespace core::utils {

enum class OperatingSystem : std::uint8_t { Darwin, Linux, Windows, FreeBsd, Unknown };

enum class Architecture : std::uint8_t { Arm64, X86_64, Arm, X86, Unknown };

/// How a toolchain spells an operating system or an architecture.
enum class TargetNaming : std::uint8_t {
    Go,   ///< runtime.GOOS / runtime.GOARCH: "darwin", "linux"; "arm64", "amd64".
    Rust, ///< std::env::consts::OS / ARCH: "macos", "linux"; "aarch64", "x86_64".
};

struct HostTarget {
    OperatingSystem os = OperatingSystem::Unknown;
    Architecture arch = Architecture::Unknown;
};

/// The target this binary was compiled for; it cannot change at run time.
[[nodiscard]] constexpr HostTarget build_host_target() noexcept {
    HostTarget target;
#if defined(__APPLE__)
    target.os = OperatingSystem::Darwin;
#elif defined(__linux__)
    target.os = OperatingSystem::Linux;
#elif defined(_WIN32)
    target.os = OperatingSystem::Windows;
#elif defined(__FreeBSD__)
    target.os = OperatingSystem::FreeBsd;
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
    target.arch = Architecture::Arm64;
#elif defined(__x86_64__) || defined(_M_X64)
    target.arch = Architecture::X86_64;
#elif defined(__arm__) || defined(_M_ARM)
    target.arch = Architecture::Arm;
#elif defined(__i386__) || defined(_M_IX86)
    target.arch = Architecture::X86;
#endif
    return target;
}

[[nodiscard]] constexpr std::string_view target_name(OperatingSystem os,
                                                     TargetNaming naming) noexcept {
    switch (os) {
        case OperatingSystem::Darwin:  return naming == TargetNaming::Go ? "darwin" : "macos";
        case OperatingSystem::Linux:   return "linux";
        case OperatingSystem::Windows: return "windows";
        case OperatingSystem::FreeBsd: return "freebsd";
        case OperatingSystem::Unknown: break;
    }
    return "unknown";
}

[[nodiscard]] constexpr std::string_view target_name(Architecture arch,
                                                     TargetNaming naming) noexcept {
    const bool go = naming == TargetNaming::Go;
    switch (arch) {
        case Architecture::Arm64:   return go ? "arm64" : "aarch64";
        case Architecture::X86_64:  return go ? "amd64" : "x86_64";
        case Architecture::Arm:     return "arm";
        case Architecture::X86:     return go ? "386" : "x86";
        case Architecture::Unknown: break;
    }
    return "unknown";
}

namespace detail {

/// Shared by the OperatingSystem and Architecture formatters: the format
/// spec names the toolchain spelling, exactly `go` or `rust`.
template <typename Value>
struct TargetNameFormatter {
    TargetNaming naming = TargetNaming::Go;

    constexpr auto parse(std::format_parse_context& ctx) {
        const auto end = std::find(ctx.begin(), ctx.end(), '}');
        const std::string_view spec(ctx.begin(), end);
        if (spec == "go") {
            naming = TargetNaming::Go;
        } else if (spec == "rust") {
            naming = TargetNaming::Rust;
        } else {
            throw std::format_error("a host target needs a naming: {:go} or {:rust}");
        }
        return end;
    }

    template <typename FormatContext>
    auto format(Value value, FormatContext& ctx) const {
        return std::ranges::copy(target_name(value, naming), ctx.out()).out;
    }
};

} // namespace detail
} // namespace core::utils

template <>
struct std::formatter<core::utils::OperatingSystem>
    : core::utils::detail::TargetNameFormatter<core::utils::OperatingSystem> {};

template <>
struct std::formatter<core::utils::Architecture>
    : core::utils::detail::TargetNameFormatter<core::utils::Architecture> {};
