#pragma once

#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::landrun {

enum class LandrunMode {
    off,
    read_only,
    workspace_write,
};

/**
 * Product-level capabilities associated with a sandbox mode.
 *
 * Keep mode meaning here rather than scattering enum comparisons throughout
 * policy compilation, tool dispatch, status rendering, and CLI parsing.
 */
struct LandrunModeDescriptor {
    LandrunMode mode;
    std::string_view cli_name;
    bool enabled;
    bool workspace_writable;
};

inline constexpr std::array<LandrunModeDescriptor, 3> kLandrunModes{{
    {LandrunMode::off, "off", false, true},
    {LandrunMode::read_only, "read-only", true, false},
    {LandrunMode::workspace_write, "workspace-write", true, true},
}};

[[nodiscard]] constexpr const LandrunModeDescriptor& describe_landrun_mode(
    LandrunMode mode) noexcept
{
    for (const auto& descriptor : kLandrunModes) {
        if (descriptor.mode == mode) return descriptor;
    }
    return kLandrunModes.front();
}

[[nodiscard]] constexpr bool landrun_enabled(LandrunMode mode) noexcept {
    return describe_landrun_mode(mode).enabled;
}

[[nodiscard]] constexpr bool landrun_workspace_writable(
    LandrunMode mode) noexcept
{
    return describe_landrun_mode(mode).workspace_writable;
}

[[nodiscard]] constexpr std::string_view landrun_mode_name(
    LandrunMode mode) noexcept
{
    return describe_landrun_mode(mode).cli_name;
}

[[nodiscard]] inline std::optional<LandrunMode> parse_landrun_mode(
    std::string_view name) noexcept
{
    for (const auto& descriptor : kLandrunModes) {
        if (descriptor.cli_name == name) return descriptor.mode;
    }
    return std::nullopt;
}

[[nodiscard]] inline std::vector<std::string> landrun_mode_names() {
    std::vector<std::string> names;
    names.reserve(kLandrunModes.size());
    for (const auto& descriptor : kLandrunModes) {
        names.emplace_back(descriptor.cli_name);
    }
    return names;
}

} // namespace core::landrun
