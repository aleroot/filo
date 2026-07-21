#include "LandrunEnvironment.hpp"
#include "LandrunSettings.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace core::landrun {
namespace {

[[nodiscard]] std::string_view variable_name(std::string_view entry) {
    const auto equals = entry.find('=');
    return equals == std::string_view::npos ? entry : entry.substr(0, equals);
}

[[nodiscard]] bool is_safe_inherited_variable(std::string_view name) {
    if (name.starts_with("LC_")) return true;
    constexpr std::array<std::string_view, 26> allowed{
        "AR", "CC", "CFLAGS", "CLICOLOR", "CLICOLOR_FORCE", "COLORTERM",
        "CMAKE_GENERATOR", "CMAKE_PREFIX_PATH", "CXX", "CXXFLAGS",
        "DEVELOPER_DIR", "GOPATH", "LANG", "LDFLAGS",
        "MACOSX_DEPLOYMENT_TARGET", "MAKEFLAGS", "NINJA_STATUS", "NO_COLOR",
        "PATH", "PKG_CONFIG_PATH", "RUSTFLAGS", "SDKROOT", "SHELL", "TERM",
        "TZ", "USER"};
    return std::ranges::find(allowed, name) != allowed.end();
}

void append_override(std::vector<std::string>& result,
                     std::unordered_set<std::string>& names,
                     std::string name,
                     const std::filesystem::path& value) {
    if (value.empty()) return;
    names.insert(name);
    result.push_back(std::move(name) + "=" + value.string());
}

} // namespace

std::vector<std::string> build_landrun_environment(
    const LandrunPolicy& policy,
    char* const inherited_environment[])
{
    std::vector<std::string> result;
    std::unordered_set<std::string> names;
    for (char* const* entry = inherited_environment; entry && *entry; ++entry) {
        const std::string_view value(*entry);
        const auto name = variable_name(value);
        if (name == "HISTFILE" || name == "HISTSIZE") continue;
        if (policy.enabled() && !is_safe_inherited_variable(name)) continue;
        if (names.insert(std::string(name)).second) result.emplace_back(value);
    }

    if (!policy.enabled()) {
        result.emplace_back("HISTFILE=/dev/null");
        result.emplace_back("HISTSIZE=0");
        return result;
    }

    const auto root = LandrunSettings::instance().runtime_root();
#if defined(__APPLE__)
    if (!names.contains("PATH")) {
        names.insert("PATH");
        result.emplace_back(
            "PATH=/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin");
    }
#else
    if (!names.contains("PATH")) {
        names.insert("PATH");
        result.emplace_back("PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin");
    }
#endif
    append_override(result, names, "HOME", root / "home");
    const auto temp = LandrunSettings::instance().effective_tmpdir(policy.mode);
    append_override(result, names, "TMPDIR", temp);
    append_override(result, names, "TMP", temp);
    append_override(result, names, "TEMP", temp);
    append_override(result, names, "XDG_CACHE_HOME", root / "cache");
    append_override(result, names, "XDG_CONFIG_HOME", root / "config");
    append_override(result, names, "XDG_DATA_HOME", root / "data");
    append_override(result, names, "XDG_STATE_HOME", root / "state");
    append_override(result, names, "XDG_RUNTIME_DIR", root / "run");
    result.emplace_back("HISTFILE=/dev/null");
    result.emplace_back("HISTSIZE=0");
    result.emplace_back("FILO_LANDRUN=1");
    result.emplace_back(policy.allow_network
        ? "FILO_SANDBOX_NETWORK_DISABLED=0"
        : "FILO_SANDBOX_NETWORK_DISABLED=1");
    return result;
}

} // namespace core::landrun
