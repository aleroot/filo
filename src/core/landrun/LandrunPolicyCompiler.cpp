#include "LandrunPolicyCompiler.hpp"

#include "LandrunSettings.hpp"

#include <algorithm>
#include <filesystem>
#include <system_error>
#include <utility>
#include <vector>

namespace core::landrun {

core::workspace::FileAccessScope landrun_temp_scope(
    const LandrunPolicyEnvironment& environment,
    LandrunMode mode)
{
    if (!landrun_enabled(mode)) return {};

    const auto excluded = [&](const std::filesystem::path& root) {
        const auto normalized = normalize_landrun_path(root);
        return std::ranges::any_of(
            environment.excluded_paths,
            [&](const auto& entry) { return is_landrun_path_within(entry, normalized); });
    };

    std::vector<std::filesystem::path> readable;
    std::vector<std::filesystem::path> writable;

    // The sandbox's private root is always fully available: it holds the
    // synthetic HOME and TMPDIR the confined process tree actually uses.
    if (!environment.runtime_root.empty()) {
        readable.push_back(environment.runtime_root);
        writable.push_back(environment.runtime_root);
    }

    // Shared temp directories are readable in every enabled mode and writable
    // only when the mode permits workspace mutation.
    const auto add_shared_temp = [&](const std::filesystem::path& root) {
        std::error_code ec;
        if (root.empty() || excluded(root)
            || !std::filesystem::is_directory(root, ec)) return;
        readable.push_back(root);
        if (landrun_workspace_writable(mode)) {
            writable.push_back(root);
        }
    };
    add_shared_temp("/tmp");
    add_shared_temp(environment.host_tmpdir);

    return core::workspace::FileAccessScope(std::move(readable), std::move(writable));
}

LandrunPolicy LandrunPolicyCompiler::build(
    const core::workspace::SessionWorkspace& workspace,
    LandrunMode mode) const
{
    LandrunPolicy policy{.mode = mode};
    if (!policy.enabled()) return policy;

    const auto& excluded_paths = environment_.excluded_paths;
    const auto excluded_root = [&](const std::filesystem::path& root) {
        const auto normalized = normalize_landrun_path(root);
        return std::ranges::any_of(excluded_paths, [&](const auto& excluded) {
            return is_landrun_path_within(excluded, normalized);
        });
    };

    const auto add_workspace = [&](const std::filesystem::path& root) {
        if (excluded_root(root)) return;
        add_readable_root(policy, root);
        if (landrun_workspace_writable(mode)) {
            add_writable_root(policy, root);
        }
    };

    add_workspace(workspace.primary());
    for (const auto& root : workspace.additional()) {
        add_workspace(root);
    }

    // Temp grants come from the shared helper so the scope handed to native
    // tools is derived from the same computation, not a parallel guess.
    const auto temp_scope = landrun_temp_scope(environment_, mode);
    for (const auto& root : temp_scope.readable_roots()) {
        add_readable_root(policy, root);
    }
    for (const auto& root : temp_scope.writable_roots()) {
        add_writable_root(policy, root);
    }

    const auto add_system_root = [&](const std::filesystem::path& root) {
        std::error_code ec;
        if (!excluded_root(root) && std::filesystem::exists(root, ec)) {
            add_readable_root(policy, root);
        }
    };
#if defined(__APPLE__)
    constexpr std::array<const char*, 10> system_roots{
        "/System", "/usr", "/bin", "/sbin", "/Library", "/Applications/Xcode.app",
        "/opt/homebrew", "/opt/local", "/private/etc", "/dev"};
#else
    constexpr std::array<const char*, 9> system_roots{
        "/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/dev",
        "/nix/store", "/snap"};
#endif
    for (const char* root : system_roots) add_system_root(root);

    // Automatic sensitive-path visibility is enforced by native tools and by
    // backends that can express it (the macOS profile). Exact/ancestor user
    // exclusions remove a root grant above. Nested exclusions become monotonic
    // deny rules so macOS can subtract them; Linux fails closed rather than
    // silently ignoring an unrepresentable policy.
    for (const auto& excluded : excluded_paths) {
        const bool nested_in_readable = std::ranges::any_of(
            policy.readable_roots, [&](const auto& root) {
                return root != excluded
                    && is_landrun_path_within(root, excluded);
            });
        const bool nested_in_writable = std::ranges::any_of(
            policy.writable_roots, [&](const auto& root) {
                return root != excluded
                    && is_landrun_path_within(root, excluded);
            });
        if (nested_in_readable) add_protected_read_path(policy, excluded);
        if (nested_in_writable) add_protected_write_path(policy, excluded);
    }
    return policy;
}

LandrunPolicyEnvironment current_landrun_environment() {
    const auto& settings = LandrunSettings::instance();
    return LandrunPolicyEnvironment{
        .excluded_paths = settings.excluded_paths(),
        .runtime_root = settings.runtime_root(),
        .host_tmpdir = settings.host_tmpdir(),
    };
}

LandrunPolicy LandrunPolicyCompiler::compile(
    const core::workspace::SessionWorkspace& workspace,
    LandrunMode mode)
{
    return LandrunPolicyCompiler(current_landrun_environment()).build(workspace, mode);
}

} // namespace core::landrun
