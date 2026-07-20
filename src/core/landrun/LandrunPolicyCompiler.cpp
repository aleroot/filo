#include "LandrunPolicyCompiler.hpp"

#include "LandrunSettings.hpp"

#include <filesystem>

namespace core::landrun {

LandrunPolicy LandrunPolicyCompiler::compile(
    const core::workspace::SessionWorkspace& workspace,
    LandrunMode mode)
{
    LandrunPolicy policy{.mode = mode};
    if (!policy.enabled()) return policy;

    const auto& settings = LandrunSettings::instance();
    const auto excluded_paths = settings.excluded_paths();
    const auto excluded_root = [&](const std::filesystem::path& root) {
        const auto normalized = normalize_landrun_path(root);
        return std::ranges::any_of(excluded_paths, [&](const auto& excluded) {
            return is_landrun_path_within(excluded, normalized);
        });
    };

    const auto add_workspace = [&](const std::filesystem::path& root) {
        if (excluded_root(root)) return;
        add_readable_root(policy, root);
        add_writable_root(policy, root);
    };

    add_workspace(workspace.primary());
    for (const auto& root : workspace.additional()) {
        add_workspace(root);
    }

    const auto runtime_root = LandrunSettings::instance().runtime_root();
    add_readable_root(policy, runtime_root);
    add_writable_root(policy, runtime_root);

    const auto add_shared_temp = [&](const std::filesystem::path& root) {
        std::error_code ec;
        if (root.empty() || excluded_root(root)
            || !std::filesystem::is_directory(root, ec)) return;
        add_readable_root(policy, root);
        add_writable_root(policy, root);
    };
    add_shared_temp("/tmp");
    add_shared_temp(settings.host_tmpdir());

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

} // namespace core::landrun
