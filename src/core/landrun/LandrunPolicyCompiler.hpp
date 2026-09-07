#pragma once

#include "LandrunPolicy.hpp"
#include "../workspace/FileAccessScope.hpp"
#include "../workspace/SessionWorkspace.hpp"

#include <filesystem>
#include <utility>
#include <vector>

namespace core::landrun {

struct LandrunPolicyEnvironment {
    std::vector<std::filesystem::path> excluded_paths;
    std::filesystem::path runtime_root;
    std::filesystem::path host_tmpdir;
};

/**
 * Temp directories the sandbox grants the process tree, split by access.
 *
 * Single source of truth, deliberately: `LandrunPolicyCompiler::build()` uses
 * it to populate the kernel policy, and the composition root uses it to build
 * the session's FileAccessScope. Because both read the same function, the
 * shell's view and the native tools' view cannot drift apart -- a divergence
 * here previously let the shell create /tmp/output while `read` denied it.
 */
[[nodiscard]] core::workspace::FileAccessScope landrun_temp_scope(
    const LandrunPolicyEnvironment& environment,
    LandrunMode mode);

/** The active process-wide landrun environment, read from LandrunSettings. */
[[nodiscard]] LandrunPolicyEnvironment current_landrun_environment();

class LandrunPolicyCompiler {
public:
    explicit LandrunPolicyCompiler(LandrunPolicyEnvironment environment)
        : environment_(std::move(environment)) {}

    [[nodiscard]] LandrunPolicy build(
        const core::workspace::SessionWorkspace& workspace,
        LandrunMode mode) const;

    /** Legacy adapter for callers not yet receiving explicit startup dependencies. */
    [[nodiscard]] static LandrunPolicy compile(
        const core::workspace::SessionWorkspace& workspace,
        LandrunMode mode);

private:
    LandrunPolicyEnvironment environment_;
};

} // namespace core::landrun
