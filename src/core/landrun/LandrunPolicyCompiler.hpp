#pragma once

#include "LandrunPolicy.hpp"
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
