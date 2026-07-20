#pragma once

#include "LandrunPath.hpp"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace core::landrun {

enum class LandrunMode {
    off,
    workspace_write,
};

/**
 * Platform-neutral policy consumed by every landrun driver.
 *
 * Drivers receive outcomes, never platform flags: explicit read/write roots,
 * monotonic protected paths, and a network decision. This keeps policy
 * construction independent from Landlock and Seatbelt implementation details.
 */
struct LandrunPolicy {
    LandrunMode mode{LandrunMode::off};
    std::vector<std::filesystem::path> readable_roots;
    std::vector<std::filesystem::path> writable_roots;
    std::vector<std::filesystem::path> protected_read_paths;
    std::vector<std::filesystem::path> protected_write_paths;
    bool allow_network{false};

    [[nodiscard]] bool enabled() const noexcept {
        return mode != LandrunMode::off;
    }

    [[nodiscard]] friend bool operator==(const LandrunPolicy&,
                                         const LandrunPolicy&) = default;
};

inline void add_writable_root(LandrunPolicy& policy,
                              const std::filesystem::path& path)
{
    auto normalized = normalize_landrun_path(path);
    if (normalized.empty()) return;
    if (std::ranges::find(policy.writable_roots, normalized)
        == policy.writable_roots.end()) {
        policy.writable_roots.push_back(std::move(normalized));
    }
}

inline void add_readable_root(LandrunPolicy& policy,
                              const std::filesystem::path& path)
{
    auto normalized = normalize_landrun_path(path);
    if (normalized.empty()) return;
    if (std::ranges::find(policy.readable_roots, normalized)
        == policy.readable_roots.end()) {
        policy.readable_roots.push_back(std::move(normalized));
    }
}

inline void add_protected_read_path(LandrunPolicy& policy,
                                    const std::filesystem::path& path)
{
    auto normalized = normalize_landrun_path(path);
    if (normalized.empty()) return;
    if (std::ranges::find(policy.protected_read_paths, normalized)
        == policy.protected_read_paths.end()) {
        policy.protected_read_paths.push_back(std::move(normalized));
    }
}

inline void add_protected_write_path(LandrunPolicy& policy,
                                     const std::filesystem::path& path)
{
    auto normalized = normalize_landrun_path(path);
    if (normalized.empty()) return;
    if (std::ranges::find(policy.protected_write_paths, normalized)
        == policy.protected_write_paths.end()) {
        policy.protected_write_paths.push_back(std::move(normalized));
    }
}

} // namespace core::landrun
