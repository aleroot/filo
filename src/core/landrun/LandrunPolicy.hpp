#pragma once

#include "LandrunMode.hpp"
#include "LandrunPath.hpp"

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace core::landrun {

/**
 * Platform-neutral policy consumed by every landrun driver.
 *
 * Drivers receive outcomes, never platform flags: explicit read/write roots,
 * monotonic protected paths, and a network decision. This keeps policy
 * construction independent from Landlock and Seatbelt implementation details.
 *
 * A policy carries two independent decisions:
 *
 *  - *Confinement* (`mode`): the user's `--sandbox` choice. It bounds what the
 *    process tree may read and write and denies child networking.
 *  - *Path protection* (`protected_*_paths`): individual files or directories
 *    subtracted from whatever is otherwise granted. Confinement contributes
 *    nested user exclusions; the session may add more (steering files the
 *    active policy withholds) without confining anything else.
 *
 * The kernel is engaged whenever either has something to enforce, so a
 * protection-only policy still launches through the helper and still reaches
 * the driver — it just leaves the rest of the host as it was.
 */
struct LandrunPolicy {
    LandrunMode mode{LandrunMode::off};
    std::vector<std::filesystem::path> readable_roots;
    std::vector<std::filesystem::path> writable_roots;
    std::vector<std::filesystem::path> protected_read_paths;
    std::vector<std::filesystem::path> protected_write_paths;
    bool allow_network{false};

    /// The user asked for a confined process tree (`--sandbox`).
    [[nodiscard]] bool confines() const noexcept {
        return landrun_enabled(mode);
    }

    /// Something is subtracted from the granted view.
    [[nodiscard]] bool protects_paths() const noexcept {
        return !protected_read_paths.empty() || !protected_write_paths.empty();
    }

    /// Anything at all for the kernel to enforce.
    [[nodiscard]] bool enabled() const noexcept {
        return confines() || protects_paths();
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
