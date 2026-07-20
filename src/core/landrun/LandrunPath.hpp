#pragma once

#include <filesystem>
#include <system_error>

namespace core::landrun {

/** Canonicalizes the existing prefix and normalizes any nonexistent suffix. */
[[nodiscard]] inline std::filesystem::path normalize_landrun_path(
    const std::filesystem::path& path)
{
    if (path.empty()) return {};
    std::error_code ec;
    auto result = std::filesystem::weakly_canonical(path, ec);
    if (!ec) return result.lexically_normal();
    ec.clear();
    result = std::filesystem::absolute(path, ec);
    return ec ? path.lexically_normal() : result.lexically_normal();
}

/** Component-wise containment; unlike string prefixes, siblings never match. */
[[nodiscard]] inline bool is_landrun_path_within(
    const std::filesystem::path& root,
    const std::filesystem::path& target)
{
    const auto normalized_root = normalize_landrun_path(root);
    const auto normalized_target = normalize_landrun_path(target);
    if (normalized_root.empty() || normalized_target.empty()) return false;

    auto root_iterator = normalized_root.begin();
    auto target_iterator = normalized_target.begin();
    while (root_iterator != normalized_root.end()
           && target_iterator != normalized_target.end()) {
        if (*root_iterator++ != *target_iterator++) return false;
    }
    return root_iterator == normalized_root.end();
}

} // namespace core::landrun
