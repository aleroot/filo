#pragma once

#include "Workspace.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <system_error>
#include <vector>

namespace core::workspace {

class SessionWorkspace {
public:
    explicit SessionWorkspace(WorkspaceSnapshot snapshot)
        : snapshot_(normalize_snapshot(std::move(snapshot))) {}

    [[nodiscard]] const WorkspaceSnapshot& snapshot() const noexcept { return snapshot_; }
    [[nodiscard]] const std::filesystem::path& primary() const noexcept { return snapshot_.primary; }
    [[nodiscard]] const std::vector<std::filesystem::path>& additional() const noexcept {
        return snapshot_.additional;
    }
    [[nodiscard]] bool enforce() const noexcept { return snapshot_.enforce; }
    [[nodiscard]] std::uint64_t version() const noexcept { return snapshot_.version; }

    [[nodiscard]] std::filesystem::path resolve_path(
        const std::filesystem::path& target_path) const
    {
        if (target_path.empty()) {
            return target_path;
        }

        if (target_path.is_absolute()) {
            return normalize_path(target_path);
        }

        if (!snapshot_.primary.empty()) {
            return normalize_path(snapshot_.primary / target_path);
        }

        return normalize_path(target_path);
    }

    [[nodiscard]] bool is_path_allowed(const std::filesystem::path& target_path) const {
        if (!snapshot_.enforce) {
            return true;
        }

        if (snapshot_.primary.empty() && snapshot_.additional.empty()) {
            return false;
        }

        const auto resolved_target = resolve_path(target_path);
        if (!snapshot_.primary.empty() && is_subpath(snapshot_.primary, resolved_target)) {
            return true;
        }

        for (const auto& add : snapshot_.additional) {
            if (is_subpath(add, resolved_target)) {
                return true;
            }
        }

        return false;
    }

    // Extends an enforced workspace with existing absolute files or
    // directories. Validation, normalization, de-duplication, and versioning
    // live here so every caller observes the same workspace invariants.
    std::size_t add_additional_paths(
        const std::vector<std::filesystem::path>& paths) {
        std::size_t added = 0;
        for (const auto& path : paths) {
            if (path.empty() || !path.is_absolute()) {
                continue;
            }

            const auto normalized = normalize_path(path);
            std::error_code ec;
            const bool is_file = std::filesystem::is_regular_file(normalized, ec);
            ec.clear();
            const bool is_directory = std::filesystem::is_directory(normalized, ec);
            if ((!is_file && !is_directory) || is_path_allowed(normalized)) {
                continue;
            }

            if (is_directory) {
                std::erase_if(snapshot_.additional, [&](const auto& existing) {
                    return is_subpath(normalized, existing);
                });
            }
            snapshot_.additional.push_back(normalized);
            ++added;
        }

        if (added > 0) {
            ++snapshot_.version;
        }
        return added;
    }

    [[nodiscard]] static WorkspaceSnapshot normalize_snapshot(WorkspaceSnapshot snapshot) {
        snapshot.primary = snapshot.primary.empty()
            ? std::filesystem::path{}
            : normalize_path(snapshot.primary);

        for (auto& dir : snapshot.additional) {
            if (!dir.empty()) {
                dir = normalize_path(dir);
            }
        }

        snapshot.additional.erase(
            std::remove_if(
                snapshot.additional.begin(),
                snapshot.additional.end(),
                [](const auto& path) { return path.empty(); }),
            snapshot.additional.end());

        return snapshot;
    }

    [[nodiscard]] static std::filesystem::path normalize_path(const std::filesystem::path& path) {
        std::error_code ec;
        auto normalized = std::filesystem::weakly_canonical(path, ec);
        if (!ec) {
            return normalized.lexically_normal();
        }

        ec.clear();
        normalized = std::filesystem::absolute(path, ec);
        if (!ec) {
            return normalized.lexically_normal();
        }

        return path.lexically_normal();
    }

private:
    [[nodiscard]] static bool is_subpath(const std::filesystem::path& root,
                                         const std::filesystem::path& target) {
        const auto normalized_root = root.lexically_normal();
        const auto normalized_target = target.lexically_normal();

        auto root_it = normalized_root.begin();
        auto target_it = normalized_target.begin();
        while (root_it != normalized_root.end() && target_it != normalized_target.end()) {
            if (*root_it != *target_it) {
                return false;
            }
            ++root_it;
            ++target_it;
        }
        return root_it == normalized_root.end();
    }

    WorkspaceSnapshot snapshot_;
};

} // namespace core::workspace
