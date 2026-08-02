#pragma once

#include "Workspace.hpp"

#include <algorithm>
#include <ranges>
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

    /**
     * True when tools may read @p target_path.
     *
     * Scope is the union of the project roots and the readable half of the
     * session scratch scope.
     */
    [[nodiscard]] bool allows_read(const std::filesystem::path& target_path) const {
        if (!snapshot_.enforce) {
            return true;
        }
        const auto resolved_target = resolve_path(target_path);
        return is_within_project_roots(resolved_target)
            || snapshot_.scratch.allows_read(resolved_target);
    }

    /**
     * True when tools may create, modify, or remove @p target_path.
     *
     * Project roots are writable whenever they are readable; whether a *tool*
     * may mutate at all is a separate question answered by the sandbox
     * capability check (see core::tools::landrun_allows_tool). Scratch, by
     * contrast, can be readable without being writable, because that is what
     * landrun's read-only mode actually grants the shell.
     */
    [[nodiscard]] bool allows_write(const std::filesystem::path& target_path) const {
        if (!snapshot_.enforce) {
            return true;
        }
        const auto resolved_target = resolve_path(target_path);
        return is_within_project_roots(resolved_target)
            || snapshot_.scratch.allows_write(resolved_target);
    }

    /**
     * True when @p target_path is in scope only because of the scratch scope.
     *
     * Lets callers tell "in scope because the user opened this project" apart
     * from "in scope because it is scratch", which matters for presentation and
     * for repo-relative logic that is meaningless outside a project root.
     */
    [[nodiscard]] bool is_scratch_path(const std::filesystem::path& target_path) const {
        const auto resolved_target = resolve_path(target_path);
        return !is_within_project_roots(resolved_target)
            && snapshot_.scratch.allows_read(resolved_target);
    }

    [[nodiscard]] const FileAccessScope& scratch() const noexcept {
        return snapshot_.scratch;
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
            if ((!is_file && !is_directory) || allows_read(normalized)) {
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

        // The scratch roots are normalized but *not* resolved through the
        // filesystem. Several of them (the sandbox's runtime root, the host
        // TMPDIR) may not exist yet when the snapshot is built, and
        // weakly_canonical() resolves a nonexistent path differently from a
        // probe taken later under that same root. Lexical normalization keeps
        // roots and probes on the same footing.
        snapshot.scratch.normalize();

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
    [[nodiscard]] bool is_within_project_roots(
        const std::filesystem::path& resolved_target) const {
        if (!snapshot_.primary.empty() && is_subpath(snapshot_.primary, resolved_target)) {
            return true;
        }
        return std::ranges::any_of(
            snapshot_.additional,
            [&](const auto& additional) { return is_subpath(additional, resolved_target); });
    }

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
