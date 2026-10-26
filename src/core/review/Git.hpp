#pragma once

#include "Types.hpp"

#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace core::review {

/// Read-only git queries against one repository.
///
/// Every query runs `git -C <repository>`, never in the process working
/// directory: the cwd is shared by every thread tab (and `/workspace change`
/// moves it), so it does not identify the repository a given session is
/// reviewing. Callers pass the session's own workspace root.
class GitClient {
public:
    explicit GitClient(std::filesystem::path repository);

    [[nodiscard]] bool is_repository(std::string* error_detail = nullptr) const;

    [[nodiscard]] std::optional<std::string> rev_parse(std::string_view ref) const;

    [[nodiscard]] std::optional<std::string>
    merge_base_with_head(std::string_view branch) const;

    [[nodiscard]] std::string uncommitted_base_ref() const;

    [[nodiscard]] std::expected<GitSnapshot, std::string>
    collect_uncommitted() const;

    [[nodiscard]] std::expected<GitSnapshot, std::string>
    collect_staged() const;

    [[nodiscard]] std::expected<GitSnapshot, std::string>
    collect_against_ref(std::string_view ref) const;

    [[nodiscard]] std::expected<GitSnapshot, std::string>
    collect_commit(std::string_view sha) const;

private:
    [[nodiscard]] std::optional<std::string>
    resolve_upstream_ref(std::string_view branch) const;

    [[nodiscard]] std::optional<std::string>
    build_untracked_patch(std::string_view paths, std::string& error_detail) const;

    [[nodiscard]] std::expected<GitSnapshot, std::string>
    collect_snapshot(std::string_view stat_args,
                     std::string_view diff_args,
                     bool include_untracked) const;

    std::filesystem::path repository_;
};

} // namespace core::review
