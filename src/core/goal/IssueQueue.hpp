#pragma once

#include "../review/MergePolicy.hpp"

#include <expected>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::goal {

struct IssueRecord {
    std::string number;
    std::string title;
    std::string url;
    std::string body;
    int priority = 3;
};

// A checkout the fixer and the reviewer share. The session checkout is not it.
// `base_ref` is the commit the branch was cut from.
struct PreparedWorkspace {
    std::filesystem::path path;
    std::string branch;
    std::string base_ref;
};

// One review of that checkout. `empty` means there was no diff to publish.
struct ReviewStep {
    core::review::ClassifiedFindings findings;
    bool empty = false;
};

// Host capabilities for one `/goal issues` run. Nothing here is stored:
// a later run asks the host again, and skips an issue that already has a PR.
struct IssueHooks {
    std::function<std::expected<PreparedWorkspace, std::string>(const IssueRecord&)> prepare;
    std::function<void(const PreparedWorkspace&)> release;
    std::function<std::expected<std::string, std::string>(
        const IssueRecord&, const PreparedWorkspace&, std::string_view retry)> fix;
    std::function<std::expected<ReviewStep, std::string>(const PreparedWorkspace&)> review;
    std::function<bool(const PreparedWorkspace&)> has_changes;
    std::function<std::expected<std::vector<std::string>, std::string>(
        const IssueRecord&, const core::review::ClassifiedFindings&)> file_follow_ups;
    std::function<bool(const IssueRecord&, const PreparedWorkspace&)> approve_publish;
    std::function<std::expected<std::string, std::string>(
        const IssueRecord&, const PreparedWorkspace&, std::string_view summary)> publish;
    /// URL of an open PR for this issue, or empty when there is none.
    std::function<std::string(const IssueRecord&)> existing_pr;
    std::function<bool()> cancelled;
    std::function<void(std::string_view)> on_event;
};

struct RunResult {
    int opened = 0;
    int blocked = 0;
    int skipped = 0;
    bool cancelled = false;
    std::string summary;
};

// Fixes `issues` one at a time. `max_attempts` is how many fix-and-review
// passes one issue gets before it is reported blocked (clamped to 1..8).
[[nodiscard]] RunResult run_issues(std::span<const IssueRecord> issues,
                                   int max_attempts,
                                   const IssueHooks& hooks);

} // namespace core::goal
