#pragma once

#include "Types.hpp"

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::review {

struct GrouperConfig {
    std::size_t prompt_budget_chars = kGroupPromptBudgetChars;
    double skip_budget_ratio = kSkipBudgetRatio;
    int plan_file_line_threshold = kPlanFileLineThreshold;
    int plan_group_line_threshold = kPlanGroupLineThreshold;
    int tiny_file_line_limit = kTinyFileLineLimit;
    std::size_t tiny_file_char_limit = kTinyFileCharLimit;
    std::size_t max_packed_files = kMaxPackedFiles;
    int max_packed_lines = kMaxPackedLines;
    std::size_t max_packed_chars = kMaxPackedChars;
};

[[nodiscard]] std::vector<FileChange> parse_unified_diff(std::string_view patch);

[[nodiscard]] ReviewPlan plan_review(std::span<const FileChange> files,
                                     const GrouperConfig& config = {});

[[nodiscard]] std::string absolute_review_path(std::string_view worktree_root,
                                               std::string_view relative_path);

/// Human-readable file list for one group, shared by prompts and progress.
[[nodiscard]] std::string join_group_paths(const ReviewGroup& group);

[[nodiscard]] std::string_view file_display_path(const FileChange& file) noexcept;

/// Family key used to bundle related files (foo.cpp + foo.hpp + test_foo.cpp).
[[nodiscard]] std::string file_family_key(std::string_view path);

/// New-file line ranges covered by unified-diff hunks in @p patch.
[[nodiscard]] std::vector<LineRange> parse_hunk_ranges(std::string_view patch);

/// Drop or snap findings whose location does not overlap the supplied diff.
/// Returns how many findings were removed as undrifted noise.
std::size_t localize_findings(std::vector<Finding>& findings,
                              std::span<const FileChange> files,
                              std::string_view worktree_root = {},
                              int snap_slack_lines = kLocalizationSnapSlackLines);

} // namespace core::review
