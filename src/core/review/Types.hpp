#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::review {

// Budgets are character-based because Filo's review path precomputes the git
// patch rather than counting tokenizer tokens. They match the previous
// /review payload cap so a single huge file still fails closed instead of
// being silently truncated mid-hunk.
inline constexpr std::size_t kGroupPromptBudgetChars = 180'000;
inline constexpr double kSkipBudgetRatio = 0.80;
inline constexpr int kPlanFileLineThreshold = 50;
inline constexpr int kPlanGroupLineThreshold = 100;
inline constexpr int kTinyFileLineLimit = 20;
inline constexpr std::size_t kTinyFileCharLimit = 8'192;
inline constexpr std::size_t kMaxPackedFiles = 4;
inline constexpr int kMaxPackedLines = 100;
inline constexpr std::size_t kMaxPackedChars = 32'768;
inline constexpr int kReviewMaxOutputTokens = 8192;
inline constexpr int kPlanMaxOutputTokens = 2048;
// Matches the read-only subagent limit used by the rest of Filo. The engine
// still falls back to one worker when the provider cannot fork safely.
inline constexpr std::size_t kMaxParallelReviewGroups = 4;
// Review is intentionally bounded even when the interactive agent is not:
// one review unit must not be able to keep requesting tools forever.
inline constexpr int kReviewMaxStepsPerTurn = 8;
inline constexpr double kReviewRotationMinContextUtilization = 0.90;
inline constexpr double kMinFindingConfidence = 0.35;
// A finding whose line range misses every hunk by more than this is dropped
// as position drift. Within the slack it is snapped onto the nearest hunk.
inline constexpr int kLocalizationSnapSlackLines = 5;
inline constexpr std::string_view kGitEmptyTreeSha =
    "4b825dc642cb6eb9a060e54bf8d69288fbee4904";
inline constexpr std::string_view kStopMarker = "[Generation stopped by user]";

enum class FileChangeKind {
    Modified,
    Added,
    Deleted,
    Renamed,
};

enum class Severity {
    Critical,
    High,
    Medium,
    Low,
};

enum class Category {
    Bug,
    Security,
    Performance,
    Maintainability,
    Test,
    Style,
    Documentation,
    Other,
};

struct LineRange {
    int start = 0;
    int end = 0;

    [[nodiscard]] bool empty() const noexcept { return start <= 0 && end <= 0; }
};

struct FileChange {
    std::string path;
    std::string old_path;
    FileChangeKind kind = FileChangeKind::Modified;
    std::string patch;
    int added_lines = 0;
    int deleted_lines = 0;
    bool untracked = false;
    std::vector<LineRange> hunks;

    [[nodiscard]] int changed_lines() const noexcept {
        return added_lines + deleted_lines;
    }
};

struct ReviewGroup {
    std::vector<FileChange> files;
    bool needs_plan = false;

    [[nodiscard]] int changed_lines() const noexcept {
        int total = 0;
        for (const auto& file : files) {
            total += file.changed_lines();
        }
        return total;
    }

    [[nodiscard]] std::size_t patch_chars() const noexcept {
        std::size_t total = 0;
        for (const auto& file : files) {
            total += file.patch.size();
        }
        return total;
    }
};

struct ReviewPlan {
    std::vector<ReviewGroup> groups;
    std::vector<std::string> skipped_paths;
    std::vector<std::string> warnings;
};

struct GitSnapshot {
    std::string worktree_root;
    std::string status;
    std::string stat;
    std::string patch;
    std::string untracked_paths;
    std::string commit_header;
};

struct RiskItem {
    std::string title;
    std::string why;
    std::string path;
};

struct Finding {
    std::string title;
    std::string body;
    std::optional<int> priority;
    std::optional<Severity> severity;
    std::optional<Category> category;
    std::optional<double> confidence;
    std::string absolute_file_path;
    int line_start = 0;
    int line_end = 0;
};

struct FailedGroup {
    std::string label;
    std::string reason;
};

struct Report {
    std::vector<Finding> findings;
    std::string overall_correctness;
    std::string overall_explanation;
    std::optional<double> overall_confidence;
    std::vector<std::string> skipped_paths;
    std::vector<std::string> warnings;
    /// Units the model could not review. Reported next to the findings so a
    /// partial review is never mistaken for a clean one.
    std::vector<FailedGroup> failed_groups;
    int groups_reviewed = 0;
    int plan_passes = 0;
};

// Domain-level lifecycle events. The engine reports what it is doing; how it
// is displayed (status pill, transcript line, log) belongs to the adapter.
enum class ProgressPhase {
    Planned,
    GroupStarted,
    RiskPass,
    GroupFinished,
    /// One unit could not be reviewed (model/transport failure, or an answer
    /// that never became JSON). The campaign keeps going: one bad file must
    /// not throw away every other file's review.
    GroupFailed,
    Finished,
};

struct Progress {
    ProgressPhase phase = ProgressPhase::GroupStarted;
    std::size_t group_index = 0; ///< 1-based; 0 for Planned.
    std::size_t group_total = 0;
    std::string label;
    int files = 0;
    int changed_lines = 0;
    int findings = 0;
    int blocking_findings = 0;
    int skipped_files = 0;
    int risk_passes = 0;
    /// Planned/Finished: paths skipped because the diff exceeded the budget.
    std::vector<std::string> skipped_paths;
    /// Finished: units that failed but did not stop the campaign.
    int failed_groups = 0;
    /// Finished only: how the campaign ended.
    bool interrupted = false;
    /// GroupFailed: why this unit failed. Finished: why the campaign failed.
    std::string failure;
};

struct CampaignInput {
    std::string task;
    std::string user_facing_hint;
    GitSnapshot snapshot;
};

struct CampaignResult {
    Report report;
    bool interrupted = false;
    std::string error;
};

[[nodiscard]] constexpr std::string_view to_string(Severity severity) noexcept {
    switch (severity) {
        case Severity::Critical: return "critical";
        case Severity::High:     return "high";
        case Severity::Medium:   return "medium";
        case Severity::Low:      return "low";
    }
    return "medium";
}

[[nodiscard]] constexpr std::string_view to_string(Category category) noexcept {
    switch (category) {
        case Category::Bug:             return "bug";
        case Category::Security:        return "security";
        case Category::Performance:     return "performance";
        case Category::Maintainability: return "maintainability";
        case Category::Test:            return "test";
        case Category::Style:           return "style";
        case Category::Documentation:   return "documentation";
        case Category::Other:           return "other";
    }
    return "other";
}

[[nodiscard]] constexpr int severity_rank(Severity severity) noexcept {
    switch (severity) {
        case Severity::Critical: return 0;
        case Severity::High:     return 1;
        case Severity::Medium:   return 2;
        case Severity::Low:      return 3;
    }
    return 2;
}

[[nodiscard]] inline Severity severity_from_priority(int priority) noexcept {
    switch (priority) {
        case 0: return Severity::Critical;
        case 1: return Severity::High;
        case 2: return Severity::Medium;
        default: return Severity::Low;
    }
}

[[nodiscard]] inline std::optional<Severity> severity_from_string(std::string_view value) {
    if (value == "critical" || value == "p0" || value == "P0") return Severity::Critical;
    if (value == "high" || value == "p1" || value == "P1") return Severity::High;
    if (value == "medium" || value == "p2" || value == "P2") return Severity::Medium;
    if (value == "low" || value == "p3" || value == "P3") return Severity::Low;
    return std::nullopt;
}

[[nodiscard]] inline std::optional<Category> category_from_string(std::string_view value) {
    if (value == "bug") return Category::Bug;
    if (value == "security") return Category::Security;
    if (value == "performance") return Category::Performance;
    if (value == "maintainability") return Category::Maintainability;
    if (value == "test") return Category::Test;
    if (value == "style") return Category::Style;
    if (value == "documentation") return Category::Documentation;
    if (value == "other") return Category::Other;
    return std::nullopt;
}

[[nodiscard]] inline Severity effective_severity(const Finding& finding) noexcept {
    if (finding.severity.has_value()) {
        return *finding.severity;
    }
    if (finding.priority.has_value()) {
        return severity_from_priority(*finding.priority);
    }
    return Severity::Medium;
}

} // namespace core::review
