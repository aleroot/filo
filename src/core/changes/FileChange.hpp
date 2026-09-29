#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace core::changes {

enum class FileChangeKind {
    Added,
    Modified,
    Deleted,
    Renamed,
};

/// Whether `FileChange::diff` could be produced. A change is always listed;
/// only the line-level detail depends on this.
enum class FileChangeContent {
    Text,         ///< `diff`, `added` and `deleted` describe the whole change
    BudgetSpent,  ///< Text, but the turn's diff budget was spent; no diff
    TooLarge,     ///< Text, but this file is beyond the per-file diff budget
    Binary,       ///< Not text; no diff
};

/// The net effect of one agent turn on one file: its state when the turn first
/// touched it compared with its state when the turn ended. Edits that cancel
/// out do not appear at all.
struct FileChange {
    FileChangeKind kind = FileChangeKind::Modified;
    FileChangeContent content = FileChangeContent::Text;
    std::string path;           ///< Workspace-relative when inside the workspace
    std::string previous_path;  ///< Renamed only
    std::string diff;           ///< Unified diff; empty unless content == Text
    std::size_t added = 0;
    std::size_t deleted = 0;

    bool operator==(const FileChange&) const = default;
};

/// The net effect of one agent turn on the workspace.
///
/// `files` is everything the turn could be observed to do. The two flags name
/// each way it may still fall short of the truth, so a summary can state its
/// gap instead of implying a complete diff it does not have. Codex's tracker
/// follows the same rule from the other direction: it drops the whole turn
/// diff when a patch was not applied exactly.
struct TurnChanges {
    std::vector<FileChange> files;
    /// A directory in scope held more files than one turn enumerates, or the
    /// turn ran out of diff budget. Files may be missing from `files`, and
    /// those that are listed may carry no `diff`.
    bool partial_enumeration = false;
    /// A tool ran whose writes cannot be enumerated up front (shell, Python,
    /// MCP, subagents). Paths the turn already knew about were reconciled
    /// afterwards; paths such a tool created from nothing are absent.
    bool unscoped_mutations = false;

    [[nodiscard]] bool empty() const noexcept { return files.empty(); }
    [[nodiscard]] bool complete() const noexcept {
        return !partial_enumeration && !unscoped_mutations;
    }

    bool operator==(const TurnChanges&) const = default;
};

[[nodiscard]] constexpr std::string_view to_string(FileChangeKind kind) noexcept {
    switch (kind) {
        case FileChangeKind::Added:    return "added";
        case FileChangeKind::Modified: return "modified";
        case FileChangeKind::Deleted:  return "deleted";
        case FileChangeKind::Renamed:  return "renamed";
    }
    return "modified";
}

[[nodiscard]] constexpr std::string_view to_string(FileChangeContent content) noexcept {
    switch (content) {
        case FileChangeContent::Text:        return "text";
        case FileChangeContent::BudgetSpent: return "budget_spent";
        case FileChangeContent::TooLarge:    return "too_large";
        case FileChangeContent::Binary:      return "binary";
    }
    return "text";
}

} // namespace core::changes
