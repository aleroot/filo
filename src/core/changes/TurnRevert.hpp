#pragma once

#include "FileChange.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace core::changes {

/// One change a revert could not honour. `reason` reads as the predicate of
/// "'<path>' …", so a report can join several of them into one sentence.
struct RevertRefusal {
    std::string path;    ///< Display path, exactly as the summary shows it
    std::string reason;  ///< Why this file was left alone
};

/// The outcome of putting a turn's files back the way they were.
///
/// A revert is all or nothing: every change is planned against the workspace
/// first and nothing is written unless each one can be honoured. Half a turn
/// boundary is a tree that matches no state anyone asked for, so one file that
/// moved on refuses the lot and names itself.
struct RevertResult {
    /// Display paths that are back where the turn found them, in the order the
    /// plan covered them. A file that was already in place counts: the tree is
    /// in the state a revert is for.
    std::vector<std::string> restored;
    /// Every change that could not be honoured. Empty when the revert ran.
    std::vector<RevertRefusal> refused;
    /// A write that failed after the plan was accepted. `restored` then holds
    /// what did get written, because the workspace is partway back.
    std::optional<std::string> error;

    /// Whether the workspace is back to the state the turn started from.
    [[nodiscard]] bool reverted() const noexcept {
        return refused.empty() && !error.has_value() && !restored.empty();
    }
    [[nodiscard]] bool empty() const noexcept {
        return restored.empty() && refused.empty() && !error.has_value();
    }
};

/// Restores the files `changes` describes to the state they had before the turn
/// that produced them. `workspace_root` resolves the display paths a summary
/// reports; a change outside it carries an absolute path and needs no root.
///
/// Strict by construction: a file is written only while its content still
/// matches the turn's "after" side line for line, terminators included. There
/// is no fuzz, because a restore that guesses where a hunk belongs is a second,
/// silent edit — the one thing an undo must never be.
///
/// What it cannot do follows from what a summary keeps. A change whose diff
/// was never built (binary, too large, or past the turn's diff budget) has no
/// recorded pre-image, so it is refused rather than approximated; deleting the
/// empty directories a reverted add left behind is left to the reader, since a
/// directory the turn did not create is not the revert's to remove.
[[nodiscard]] RevertResult revert_turn(const TurnChanges& changes,
                                       const std::filesystem::path& workspace_root);

} // namespace core::changes
