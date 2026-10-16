#pragma once

#include "FileChange.hpp"

#include <filesystem>
#include <span>
#include <string>
#include <vector>

namespace core::changes {

/// A file whose net change across the conversation could not be established.
/// `reason` reads as the predicate of "'<path>' …", like a RevertRefusal.
struct ConversationChangeGap {
    std::string path;    ///< Display path, as the file is named now
    std::string reason;
};

/// The net effect of a whole conversation on the workspace.
struct ConversationChanges {
    /// One entry per file, comparing its state before the first turn touched
    /// it with its state now. Edits that cancel out across turns do not appear.
    /// The flags are those any of the turns raised.
    TurnChanges changes;
    /// Files that are not in `changes` because their history could not be
    /// unwound, in the order `changes` would have listed them.
    std::vector<ConversationChangeGap> gaps;
};

/// Nets `turns` (oldest first) into one change per file.
///
/// A turn keeps diffs, not snapshots, so the state before the conversation is
/// recovered the way a revert would recover it: each file as it is now, with
/// every turn's diff unwound newest first. Nothing is written. Unwinding is as
/// strict as a revert — a diff whose after side no longer matches means the
/// file was edited outside the conversation over lines a turn wrote, and that
/// file becomes a gap rather than a comparison that silently folds someone
/// else's edit into it. An outside edit away from every turn's lines survives
/// unwinding, so it sits on both sides and the diff stays the conversation's.
///
/// A reverted turn is skipped: the workspace no longer holds its changes.
/// `workspace_root` resolves the display paths the turns report.
[[nodiscard]] ConversationChanges conversation_changes(
    std::span<const TurnChanges> turns,
    const std::filesystem::path& workspace_root);

} // namespace core::changes
