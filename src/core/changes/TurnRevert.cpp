#include "TurnRevert.hpp"

#include "StoredDiff.hpp"
#include "../utils/InterprocessFile.hpp"

#include <format>
#include <system_error>

namespace core::changes {
namespace {

namespace fs = std::filesystem;
using detail::absolute_in;
using detail::parse_hunks;
using detail::reverse_apply;
using core::tools::detail::join_diff_lines;

enum class ContentAction { Keep, Write, Remove };

/// What one file needs in order to be back where the turn found it.
struct Restoration {
    std::string display_path;
    /// Where the file belongs once the revert is done.
    fs::path origin;
    /// Where it is now, when that is somewhere else: a rename to undo.
    std::optional<fs::path> misplaced;
    ContentAction action = ContentAction::Keep;
    std::string content;  ///< Write only
};

/// The file whole, or the reason a revert gives for not reading it.
std::optional<std::string> read_whole(const fs::path& path, std::string& reason) {
    auto content = detail::read_whole(path);
    if (content) {
        return std::move(*content);
    }
    switch (content.error()) {
        case detail::ReadFailure::Missing:    reason = "was deleted since that turn"; break;
        case detail::ReadFailure::TooLarge:   reason = "grew past what a revert reads"; break;
        case detail::ReadFailure::Unreadable: reason = "could not be read"; break;
    }
    return std::nullopt;
}

[[nodiscard]] std::string_view undiffed_phrase(FileChangeContent content) noexcept {
    switch (content) {
        case FileChangeContent::Binary:
            return "is a binary file, whose previous content was never kept";
        case FileChangeContent::TooLarge:
            return "is too large for the diff a revert reads";
        case FileChangeContent::BudgetSpent:
            return "kept no diff, because that turn's diff budget was spent";
        case FileChangeContent::Text:
            break;
    }
    return "kept no diff";
}

/// Plans one file's part of the revert, or says why it cannot be honoured.
std::optional<Restoration> plan_restoration(const FileChange& change,
                                            const fs::path& root,
                                            std::string& reason) {
    const fs::path target = absolute_in(root, change.path);
    Restoration step{.display_path = change.path, .origin = target};
    std::error_code ec;
    const bool target_exists = fs::exists(target, ec);

    // An empty diff with text content means the two sides were byte-identical:
    // a rename that edited nothing, or a file the turn created or deleted
    // while it was empty. Each of those has an exact pre-image without a hunk.
    // Any other change without a diff (binary, too large, budget spent) has
    // none, and is refused rather than approximated.
    if (change.diff.empty()) {
        if (change.content != FileChangeContent::Text) {
            reason = std::string(undiffed_phrase(change.content));
            return std::nullopt;
        }
        switch (change.kind) {
            case FileChangeKind::Renamed: {
                if (change.previous_path.empty()) {
                    reason = "kept no record of where it was moved from";
                    return std::nullopt;
                }
                step.origin = absolute_in(root, change.previous_path);
                if (target_exists) {
                    step.misplaced = target;
                } else if (!fs::exists(step.origin, ec)) {
                    reason = "was deleted since that turn";
                    return std::nullopt;
                }
                return step;
            }
            case FileChangeKind::Added: {
                if (!target_exists) {
                    return step;  // Already gone; nothing to undo.
                }
                const auto current = read_whole(target, reason);
                if (!current) {
                    return std::nullopt;  // read_whole named the reason.
                }
                if (!current->empty()) {
                    reason = "changed since that turn";
                    return std::nullopt;
                }
                step.action = ContentAction::Remove;
                return step;
            }
            case FileChangeKind::Deleted: {
                if (target_exists) {
                    std::string unreadable;
                    const auto current = read_whole(target, unreadable);
                    if (!current || !current->empty()) {
                        reason = "was recreated with different content since that turn";
                        return std::nullopt;
                    }
                    return step;  // Already back; nothing to write.
                }
                step.action = ContentAction::Write;  // An empty file, restored.
                return step;
            }
            case FileChangeKind::Modified:
                break;
        }
        reason = std::string(undiffed_phrase(change.content));
        return std::nullopt;
    }

    const auto hunks = parse_hunks(change.diff);
    if (!hunks) {
        reason = "kept a diff this build cannot read";
        return std::nullopt;
    }

    switch (change.kind) {
        case FileChangeKind::Added: {
            const auto current = read_whole(target, reason);
            if (!current) {
                return std::nullopt;
            }
            const auto restored = reverse_apply(*current, *hunks);
            // The turn created this file, so undoing it leaves nothing. Text
            // still standing after the hunks went back means the file is not
            // only what the turn wrote.
            if (!restored || !restored->empty()) {
                reason = "changed since that turn";
                return std::nullopt;
            }
            step.action = ContentAction::Remove;
            return step;
        }
        case FileChangeKind::Deleted: {
            // Deleting a file diffs its whole content, so one hunk holds the
            // pre-image. More than one means gaps this revert would not see.
            if (hunks->size() != 1) {
                reason = "kept no diff of its whole previous content";
                return std::nullopt;
            }
            std::string content = join_diff_lines(hunks->front().before);
            std::error_code ec;
            if (fs::exists(target, ec)) {
                std::string unused;
                const auto current = read_whole(target, unused);
                if (!current || *current != content) {
                    reason = "was recreated with different content since that turn";
                    return std::nullopt;
                }
                return step;  // Already back; nothing to write.
            }
            step.action = ContentAction::Write;
            step.content = std::move(content);
            return step;
        }
        case FileChangeKind::Renamed: {
            if (change.previous_path.empty()) {
                reason = "kept no record of where it was moved from";
                return std::nullopt;
            }
            step.origin = absolute_in(root, change.previous_path);
            std::error_code ec;
            if (!fs::exists(target, ec)) {
                if (fs::exists(step.origin, ec)) {
                    return step;  // Already back; nothing to write.
                }
                reason = "was deleted since that turn";
                return std::nullopt;
            }
            if (step.origin != target && fs::exists(step.origin, ec)) {
                reason = std::format("cannot be moved back: '{}' exists again",
                                     change.previous_path);
                return std::nullopt;
            }
            const auto current = read_whole(target, reason);
            if (!current) {
                return std::nullopt;
            }
            const auto restored = reverse_apply(*current, *hunks);
            if (!restored) {
                reason = "changed since that turn";
                return std::nullopt;
            }
            step.misplaced = target;
            step.action = ContentAction::Write;
            step.content = std::move(*restored);
            return step;
        }
        case FileChangeKind::Modified: {
            const auto current = read_whole(target, reason);
            if (!current) {
                return std::nullopt;
            }
            const auto restored = reverse_apply(*current, *hunks);
            if (!restored) {
                reason = "changed since that turn";
                return std::nullopt;
            }
            step.action = ContentAction::Write;
            step.content = std::move(*restored);
            return step;
        }
    }
    reason = "is not a change a revert understands";
    return std::nullopt;
}

/// Writes one planned step. The rename goes first, so a file that has to move
/// back is written at the path it is restored to.
bool perform(const Restoration& step, std::string& error) {
    if (step.misplaced && *step.misplaced != step.origin) {
        std::error_code ec;
        if (const auto parent = step.origin.parent_path(); !parent.empty()) {
            fs::create_directories(parent, ec);
        }
        fs::rename(*step.misplaced, step.origin, ec);
        if (ec) {
            // A rename across filesystems fails; copy and remove, as the move
            // tool does, so a revert is not refused by a mount point.
            fs::copy_file(*step.misplaced, step.origin,
                          fs::copy_options::overwrite_existing, ec);
            if (ec) {
                error = std::format("could not be moved back: {}", ec.message());
                return false;
            }
            fs::remove(*step.misplaced, ec);
        }
    }
    switch (step.action) {
        case ContentAction::Keep:
            return true;
        case ContentAction::Remove: {
            std::error_code ec;
            fs::remove(step.origin, ec);
            if (ec) {
                error = std::format("could not be removed: {}", ec.message());
                return false;
            }
            return true;
        }
        case ContentAction::Write:
            // The same crash-safe write the rest of Filo uses: it keeps the
            // file's mode and never leaves a half-written source file behind.
            if (core::utils::atomic_write_file(step.origin, step.content, &error)) {
                return true;
            }
            error = std::format("could not be written: {}", error);
            return false;
    }
    return false;
}

} // namespace

RevertResult revert_turn(const TurnChanges& changes, const fs::path& workspace_root) {
    RevertResult result;
    std::vector<Restoration> plan;
    plan.reserve(changes.files.size());
    for (const auto& change : changes.files) {
        std::string reason;
        if (auto step = plan_restoration(change, workspace_root, reason)) {
            plan.push_back(std::move(*step));
            continue;
        }
        result.refused.push_back(RevertRefusal{.path = change.path, .reason = std::move(reason)});
    }
    // Nothing is written until every change is known to be honourable.
    if (!result.refused.empty()) {
        return result;
    }

    for (const auto& step : plan) {
        std::string error;
        if (!perform(step, error)) {
            result.error = std::format("{} {}.", step.display_path, error);
            return result;
        }
        result.restored.push_back(step.display_path);
    }
    return result;
}

} // namespace core::changes
