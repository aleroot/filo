#include "TurnRevert.hpp"

#include "StoredDiff.hpp"
#include "WorkspaceMutationGate.hpp"
#include "RecoveryStore.hpp"
#include "../utils/InterprocessFile.hpp"
#include "../utils/PathUtils.hpp"
#include "../utils/ProcessUtils.hpp"


#include <format>
#include <map>
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

using FileState = std::optional<std::string>;
std::expected<FileState, std::string> file_state(const fs::path& path) {
    std::error_code ec;
    const auto status = fs::symlink_status(path, ec);
    if (status.type() == fs::file_type::not_found) return FileState{};
    if (ec || !fs::is_regular_file(status))
        return std::unexpected("is not a readable regular file");
    const auto bytes = detail::read_whole(path);
    if (!bytes) return std::unexpected("could not be read for restoration");
    return FileState{std::move(*bytes)};
}

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

// The caller already holds the exclusive workspace lease and has stopped shells.
RevertResult restore_checkpoint(const TurnChanges& changes, const fs::path& workspace) {
    RevertResult result;
    const auto fail = [&](std::string path, std::string reason) {
        result.refused.push_back({.path = std::move(path), .reason = std::move(reason)});
    };
    RecoveryStore store(changes.recovery_path);
    auto loaded = store.load(workspace);
    if (!loaded) { fail(changes.recovery_path, loaded.error()); return result; }
    auto& manifest = *loaded;
    if (manifest.phase == RecoveryPhase::Recording || manifest.phase == RecoveryPhase::Reverted) {
        fail(changes.recovery_path, "requires interruption reconciliation or was already reverted");
        return result;
    }
    for (int group : manifest.groups) {
        if (core::utils::process::group_may_exist(group)) {
            fail(changes.recovery_path, "still has a possibly live recorded process group");
            return result;
        }
    }
    for (const auto& change : changes.files) {
        const auto path = fs::path(change.path).is_absolute() ? fs::path(change.path) : workspace / change.path;
        if (!manifest.entries.contains(core::utils::path::normalized(path))) fail(change.path, "has no durable original");
    }
    if (!result.refused.empty()) return result;
    struct Step { fs::path path; RecoveryImage before; RecoveryImage after; std::string bytes; };
    std::vector<Step> plan;
    for (const auto& [path, entry] : manifest.entries) {
        if (!entry.after_known) { fail(path.string(), "has no verified post-image"); continue; }
        if (entry.before == entry.after) continue;
        auto current = RecoveryStore::read(path);
        if (!current || current->image != entry.after) {
            fail(path.string(), "changed since the recovery checkpoint"); continue;
        }
        auto bytes = store.original(entry.before);
        if (!bytes) { fail(path.string(), bytes.error()); continue; }
        plan.push_back({path, entry.before, entry.after, std::move(*bytes)});
    }
    if (!result.refused.empty()) return result;
    for (const auto& step : plan) {
        const auto current = RecoveryStore::read(step.path);
        if (!current || current->image != step.after) {
            result.error = step.path.string() + " changed during restoration.";
            return result;
        }
        std::string error;
        if (step.before.hash.empty()) {
            std::error_code ec;
            fs::remove(step.path, ec);
            if (ec) error = ec.message();
        } else {
            (void)core::utils::atomic_write_file(step.path, step.bytes, &error,
                                               static_cast<fs::perms>(step.before.mode));
        }
        if (!error.empty()) { result.error = std::move(error); return result; }
        const auto relative = core::utils::path::relative_within(step.path, manifest.workspace);
        result.restored.push_back(relative ? relative->generic_string() : step.path.string());
    }
    for (const auto& step : plan) {
        auto current = RecoveryStore::read(step.path);
        if (!current || current->image != step.before) {
            result.error = step.path.string() + " failed restoration verification.";
            return result;
        }
    }
    manifest.phase = RecoveryPhase::Reverted;
    try { store.save(manifest); } catch (const std::exception& error) { result.error = error.what(); }
    return result;
}

} // namespace

RevertResult revert_turn(const TurnChanges& changes, const fs::path& workspace_root, WorkspaceMutationGate& mutations,
                         const QuiesceWriters& quiesce_writers) {
    RevertResult result;
    if (changes.files.empty()) return result;
    std::vector<fs::path> roots{workspace_root};
    for (const auto& file : changes.files) {
        roots.push_back(absolute_in(workspace_root, file.path));
        if (!file.previous_path.empty()) roots.push_back(absolute_in(workspace_root, file.previous_path));
    }
    auto lease = mutations.acquire(
        roots, WorkspaceMutationGate::Access::Exclusive, std::chrono::milliseconds(0));
    if (!lease) {
        result.refused.push_back({.path = workspace_root.string(), .reason = "has an active writer"});
        return result;
    }
    if (quiesce_writers && !quiesce_writers(roots)) {
        result.refused.push_back({.path = workspace_root.string(), .reason = "still has a managed writer"});
        return result;
    }
    if (!changes.recovery_path.empty() && fs::exists(fs::path(changes.recovery_path) / "manifest.json"))
        return restore_checkpoint(changes, workspace_root);
    std::vector<Restoration> plan;
    std::map<fs::path, FileState> observed;
    plan.reserve(changes.files.size());
    for (const auto& change : changes.files) {
        for (const auto& path : {absolute_in(workspace_root, change.path),
                                absolute_in(workspace_root, change.previous_path.empty()
                                    ? change.path : change.previous_path)}) {
            const auto state = file_state(path);
            if (!state) {
                result.refused.push_back({.path = path.string(), .reason = state.error()});
                return result;
            }
            observed.try_emplace(path, *state);
        }
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

    for (const auto& [path, before] : observed) {
        const auto now = file_state(path);
        if (!now || *now != before) {
            result.refused.push_back({.path = path.string(), .reason = "changed while planning restoration"});
        }
    }
    if (!result.refused.empty()) return result;

    for (const auto& step : plan) {
        const auto target = step.misplaced.value_or(step.origin);
        const auto now = file_state(target);
        if (!now || *now != observed.at(target)) {
            result.error = step.display_path + " changed during restoration.";
            return result;
        }
        std::string error;
        if (!perform(step, error)) {
            result.error = std::format("{} {}.", step.display_path, error);
            return result;
        }
        result.restored.push_back(step.display_path);
    }
    for (const auto& step : plan) {
        const auto expected = step.action == ContentAction::Remove ? FileState{}
            : step.action == ContentAction::Write ? FileState{step.content}
            : observed.at(step.misplaced.value_or(step.origin));
        const auto actual = file_state(step.origin);
        if (!actual || *actual != expected || (step.misplaced && fs::exists(*step.misplaced))) {
            result.error = step.display_path + " did not match the restored contents.";
            return result;
        }
    }
    return result;
}

} // namespace core::changes
