#include "TurnRecovery.hpp"

#include "WorkspaceMutationGate.hpp"
#include "WorkspaceInventory.hpp"
#include "../tools/ToolDiffUtils.hpp"
#include "../utils/PathUtils.hpp"
#include "../utils/ProcessUtils.hpp"

#include <algorithm>
#include <chrono>
#include <ranges>
#include <stdexcept>

namespace core::changes {
namespace fs = std::filesystem;
using core::utils::path::normalized;
namespace {
std::string display(const fs::path& root, const fs::path& path) {
    const auto within = core::utils::path::relative_within(path, root);
    return within ? within->generic_string() : path.string();
}
}

TurnRecovery::TurnRecovery(fs::path directory, fs::path workspace, Clock clock)
    : store_(std::move(directory)), manifest_{.workspace = normalized(workspace)} {
    if (clock) clock_ = std::move(clock);
}

bool TurnRecovery::partial() const {
    std::lock_guard lock(mutex_);
    return manifest_.partial;
}

std::optional<RecoveryImage> TurnRecovery::captured(const fs::path& path) const {
    std::lock_guard lock(mutex_);
    auto it = manifest_.entries.find(path);
    if (it == manifest_.entries.end()) it = manifest_.entries.find(normalized(path));
    if (it == manifest_.entries.end() || !store_.has_object(it->second.before)) return std::nullopt;
    return it->second.before;
}

void TurnRecovery::before(std::span<const fs::path> paths) {
    std::lock_guard lock(mutex_);
    capture_unlocked(paths, false);
    if (!paths.empty()) store_.save(manifest_); // Retry an earlier failed commit, too.
}

void TurnRecovery::capture_unlocked(std::span<const fs::path> paths, bool canonical) {
    // Taken before any read, so a stamp is trusted only if the file had
    // settled before its bytes were looked at.
    const auto observed = clock_();
    bool stored = false;
    for (const auto& requested : paths) {
        const auto path = canonical ? requested : normalized(requested);
        if (manifest_.entries.contains(path) || manifest_.uncaptured.contains(path)) continue;
        if (manifest_.entries.size() + manifest_.uncaptured.size() >= kMaxFiles) {
            manifest_.partial = true;
            continue;
        }
        if (adopt_unchanged_unlocked(path)) continue;
        if (!manifest_saved_) {
            // A Recording manifest on disk before the first object: another
            // turn's `retain` then leaves the store alone.
            store_.save(manifest_);
            manifest_saved_ = true;
        }
        auto original = store_.store_original(path, kMaxBytes - bytes_);
        if (!original) {
            manifest_.uncaptured.insert(path);
            manifest_.partial = true;
            continue;
        }
        stored = true;
        bytes_ += original->stamp.size;
        manifest_.entries.emplace(path, RecoveryEntry{
            .before = std::move(original->image),
            .before_stamp = original->stamp.settled_identity(observed)});
    }
    // Before any manifest refers to them.
    if (stored) store_.sync_objects();
}

bool TurnRecovery::adopt_unchanged_unlocked(const fs::path& path) {
    if (!previous_entries_) previous_entries_ = previous_turn_entries();
    const auto previous = previous_entries_->find(path);
    if (previous == previous_entries_->end()) return false;
    const auto stamp = previous->second.latest_stamp();
    // A stamp is never 0 for a file it witnessed, and identity is 0 for
    // anything but a regular file, so equality also proves the file is there.
    if (stamp == 0 || core::utils::FileStamp::of(path).identity != stamp) return false;
    // after() identifies edited bytes without storing them. A matching stamp
    // proves what the file holds, not that its post-image is in the store.
    // Capture it normally if there is no object to share (also repairs missing
    // references forwarded by earlier versions).
    if (!store_.has_object(previous->second.latest())) return false;
    // The earlier turn's node moves over whole: no path or hash is copied.
    auto node = previous_entries_->extract(previous);
    auto& entry = node.mapped();
    entry = RecoveryEntry{.before = std::move(entry.after_known ? entry.after : entry.before), .before_stamp = stamp};
    manifest_.entries.insert(std::move(node));
    return true;
}

RecoveryEntries TurnRecovery::previous_turn_entries() const {
    // Turn directories sort by creation time: the latest earlier turn of this
    // session is the last one below this one.
    const auto& self = store_.directory();
    auto turns = RecoveryStore::turn_directories(self.parent_path()).value_or(std::vector<fs::path>{});
    std::erase_if(turns, [&](const fs::path& turn) { return turn.filename() >= self.filename(); });
    for (const auto& directory : turns | std::views::reverse) {
        const RecoveryStore earlier(directory);
        auto manifest = earlier.load(manifest_.workspace);
        if (!manifest) continue;
        // A turn with a store of its own shares nothing with this one.
        if (!earlier.shares_objects()) return {};
        return std::move(manifest->entries);
    }
    return {};
}

void TurnRecovery::after(std::span<const fs::path> paths, std::span<const fs::path> created) {
    std::lock_guard lock(mutex_);
    const auto observed = clock_();
    record_absent_unlocked(created);
    for (const auto& path : paths) {
        auto it = manifest_.entries.find(path);
        if (it == manifest_.entries.end()) it = manifest_.entries.find(normalized(path));
        if (it == manifest_.entries.end()) continue;
        auto& entry = it->second;
        const auto stamp = core::utils::FileStamp::of(it->first);
        std::error_code ec;
        if (entry.before.hash.empty() && stamp.kind == core::utils::FileStamp::Kind::Other &&
            fs::is_directory(fs::symlink_status(it->first, ec))) {
            manifest_.entries.erase(it);
            continue;
        }
        if (stamp.identity != 0 && stamp.identity == entry.latest_stamp()) {
            // Still the bytes it last held: nothing to read or hash.
            entry.after = entry.latest();
            entry.after_stamp = stamp.identity;
            entry.after_known = true;
            continue;
        }
        auto current = RecoveryStore::identify(it->first);
        if (!current) {
            manifest_.partial = true;
            entry.after_known = false;
            continue;
        }
        entry.after = std::move(current->image);
        entry.after_stamp = current->stamp.settled_identity(observed);
        entry.after_known = true;
    }
    if (!paths.empty() || !created.empty()) store_.save(manifest_);
}

void TurnRecovery::record_absent_unlocked(std::span<const fs::path> paths) {
    for (const auto& path : paths) {
        const auto key = normalized(path);
        if (manifest_.entries.contains(key) || manifest_.uncaptured.contains(key)) continue;
        if (manifest_.entries.size() + manifest_.uncaptured.size() >= kMaxFiles) {
            manifest_.partial = true;
            break;
        }
        manifest_.entries.try_emplace(key);
    }
}

void TurnRecovery::before_workspace(std::span<const fs::path> paths,
                                   std::span<const fs::path> roots, bool complete) {
    std::lock_guard lock(mutex_);
    capture_unlocked(paths, true);  // An inventory lists canonical paths.
    manifest_.roots.assign(roots.begin(), roots.end());
    manifest_.inventory_complete = complete;
    manifest_.partial |= !complete;
    store_.save(manifest_);
}

void TurnRecovery::process_group(int group) {
    if (group <= 1) return;
    std::lock_guard lock(mutex_);
    if (!std::ranges::contains(manifest_.groups, group)) {
        if (manifest_.groups.size() >= kMaxFiles) throw std::runtime_error("Process recovery limit reached before command dispatch.");
        manifest_.groups.push_back(group);
        store_.save(manifest_); // Before the shell receives the user command.
    }
}

void TurnRecovery::finish() {
    std::lock_guard lock(mutex_);
    if (manifest_.entries.empty()) return;
    manifest_.phase = RecoveryPhase::Complete;
    store_.save(manifest_);
    store_.retain(manifest_.workspace);
}

void TurnRecovery::mark_ambiguous() {
    store_.mark_ambiguous();
}

std::expected<TurnChanges, std::string> TurnRecovery::reconcile(
    const fs::path& directory, const fs::path& workspace, WorkspaceMutationGate& mutations,
    const std::function<bool(const fs::path&)>& observable) {
    RecoveryStore store(directory);
    auto loaded = store.load(workspace);
    if (!loaded) return std::unexpected(loaded.error());
    return TurnRecovery(std::move(store), std::move(*loaded)).reconcile(mutations, observable);
}

TurnRecovery::TurnRecovery(RecoveryStore store, RecoveryManifest manifest)
    : store_(std::move(store)), manifest_(std::move(manifest)) {}

std::expected<TurnChanges, std::string> TurnRecovery::reconcile(
    WorkspaceMutationGate& mutations, const std::function<bool(const fs::path&)>& observable) {
    std::vector<fs::path> roots;
    for (const auto& [path, _] : manifest_.entries) roots.push_back(path);
    auto lease = mutations.acquire(roots,
                                   std::chrono::milliseconds(0));
    if (!lease) return std::unexpected(WorkspaceMutationGate::failure_message(lease.error()));
    for (int group : manifest_.groups) {
        if (core::utils::process::group_may_exist(group))
            return std::unexpected("Interrupted work still has a possibly live process group; stop it before recovery.");
    }
    if (!manifest_.roots.empty()) {
        auto inventory = workspace_inventory(manifest_.roots, kMaxFiles, [&](const auto& path) {
            return can_observe(path) && (!observable || observable(path));
        });
        manifest_.partial |= !inventory.complete;
        if (manifest_.phase == RecoveryPhase::Recording && manifest_.inventory_complete) {
            record_absent_unlocked(inventory.paths);
        }
    }
    TurnChanges changes;
    changes.recovery_path = store_.directory().string();
    changes.partial_enumeration = manifest_.partial;
    changes.unscoped_mutations = true; // Interruption cannot establish exclusive authorship.
    changes.ambiguous_ownership = manifest_.ambiguous_ownership;
    changes.workspace_observed = !manifest_.roots.empty();
    std::size_t diff_budget = 1024 * 1024;
    for (auto& [path, entry] : manifest_.entries) {
        if (observable && !observable(path)) { changes.partial_enumeration = true; continue; }
        auto current = RecoveryStore::read(path);
        if (!current) return std::unexpected(current.error());
        if (manifest_.phase != RecoveryPhase::Recording && (!entry.after_known || current->image != entry.after))
            return std::unexpected("Files changed after the recovery checkpoint; review them before recovery.");
        entry.after = current->image;
        entry.after_known = true;
        if (entry.before == entry.after) continue;
        FileChange change{.kind = entry.before.hash.empty() ? FileChangeKind::Added
            : entry.after.hash.empty() ? FileChangeKind::Deleted : FileChangeKind::Modified,
            .path = display(manifest_.workspace, path)};
        auto original = store_.original(entry.before);
        if (!original) return std::unexpected(original.error());
        if (core::tools::detail::is_text_like_for_diff(*original) &&
            core::tools::detail::is_text_like_for_diff(current->bytes)) {
            auto diff = core::tools::detail::build_unified_diff_detailed(change.path, *original, current->bytes);
            if (diff.has_diff() && diff.diff->size() <= diff_budget) {
                change.diff = std::move(*diff.diff);
                diff_budget -= change.diff.size();
                if (diff.counts) { change.added = diff.counts->first; change.deleted = diff.counts->second; }
            } else change.content = FileChangeContent::TooLarge;
        } else change.content = FileChangeContent::Binary;
        changes.files.push_back(std::move(change));
    }
    manifest_.phase = RecoveryPhase::Interrupted;
    store_.save(manifest_);
    if (changes.files.empty() && !changes.partial_enumeration) store_.acknowledge();
    return changes;
}

std::expected<std::vector<TurnChanges>, std::string> TurnRecovery::pending(
    const fs::path& root, const fs::path& workspace, WorkspaceMutationGate& mutations,
    std::span<const std::string> attached,
    const std::function<bool(const fs::path&)>& observable) {
    std::error_code ec;
    if (root.empty() || !fs::exists(root, ec)) return std::vector<TurnChanges>{};
    const auto paths = RecoveryStore::turn_directories(root);
    if (!paths) return std::unexpected(paths.error());
    std::vector<TurnChanges> changes;
    for (const auto& path : *paths) {
        if (fs::exists(path / "attached", ec) || std::ranges::contains(attached, path.string())) continue;
        RecoveryStore store(path);
        auto loaded = store.load(workspace);
        if (!loaded) return std::unexpected(loaded.error());
        if (loaded->phase == RecoveryPhase::Reverted) continue;
        auto recovered = TurnRecovery(std::move(store), std::move(*loaded)).reconcile(mutations, observable);
        if (!recovered) return std::unexpected(recovered.error());
        if (!recovered->empty()) changes.push_back(std::move(*recovered));
    }
    return changes;
}

} // namespace core::changes
