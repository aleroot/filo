#include "TurnRecovery.hpp"

#include "WorkspaceMutationGate.hpp"
#include "WorkspaceInventory.hpp"
#include "../tools/ToolDiffUtils.hpp"
#include "../utils/PathUtils.hpp"
#include "../utils/ProcessUtils.hpp"

#include <algorithm>
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

TurnRecovery::TurnRecovery(fs::path directory, fs::path workspace)
    : store_(std::move(directory)), manifest_{.workspace = normalized(workspace)} {}

bool TurnRecovery::partial() const {
    std::lock_guard lock(mutex_);
    return manifest_.partial;
}

void TurnRecovery::before(std::span<const fs::path> paths) {
    std::lock_guard lock(mutex_);
    capture_unlocked(paths);
    if (!paths.empty()) store_.save(manifest_); // Retry an earlier failed commit, too.
}

void TurnRecovery::capture_unlocked(std::span<const fs::path> paths) {
    for (const auto& requested : paths) {
        const auto path = normalized(requested);
        if (manifest_.entries.contains(path) || manifest_.uncaptured.contains(path)) continue;
        if (manifest_.entries.size() + manifest_.uncaptured.size() >= kMaxFiles) {
            manifest_.partial = true;
            continue;
        }
        auto original = RecoveryStore::read(path);
        if (!original || original->bytes.size() > kMaxBytes - bytes_) {
            manifest_.uncaptured.insert(path);
            manifest_.partial = true;
            continue;
        }
        store_.put_original(*original);
        bytes_ += original->bytes.size();
        manifest_.entries.emplace(path, RecoveryEntry{.before = original->image});
    }
}

void TurnRecovery::after(std::span<const fs::path> paths, std::span<const fs::path> created) {
    std::lock_guard lock(mutex_);
    record_absent_unlocked(created);
    for (const auto& path : paths) {
        const auto it = manifest_.entries.find(normalized(path));
        if (it == manifest_.entries.end()) continue;
        std::error_code ec;
        if (it->second.before.hash.empty() && fs::is_directory(fs::symlink_status(path, ec))) {
            manifest_.entries.erase(it);
            continue;
        }
        auto current = RecoveryStore::read(path);
        if (!current) {
            manifest_.partial = true;
            it->second.after_known = false;
            continue;
        }
        it->second.after = current->image;
        it->second.after_known = true;
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
    capture_unlocked(paths);
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
    auto roots = manifest_.roots;
    roots.push_back(manifest_.workspace);
    for (const auto& [path, _] : manifest_.entries) roots.push_back(path);
    auto lease = mutations.acquire(roots, WorkspaceMutationGate::Access::Exclusive,
                                   std::chrono::milliseconds(0));
    if (!lease) return std::unexpected(lease.error());
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
    std::vector<fs::path> paths;
    std::error_code ec;
    if (root.empty() || !fs::exists(root, ec)) return std::vector<TurnChanges>{};
    for (fs::directory_iterator it(root, ec), end; !ec && it != end; it.increment(ec)) {
        if (it->is_directory(ec) && !it->is_symlink(ec)) paths.push_back(normalized(it->path()));
    }
    if (ec) return std::unexpected(ec.message());
    std::ranges::sort(paths);
    std::vector<TurnChanges> changes;
    for (const auto& path : paths) {
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
