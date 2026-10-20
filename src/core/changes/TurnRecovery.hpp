#pragma once

#include "FileChange.hpp"
#include "RecoveryStore.hpp"

#include <expected>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace core::changes {

class WorkspaceMutationGate;

// Durable pre-images are independent of presentation diffs. A manifest is
// committed before a tool starts, and again after it returns. On interruption
// its complete before-inventory can be compared with disk without replaying a
// command or relying on a tool's final status.
class TurnRecovery final {
public:
    static constexpr std::size_t kMaxFileBytes = RecoveryStore::kMaxFileBytes;
    static constexpr std::size_t kMaxBytes = 64 * 1024 * 1024;
    static constexpr std::size_t kMaxFiles = RecoveryStore::kMaxFiles;
    static constexpr std::size_t kRetainedTurns = RecoveryStore::kRetainedTurns;

    TurnRecovery(std::filesystem::path directory, std::filesystem::path workspace);
    void before(std::span<const std::filesystem::path> paths);
    void before_workspace(std::span<const std::filesystem::path> paths,
                          std::span<const std::filesystem::path> roots, bool complete);
    void after(std::span<const std::filesystem::path> paths,
               std::span<const std::filesystem::path> created = {});
    void process_group(int group);
    void finish();
    [[nodiscard]] const std::filesystem::path& directory() const noexcept { return store_.directory(); }
    [[nodiscard]] bool partial() const;
    /// The pre-image held for `path`, when this turn captured its bytes.
    [[nodiscard]] std::optional<RecoveryImage> captured(const std::filesystem::path& path) const;
    [[nodiscard]] std::expected<std::string, std::string> original(const RecoveryImage& image) const {
        return store_.original(image);
    }

    [[nodiscard]] bool can_observe(const std::filesystem::path& path) const { return !store_.is_internal(path); }
    [[nodiscard]] static std::expected<TurnChanges, std::string> reconcile(
        const std::filesystem::path& directory, const std::filesystem::path& workspace,
        WorkspaceMutationGate& mutations,
        const std::function<bool(const std::filesystem::path&)>& observable = {});
    [[nodiscard]] static std::expected<std::vector<TurnChanges>, std::string> pending(
        const std::filesystem::path& session_root, const std::filesystem::path& workspace,
        WorkspaceMutationGate& mutations,
        std::span<const std::string> attached = {},
        const std::function<bool(const std::filesystem::path&)>& observable = {});

private:
    TurnRecovery(RecoveryStore store, RecoveryManifest manifest);
    [[nodiscard]] std::expected<TurnChanges, std::string> reconcile(
        WorkspaceMutationGate& mutations,
        const std::function<bool(const std::filesystem::path&)>& observable);
    /// `canonical`: the paths are already normalized, as an inventory's are.
    void capture_unlocked(std::span<const std::filesystem::path> paths, bool canonical);
    /// Takes over the previous turn's entry for a file that has not changed
    /// since: its image is shared instead of being read and stored again.
    [[nodiscard]] bool adopt_unchanged_unlocked(const std::filesystem::path& path);
    /// The entries of the latest earlier turn of this session that shares
    /// this one's object store; empty when there is none.
    [[nodiscard]] RecoveryEntries previous_turn_entries() const;
    void record_absent_unlocked(std::span<const std::filesystem::path> paths);
    RecoveryStore store_;
    RecoveryManifest manifest_;
    std::size_t bytes_ = 0;  ///< Written by this turn; shared objects cost nothing.
    std::optional<RecoveryEntries> previous_entries_;  ///< Looked up on the first capture.
    bool manifest_saved_ = false;  ///< Must precede the first stored object.
    mutable std::mutex mutex_;
};

} // namespace core::changes
