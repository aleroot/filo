#pragma once

#include "ToolRecoveryTypes.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace core::memory {

/**
 * Persistence boundary for tool-recovery lessons.
 *
 * This is one typed memory port — procedural lessons about tool contracts —
 * not a generic key-value store. The agent loop uses recall() when a call
 * fails with a known issue and record() when a failure is followed by a
 * corrected success. Implementations must be safe for concurrent use from
 * scheduler threads.
 */
class ToolRecoveryMemory {
public:
    virtual ~ToolRecoveryMemory() = default;

    /// Returns a lesson's hint only once it has reached the evidence
    /// threshold (proven, not merely observed once).
    [[nodiscard]] virtual std::optional<std::string> recall(
        const RecoveryKey& key) = 0;

    /// Proven runtime-failure lessons for a tool's current schema version.
    /// Runtime failures cannot blame a specific parameter at failure time, so
    /// recall is scoped per tool rather than per parameter.
    [[nodiscard]] virtual std::vector<std::string> recall_runtime_hints(
        const std::string& tool,
        const std::string& schema_fingerprint) = 0;

    /// Records one successful-recovery observation. Identical evidence
    /// strengthens the lesson; conflicting evidence resets it.
    virtual void record(const RecoveryLesson& lesson) = 0;
};

/// Inert implementation: forgets everything, recalls nothing. Used as the
/// default so the agent loop needs no null checks, and whenever the feature is
/// disabled by configuration.
class NullToolRecoveryMemory final : public ToolRecoveryMemory {
public:
    [[nodiscard]] std::optional<std::string> recall(
        const RecoveryKey&) override {
        return std::nullopt;
    }

    [[nodiscard]] std::vector<std::string> recall_runtime_hints(
        const std::string&, const std::string&) override {
        return {};
    }

    void record(const RecoveryLesson&) override {}
};

/**
 * Local, bounded, JSON-file-backed recovery memory.
 *
 * Semantics (ported from the proven Lampo tool-recovery store):
 *  - The first matching recovery creates a candidate (evidence 1).
 *  - A second identical recovery activates it for recall (evidence 2).
 *  - A conflicting hint for the same key resets evidence to 1.
 *  - The store is bounded; eviction removes the least proven (then least
 *    recently verified) lessons first.
 *  - Only parameter names and hint text are persisted — never argument
 *    values, tool results, or conversation content.
 *
 * Concurrency: every operation is a self-contained transaction that reads the
 * current on-disk state under an InterprocessFileLock (which also serializes
 * threads within this process). No state is cached between calls, so several
 * filo processes (and standalone TaskService callers that build their own
 * MemorySystem) merge their lessons instead of overwriting each other.
 * Agents that share one MemorySystem already share this port.
 * Operations run only on tool failure, so the I/O cost stays negligible.
 */
class FileToolRecoveryMemory final : public ToolRecoveryMemory {
public:
    /// On-disk schema version. Files written by a newer, incompatible version
    /// are ignored rather than misparsed.
    static constexpr int kVersion = 1;
    static constexpr std::size_t kMaxEntries = 64;
    static constexpr int kMinEvidence = 2;
    static constexpr int kMaxEvidence = 32;
    /// Upper bound on runtime hints merged into a single failure payload.
    static constexpr std::size_t kMaxRuntimeHints = 2;

    explicit FileToolRecoveryMemory(
        std::filesystem::path path = default_path());

    [[nodiscard]] static std::filesystem::path default_path();

    [[nodiscard]] std::optional<std::string> recall(
        const RecoveryKey& key) override;

    [[nodiscard]] std::vector<std::string> recall_runtime_hints(
        const std::string& tool,
        const std::string& schema_fingerprint) override;

    void record(const RecoveryLesson& lesson) override;

    /// Number of stored lessons, proven and candidate alike.
    [[nodiscard]] std::size_t entry_count() const;

    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return path_;
    }

private:
    struct Entry {
        RecoveryKey key;
        std::string hint;
        int evidence = 1;
        std::string updated_at;
    };

    enum class LoadStatus {
        Ok,
        /// File was written by a newer, incompatible layout. Must not be
        /// overwritten by this version.
        UnsupportedVersion,
    };

    struct LoadedStore {
        LoadStatus status = LoadStatus::Ok;
        std::vector<Entry> entries;
    };

    /// Reads current state. Caller must hold the file lock.
    [[nodiscard]] LoadedStore load_unlocked() const;
    /// Replaces stored state. Caller must hold the file lock.
    void save_unlocked(const std::vector<Entry>& entries) const;
    /// Drops the weakest lessons until the bound is satisfied.
    static void evict_overflow(std::vector<Entry>& entries);

    std::filesystem::path path_;
};

/// Builds the recovery memory an execution root should inject: the persistent
/// store when the feature is enabled, otherwise the inert null object.
[[nodiscard]] std::shared_ptr<ToolRecoveryMemory> make_tool_recovery_memory(
    bool enabled);

} // namespace core::memory
