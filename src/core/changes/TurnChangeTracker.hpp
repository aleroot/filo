#pragma once

#include "FileChange.hpp"
#include "MutationScope.hpp"

#include <cstdint>
#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace core::changes {

/// Net file changes made during one agent turn.
///
/// The turn's own file tools report what they touch through `track`, which
/// observes the scope before and after the mutation. The first observation of
/// a file becomes its baseline; later observations only move its current
/// state. `changes()` then diffs baseline against current, so repeated edits
/// collapse into one diff and a file restored to its baseline disappears.
///
/// A scope the tracker cannot enumerate (shell, Python, MCP, subagents) is
/// still observed: every path the turn already knows is reconciled against
/// the filesystem, so a later untracked edit cannot leave an earlier diff
/// stale. Paths such a tool creates from nothing stay invisible, and the
/// summary says so rather than implying completeness.
///
/// Both resources a turn can consume without limit are bounded: how much
/// content a directory scan retains, and how much diff text the turn emits.
/// Exhausting either degrades files to a listed-without-detail entry and
/// marks the summary partial.
///
/// Thread-safe: tool calls with disjoint scopes may run concurrently.
class TurnChangeTracker {
public:
    /// Only the first `kMaxFilesPerDirectory` files of a directory in a scope
    /// are enumerated.
    static constexpr std::size_t kMaxFilesPerDirectory = 1000;
    /// File content one turn retains for diffing, across every file it saw.
    /// Past this a file is still listed and still detected as changed, only
    /// its detail is given up.
    static constexpr std::size_t kMaxTrackedBytesPerTurn = 16 * 1024 * 1024;
    /// Diff text one turn emits across all its files.
    static constexpr std::size_t kMaxDiffBytesPerTurn = 1024 * 1024;
    /// Largest file whose lines are still counted when its content is not
    /// retained.
    static constexpr std::size_t kMaxCountedFileBytes = 8 * 1024 * 1024;
    /// Distinct tool names one summary keeps as the reason its file list may be
    /// incomplete. Past this the list is short, and the summary still says so.
    static constexpr std::size_t kMaxUnscopedToolNames = 8;

    /// Resource bounds for one turn's summary. Exhausting any of them degrades
    /// detail and marks the summary partial; it never drops a file silently.
    struct Limits {
        std::size_t max_files_per_directory = kMaxFilesPerDirectory;
        std::size_t max_tracked_bytes = kMaxTrackedBytesPerTurn;
        std::size_t max_diff_bytes = kMaxDiffBytesPerTurn;
    };

    /// Files beneath `display_root` are reported relative to it.
    explicit TurnChangeTracker(std::filesystem::path display_root);
    /// Same, with the resource bounds overridden.
    TurnChangeTracker(std::filesystem::path display_root, Limits limits);

    /// Records that something wrote where no scope can say: a shell command, a
    /// script, a hook. Paths the turn already knows are reconciled, so a later
    /// diff cannot be left stale; paths created from nothing stay invisible,
    /// and the summary names `tool` as the reason.
    void note_unscoped_mutation(std::string tool);

    /// One more look before a report, for a turn that knows something wrote
    /// where it could not follow: a hook runs on a detached thread, so its
    /// edits can land after the last observation. A turn with nothing unscoped
    /// is left alone — reconciling one would pick up edits made by hand and
    /// blame them on the agent.
    void settle();

    /// Runs `mutate` between two observations of `scope`. The second
    /// observation also runs when `mutate` throws: it may have changed files.
    template <typename Mutation>
    auto track(const MutationScope& scope, Mutation&& mutate)
        -> std::invoke_result_t<Mutation&&> {
        if (scope.empty()) {
            return std::forward<Mutation>(mutate)();
        }
        observe(scope, Observation::Before);
        try {
            auto result = std::forward<Mutation>(mutate)();
            observe(scope, Observation::After);
            return result;
        } catch (...) {
            observe(scope, Observation::After);
            throw;
        }
    }

    /// Net changes so far, sorted by path, with the fidelity flags that say
    /// what the summary could not establish.
    [[nodiscard]] TurnChanges changes() const;

private:
    /// Where an observation sits relative to the mutation it describes. A path
    /// met for the first time after a mutation cannot be assumed to have
    /// existed before it — that assumption is what turned a directory moved
    /// onto an existing one into a report of deletions.
    enum class Observation { Before, After };

    struct Snapshot {
        enum class State { Missing, Text, Opaque };
        State state = State::Missing;
        std::string text;               ///< State::Text only
        FileChangeContent opaque_as = FileChangeContent::Binary;  ///< State::Opaque only
        std::uint64_t fingerprint = 0;  ///< State::Opaque only: identifies content
        /// Lines the file held. Known even when the bytes were not retained,
        /// so a change too large to diff can still be counted.
        std::optional<std::size_t> lines;
        /// Cheap size+mtime identity of the file this came from; 0 when absent.
        /// Lets an unscoped mutation be reconciled with one stat per known
        /// path instead of a read.
        std::uint64_t stamp = 0;

        /// Whether both snapshots hold the same bytes. `stamp` is deliberately
        /// excluded: a file rewritten with identical content is not a change.
        [[nodiscard]] bool same_content_as(const Snapshot& other) const {
            if (state != other.state) {
                return false;
            }
            switch (state) {
                case State::Missing: return true;
                case State::Text:    return text == other.text;
                case State::Opaque:  return opaque_as == other.opaque_as
                                        && fingerprint == other.fingerprint;
            }
            return false;
        }
    };

    using Key = std::string;  // canonical absolute path

    /// The files a scope stands for, and whether all of them fit the cap.
    struct Expansion {
        std::vector<Key> keys;
        bool complete = true;
    };

    /// What one look at a path tells us: enough to detect a change without
    /// reading it, and to decide whether reading it is affordable.
    struct Stat {
        bool exists = false;
        bool regular = false;
        std::uint64_t size = 0;
        std::uint64_t stamp = 0;  ///< size+mtime identity; 0 when unavailable
    };

    /// What one bounded read says about a file whose bytes are not kept.
    struct Scan {
        std::size_t lines = 0;
        bool text = true;
        /// Identity of the bytes that were read, so a file whose content is not
        /// retained can still be compared by content rather than by mtime.
        std::uint64_t fingerprint = 0;
    };

    /// Reads a file far enough to count its lines and tell text from binary,
    /// retaining nothing. `nullopt` when it cannot be read or is bigger than
    /// `kMaxCountedFileBytes`, where the read would cost more than the number.
    [[nodiscard]] static std::optional<Scan> scan_without_retaining(
        const std::filesystem::path& path);

    [[nodiscard]] static Stat stat_of(const std::filesystem::path& path);

    /// Reads one file, charging its content against the turn's budget.
    /// `reclaimable` is content already held for this path that the caller is
    /// about to release, so it is available to the new read.
    [[nodiscard]] Snapshot read_unlocked(const std::filesystem::path& path,
                                        std::size_t reclaimable = 0);
    /// Content held for `key` that a re-read would release.
    [[nodiscard]] std::size_t reclaimable_for_unlocked(const Key& key) const;
    /// Content the turn may still retain. Saturates at zero.
    [[nodiscard]] std::size_t remaining_bytes_unlocked() const;
    /// Replaces a path's current state, releasing the content it supersedes.
    void store_current_unlocked(const Key& key, Snapshot snapshot);
    [[nodiscard]] Key key_for(const std::filesystem::path& path) const;
    [[nodiscard]] Expansion expand_unlocked(const Key& root) const;
    void observe(const MutationScope& scope, Observation when);
    void note_unscoped_tool_unlocked(std::string_view tool);
    void reconcile_known_unlocked();
    void record_move_unlocked(const Key& source, const Key& destination);
    [[nodiscard]] std::string display_path(const Key& key) const;
    /// The most informative reason neither side of a change can be diffed.
    [[nodiscard]] static FileChangeContent no_diff_reason(const Snapshot& before,
                                                          const Snapshot& after);
    [[nodiscard]] FileChange describe(FileChangeKind kind,
                                      const Key& from,
                                      const Key& to,
                                      std::size_t& diff_budget) const;

    std::filesystem::path display_root_;
    Limits limits_;
    mutable std::mutex mutex_;
    std::map<Key, Snapshot> baseline_;
    std::map<Key, Snapshot> current_;
    std::map<Key, Key> origin_;  // destination -> baseline path it was moved from
    std::size_t tracked_bytes_ = 0;  ///< Text retained across baseline_ and current_
    bool partial_enumeration_ = false;
    bool unscoped_mutations_ = false;
    std::vector<std::string> unscoped_tools_;  ///< Distinct, in first-run order
};

} // namespace core::changes
