#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace core::history {

// -----------------------------------------------------------------------------
// PromptHistoryStore — persists command-line input history across sessions.
//
// Storage format: {"version":2,"entries":[text...],"meta":[{session,workspace}...]}.
//                 `entries` stays a plain string array so older builds keep
//                 reading (and, at worst, drop `meta` — scoping degrades, no
//                 prompt is lost). `meta` is parallel to `entries`; a length
//                 mismatch means an older build rewrote the file, so it is
//                 ignored as a whole.
// File location:  $XDG_DATA_HOME/filo/history.json
//                 (or ~/.local/share/filo/history.json)
//
// Features:
//   - Every entry remembers the session (thread) and workspace it was sent
//     from, so navigation can prefer the composer's own thread
//   - Consecutive duplicates from the same session are collapsed
//   - Size limit with FIFO eviction (default: 10,000 entries)
//   - Crash-safe atomic writes (unique temp file → fsync → rename)
//   - Cross-process-safe append and clear operations
// -----------------------------------------------------------------------------

struct PromptHistoryEntry {
    std::string text;
    std::string session_id = {};  // empty for legacy entries
    std::string workspace = {};   // primary workspace root; empty for legacy

    bool operator==(const PromptHistoryEntry&) const = default;
};

// Where a prompt is being typed: the visible thread's session and its primary
// workspace root. Empty fields match nothing.
struct PromptHistoryScope {
    std::string session_id;
    std::string workspace;
};

class PromptHistoryStore {
public:
    explicit PromptHistoryStore(std::filesystem::path history_file);

    // Load history from disk. Returns true on success (empty file is success).
    [[nodiscard]] bool load(std::string* error = nullptr);

    // Save history to disk atomically.
    [[nodiscard]] bool save(std::string* error = nullptr) const;

    // Reload, append one entry, and save as one cross-process transaction.
    [[nodiscard]] bool append_and_save(PromptHistoryEntry entry,
                                       std::string* error = nullptr);
    [[nodiscard]] bool append_and_save(std::string_view text,
                                       std::string* error = nullptr);

    // Reload, relabel every entry of session @p from as @p to, and save as one
    // cross-process transaction. A thread whose session id changes (/clear,
    // branch, rotation) keeps its own prompt history this way.
    [[nodiscard]] bool rename_session_and_save(std::string_view from,
                                               std::string_view to,
                                               std::string* error = nullptr);

    // Clear in-memory and on-disk history as one cross-process transaction.
    [[nodiscard]] bool clear_and_save(std::string* error = nullptr);

    // Reload, remove the entry at @p index (0-based, oldest first), and save
    // as one cross-process transaction. Returns false if @p index is out of
    // range or the save fails; sets @p error when provided.
    [[nodiscard]] bool remove_at_and_save(std::size_t index,
                                          std::string* error = nullptr);

    // Add a new entry. Empty text is ignored. A repeat of the newest entry
    // from the same session is collapsed.
    void add(PromptHistoryEntry entry);
    void add(std::string_view text);

    // Entry texts (oldest first).
    [[nodiscard]] std::vector<std::string> entries() const;

    // Entries with their scope (oldest first).
    [[nodiscard]] const std::vector<PromptHistoryEntry>& records() const noexcept {
        return entries_;
    }

    // Get entries in reverse order (newest first) for navigation.
    [[nodiscard]] std::vector<std::string> entries_newest_first() const;

    // Clear all history.
    void clear();

    // Set maximum number of entries to keep (0 = unlimited).
    void set_max_entries(std::size_t max_entries) noexcept {
        max_entries_ = max_entries;
    }

    [[nodiscard]] std::size_t max_entries() const noexcept {
        return max_entries_;
    }

    [[nodiscard]] std::size_t size() const noexcept {
        return entries_.size();
    }

    [[nodiscard]] bool empty() const noexcept {
        return entries_.empty();
    }

    // Get the history file path.
    [[nodiscard]] const std::filesystem::path& path() const noexcept {
        return history_file_;
    }

    // XDG-aware default history file path.
    [[nodiscard]] static std::filesystem::path default_history_path();

    // Get current UTC time formatted as ISO 8601.
    [[nodiscard]] static std::string now_iso8601();

private:
    [[nodiscard]] static std::string to_json(const std::vector<PromptHistoryEntry>& entries);
    [[nodiscard]] static std::optional<std::vector<PromptHistoryEntry>> from_json(
        std::string_view json);

    void enforce_size_limit();
    [[nodiscard]] bool load_unlocked(std::string* error = nullptr);
    [[nodiscard]] bool save_unlocked(std::string* error = nullptr) const;

    std::filesystem::path history_file_;
    std::vector<PromptHistoryEntry> entries_;
    std::size_t max_entries_ = 10000;  // Default: keep last 10k entries
};

// -----------------------------------------------------------------------------
// PersistentPromptHistory — Up/Down navigation over the store, per thread.
//
// Candidates for a scope, newest first and deduplicated by text:
//   1. prompts sent from this session (the visible thread),
//   2. prompts sent from other sessions in the same workspace,
//   3. everything else.
// Each session browses independently: switching tabs mid-browse and back
// resumes where it was, and Down still restores the draft typed before the
// first Up. The candidate list is snapshotted when browsing starts, so prompts
// sent elsewhere never shift what Up/Down show.
//
// Navigation never destroys typed text: once the composer no longer holds the
// recalled entry (the user edited it), Up starts a fresh browse that keeps the
// edited text as the draft, and Down does nothing.
//
// Not thread-safe except rename_session() and forget_session(), which only
// queue the change; the UI-thread calls apply it before touching the store.
// -----------------------------------------------------------------------------

class PersistentPromptHistory {
public:
    explicit PersistentPromptHistory(std::shared_ptr<PromptHistoryStore> store);

    // Move to the previous (older) entry. Returns true if navigation occurred.
    [[nodiscard]] bool navigate_prev(const PromptHistoryScope& scope,
                                     std::string& input, int& cursor);

    // Move to the next (newer) entry, ending on the saved draft. Returns true
    // if navigation occurred.
    [[nodiscard]] bool navigate_next(const PromptHistoryScope& scope,
                                     std::string& input, int& cursor);

    // Record a sent (or stashed) prompt for @p scope and persist it.
    void save(const PromptHistoryScope& scope, std::string_view text);

    // A thread's session id changed from @p from to @p to. Safe from any
    // thread: applied by the next UI-thread call.
    void rename_session(std::string from, std::string to);

    // A thread closed: drop its browsing state (its entries stay). Safe from
    // any thread: applied by the next UI-thread call.
    void forget_session(std::string session_id);

    // Reload from disk (after external modifications). Browsing state stays.
    void reload();

    // Clear all history (both memory and disk).
    void clear();

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] bool empty() const noexcept;

private:
    struct Navigation {
        std::vector<std::string> candidates;  // newest first
        int index = -1;                       // -1 = showing the draft
        std::string draft;                    // input when browsing started
        std::string shown;                    // text last placed in the input
    };

    [[nodiscard]] std::vector<std::string> candidates_for(
        const PromptHistoryScope& scope, std::string_view draft) const;
    void apply_pending_changes();

    std::shared_ptr<PromptHistoryStore> store_;
    std::unordered_map<std::string, Navigation> navigation_;  // by session id
    std::mutex pending_mutex_;
    std::vector<std::pair<std::string, std::string>> pending_renames_;
    std::vector<std::string> pending_forgets_;
};

} // namespace core::history
