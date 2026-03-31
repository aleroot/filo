#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::history {

// -----------------------------------------------------------------------------
// PromptHistoryStore — persists command-line input history across sessions.
//
// Storage format: JSON array of strings, one per history entry.
// File location:  $XDG_DATA_HOME/filo/history.json
//                 (or ~/.local/share/filo/history.json)
//
// Features:
//   - Automatic deduplication (consecutive duplicates are collapsed)
//   - Size limit with FIFO eviction (default: 10,000 entries)
//   - Atomic writes (temp file → rename)
//   - Thread-safe save operations
// -----------------------------------------------------------------------------

struct PromptHistoryEntry {
    std::string text;
    std::string timestamp;  // ISO 8601 format
};

class PromptHistoryStore {
public:
    explicit PromptHistoryStore(std::filesystem::path history_file);

    // Load history from disk. Returns true on success (empty file is success).
    [[nodiscard]] bool load(std::string* error = nullptr);

    // Save history to disk atomically.
    [[nodiscard]] bool save(std::string* error = nullptr) const;

    // Add a new entry. Empty strings are ignored. Consecutive duplicates are collapsed.
    void add(std::string_view text);

    // Get all entries (oldest first).
    [[nodiscard]] const std::vector<std::string>& entries() const noexcept {
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
    [[nodiscard]] static std::string to_json(const std::vector<std::string>& entries);
    [[nodiscard]] static std::optional<std::vector<std::string>> from_json(std::string_view json);

    void enforce_size_limit();

    std::filesystem::path history_file_;
    std::vector<std::string> entries_;
    std::size_t max_entries_ = 10000;  // Default: keep last 10k entries
};

// -----------------------------------------------------------------------------
// PersistentPromptHistory — combines in-memory navigation with persistence.
//
// This is a drop-in replacement for the existing PromptHistory struct in
// MainApp.cpp, adding automatic persistence via PromptHistoryStore.
// -----------------------------------------------------------------------------

class PersistentPromptHistory {
public:
    explicit PersistentPromptHistory(std::shared_ptr<PromptHistoryStore> store);

    // Navigation: move to previous (older) entry. Returns true if navigation occurred.
    // Updates input and cursor position.
    [[nodiscard]] bool navigate_prev(std::string& input, int& cursor);

    // Navigation: move to next (newer) entry. Returns true if navigation occurred.
    // Updates input and cursor position.
    [[nodiscard]] bool navigate_next(std::string& input, int& cursor);

    // Save a new entry to history and persist to disk.
    void save(std::string_view text);

    // Reload from disk (useful after external modifications).
    void reload();

    // Clear all history (both memory and disk).
    void clear();

    // Get total number of entries.
    [[nodiscard]] std::size_t size() const noexcept;

    // Check if history is empty.
    [[nodiscard]] bool empty() const noexcept;

private:
    std::shared_ptr<PromptHistoryStore> store_;
    int idx_ = -1;                     // -1 = viewing current (unsaved) input
    std::string saved_input_;          // snapshot of current input when browsing starts
};

} // namespace core::history
