#pragma once

#include "SessionData.hpp"
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::session {

/// Brief metadata for listing sessions without loading full message history.
struct SessionInfo {
    std::string           session_id;
    std::string           created_at;
    std::string           last_active_at;
    std::string           provider;
    std::string           model;
    std::string           mode;
    int32_t               turn_count = 0;
    std::filesystem::path path;
};

// ---------------------------------------------------------------------------
// SessionStore — persists sessions as JSON files under sessions_dir.
//
// File naming: session-YYYYMMDD-HHMMSS-<8hex-id>.json
//   (lexicographic sort equals chronological sort)
//
// Thread-safety:
//   save()       — atomic write (temp file → rename); safe to call from any thread.
//   list/load    — read-only; safe for concurrent reads.
//   remove()     — deletes one file; avoid concurrent remove + list.
// ---------------------------------------------------------------------------
class SessionStore {
public:
    explicit SessionStore(std::filesystem::path sessions_dir);

    // Serialize and atomically write session data (.tmp → rename to .json).
    bool save(const SessionData& data, std::string* error = nullptr) const;

    // Load by 8-char hex session ID.
    [[nodiscard]] std::optional<SessionData> load_by_id(std::string_view session_id) const;

    // Load by 1-based index from the sorted list (1 = most recent).
    [[nodiscard]] std::optional<SessionData> load_by_index(int index) const;

    // Load the most recent session.
    [[nodiscard]] std::optional<SessionData> load_most_recent() const;

    // Unified load: tries integer index first, then session ID.
    [[nodiscard]] std::optional<SessionData> load(std::string_view id_or_index) const;

    // List all sessions sorted by last-active time (most recent first).
    [[nodiscard]] std::vector<SessionInfo> list() const;

    // Delete a session file by ID (returns true even when not found).
    bool remove(std::string_view session_id, std::string* error = nullptr) const;

    // Compute the canonical file path for the given session data.
    [[nodiscard]] std::filesystem::path compute_path(const SessionData& data) const;

    // ── Static helpers ───────────────────────────────────────────────────────

    /// Generate a random 8-hex-character session ID.
    [[nodiscard]] static std::string generate_id();

    /// Current UTC time formatted as ISO 8601.
    [[nodiscard]] static std::string now_iso8601();

    /// XDG-aware default session storage directory.
    [[nodiscard]] static std::filesystem::path default_sessions_dir();

    [[nodiscard]] const std::filesystem::path& sessions_dir() const noexcept {
        return sessions_dir_;
    }

private:
    [[nodiscard]] static std::string    to_json(const SessionData& data);
    [[nodiscard]] static std::optional<SessionData> from_json(std::string_view json);
    bool ensure_dir(std::string* error = nullptr) const;

    std::filesystem::path sessions_dir_;
};

} // namespace core::session
