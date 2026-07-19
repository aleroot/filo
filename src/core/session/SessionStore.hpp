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
    std::string           name;
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
// Concurrency safety:
//   save()       — cross-process lock + unique fsynced temp file + atomic rename.
//   list/load    — read-only; safe for concurrent reads.
//   remove()     — serialized with save and rejects actively leased sessions.
// ---------------------------------------------------------------------------
class SessionStore {
public:
    explicit SessionStore(std::filesystem::path sessions_dir);

    // Serialize and atomically write session data under a cross-process lock.
    bool save(const SessionData& data, std::string* error = nullptr) const;

    // Load by 8-char hex session ID.
    [[nodiscard]] std::optional<SessionData> load_by_id(std::string_view session_id) const;

    // Load by user-assigned name (most recently active match wins).
    [[nodiscard]] std::optional<SessionData> load_by_name(std::string_view name) const;

    // Load by 1-based index from the sorted list (1 = most recent).
    [[nodiscard]] std::optional<SessionData> load_by_index(int index) const;

    // Load the most recent session.
    [[nodiscard]] std::optional<SessionData> load_most_recent() const;

    // Load the most recent session whose working_dir matches @p working_dir.
    // This is what powers `--continue`: it scopes to the current project so a
    // session from another repo is never silently resumed.
    [[nodiscard]] std::optional<SessionData> load_most_recent_for_project(
        std::string_view working_dir) const;

    // Unified load: tries integer index first, then session ID, then name.
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
