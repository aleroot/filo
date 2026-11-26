#pragma once

#include "../persistence/AmendedFile.hpp"

#include "SessionData.hpp"
#include <chrono>
#include <filesystem>
#include <memory>
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
    std::string           working_dir; ///< project directory at session start
    std::string           provider;
    std::string           model;
    std::string           mode;
    /// Collapsed first user-message preview used as the thread title fallback.
    std::string           preview;
    int32_t               turn_count = 0;
    std::filesystem::path path;
};

// ---------------------------------------------------------------------------
// SessionStore — persists sessions as gzip-compressed JSON under sessions_dir.
//
// File naming: session-YYYYMMDD-HHMMSS-<8hex-id>.json
//   (lexicographic sort equals chronological sort). The name stays `.json`
//   so a session keeps one path, lease and lock whether it was saved before
//   compression (plain JSON, still read) or after; `gunzip -c < file` shows it.
//
// Incremental saves: a conversation grows by a turn at a time, so rewriting
//   the whole file at every save wrote the conversation over and over —
//   quadratic in its length. After its first save in a process, a store
//   records only what changed (core::persistence::AmendedFile): the session
//   without the messages it keeps unchanged, compressed.
//
// Concurrency safety:
//   save()       — cross-process lock; a durable record, or a whole snapshot
//                  through a unique fsynced temp file + atomic rename.
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

    // The load_* functions prepare this store to save what they return as an
    // append to the file. A caller that only reads a session (another
    // conversation it refers to) reads it with these instead.
    [[nodiscard]] std::optional<SessionData> read_by_id(std::string_view session_id) const;
    [[nodiscard]] std::optional<SessionData> read_by_name(std::string_view name) const;

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
    // Rows come from a stat-validated catalogue cache in sessions_dir, so
    // unchanged sessions are never re-read.
    [[nodiscard]] std::vector<SessionInfo> list() const;

    // Delete a session by ID with everything stored for it: journal, revert
    // checkpoints, offloaded tool results and lock files (returns true even
    // when not found).
    bool remove(std::string_view session_id, std::string* error = nullptr) const;

    /// Rewrites a session file saved before sessions were compressed in the
    /// compressed form, keeping its contents and modification time. False
    /// when there was nothing to do or it failed (then @p error says why).
    bool compress_saved(const std::filesystem::path& path, std::string* error = nullptr) const;

    // Compute the canonical file path for the given session data.
    [[nodiscard]] std::filesystem::path compute_path(const SessionData& data) const;

    /// Shared lossless message encoding for snapshots and execution journals.
    [[nodiscard]] static std::string message_json(const core::llm::Message& message);

    // ── Static helpers ───────────────────────────────────────────────────────

    /// Generate a random 8-hex-character session ID.
    [[nodiscard]] static std::string generate_id();

    /// Current UTC time formatted as ISO 8601.
    [[nodiscard]] static std::string now_iso8601();

    /// Format an arbitrary UTC time point as ISO 8601 (shared by now_iso8601
    /// and callers that persist in-memory timestamps, e.g. live-thread
    /// activity).
    [[nodiscard]] static std::string to_iso8601(
        std::chrono::system_clock::time_point when);

    /// Normalize a working-directory string into a canonical absolute path so
    /// that two logically-identical directories compare equal regardless of
    /// relative form, trailing separators, or symlinks. Falls back to a lexical
    /// normalization when the filesystem resolution fails (e.g. a deleted dir).
    /// An empty @p dir yields an empty path.
    [[nodiscard]] static std::filesystem::path canonicalize_working_dir(std::string_view dir);

    /// True when @p a and @p b resolve to the same project directory. Returns
    /// true when either side is empty (the working dir is unknown, e.g. for a
    /// legacy session written before working_dir was tracked) so callers never
    /// raise a spurious mismatch warning.
    [[nodiscard]] static bool working_dirs_match(std::string_view a, std::string_view b);

    /// Returns a human-readable warning when a resumed session's working
    /// directory differs from the process's current directory, or std::nullopt
    /// when they agree (or are unknown). Centralizes the message so every
    /// resume entry point (TUI startup, /resume, session picker, prompter)
    /// stays in sync — one place to change the wording.
    [[nodiscard]] static std::optional<std::string>
    working_dir_mismatch_notice(std::string_view session_dir, std::string_view current_dir);

    /// XDG-aware default session storage directory.
    [[nodiscard]] static std::filesystem::path default_sessions_dir();

    /// Where offloaded tool results of the sessions in @p sessions_dir live:
    /// `tool-results/<session-id>/` beside that directory.
    [[nodiscard]] static std::filesystem::path tool_results_dir(
        const std::filesystem::path& sessions_dir);

    /// The offloaded tool results directory of session @p session_id, or an
    /// empty path when that ID never names one.
    [[nodiscard]] static std::filesystem::path tool_results_dir(
        const std::filesystem::path& sessions_dir, std::string_view session_id);

    /// Where the media snapshots of the sessions in @p sessions_dir live
    /// (see core::llm::MediaSnapshots): `media/<session-id>/` beside it.
    [[nodiscard]] static std::filesystem::path media_dir(
        const std::filesystem::path& sessions_dir);

    /// The media snapshot directory of session @p session_id, or an empty
    /// path when that ID never names one.
    [[nodiscard]] static std::filesystem::path media_dir(
        const std::filesystem::path& sessions_dir, std::string_view session_id);

    /// The session ID a saved session's file name carries (`…-<id>.json`),
    /// or nullopt when @p file_name does not name a saved session.
    [[nodiscard]] static std::optional<std::string_view> session_id_from_filename(
        std::string_view file_name);

    [[nodiscard]] const std::filesystem::path& sessions_dir() const noexcept {
        return sessions_dir_;
    }

private:
    // Builds one catalogue row from a bounded read of the file (header, first
    // user message, legacy trailing stats) without decoding the conversation.
    [[nodiscard]] std::optional<SessionInfo> read_session_header(
        const std::filesystem::path& path) const;
    // The catalogue row of a file amended by a log, from a scan of its
    // snapshot and a replay of its (small) records; nullopt when that cannot
    // tell, and the caller decodes the whole session instead.
    [[nodiscard]] std::optional<SessionInfo> read_amended_header(
        const std::filesystem::path& path) const;
    [[nodiscard]] std::optional<SessionData> load_by_path(
        const std::filesystem::path& path,
        core::persistence::AmendedFile::Position* position = nullptr) const;
    /// Remembers that the file at @p path holds @p data as `read` left it at
    /// @p position, so the first save of a session opened through this store
    /// appends to it without reading the whole file again.
    void remember(const std::filesystem::path& path, const SessionData& data,
                  core::persistence::AmendedFile::Position position) const;
    /// load_by_path, then remember() what it read.
    [[nodiscard]] std::optional<SessionData> load_and_remember(
        const std::filesystem::path& path) const;
    [[nodiscard]] std::optional<SessionData> find_by_id(std::string_view session_id,
                                                        bool remember_for_save) const;
    [[nodiscard]] std::optional<SessionData> find_by_name(std::string_view name,
                                                          bool remember_for_save) const;

    [[nodiscard]] static std::string    to_json(const SessionData& data);
    /// @p keep, when given, receives how many saved messages a log record
    /// keeps; it is left alone when the document has none.
    [[nodiscard]] static std::optional<SessionData> from_json(std::string_view json, std::size_t* keep = nullptr);
    bool ensure_dir(std::string* error = nullptr) const;
    /// Appends what changed since this store's last save of the file at
    /// @p path; false when only a whole snapshot can commit @p data.
    [[nodiscard]] bool append(const std::filesystem::path& path, const SessionData& data,
                              const std::vector<std::string>& messages) const;

    struct Log;
    std::filesystem::path sessions_dir_;
    std::shared_ptr<Log> log_;  ///< Shared by copies: they save the same files.
};

} // namespace core::session
