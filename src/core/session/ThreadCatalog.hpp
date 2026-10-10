#pragma once

// ThreadCatalog — pure, UI-agnostic helpers for first-class thread management.
//
// Threads and sessions deliberately have different lifecycles: a thread is a
// live in-process runtime, while SessionData is the persisted conversation it
// owns. This module only provides catalogue presentation for their shared
// SessionInfo projection: titles, previews, recency groups, filtering, and
// relative timestamps. Keeping it free of FTXUI/TUI dependencies makes it
// unit-testable and reusable from CLI (`--list-sessions`) and both TUI pickers.

#include "SessionStore.hpp"

#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::session {

/// Recency buckets used by the thread browser (agentty-compatible grouping).
enum class ThreadRecencyGroup {
    Today,
    Yesterday,
    LastWeek,
    LastMonth,
    Older,
};

[[nodiscard]] std::string_view to_string(ThreadRecencyGroup group) noexcept;

/// Collapse whitespace and truncate for single-line previews/titles.
[[nodiscard]] std::string collapse_preview(std::string_view text,
                                           std::size_t max_chars = 72);

/// Best-effort first non-synthetic user message text from a conversation.
[[nodiscard]] std::string first_user_message_preview(
    const std::vector<core::llm::Message>& messages,
    std::size_t max_chars = 72);

/// User-facing title: explicit name → first-user preview → short id fallback.
[[nodiscard]] std::string thread_display_title(const SessionInfo& info);

/// "provider/model", or the model alone when either part is missing.
[[nodiscard]] std::string thread_model_label(std::string_view provider, std::string_view model);

/// When the thread was last used: last_active_at, else created_at.
[[nodiscard]] const std::string& thread_activity_timestamp(const SessionInfo& info) noexcept;

/// Parse Filo's ISO-8601 timestamps (`YYYY-MM-DDTHH:MM:SSZ`).
[[nodiscard]] std::optional<std::chrono::system_clock::time_point>
parse_iso8601_utc(std::string_view iso);

/// Human relative time ("just now", "5m ago", "2h ago", "3d ago", or date).
[[nodiscard]] std::string format_relative_time(
    std::string_view iso_timestamp,
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now());

/// Compact local time: "14:05" today, "yesterday 14:05", "Tue 14:05" within
/// the week, "Oct 12" this year, "Oct 12 2025" before.
[[nodiscard]] std::string format_session_moment(
    std::string_view iso_timestamp,
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now());

/// Age by local calendar day, consistent with format_session_moment: "5m ago",
/// "3h ago" (same day), "yesterday", "4d ago", "3w ago", "5mo ago", "2y ago".
/// Unlike format_relative_time, "yesterday" always means the previous date.
[[nodiscard]] std::string format_session_age(
    std::string_view iso_timestamp,
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now());

/// Map an activity timestamp into a recency group.
[[nodiscard]] ThreadRecencyGroup thread_recency_group(
    std::string_view iso_timestamp,
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now());

struct ThreadGroup {
    ThreadRecencyGroup group = ThreadRecencyGroup::Older;
    std::string label;
    /// Indices into the original (filtered) sessions vector, newest first.
    std::vector<std::size_t> indices;
};

/// Case-insensitive filter over id, name, preview, working_dir, provider, model.
[[nodiscard]] std::vector<SessionInfo> filter_threads(
    const std::vector<SessionInfo>& sessions,
    std::string_view query);

/// Pin the current primary thread (queue head) first, then order the rest by
/// most recent activity. The id is runtime identity, not a display name.
void order_active_threads(std::vector<SessionInfo>& threads,
                          std::string_view primary_session_id);

/// Group sessions (already sorted newest-first) into recency buckets.
[[nodiscard]] std::vector<ThreadGroup> group_threads_by_recency(
    const std::vector<SessionInfo>& sessions,
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now());

/// True when @p session matches the currently active thread id.
[[nodiscard]] bool is_current_thread(const SessionInfo& info,
                                     std::string_view current_session_id) noexcept;

} // namespace core::session
