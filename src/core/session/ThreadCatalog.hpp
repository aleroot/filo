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

/// Parse Filo's ISO-8601 timestamps (`YYYY-MM-DDTHH:MM:SSZ`).
[[nodiscard]] std::optional<std::chrono::system_clock::time_point>
parse_iso8601_utc(std::string_view iso);

/// Human relative time ("just now", "5m ago", "2h ago", "3d ago", or date).
[[nodiscard]] std::string format_relative_time(
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

/// Pin the process's main thread first, then order all other live threads by
/// most recent activity. The main id is runtime identity, not a display name.
void order_active_threads(std::vector<SessionInfo>& threads,
                          std::string_view main_session_id);

/// Group sessions (already sorted newest-first) into recency buckets.
[[nodiscard]] std::vector<ThreadGroup> group_threads_by_recency(
    const std::vector<SessionInfo>& sessions,
    std::chrono::system_clock::time_point now = std::chrono::system_clock::now());

/// True when @p session matches the currently active thread id.
[[nodiscard]] bool is_current_thread(const SessionInfo& info,
                                     std::string_view current_session_id) noexcept;

} // namespace core::session
