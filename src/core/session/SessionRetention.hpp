#pragma once

#include <chrono>
#include <cstddef>
#include <optional>
#include <stop_token>
#include <string>
#include <vector>

namespace core::session {

class SessionStore;

/// Revert checkpoints are the bulk of what sessions store, and are rarely
/// wanted once a conversation has gone quiet.
inline constexpr std::chrono::days kDefaultRevertHistoryRetention{7};

/// How long stored data outlives its last use. nullopt keeps it.
struct RetentionPolicy {
    /// Saved conversations idle longer than this are deleted with all their
    /// data. Named (`/rename`d) conversations are always kept.
    std::optional<std::chrono::days> sessions;
    /// Revert checkpoints of conversations idle longer than this are deleted;
    /// the conversation stays and `/revert` falls back to its saved diffs.
    std::optional<std::chrono::days> revert_history = kDefaultRevertHistoryRetention;
};

/// Parses a `*_days` setting: "0" keeps forever, a positive count is that many
/// days, and an absent or malformed value yields @p fallback.
[[nodiscard]] std::optional<std::chrono::days> parse_retention_days(
    const std::optional<std::string>& value, std::optional<std::chrono::days> fallback);

struct RetentionReport {
    std::size_t sessions = 0;     ///< Conversations deleted.
    std::size_t checkpoints = 0;  ///< Conversations whose revert history was deleted.
    std::size_t leftovers = 0;    ///< Journals, tool results and lock files of no conversation.
    std::vector<std::string> errors;
    [[nodiscard]] bool empty() const noexcept {
        return sessions == 0 && checkpoints == 0 && leftovers == 0 && errors.empty();
    }
};

/// Deletes what @p policy no longer keeps from @p store's directory and the
/// tool results beside it. Anything a Filo process holds open (its session
/// lease) is skipped, so is anything whose age cannot be established, and
/// one pass runs at a time across processes. Returns between items once
/// @p stop is requested.
RetentionReport apply_retention(const SessionStore& store, const RetentionPolicy& policy,
                                std::chrono::system_clock::time_point now,
                                std::stop_token stop = {});

} // namespace core::session
