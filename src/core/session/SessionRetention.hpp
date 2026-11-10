#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
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
    std::size_t compressed = 0;   ///< Conversations saved before compression, now compressed.
    std::vector<std::string> errors;
    [[nodiscard]] bool empty() const noexcept {
        return sessions == 0 && checkpoints == 0 && leftovers == 0 && compressed == 0 && errors.empty();
    }
};

/// Bytes stored for saved conversations, by what they are kept for.
struct StorageUsage {
    std::uintmax_t conversations = 0;   ///< Saved conversations and their lock files.
    std::uintmax_t revert_history = 0;  ///< File checkpoints behind `/revert`.
    std::uintmax_t journals = 0;        ///< Execution journals.
    std::uintmax_t tool_output = 0;     ///< Offloaded tool results.
    std::size_t conversation_count = 0;
    /// False when something could not be measured or the measurement was
    /// stopped, so the figures are a lower bound.
    bool complete = true;
    [[nodiscard]] std::uintmax_t total() const noexcept {
        return conversations + revert_history + journals + tool_output;
    }
};

/// Measures what @p store's directory and the tool results beside it hold.
/// Reads only; returns early, incomplete, once @p stop is requested.
[[nodiscard]] StorageUsage measure_storage(const SessionStore& store, std::stop_token stop = {});

/// Deletes what @p policy no longer keeps from @p store's directory and the
/// tool results beside it, then compresses conversations saved before
/// sessions were. Anything a Filo process holds open (its session lease) is
/// not deleted, nor is anything whose age cannot be established, and one pass
/// runs at a time across processes. Returns between items once @p stop is
/// requested.
RetentionReport apply_retention(const SessionStore& store, const RetentionPolicy& policy,
                                std::chrono::system_clock::time_point now,
                                std::stop_token stop = {});

} // namespace core::session
