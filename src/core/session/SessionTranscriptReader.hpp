#pragma once

#include "SessionStore.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <optional>
#include <string>
#include <string_view>

namespace core::session {

inline constexpr std::size_t kMinTranscriptPageBytes = 4;
inline constexpr std::size_t kDefaultTranscriptPageBytes = 16 * 1024;
inline constexpr std::size_t kMaxTranscriptPageBytes = 32 * 1024;

struct TranscriptReadOptions {
    /// Entry number to start at (0 is the compaction summary); negative counts
    /// back from the last message. Unset starts at the beginning.
    std::optional<std::int64_t> from = std::nullopt;
    /// next_cursor of a previous page. Continues that read with its options,
    /// so it cannot be combined with @ref from.
    std::string_view cursor = {};
    /// Entry bytes per page; the header on a read's first page is extra.
    std::size_t limit = kDefaultTranscriptPageBytes;
    bool include_tool_output = false;
};

struct SessionTranscriptPage {
    std::string session_id;
    std::string content;
    /// Numbers of the first and last entries the page touches; an entry
    /// longer than a page spans several, continuing where the last one ended.
    std::size_t first_message = 0;
    std::size_t last_message = 0;
    /// Number of the newest message saved when this page was read.
    std::size_t total_messages = 0;
    /// Empty once complete.
    std::string next_cursor;
    bool complete = false;
    /// Whether the page carries tool results (requested, or by its cursor).
    bool includes_tool_output = false;
};

/// Provider-neutral saved transcript paging over the numbered entries of
/// render_transcript_entries(). A cursor names a position in an entry plus a
/// fingerprint of everything before it, so messages appended to a live
/// conversation never invalidate it while a compaction or rewrite of what was
/// already read does. The caller adapts requests and responses to its
/// transport; this reader has no JSON or tool dependencies.
class SessionTranscriptReader final {
public:
    explicit SessionTranscriptReader(SessionStore store);

    [[nodiscard]] std::expected<SessionTranscriptPage, std::string> read(
        std::string_view session_id,
        std::string_view current_session_id,
        const TranscriptReadOptions& options = {}) const;

private:
    SessionStore store_;
};

} // namespace core::session
