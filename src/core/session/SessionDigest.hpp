#pragma once

#include "SessionData.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace core::session {

// A bounded, text-only digest of a saved conversation, inlined into a prompt
// by a `#` reference: header, compaction summary, user/assistant turns and
// tool calls by name, so the model can build on it without resuming it. Tool
// output is never copied: it is the bulk of a transcript and the files it
// describes may have changed since. The digest ends by saying whether it is
// complete and, when it is not, which message the `conversation` reader should
// start from.

struct SessionDigestOptions {
    /// Budget for the whole rendered block; the oldest turns are dropped first
    /// (the opening user request is kept, shortened when necessary).
    /// Returns an empty block when the budget cannot fit its framing.
    std::size_t max_bytes = 32 * 1024;
    /// Per-message clamp so a single long answer cannot crowd out the rest.
    std::size_t max_message_bytes = 4 * 1024;
};

/// Renders @p session as a `[Context for <label>]` ... `[/Context]` block
/// ending in a newline.
[[nodiscard]] std::string render_session_digest(const SessionData& session,
                                                std::string_view label,
                                                const SessionDigestOptions& options = {});

/// Cap on each tool result a transcript includes on request.
inline constexpr std::size_t kMaxTranscriptToolOutputBytes = 2 * 1024;

struct TranscriptOptions {
    /// Adds each tool call's saved result, clamped to
    /// kMaxTranscriptToolOutputBytes, under the assistant message that made it.
    bool include_tool_output = false;
};

/// One numbered transcript entry, ending in a newline. Number 0 is the
/// compaction summary (present only after a compaction); visible user and
/// assistant messages are numbered from 1 in saved order. The digest and the
/// reader number identically, and appending to a conversation never renumbers
/// earlier entries, so a number the digest cites stays readable.
struct TranscriptEntry {
    std::size_t number = 0;
    std::string text;
};

/// Identity lines opening a transcript: title, id, project, dates and model.
[[nodiscard]] std::string render_transcript_header(const SessionData& session);

/// Every entry, complete. Excludes system records and synthetic context, and
/// tool results unless requested.
[[nodiscard]] std::vector<TranscriptEntry> render_transcript_entries(
    const SessionData& session,
    const TranscriptOptions& options = {});

/// The header followed by every entry: what a full read returns.
[[nodiscard]] std::string render_session_transcript(const SessionData& session,
                                                    const TranscriptOptions& options = {});

} // namespace core::session
