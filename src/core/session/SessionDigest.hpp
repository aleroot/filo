#pragma once

#include "SessionData.hpp"

#include <cstddef>
#include <string>
#include <string_view>

namespace core::session {

// A bounded, text-only digest of a saved conversation, inlined into a prompt
// by a `#` reference: header, compaction summary, user/assistant turns and
// tool calls by name, so the model can build on it without resuming it. Tool
// output is never copied: it is the bulk of a transcript and the files it
// describes may have changed since.

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

/// Complete saved user/assistant text and compaction summary for paged reads.
/// Excludes tool results, system records, and synthetic context.
[[nodiscard]] std::string render_session_transcript(const SessionData& session);

} // namespace core::session
