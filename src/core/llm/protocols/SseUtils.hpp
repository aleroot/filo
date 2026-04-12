#pragma once

#include <string>
#include <string_view>

namespace core::llm::protocols::sse {

struct EnvelopeView {
    std::string_view event;
    std::string_view data;
};

struct ParsedEventView {
    std::string_view event;
    std::string_view data;
    bool             is_done = false;
};

// Parse an SSE envelope with optional `event:` and one-or-more `data:` lines.
// - Accepts both `field:value` and `field: value`.
// - Supports CRLF input (`\r\n`) and joins multiline `data:` payloads with '\n'.
// - Reuses `data_scratch` for allocations when multiple data lines are present.
// - When `allow_unframed_payload` is true, accepts already-unwrapped `[DONE]` or JSON payloads.
[[nodiscard]] bool parse_envelope(std::string_view raw_event,
                                  EnvelopeView& out,
                                  std::string& data_scratch,
                                  bool allow_unframed_payload = true);

// Returns true when `payload` is the SSE completion sentinel.
[[nodiscard]] bool is_done_sentinel(std::string_view payload) noexcept;

// High-level parse helper for protocol parse_event methods.
// - Produces trimmed event/data views and classifies `[DONE]`.
// - `out` string_views reference `raw_event` and/or `data_scratch`.
[[nodiscard]] bool parse_event_payload(std::string_view raw_event,
                                       ParsedEventView& out,
                                       std::string& data_scratch,
                                       bool allow_unframed_payload = true);

// Convenience overload that reuses an internal thread-local scratch buffer.
// Suitable for per-event protocol parsing where views are consumed immediately.
[[nodiscard]] bool parse_event_payload(std::string_view raw_event,
                                       ParsedEventView& out,
                                       bool allow_unframed_payload = true);

} // namespace core::llm::protocols::sse
