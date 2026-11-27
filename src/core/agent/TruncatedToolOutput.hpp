#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>

namespace core::agent::tool_output_history {

// Field names of the envelope `prepare_for_history` writes when an oversized
// tool result is cut to a head and a tail. Shared by the writer and
// `reassemble_truncated_output`, so the two cannot drift apart.
namespace envelope {
inline constexpr std::string_view kTruncated = "truncated";
inline constexpr std::string_view kError = "error";
inline constexpr std::string_view kTool = "tool";
inline constexpr std::string_view kOriginalChars = "original_chars";
inline constexpr std::string_view kKeptChars = "kept_chars";
inline constexpr std::string_view kDigest = "digest_fnv1a64";
inline constexpr std::string_view kHead = "head";
inline constexpr std::string_view kTail = "tail";
} // namespace envelope

// What the tool returned, rebuilt from a truncation envelope.
struct TruncatedToolOutput {
    // The tool's own result shape. When the original was a JSON object it is
    // again a valid JSON object: the head is closed where it was cut, and when
    // the cut fell inside a top-level string (a command's `output`, a file's
    // `content`) the tail's end of that string and the fields after it are
    // joined back on, with an omission note in between. Plain-text results
    // come back as text with the same note.
    std::string payload;
    std::size_t original_chars = 0;
    std::size_t omitted_chars = 0;
};

// "51 KB omitted": how a reader is told what the history no longer holds.
[[nodiscard]] std::string describe_omission(std::size_t omitted_chars);

// Returns nothing when `payload` is not a truncation envelope, or when its
// head cannot be closed into valid JSON.
[[nodiscard]] std::optional<TruncatedToolOutput> reassemble_truncated_output(
    std::string_view payload);

} // namespace core::agent::tool_output_history
