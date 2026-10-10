#include "SessionTranscriptReader.hpp"

#include "SessionDigest.hpp"
#include "SessionId.hpp"
#include "../utils/Crypto.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <utility>

namespace core::session {
namespace {

[[nodiscard]] bool is_utf8_continuation(char byte) noexcept {
    return (static_cast<unsigned char>(byte) & 0xC0U) == 0x80U;
}

[[nodiscard]] std::string transcript_revision(std::string_view transcript) {
    std::string revision;
    revision.reserve(64);
    constexpr std::string_view kHex = "0123456789abcdef";
    for (const auto byte : core::utils::crypto::sha256(transcript)) {
        revision.push_back(kHex[byte >> 4]);
        revision.push_back(kHex[byte & 15]);
    }
    return revision;
}

} // namespace

SessionTranscriptReader::SessionTranscriptReader(SessionStore store)
    : store_(std::move(store)) {}

std::expected<SessionTranscriptPage, std::string> SessionTranscriptReader::read(
    std::string_view session_id,
    std::string_view current_session_id,
    const TranscriptReadOptions& options) const {
    if (!is_session_id(session_id)) {
        return std::unexpected("An 8-hex session_id is required.");
    }
    if (options.limit < 4 || options.limit > kMaxTranscriptPageBytes) {
        return std::unexpected("Use a page limit between 4 and 8192 bytes.");
    }
    const auto id = core::utils::str::to_lower_ascii_copy(session_id);
    if (id == current_session_id) {
        return std::unexpected("A conversation cannot reference itself.");
    }
    const auto session = store_.load_by_id(id);
    if (!session) {
        return std::unexpected("Saved conversation unavailable; it may have been deleted.");
    }
    const auto transcript = render_session_transcript(*session);
    auto revision = transcript_revision(transcript);
    if (!options.revision.empty() && options.revision != revision) {
        return std::unexpected(
            "Conversation changed during pagination. Restart from offset 0 without revision.");
    }
    const auto start = options.offset;
    if (start > transcript.size()
        || (start < transcript.size() && is_utf8_continuation(transcript[start]))) {
        return std::unexpected("Offset is out of range or not a UTF-8 boundary; use next_offset.");
    }
    auto end = start + std::min(static_cast<std::uint64_t>(options.limit), transcript.size() - start);
    while (end < transcript.size() && end > start && is_utf8_continuation(transcript[end])) --end;
    return SessionTranscriptPage{
        .session_id = session->session_id,
        .revision = std::move(revision),
        .content = transcript.substr(start, end - start),
        .offset = start,
        .next_offset = end,
        .total_bytes = transcript.size(),
        .complete = end == transcript.size(),
    };
}

} // namespace core::session
