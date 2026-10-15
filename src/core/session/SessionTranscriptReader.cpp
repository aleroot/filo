#include "SessionTranscriptReader.hpp"

#include "SessionDigest.hpp"
#include "SessionId.hpp"
#include "../utils/Crypto.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <charconv>
#include <format>
#include <utility>
#include <vector>

namespace core::session {
namespace {

[[nodiscard]] bool is_utf8_continuation(char byte) noexcept {
    return (static_cast<unsigned char>(byte) & 0xC0U) == 0x80U;
}

/// A position inside entry @ref number, bound to the text before it.
struct Cursor {
    std::size_t number = 0;
    std::size_t offset = 0;
    bool include_tool_output = false;
    std::string fingerprint;
};

/// Hash of every transcript byte before (@p index, @p offset): unchanged by
/// appended messages, changed by anything that rewrites what was already read.
[[nodiscard]] std::string fingerprint(const std::vector<TranscriptEntry>& entries,
                                      std::size_t index, std::size_t offset) {
    std::string prefix;
    for (std::size_t i = 0; i < index; ++i) prefix += entries[i].text;
    prefix.append(entries[index].text, 0, offset);
    constexpr std::string_view kHex = "0123456789abcdef";
    std::string out;
    const auto digest = core::utils::crypto::sha256(prefix);
    for (std::size_t i = 0; i < 8; ++i) {
        out.push_back(kHex[digest[i] >> 4]);
        out.push_back(kHex[digest[i] & 15]);
    }
    return out;
}

[[nodiscard]] std::string format_cursor(const Cursor& cursor) {
    return std::format("{}.{}.{}.{}", cursor.number, cursor.offset,
                       cursor.include_tool_output ? 1 : 0, cursor.fingerprint);
}

[[nodiscard]] std::optional<Cursor> parse_cursor(std::string_view text) {
    const auto number = [&text](std::size_t& out) {
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), out);
        if (error != std::errc{} || end == text.data() || end == text.data() + text.size()
            || *end != '.') {
            return false;
        }
        text.remove_prefix(static_cast<std::size_t>(end - text.data()) + 1);
        return true;
    };
    Cursor cursor;
    std::size_t tool_output = 0;
    if (!number(cursor.number) || !number(cursor.offset) || !number(tool_output)
        || tool_output > 1 || text.size() != 16
        || !std::ranges::all_of(text, [](char c) {
               return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
           })) {
        return std::nullopt;
    }
    cursor.include_tool_output = tool_output == 1;
    cursor.fingerprint = std::string(text);
    return cursor;
}

/// Where a read starts: an entry index and a byte offset inside it.
struct Start {
    std::size_t index = 0;
    std::size_t offset = 0;
};

[[nodiscard]] std::expected<Start, std::string> resume(
    const std::vector<TranscriptEntry>& entries, const Cursor& cursor) {
    constexpr std::string_view kRewritten =
        "The conversation was rewritten since this cursor (for example compacted). "
        "Restart without cursor.";
    // The fingerprint proves the offset is one this reader produced, so it
    // needs no separate UTF-8 boundary check.
    const auto entry = std::ranges::find(entries, cursor.number, &TranscriptEntry::number);
    if (entry == entries.end() || cursor.offset >= entry->text.size()) {
        return std::unexpected(std::string(kRewritten));
    }
    const auto index = static_cast<std::size_t>(entry - entries.begin());
    if (fingerprint(entries, index, cursor.offset) != cursor.fingerprint) {
        return std::unexpected(std::string(kRewritten));
    }
    return Start{.index = index, .offset = cursor.offset};
}

[[nodiscard]] std::expected<Start, std::string> start_from(
    const std::vector<TranscriptEntry>& entries, std::int64_t from) {
    const auto last = entries.empty() ? std::size_t{0} : entries.back().number;
    // Negative counts back from the newest message, never into the summary.
    const auto number = from < 0
        ? static_cast<std::size_t>(std::max<std::int64_t>(
              1, static_cast<std::int64_t>(last) + 1 + from))
        : static_cast<std::size_t>(from);
    const auto entry = std::ranges::find_if(entries, [number](const TranscriptEntry& e) {
        return e.number >= number;
    });
    if (entry == entries.end()) {
        return std::unexpected(std::format(
            "from={} is past the last message ({}).", from, last));
    }
    return Start{.index = static_cast<std::size_t>(entry - entries.begin())};
}

/// Bytes of @p text that fit @p budget, ending on a line break when one falls
/// in the second half and never inside a UTF-8 sequence. Always at least one
/// byte, so a cursor advances even over malformed text.
[[nodiscard]] std::size_t slice_length(std::string_view text, std::size_t budget) {
    std::size_t cut = budget;
    while (cut > 0 && is_utf8_continuation(text[cut])) --cut;
    if (cut == 0) return budget;
    if (const auto line_end = text.rfind('\n', cut - 1);
        line_end != std::string_view::npos && line_end + 1 >= cut / 2) {
        cut = line_end + 1;
    }
    return cut;
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
    if (options.limit < kMinTranscriptPageBytes || options.limit > kMaxTranscriptPageBytes) {
        return std::unexpected(std::format("Use a page limit between {} and {} bytes.",
                                           kMinTranscriptPageBytes, kMaxTranscriptPageBytes));
    }
    std::optional<Cursor> cursor;
    if (!options.cursor.empty()) {
        if (options.from.has_value()) {
            return std::unexpected("Pass either from or cursor, not both.");
        }
        cursor = parse_cursor(options.cursor);
        if (!cursor) return std::unexpected("Invalid cursor; pass next_cursor unchanged.");
    }
    const auto id = core::utils::str::to_lower_ascii_copy(session_id);
    if (id == current_session_id) {
        return std::unexpected("A conversation cannot reference itself.");
    }
    const auto session = store_.load_by_id(id);
    if (!session) {
        return std::unexpected("Saved conversation unavailable; it may have been deleted.");
    }

    const bool include_tool_output =
        cursor ? cursor->include_tool_output : options.include_tool_output;
    const auto entries =
        render_transcript_entries(*session, {.include_tool_output = include_tool_output});
    Start start;
    if (cursor) {
        auto resumed = resume(entries, *cursor);
        if (!resumed) return std::unexpected(std::move(resumed.error()));
        start = *resumed;
    } else if (options.from.has_value()) {
        auto from = start_from(entries, *options.from);
        if (!from) return std::unexpected(std::move(from.error()));
        start = *from;
    }

    SessionTranscriptPage page{
        .session_id = session->session_id,
        .content = cursor ? std::string{} : render_transcript_header(*session),
        .total_messages = entries.empty() ? 0 : entries.back().number,
        .includes_tool_output = include_tool_output,
    };
    // Whole entries while they fit; an entry larger than what is left either
    // waits for the next page or, alone on this one, is cut and continued.
    std::size_t budget = options.limit;
    std::size_t index = start.index;
    std::size_t offset = start.offset;
    bool touched = false;
    while (index < entries.size()) {
        const auto rest = std::string_view(entries[index].text).substr(offset);
        if (!touched) page.first_message = entries[index].number;
        if (rest.size() <= budget) {
            page.content += rest;
            budget -= rest.size();
            page.last_message = entries[index].number;
            touched = true;
            ++index;
            offset = 0;
            continue;
        }
        if (!touched) {
            const auto length = slice_length(rest, budget);
            page.content += rest.substr(0, length);
            page.last_message = entries[index].number;
            offset += length;
        }
        break;
    }
    page.complete = index == entries.size();
    if (!page.complete) {
        page.next_cursor = format_cursor({
            .number = entries[index].number,
            .offset = offset,
            .include_tool_output = include_tool_output,
            .fingerprint = fingerprint(entries, index, offset),
        });
    }
    return page;
}

} // namespace core::session
