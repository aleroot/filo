#include "SessionReference.hpp"

#include "../context/MentionSyntax.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <charconv>

namespace core::session {

namespace {

namespace ascii = core::utils::ascii;

[[nodiscard]] constexpr bool is_space(char ch) noexcept {
    return ascii::is_space(static_cast<unsigned char>(ch));
}

[[nodiscard]] constexpr bool is_digit(char ch) noexcept {
    return ascii::is_digit(static_cast<unsigned char>(ch));
}

[[nodiscard]] bool is_session_id_shaped(std::string_view text) noexcept {
    constexpr std::size_t kSessionIdLength = 8;
    return text.size() == kSessionIdLength && std::ranges::all_of(text, [](char ch) {
               return is_digit(ch) || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F');
           });
}

[[nodiscard]] bool is_sigil_at_boundary(std::string_view input, std::size_t pos) noexcept {
    return input[pos] == kSessionReferenceSigil && core::context::is_mention_boundary(input, pos);
}

[[nodiscard]] std::size_t end_of_word(std::string_view input, std::size_t from) noexcept {
    while (from < input.size() && !is_space(input[from])) ++from;
    return from;
}

} // namespace

std::optional<ActiveSessionReference> find_active_session_reference(std::string_view input,
                                                                    std::size_t cursor) {
    cursor = std::min(cursor, input.size());
    // A title search is a few words; scanning further back would revive a
    // `#` from an earlier sentence.
    constexpr std::size_t kMaxQueryBytes = 80;

    for (std::size_t pos = cursor; pos-- > 0;) {
        if (input[pos] == '\n' || cursor - pos > kMaxQueryBytes + 1) return std::nullopt;
        if (!is_sigil_at_boundary(input, pos)) continue;

        // The nearest `#` decides: `# Heading`, `##` and `#word ` never open.
        const std::string_view query = input.substr(pos + 1, cursor - pos - 1);
        if (!query.empty()
            && (is_space(query.front()) || query.front() == kSessionReferenceSigil
                || is_space(query.back()))) {
            return std::nullopt;
        }
        if (is_inside_code_fence(input, pos)) return std::nullopt;

        return ActiveSessionReference{
            .replace_begin = pos,
            .replace_end = end_of_word(input, cursor),
            .query = std::string(query),
        };
    }
    return std::nullopt;
}

SessionReferenceCompletion apply_session_reference_completion(
    std::string_view input,
    const ActiveSessionReference& active,
    std::string_view session_id) {
    const std::size_t begin = std::min(active.replace_begin, input.size());
    const std::size_t end = std::clamp(active.replace_end, begin, input.size());

    SessionReferenceCompletion out;
    out.text.reserve(input.size() + session_id.size() + 2);
    out.text.append(input.substr(0, begin));
    out.text.push_back(kSessionReferenceSigil);
    out.text.append(session_id);
    out.cursor = out.text.size() + 1;  // past the space that follows the id
    // A following space closes the picker (see find_active_session_reference).
    if (end >= input.size() || input[end] != ' ') out.text.push_back(' ');
    out.text.append(input.substr(end));
    return out;
}

std::optional<int> parse_session_ordinal(std::string_view query) noexcept {
    constexpr std::size_t kMaxDigits = 4;
    const bool from_oldest = query.starts_with('-');
    const std::string_view digits = from_oldest ? query.substr(1) : query;
    if (digits.empty() || digits.size() > kMaxDigits || digits.front() == '0'
        || !std::ranges::all_of(digits, is_digit)) {
        return std::nullopt;
    }
    int value = 0;  // at most 4 plain digits: from_chars cannot fail or overflow
    std::from_chars(digits.data(), digits.data() + digits.size(), value);
    return from_oldest ? -value : value;
}

std::vector<std::string> session_reference_terms(std::string_view query) {
    if (parse_session_ordinal(query).has_value()) return {};
    std::vector<std::string> terms;
    std::string current;
    const auto flush = [&] {
        if (!current.empty()) terms.push_back(core::utils::str::to_lower_ascii_copy(current));
        current.clear();
    };
    for (const char ch : query) {
        if (is_space(ch) || ch == '"') {
            flush();
        } else {
            current.push_back(ch);
        }
    }
    flush();
    return terms;
}

bool session_reference_accepts_on_enter(std::string_view query,
                                        std::string_view selected_session_id) noexcept {
    if (const auto ordinal = parse_session_ordinal(query)) return *ordinal < 0;
    return query != selected_session_id;
}

std::optional<SessionReferenceToken> parse_session_reference_token(std::string_view input,
                                                                   std::size_t pos) {
    if (pos >= input.size() || !is_sigil_at_boundary(input, pos)) return std::nullopt;

    const std::size_t start = pos + 1;
    if (start < input.size() && input[start] == '"') {
        const std::size_t close = input.find_first_of("\"\n", start + 1);
        if (close == std::string_view::npos || input[close] != '"') return std::nullopt;
        std::string target = core::utils::str::trim_ascii_copy(
            input.substr(start + 1, close - start - 1));
        if (target.empty()) return std::nullopt;
        return SessionReferenceToken{.target = std::move(target), .quoted = true, .end = close + 1};
    }

    const std::size_t end = end_of_word(input, start);
    std::size_t id_end = end;
    while (id_end > start && core::context::is_trailing_mention_punctuation(input[id_end - 1])) {
        --id_end;
    }

    const std::string_view id = input.substr(start, id_end - start);
    if (!is_session_id_shaped(id)) return std::nullopt;
    return SessionReferenceToken{
        .target = core::utils::str::to_lower_ascii_copy(id),
        .quoted = false,
        .end = end,
        .trailing_suffix = std::string(input.substr(id_end, end - id_end)),
    };
}

bool is_inside_code_fence(std::string_view input, std::size_t pos) noexcept {
    constexpr std::string_view kFence = "```";
    bool inside = false;
    const std::size_t limit = std::min(pos, input.size());
    for (std::size_t at = input.find(kFence); at != std::string_view::npos && at < limit;
         at = input.find(kFence, at + kFence.size())) {
        inside = !inside;
    }
    return inside;
}

std::optional<SessionData> resolve_session_reference(const SessionStore& store,
                                                     const SessionReferenceToken& token,
                                                     std::string_view current_session_id) {
    std::optional<SessionData> session;
    if (is_session_id_shaped(token.target)) {
        session = store.load_by_id(core::utils::str::to_lower_ascii_copy(token.target));
    }
    if (!session.has_value() && token.quoted) session = store.load_by_name(token.target);
    if (!session.has_value() || session->session_id == current_session_id) return std::nullopt;
    return session;
}

} // namespace core::session
