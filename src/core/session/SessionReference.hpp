#pragma once

#include "SessionData.hpp"
#include "SessionStore.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::session {

// ---------------------------------------------------------------------------
// Conversation references: the `#` syntax in a prompt.
//
// `@` stays for files; `#` pulls in another saved conversation. Typing `#`
// opens a picker (SessionReferenceCatalogue) that matches session titles, and
// accepting a row inserts the session's id. On submit the token is replaced by
// a digest of that conversation (SessionDigest).
//
// `#` is common in prose and code (`# Heading`, `#123`, `#include`, `#fff`),
// so a sent token only becomes a reference when it is exactly an existing
// session's 8-hex id, or an explicit `#"session name"`. Anything else is left
// untouched, and nothing is recognised inside ``` fences. In the picker, `#2`
// or `#-1` select by position (see parse_session_ordinal).
// ---------------------------------------------------------------------------

inline constexpr char kSessionReferenceSigil = '#';

// ---- While typing: the picker ---------------------------------------------

/// The `#query` the cursor is in, for the reference picker. The query may
/// contain spaces (multi-word title search) but never a newline; it must start
/// right after `#` (so `# Heading` and `##` never open the picker) and must
/// not end in whitespace (so the picker closes after an accepted reference).
struct ActiveSessionReference {
    std::size_t replace_begin = 0; ///< index of '#'
    std::size_t replace_end = 0;   ///< end of the word under the cursor
    std::string query;             ///< text between '#' and the cursor
};

[[nodiscard]] std::optional<ActiveSessionReference> find_active_session_reference(
    std::string_view input,
    std::size_t cursor);

struct SessionReferenceCompletion {
    std::string text;
    std::size_t cursor = 0;
};

/// Replaces the active `#query` with `#<session_id> ` and puts the cursor
/// after the space.
[[nodiscard]] SessionReferenceCompletion apply_session_reference_completion(
    std::string_view input,
    const ActiveSessionReference& active,
    std::string_view session_id);

/// A positional shortcut typed in the picker, numbered like `/resume`:
/// `#1` is the newest other conversation in this project, `#2` the one before
/// it, and `#-1` the oldest. Accepts `-?[1-9][0-9]{0,3}`; anything else
/// (`#0`, `#12345`, `#1a`) is nullopt. Ordinals are a picker affordance only:
/// accepting the row inserts the stable `#<id>`, and a bare `#3` that is sent
/// stays plain text, because the numbering shifts with every new session.
[[nodiscard]] std::optional<int> parse_session_ordinal(std::string_view query) noexcept;

/// Lowercased terms of a title query, split on whitespace and quotes; empty
/// for an ordinal query, which matches by position rather than by text.
[[nodiscard]] std::vector<std::string> session_reference_terms(std::string_view query);

/// Whether Enter accepts the picker row for @p query (Tab always does).
/// Enter sends instead when the selected id is already typed out, and for a
/// positive number, which is as likely an issue at the end of a sentence.
[[nodiscard]] bool session_reference_accepts_on_enter(std::string_view query,
                                                      std::string_view selected_session_id) noexcept;

// ---- On submit: resolution ------------------------------------------------

/// A `#...` token that names a session reference at @p pos (which must hold
/// '#' at a token boundary). Unquoted: an 8-hex id, trailing punctuation
/// excluded. Quoted: `#"session name"` on one line.
struct SessionReferenceToken {
    std::string target;
    bool quoted = false;
    std::size_t end = 0;           ///< one past the token in the input
    std::string trailing_suffix;   ///< punctuation that followed an unquoted id
};

[[nodiscard]] std::optional<SessionReferenceToken> parse_session_reference_token(
    std::string_view input,
    std::size_t pos);

/// True when @p pos lies inside a ``` fenced block of @p input.
[[nodiscard]] bool is_inside_code_fence(std::string_view input, std::size_t pos) noexcept;

/// Loads the session @p token names, or nullopt when it names none or names
/// @p current_session_id (a conversation cannot reference itself).
[[nodiscard]] std::optional<SessionData> resolve_session_reference(
    const SessionStore& store,
    const SessionReferenceToken& token,
    std::string_view current_session_id);

} // namespace core::session
