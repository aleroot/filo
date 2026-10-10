#pragma once

#include "SessionStore.hpp"

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace core::session {

/// One `#` picker row. Text fields are display-ready.
struct SessionReferenceSuggestion {
    std::string session_id;
    std::string title;
    std::string project;        ///< basename of the session's working dir
    std::string model;          ///< "provider/model" or model alone
    std::string last_active_at; ///< ISO 8601, as stored
    int turn_count = 0;
    int ordinal = 0;            ///< `#N` shortcut in this project; 0 elsewhere
};

/// The conversations a `#` picker can offer, built once per catalogue refresh
/// so the per-keystroke search does no I/O: the current session is removed and
/// the sessions of the project are numbered newest-first, like `/resume`.
class SessionReferenceCatalogue {
public:
    SessionReferenceCatalogue() = default;
    /// @p sessions in SessionStore::list() order (most recent first).
    SessionReferenceCatalogue(std::vector<SessionInfo> sessions,
                              std::string_view project_dir,
                              std::string_view current_session_id);

    /// Rows for @p query. An ordinal (see parse_session_ordinal) yields that
    /// one conversation of this project. Otherwise every term must match the
    /// title, id, project or model; ranking is id prefix, title prefix, every
    /// term at a word start of the title, then any match, with recency kept
    /// within a tier. A bare number that is not an ordinal in range (`#123` is
    /// an issue), and prose typed after an accepted `#<id>`, yield nothing.
    [[nodiscard]] std::vector<SessionReferenceSuggestion> search(std::string_view query,
                                                                 std::size_t max_results) const;

    [[nodiscard]] std::size_t project_size() const noexcept { return project_rows_.size(); }

private:
    [[nodiscard]] std::vector<SessionReferenceSuggestion> search_ordinal(int ordinal) const;
    [[nodiscard]] std::vector<SessionReferenceSuggestion> search_terms(
        const std::vector<std::string>& terms,
        std::size_t max_results) const;
    [[nodiscard]] SessionReferenceSuggestion row(std::size_t index) const;
    [[nodiscard]] bool contains_id(std::string_view session_id) const noexcept;

    std::vector<SessionInfo> sessions_;
    std::vector<int> ordinals_;              ///< parallel to sessions_
    std::vector<std::size_t> project_rows_;  ///< indexes into sessions_, newest first
};

} // namespace core::session
