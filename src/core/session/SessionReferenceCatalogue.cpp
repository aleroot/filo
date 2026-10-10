#include "SessionReferenceCatalogue.hpp"

#include "SessionReference.hpp"
#include "ThreadCatalog.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <format>
#include <optional>
#include <unordered_map>
#include <utility>

namespace core::session {

namespace {

namespace ascii = core::utils::ascii;
namespace str = core::utils::str;

[[nodiscard]] std::string project_name(std::string_view working_dir) {
    if (working_dir.empty()) return {};
    auto path = std::filesystem::path(working_dir).lexically_normal();
    if (!path.has_filename()) path = path.parent_path();
    return path.filename().string();
}

// `123` or `-4`: a number that was not an ordinal in range.
[[nodiscard]] bool is_numeric_term(std::string_view term) noexcept {
    if (term.starts_with('-')) term.remove_prefix(1);
    return !term.empty() && std::ranges::all_of(term, [](char ch) {
        return ascii::is_digit(static_cast<unsigned char>(ch));
    });
}

[[nodiscard]] bool occurs_at_word_start(std::string_view haystack, std::string_view term) noexcept {
    for (std::size_t at = haystack.find(term); at != std::string_view::npos;
         at = haystack.find(term, at + 1)) {
        if (at == 0 || !ascii::is_alnum(static_cast<unsigned char>(haystack[at - 1]))) return true;
    }
    return false;
}

enum class MatchTier : std::size_t { IdPrefix, TitlePrefix, WordStarts, Anywhere, Count };

/// How well @p row matches the (lowercased, non-empty) @p terms; nullopt when
/// some term matches nowhere.
[[nodiscard]] std::optional<MatchTier> match_tier(const SessionReferenceSuggestion& row,
                                                  const std::vector<std::string>& terms,
                                                  std::string_view phrase) {
    const std::string title = str::to_lower_ascii_copy(row.title);
    const std::string haystack = std::format("{}\n{}\n{}\n{}",
                                             title,
                                             row.session_id,
                                             str::to_lower_ascii_copy(row.project),
                                             str::to_lower_ascii_copy(row.model));
    const auto everywhere = [&](auto&& matches) { return std::ranges::all_of(terms, matches); };

    if (!everywhere([&](const std::string& term) { return haystack.contains(term); })) {
        return std::nullopt;
    }
    if (terms.size() == 1 && row.session_id.starts_with(terms.front())) return MatchTier::IdPrefix;
    if (title.starts_with(phrase)) return MatchTier::TitlePrefix;
    if (everywhere([&](const std::string& term) { return occurs_at_word_start(title, term); })) {
        return MatchTier::WordStarts;
    }
    return MatchTier::Anywhere;
}

} // namespace

SessionReferenceCatalogue::SessionReferenceCatalogue(std::vector<SessionInfo> sessions,
                                                     std::string_view project_dir,
                                                     std::string_view current_session_id)
    : sessions_(std::move(sessions)) {
    std::erase_if(sessions_, [&](const SessionInfo& info) {
        return info.session_id.empty() || info.session_id == current_session_id;
    });
    ordinals_.assign(sessions_.size(), 0);
    if (project_dir.empty()) return;

    // Canonicalising touches the filesystem; sessions share a handful of
    // working dirs, so resolve each distinct one once.
    const auto project = SessionStore::canonicalize_working_dir(project_dir);
    std::unordered_map<std::string_view, bool> in_project;
    for (std::size_t i = 0; i < sessions_.size(); ++i) {
        const std::string_view dir = sessions_[i].working_dir;
        if (dir.empty()) continue;
        auto [it, inserted] = in_project.try_emplace(dir, false);
        if (inserted) it->second = SessionStore::canonicalize_working_dir(dir) == project;
        if (!it->second) continue;
        project_rows_.push_back(i);
        ordinals_[i] = static_cast<int>(project_rows_.size());
    }
}

std::vector<SessionReferenceSuggestion> SessionReferenceCatalogue::search(
    std::string_view query,
    std::size_t max_results) const {
    if (max_results == 0) return {};
    if (const auto ordinal = parse_session_ordinal(query)) return search_ordinal(*ordinal);

    const auto terms = session_reference_terms(query);
    if (terms.size() == 1 && is_numeric_term(terms.front())) {
        return {};  // `#123` is an issue or PR number, not a title search.
    }
    if (terms.size() > 1 && contains_id(terms.front())) {
        return {};  // Prose typed after an accepted `#<id>`.
    }
    return search_terms(terms, max_results);
}

std::vector<SessionReferenceSuggestion> SessionReferenceCatalogue::search_ordinal(
    int ordinal) const {
    const auto count = static_cast<int>(project_rows_.size());
    const int position = ordinal > 0 ? ordinal - 1 : count + ordinal;
    if (position < 0 || position >= count) return {};
    return {row(project_rows_[static_cast<std::size_t>(position)])};
}

std::vector<SessionReferenceSuggestion> SessionReferenceCatalogue::search_terms(
    const std::vector<std::string>& terms,
    std::size_t max_results) const {
    std::string phrase;
    for (const auto& term : terms) {
        if (!phrase.empty()) phrase.push_back(' ');
        phrase += term;
    }

    std::array<std::vector<SessionReferenceSuggestion>, std::to_underlying(MatchTier::Count)> tiers;
    for (std::size_t i = 0; i < sessions_.size(); ++i) {
        auto suggestion = row(i);
        const auto tier = terms.empty() ? std::optional{MatchTier::IdPrefix}
                                        : match_tier(suggestion, terms, phrase);
        if (tier.has_value()) tiers[std::to_underlying(*tier)].push_back(std::move(suggestion));
    }

    std::vector<SessionReferenceSuggestion> out;
    out.reserve(max_results);
    for (auto& tier : tiers) {
        for (auto& suggestion : tier) {
            if (out.size() == max_results) return out;
            out.push_back(std::move(suggestion));
        }
    }
    return out;
}

SessionReferenceSuggestion SessionReferenceCatalogue::row(std::size_t index) const {
    const SessionInfo& info = sessions_[index];
    return {
        .session_id = info.session_id,
        .title = str::collapse_ascii_whitespace_copy(thread_display_title(info)),
        .project = project_name(info.working_dir),
        .model = thread_model_label(info.provider, info.model),
        .last_active_at = thread_activity_timestamp(info),
        .turn_count = info.turn_count,
        .ordinal = ordinals_[index],
    };
}

bool SessionReferenceCatalogue::contains_id(std::string_view session_id) const noexcept {
    return std::ranges::contains(sessions_, session_id, &SessionInfo::session_id);
}

} // namespace core::session
