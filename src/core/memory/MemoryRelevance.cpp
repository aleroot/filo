#include "MemoryRelevance.hpp"

#include "uni_algo/case.h"
#include "uni_algo/norm.h"
#include "uni_algo/ranges_grapheme.h"
#include "uni_algo/ranges_word.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace core::memory {
namespace {

/**
 * BM25's two shape parameters, at the values Lucene and Elasticsearch ship as
 * their defaults (Robertson and Spark-Jones; Xapian uses k1 = 1, b = 0.5).
 *
 * b is the one that matters here: Filo's memories run from a one-line preference
 * to a 6 KB investigation, and without length normalisation the long ones win
 * simply by containing more words. Measured on 323 recorded turns, b = 0.75 and
 * b = 1.0 both cut the correlation between an entry's length and how often it is
 * recalled (Spearman +0.118 and lower, against +0.178 for a plain IDF sum) at no
 * cost in recall, so the standard value is kept rather than a tuned one.
 */
constexpr double kBm25K1 = 1.2;
constexpr double kBm25B = 0.75;

/// Bound on the text memories are ranked against, in graphemes so the cut can
/// never split a character. The opening prompt plus its attachments is where the
/// identifiers worth matching live; ranking against more only dilutes the
/// weights, and an unbounded query would make the score depend on how much file
/// content happened to be attached.
constexpr std::size_t kMaxQueryGraphemes = 4000;

/// Heterogeneous lookup, so a word from the analyzer (a view into the folded
/// text) can be tested against the query's terms without allocating a string.
struct TransparentStringHash {
    using is_transparent = void;
    [[nodiscard]] std::size_t operator()(std::string_view value) const noexcept {
        return std::hash<std::string_view>{}(value);
    }
};

using TermSet = std::unordered_set<std::string, TransparentStringHash, std::equal_to<>>;

/**
 * The analyzer, shared by queries and memories so both sides produce the same
 * terms: NFC (an accented letter matches whichever composed form it was typed
 * in), case folding (an identifier matches prose that names it), then UAX #29
 * word boundaries (so `load_for_prompt` stays one term and "MemoryStore" is not
 * split, while punctuation and spaces are dropped).
 *
 * There is deliberately no stopword list and no minimum term length. A hand
 * written English list would silently drop Italian words and code identifiers,
 * and it is not needed: BM25's IDF already scores a term that appears in every
 * memory near zero. Elasticsearch reaches the same conclusion from the other
 * side — its standard analyzer ships with the stop filter disabled.
 */
[[nodiscard]] std::string analyze(std::string_view text) {
    return una::cases::to_casefold_utf8(una::norm::to_nfc_utf8(text));
}

/// What one memory contributes to the score: how often each query term occurs in
/// it, and its length in terms for BM25's length normalisation.
struct DocumentTerms {
    std::vector<std::pair<std::string, std::size_t>> query_term_counts;
    std::size_t length = 0;
};

[[nodiscard]] DocumentTerms document_terms(std::string_view content, const TermSet& query_terms) {
    const std::string folded = analyze(content);
    std::unordered_map<std::string_view, std::size_t> counts;
    DocumentTerms result;
    for (const std::string_view word : folded | una::views::word_only::utf8) {
        ++result.length;
        ++counts[word];
    }
    for (const auto& [term, count] : counts) {
        if (query_terms.contains(term)) {
            result.query_term_counts.emplace_back(std::string(term), count);
        }
    }
    return result;
}

[[nodiscard]] std::string bounded_prefix(std::string_view text) {
    std::string bounded;
    std::size_t graphemes = 0;
    for (const std::string_view cluster : text | una::views::grapheme::utf8) {
        if (graphemes++ == kMaxQueryGraphemes) break;
        bounded += cluster;
    }
    return bounded;
}

} // namespace

bool prompt_precedes(const MemoryEntry& lhs, const MemoryEntry& rhs) noexcept {
    if (lhs.last_used_at != rhs.last_used_at) return lhs.last_used_at > rhs.last_used_at;
    if (lhs.created_at != rhs.created_at) return lhs.created_at > rhs.created_at;
    return lhs.id < rhs.id;
}

std::vector<std::size_t> rank_memories(std::span<const MemoryEntry> candidates,
                                       std::string_view query) {
    std::vector<std::size_t> order(candidates.size());
    for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
    const auto by_recency = [&](std::size_t lhs, std::size_t rhs) {
        return prompt_precedes(candidates[lhs], candidates[rhs]);
    };

    TermSet query_terms;
    {
        const std::string folded = analyze(query);
        for (const std::string_view word : folded | una::views::word_only::utf8) {
            query_terms.emplace(word);
        }
    }
    if (query_terms.empty() || candidates.empty()) {
        std::ranges::sort(order, by_recency);
        return order;
    }

    std::vector<DocumentTerms> documents;
    documents.reserve(candidates.size());
    std::unordered_map<std::string, std::size_t> document_frequency;
    std::size_t total_length = 0;
    for (const auto& candidate : candidates) {
        documents.push_back(document_terms(candidate.content, query_terms));
        total_length += documents.back().length;
        for (const auto& [term, count] : documents.back().query_term_counts) {
            ++document_frequency[term];
        }
    }
    const double population = static_cast<double>(candidates.size());
    const double average_length = total_length > 0
        ? static_cast<double>(total_length) / population
        : 1.0;

    std::vector<double> score(candidates.size(), 0.0);
    for (std::size_t i = 0; i < documents.size(); ++i) {
        const auto& document = documents[i];
        for (const auto& [term, frequency] : document.query_term_counts) {
            const double df = static_cast<double>(document_frequency[term]);
            const double idf = std::log(1.0 + (population - df + 0.5) / (df + 0.5));
            const double f = static_cast<double>(frequency);
            const double length_penalty =
                1.0 - kBm25B +
                kBm25B * (static_cast<double>(document.length) / average_length);
            score[i] += idf * (f * (kBm25K1 + 1.0)) / (f + kBm25K1 * length_penalty);
        }
    }

    std::ranges::sort(order, [&](std::size_t lhs, std::size_t rhs) {
        if (score[lhs] != score[rhs]) return score[lhs] > score[rhs];
        return by_recency(lhs, rhs);
    });
    return order;
}

std::vector<std::size_t> select_memories(std::span<const MemoryEntry> candidates,
                                         std::string_view query,
                                         std::size_t max_entries,
                                         std::size_t max_block_chars) {
    if (max_entries == 0) return {};
    std::vector<std::size_t> chosen;
    chosen.reserve(std::min(max_entries, candidates.size()));
    std::size_t used = 0;
    for (const std::size_t index : rank_memories(candidates, query)) {
        if (chosen.size() == max_entries) break;
        const std::size_t size = candidates[index].content.size();
        if (max_block_chars != 0 && !chosen.empty() && used + size > max_block_chars) {
            continue;  // A shorter memory further down still fits; do not stop here.
        }
        chosen.push_back(index);
        used += size;
    }
    return chosen;
}

std::string conversation_relevance_query(
    std::span<const core::llm::Message> conversation) {
    for (const auto& message : conversation) {
        if (message.role != "user" || message.synthetic) continue;
        // The expanded content, not input_text: attachments are part of what the
        // user put in front of the agent, and they carry file paths and symbols.
        const std::string_view text = !message.content.empty()
            ? std::string_view(message.content)
            : std::string_view(message.input_text);
        if (text.empty()) continue;
        return bounded_prefix(text);
    }
    return {};
}

} // namespace core::memory
