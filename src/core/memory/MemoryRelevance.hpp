#pragma once

#include "MemoryStore.hpp"
#include "../llm/Models.hpp"

#include <cstddef>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::memory {

/**
 * Prompt-projection ranking for semantic memories: Okapi BM25 over the
 * conversation's opening prompt.
 *
 * The block used to be ordered by recency alone, so which memories a turn saw
 * was decided by when they were last recalled, not by what the user asked.
 * Measured over 323 recorded turns whose project had memories: 65.3% of turns
 * had more candidates than the block carries, and BM25 ranking raised "the block
 * names a file this turn edited" from 26.6% to 31.9%, staying ahead of recency
 * at every block budget (26.6% vs 21.4% at 24k characters, McNemar
 * chi2 = 6.92).
 *
 * Scoring is BM25 rather than a bespoke similarity because BM25 is what Lucene,
 * Elasticsearch and Xapian rank with, and its two terms are exactly the failure
 * modes of a hand-rolled score: term frequency saturates, so one memory that
 * repeats an identifier cannot drown out another that states it once, and
 * document length is normalised, so a 6 KB investigation is not rewarded for
 * being long. Both are measured against this corpus in the comment on the
 * parameters below.
 *
 * Ranking is lexical and deterministic — no model call, no index, no I/O — so
 * it costs microseconds and never differs between two runs on the same inputs.
 */

/// Recency order: most recently recalled first, then newest first, then id.
/// Single source of truth for "which memory comes first", shared by the prompt
/// projection, relevance tie-breaks, and listing, so they cannot drift apart.
[[nodiscard]] bool prompt_precedes(const MemoryEntry& lhs, const MemoryEntry& rhs) noexcept;

/**
 * Orders candidates best-first for `query` and returns their indices.
 *
 * Candidates that match nothing keep recency order and still fill the tail: an
 * unrelated memory is better than an empty block. An empty query, or one that
 * matches no candidate, is pure recency.
 */
[[nodiscard]] std::vector<std::size_t> rank_memories(
    std::span<const MemoryEntry> candidates,
    std::string_view query);

/**
 * rank_memories plus the prompt budget, in render order.
 *
 * Whole entries only: cutting one mid-sentence destroyed the reason to recall
 * it (47.6% of the useful entries lost their file mention at a 1200-character
 * cut). An entry that does not fit is skipped rather than ending the search, so
 * the budget is spent on memories that do fit. The best-ranked entry is always
 * taken, even alone over budget, because an empty block recalls nothing.
 * `max_block_chars == 0` disables the budget; `max_entries == 0` selects none.
 */
[[nodiscard]] std::vector<std::size_t> select_memories(
    std::span<const MemoryEntry> candidates,
    std::string_view query,
    std::size_t max_entries,
    std::size_t max_block_chars);

/**
 * The text one conversation's memories are ranked against: the prompt that
 * opened it, attachments included, because that is where the identifiers worth
 * matching live.
 *
 * Derived from the conversation rather than from the turn in flight, because the
 * memory block sits in the cached prompt prefix: a query that changed every turn
 * would rewrite that prefix every turn and cost the provider's prompt cache. The
 * choice is measured, not merely convenient — ranking against the opening prompt
 * recalled as many relevant memories as ranking against the current one (27.2%
 * both, at the shipped budget).
 *
 * Empty while the conversation has no real user prompt yet.
 */
[[nodiscard]] std::string conversation_relevance_query(
    std::span<const core::llm::Message> conversation);

} // namespace core::memory
