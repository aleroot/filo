#pragma once

#include "../llm/Models.hpp"

#include <cstddef>
#include <vector>

namespace core::agent {

struct SemanticHistoryEditPolicy {
    std::size_t protected_tail_messages = 8;
    std::size_t minimum_result_chars = 2048;
};

struct SemanticHistoryEditResult {
    std::vector<core::llm::Message> messages;
    std::size_t superseded_results = 0;
    std::size_t characters_removed = 0;
};

/**
 * Request-scoped semantic context editor.
 *
 * It preserves the durable transcript and the tool-call/result structure, but
 * replaces large, successful results of repeated identical observation calls
 * when a newer result is already present. The latest results, errors, signed
 * continuation state, and a protected recent tail are never modified.
 */
class SemanticHistoryEditor {
public:
    [[nodiscard]] static SemanticHistoryEditResult edit(
        const std::vector<core::llm::Message>& history,
        SemanticHistoryEditPolicy policy = {});
};

} // namespace core::agent
