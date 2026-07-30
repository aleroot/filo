#pragma once

#include "../llm/Models.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace core::agent {

struct HistoryCompactionPolicy {
    std::size_t summary_input_tokens = 24'000;
    std::size_t retained_user_tokens = 20'000;
    std::size_t retained_head_user_tokens = 2'000;
    std::size_t execution_checkpoint_tokens = 8'000;
};

struct HistoryCompactionPlan {
    // Provider-neutral text-only history used by the summarizer.
    std::vector<core::llm::Message> summary_history;
    // Genuine user input retained verbatim when possible and replay-safe on
    // every provider. System state and tool protocol messages are rebuilt.
    std::vector<core::llm::Message> retained_history;
    // Deterministic recent execution state appended outside the lossy model
    // summary. This preserves tool arguments, results, and offload references.
    std::string execution_checkpoint;
};

class HistoryCompactionPlanner final {
public:
    [[nodiscard]] static HistoryCompactionPolicy policy_for_context_window(
        int32_t max_context_tokens) noexcept;

    [[nodiscard]] static HistoryCompactionPlan plan(
        const std::vector<core::llm::Message>& history,
        std::string_view existing_summary = {},
        HistoryCompactionPolicy policy = {});
};

} // namespace core::agent
