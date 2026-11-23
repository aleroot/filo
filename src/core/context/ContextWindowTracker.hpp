#pragma once

#include "../llm/LLMProvider.hpp"

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

namespace core::context {

struct ContextWindowSnapshot {
    std::size_t estimated_context_tokens = 0;
    std::size_t metered_context_tokens = 0;
    int32_t     max_context_tokens = 0;
    int32_t     remaining_pct = -1;
};

struct CompactionTriggerPolicy {
    static constexpr int kDefaultFixedTokenThreshold = 25'000;
    static constexpr double kDefaultContextUsageRatio = 0.75;

    int configured_token_threshold = kDefaultFixedTokenThreshold;
    double context_usage_ratio = kDefaultContextUsageRatio;
    bool use_model_aware_default_threshold = true;

    /// The policy an `auto_compact_threshold` setting describes: @p percent
    /// (1-100) is a share of the model's window, else @p tokens is a fixed
    /// count (capped by the default share) unless it was never set
    /// explicitly, which selects the model-aware default. 0 tokens disables.
    [[nodiscard]] static constexpr CompactionTriggerPolicy from_setting(
        int tokens, bool explicit_tokens, int percent) noexcept {
        if (percent > 0) {
            return {
                .configured_token_threshold = kDefaultFixedTokenThreshold,
                .context_usage_ratio = static_cast<double>(percent) / 100.0,
                .use_model_aware_default_threshold = true,
            };
        }
        return {
            .configured_token_threshold = tokens,
            .use_model_aware_default_threshold = !explicit_tokens,
        };
    }
};

struct CompactionDecision {
    bool        should_compact = false;
    std::size_t estimated_context_tokens = 0;
    int         effective_token_threshold = 0;
    int32_t     max_context_tokens = 0;
};

class ContextWindowTracker {
public:
    [[nodiscard]] static std::size_t estimate_tokens(
        const std::vector<core::llm::Message>& history) noexcept;

    [[nodiscard]] static int32_t resolve_max_context_tokens(
        const std::shared_ptr<core::llm::LLMProvider>& provider,
        std::string_view model) noexcept;

    [[nodiscard]] static ContextWindowSnapshot snapshot(
        const std::vector<core::llm::Message>& history,
        const std::shared_ptr<core::llm::LLMProvider>& provider,
        std::string_view model,
        std::size_t metered_tokens_to_exclude = 0) noexcept;

    [[nodiscard]] static int effective_compaction_threshold(
        const CompactionTriggerPolicy& policy,
        int32_t max_context_tokens) noexcept;

    [[nodiscard]] static CompactionDecision compaction_decision(
        const std::vector<core::llm::Message>& history,
        const std::shared_ptr<core::llm::LLMProvider>& provider,
        std::string_view model,
        const CompactionTriggerPolicy& policy) noexcept;
};

} // namespace core::context
