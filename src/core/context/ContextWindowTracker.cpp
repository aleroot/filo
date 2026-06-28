#include "ContextWindowTracker.hpp"

#include "../llm/ModelRegistry.hpp"

#include <algorithm>
#include <cmath>

namespace core::context {

std::size_t ContextWindowTracker::estimate_tokens(
    const std::vector<core::llm::Message>& history) noexcept {
    std::size_t total_chars = 0;
    for (const auto& msg : history) {
        total_chars += msg.content.size();
        for (const auto& tc : msg.tool_calls) {
            total_chars += tc.function.name.size() + tc.function.arguments.size();
        }
    }
    return total_chars / 4 + 1;
}

int32_t ContextWindowTracker::resolve_max_context_tokens(
    const std::shared_ptr<core::llm::LLMProvider>& provider,
    std::string_view model) noexcept {
    int32_t max_context_tokens = 0;
    if (provider) {
        max_context_tokens = provider->max_context_size();
    }
    if (max_context_tokens == 0 && !model.empty()) {
        max_context_tokens = core::llm::get_max_context_size(model);
    }
    if (max_context_tokens == 0 && provider) {
        const std::string last_model = provider->get_last_model();
        if (!last_model.empty() && last_model != model) {
            max_context_tokens = core::llm::get_max_context_size(last_model);
        }
    }
    return max_context_tokens;
}

ContextWindowSnapshot ContextWindowTracker::snapshot(
    const std::vector<core::llm::Message>& history,
    const std::shared_ptr<core::llm::LLMProvider>& provider,
    std::string_view model) noexcept {
    const int32_t max_context_tokens =
        resolve_max_context_tokens(provider, model);
    const std::size_t estimated_context_tokens = estimate_tokens(history);

    int32_t remaining_pct = -1;
    if (max_context_tokens > 0) {
        const double remaining = static_cast<double>(max_context_tokens)
            - static_cast<double>(estimated_context_tokens);
        remaining_pct = remaining <= 0.0
            ? 0
            : static_cast<int32_t>(std::ceil(
                (remaining / static_cast<double>(max_context_tokens)) * 100.0));
    }

    return ContextWindowSnapshot{
        .estimated_context_tokens = estimated_context_tokens,
        .max_context_tokens = max_context_tokens,
        .remaining_pct = remaining_pct,
    };
}

int ContextWindowTracker::effective_compaction_threshold(
    const CompactionTriggerPolicy& policy,
    int32_t max_context_tokens) noexcept {
    if (policy.configured_token_threshold <= 0) {
        return policy.configured_token_threshold;
    }
    if (max_context_tokens <= 0) {
        return policy.configured_token_threshold;
    }

    const double ratio = std::clamp(policy.context_usage_ratio, 0.0, 1.0);
    const int ratio_threshold =
        static_cast<int>(std::floor(static_cast<double>(max_context_tokens) * ratio));
    if (policy.use_model_aware_default_threshold
        && policy.configured_token_threshold
            == CompactionTriggerPolicy::kDefaultFixedTokenThreshold) {
        return ratio_threshold;
    }
    return std::min(policy.configured_token_threshold, ratio_threshold);
}

CompactionDecision ContextWindowTracker::compaction_decision(
    const std::vector<core::llm::Message>& history,
    const std::shared_ptr<core::llm::LLMProvider>& provider,
    std::string_view model,
    const CompactionTriggerPolicy& policy) noexcept {
    const int32_t max_context_tokens =
        resolve_max_context_tokens(provider, model);
    const std::size_t estimated_context_tokens = estimate_tokens(history);
    const int effective_threshold =
        effective_compaction_threshold(policy, max_context_tokens);

    return CompactionDecision{
        .should_compact = effective_threshold > 0
            && estimated_context_tokens >= static_cast<std::size_t>(effective_threshold),
        .estimated_context_tokens = estimated_context_tokens,
        .effective_token_threshold = effective_threshold,
        .max_context_tokens = max_context_tokens,
    };
}

} // namespace core::context
