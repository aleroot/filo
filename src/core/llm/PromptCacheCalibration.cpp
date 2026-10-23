#include "PromptCacheCalibration.hpp"

#include <algorithm>
#include <format>

namespace core::llm {

PromptCacheCalibration& PromptCacheCalibration::shared() {
    static PromptCacheCalibration instance;
    return instance;
}

void PromptCacheCalibration::observe(std::string_view key,
                                     std::chrono::seconds gap,
                                     const TokenUsage& usage) {
    if (gap.count() <= 0 || usage.prompt_tokens < kMinimumObservedPromptTokens) {
        return;
    }
    // A hit means most of the prompt came back from the cache; a few cached
    // tokens only show that a short shared prefix (the system prompt) survived.
    const bool hit = static_cast<std::int64_t>(usage.cached_prompt_tokens) * 2
        >= static_cast<std::int64_t>(usage.prompt_tokens);
    if (!hit) {
        return;
    }
    std::lock_guard lock(mutex_);
    auto& longest = longest_hit_gap_[std::string(key)];
    longest = std::max(longest, gap);
}

PromptCacheProfile PromptCacheCalibration::refine(std::string_view key,
                                                  PromptCacheProfile profile) const {
    if (!profile.remote() || profile.basis == PromptCacheBasis::Documented) {
        return profile;
    }
    std::lock_guard lock(mutex_);
    const auto it = longest_hit_gap_.find(key);
    if (it != longest_hit_gap_.end() && it->second > profile.ttl) {
        profile.ttl = it->second;
        profile.basis = PromptCacheBasis::Observed;
    }
    return profile;
}

std::string PromptCacheCalibration::key(std::string_view provider,
                                        std::string_view model) {
    return std::format("{}\n{}", provider, model);
}

void PromptCacheCalibration::clear() {
    std::lock_guard lock(mutex_);
    longest_hit_gap_.clear();
}

} // namespace core::llm
