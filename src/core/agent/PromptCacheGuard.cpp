#include "PromptCacheGuard.hpp"

#include "../config/ConfigManager.hpp"

#include <algorithm>

namespace core::agent {

PromptCacheGuardSettings prompt_cache_guard_settings(
    const core::config::AppConfig& config) {
    PromptCacheGuardSettings settings;
    if (config.prompt_cache_guard_tokens.has_value()) {
        settings.min_context_tokens = *config.prompt_cache_guard_tokens;
    }
    if (config.prompt_cache_notice_seconds.has_value()) {
        settings.notice_lead = std::chrono::seconds{*config.prompt_cache_notice_seconds};
    }
    if (config.prompt_cache_keep_warm == kPromptCacheKeepWarmAuto) {
        settings.keep_warm = PromptCacheKeepWarm::Auto;
    } else if (config.prompt_cache_keep_warm == kPromptCacheKeepWarmCompact) {
        settings.keep_warm = PromptCacheKeepWarm::Compact;
    }
    return settings;
}

PromptCacheAssessment PromptCacheGuard::assess(
    const PromptCacheState& state,
    std::chrono::system_clock::time_point now) noexcept {
    PromptCacheAssessment assessment{
        .context_tokens = state.context_tokens,
        .profile = state.profile,
    };
    if (!state.profile.remote() || !state.last_exchange_at.has_value()) {
        return assessment;
    }
    // A clock that stepped backwards reads as "just used", never as expired.
    assessment.idle = std::max(
        std::chrono::seconds::zero(),
        std::chrono::duration_cast<std::chrono::seconds>(now - *state.last_exchange_at));
    if (assessment.idle >= state.profile.ttl) {
        assessment.warmth = PromptCacheWarmth::Expired;
    } else {
        assessment.warmth = PromptCacheWarmth::Warm;
        assessment.remaining = state.profile.ttl - assessment.idle;
    }
    return assessment;
}

bool PromptCacheGuard::should_confirm_send(
    const PromptCacheAssessment& assessment) const noexcept {
    return assessment.expired() && is_heavy(assessment.context_tokens);
}

std::optional<PromptCacheDeadlines> PromptCacheGuard::deadlines(
    const PromptCacheState& state) const noexcept {
    if (!state.profile.remote()
        || !state.last_exchange_at.has_value()
        || !is_heavy(state.context_tokens)) {
        return std::nullopt;
    }
    const auto lead = settings_.notice_lead_for(state.profile.ttl);
    if (lead <= std::chrono::seconds::zero() || lead >= state.profile.ttl) {
        return std::nullopt;
    }
    const auto expiry = *state.last_exchange_at + state.profile.ttl;
    return PromptCacheDeadlines{
        .notice = expiry - lead,
        .expiry = expiry,
    };
}

} // namespace core::agent
