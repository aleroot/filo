#include "PromptCacheTracker.hpp"

#include "../llm/LLMProvider.hpp"

#include <algorithm>
#include <utility>

namespace core::agent {

void PromptCacheTracker::record(Exchange exchange) {
    const auto profile = exchange.provider
        ? exchange.provider->prompt_cache_profile(exchange.model)
        : core::llm::PromptCacheProfile::none();
    if (!profile.remote()) {
        reset();
        return;
    }
    const bool same_cache = last_exchange_at_.has_value()
        && provider_.lock() == exchange.provider
        && model_ == exchange.model;
    // Only a provider bound to one endpoint can teach a lifetime; an
    // aggregate (the router) may have served the two requests from
    // different caches.
    if (same_cache && exchange.provider->metadata().has_value()) {
        calibration_->observe(
            core::llm::PromptCacheCalibration::key(exchange.provider_name, exchange.model),
            std::chrono::duration_cast<std::chrono::seconds>(
                exchange.started_at - *last_exchange_at_),
            exchange.usage);
    }
    last_exchange_at_ = exchange.started_at;
    provider_ = exchange.provider;
    provider_name_ = std::move(exchange.provider_name);
    model_ = std::move(exchange.model);
    prompt_tokens_ = exchange.usage.prompt_tokens;
    replay_request_ = PromptCacheGuard::can_keep_warm(profile)
        ? std::move(exchange.request)
        : nullptr;
}

void PromptCacheTracker::restore(Clock::time_point last_exchange_at,
                                 const std::shared_ptr<core::llm::LLMProvider>& provider,
                                 std::string provider_name,
                                 std::string model) {
    reset();
    if (!provider || !provider->prompt_cache_profile(model).remote()) {
        return;
    }
    last_exchange_at_ = last_exchange_at;
    provider_ = provider;
    provider_name_ = std::move(provider_name);
    model_ = std::move(model);
}

PromptCacheState PromptCacheTracker::state(
    const std::shared_ptr<core::llm::LLMProvider>& provider,
    std::string_view model,
    std::int64_t estimated_context_tokens) const {
    // A switched provider or model starts a cache of its own; the old one's
    // age says nothing about what the next request costs.
    if (!last_exchange_at_.has_value() || !provider
        || provider_.lock() != provider || model_ != model) {
        return {};
    }
    return PromptCacheState{
        .last_exchange_at = last_exchange_at_,
        .context_tokens = std::max(prompt_tokens_, estimated_context_tokens),
        .profile = calibration_->refine(calibration_key(),
                                        provider->prompt_cache_profile(model_)),
    };
}

std::optional<PromptCacheTracker::Replay> PromptCacheTracker::replay() const {
    auto provider = provider_.lock();
    if (!replay_request_ || !provider) {
        return std::nullopt;
    }
    return Replay{.request = replay_request_, .provider = std::move(provider), .model = model_};
}

void PromptCacheTracker::renewed(const std::shared_ptr<const core::llm::ChatRequest>& request,
                                 Clock::time_point started_at) noexcept {
    if (request && replay_request_ == request) {
        last_exchange_at_ = started_at;
    }
}

void PromptCacheTracker::reset() noexcept {
    last_exchange_at_.reset();
    provider_.reset();
    provider_name_.clear();
    model_.clear();
    prompt_tokens_ = 0;
    replay_request_.reset();
}

std::string PromptCacheTracker::calibration_key() const {
    return core::llm::PromptCacheCalibration::key(provider_name_, model_);
}

} // namespace core::agent
