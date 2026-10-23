#pragma once

#include "PromptCacheGuard.hpp"
#include "../llm/Models.hpp"
#include "../llm/PromptCacheCalibration.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace core::llm {
class LLMProvider;
}

namespace core::agent {

/// Remembers which provider cache a conversation lives in: who holds it,
/// when it was last read, and — only where a replay renews it — the request
/// to replay. Turns that history into a PromptCacheState for the policy.
///
/// Not synchronized: the owner (Agent) guards it with its history lock.
/// Providers are held weakly so the bookkeeping never keeps a switched-away
/// provider alive.
class PromptCacheTracker {
public:
    using Clock = std::chrono::system_clock;

    /// One successful model request, as the prompt cache saw it.
    struct Exchange {
        std::shared_ptr<core::llm::LLMProvider> provider;
        std::string provider_name;
        std::string model;
        Clock::time_point started_at;
        std::shared_ptr<const core::llm::ChatRequest> request;
        core::llm::TokenUsage usage;
    };

    /// What refresh needs to re-send the last request.
    struct Replay {
        std::shared_ptr<const core::llm::ChatRequest> request;
        std::shared_ptr<core::llm::LLMProvider> provider;
        std::string model;
    };

    explicit PromptCacheTracker(
        core::llm::PromptCacheCalibration& calibration = core::llm::PromptCacheCalibration::shared())
        : calibration_(&calibration) {}

    /// Records a successful request, teaching the calibration what an idle
    /// gap before it revealed about the cache's lifetime.
    void record(Exchange exchange);

    /// Seeds the last exchange of a resumed session.
    void restore(Clock::time_point last_exchange_at,
                 const std::shared_ptr<core::llm::LLMProvider>& provider,
                 std::string provider_name,
                 std::string model);

    /// The cache the next request to @p provider / @p model would read;
    /// untracked when that is not the cache the conversation last used.
    [[nodiscard]] PromptCacheState state(
        const std::shared_ptr<core::llm::LLMProvider>& provider,
        std::string_view model,
        std::int64_t estimated_context_tokens) const;

    [[nodiscard]] std::optional<Replay> replay() const;

    /// A replay of @p request that started at @p started_at renewed the
    /// cache — unless a turn ran meanwhile and renewed it later.
    void renewed(const std::shared_ptr<const core::llm::ChatRequest>& request,
                 Clock::time_point started_at) noexcept;

    /// History was rewritten (compaction, clear, load): the old prefix is gone.
    void reset() noexcept;

private:
    [[nodiscard]] std::string calibration_key() const;

    core::llm::PromptCacheCalibration* calibration_;
    std::optional<Clock::time_point> last_exchange_at_;
    std::weak_ptr<core::llm::LLMProvider> provider_;
    std::string provider_name_;
    std::string model_;
    std::int64_t prompt_tokens_ = 0;
    std::shared_ptr<const core::llm::ChatRequest> replay_request_;
};

} // namespace core::agent
