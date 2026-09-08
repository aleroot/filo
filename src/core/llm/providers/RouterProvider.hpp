#pragma once

#include "../LLMProvider.hpp"
#include "../ProviderManager.hpp"
#include "../routing/ProviderHealth.hpp"
#include "../routing/RouterEngine.hpp"

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace core::llm::providers {

class RouterProvider final : public LLMProvider {
public:
    RouterProvider(core::llm::ProviderManager& provider_manager,
                   std::shared_ptr<core::llm::routing::RouterEngine> router_engine,
                   std::unordered_map<std::string, std::string> provider_default_models,
                   bool isolate_target_requests = false,
                   std::shared_ptr<core::llm::routing::ProviderHealthRegistry> health = nullptr);

    void stream_response(const ChatRequest& request,
                         std::function<void(const StreamChunk&)> callback) override;

    [[nodiscard]] std::string get_last_model() const override;
    [[nodiscard]] ProviderCapabilities capabilities() const override;
    [[nodiscard]] std::shared_ptr<LLMProvider> fork_for_parallel_request() const override;
    [[nodiscard]] bool should_estimate_cost() const override;
    void cancel() override;
    void reset_conversation_state() override;

    [[nodiscard]] std::string active_policy() const;
    [[nodiscard]] bool set_active_policy(std::string policy_name);
    [[nodiscard]] std::string last_route_summary() const;
    [[nodiscard]] std::string last_guardrail_summary() const;

    /**
     * @brief Parks the current stream until a provider rate-limit reset.
     *
     * Emits non-error status chunks (waiting notice, throttled countdown
     * updates) so the transcript shows why the turn is paused.  Sleeps in
     * short ticks on the system clock so a laptop suspend naturally eats
     * into (or exhausts) the remaining wait.
     *
     * @return true when the reset moment was reached and routing should be
     *         retried; false when the wait was cancelled.
     */
    [[nodiscard]] bool wait_for_reset(
        std::chrono::system_clock::time_point target,
        const std::string& provider,
        const std::function<void(const StreamChunk&)>& callback);

private:
    core::llm::ProviderManager& provider_manager_;
    std::shared_ptr<core::llm::routing::RouterEngine> router_engine_;
    std::unordered_map<std::string, std::string> provider_default_models_;

    // Cross-request provider health (rate-limit cooldowns, breaker state).
    // Shared with forked instances so parallel requests pool their knowledge.
    std::shared_ptr<core::llm::routing::ProviderHealthRegistry> health_;

    // Set by cancel() to abort an active wait_for_reset park.
    std::atomic<bool> wait_aborted_{false};

    mutable std::mutex state_mutex_;
    std::string last_model_;
    std::string last_route_summary_;
    std::string last_guardrail_summary_;
    bool last_should_estimate_cost_ = true;
    bool isolate_target_requests_ = false;
    std::shared_ptr<LLMProvider> active_isolated_provider_;
};

} // namespace core::llm::providers
