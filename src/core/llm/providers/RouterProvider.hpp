#pragma once

#include "../LLMProvider.hpp"
#include "../ProviderManager.hpp"
#include "../routing/RouterEngine.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace core::llm::providers {

class RouterProvider final : public LLMProvider {
public:
    RouterProvider(core::llm::ProviderManager& provider_manager,
                   std::shared_ptr<core::llm::routing::RouterEngine> router_engine,
                   std::unordered_map<std::string, std::string> provider_default_models);

    void stream_response(const ChatRequest& request,
                         std::function<void(const StreamChunk&)> callback) override;

    [[nodiscard]] std::string get_last_model() const override;
    [[nodiscard]] bool should_estimate_cost() const override;

    [[nodiscard]] std::string active_policy() const;
    [[nodiscard]] std::string last_route_summary() const;
    [[nodiscard]] std::string last_guardrail_summary() const;

private:
    core::llm::ProviderManager& provider_manager_;
    std::shared_ptr<core::llm::routing::RouterEngine> router_engine_;
    std::unordered_map<std::string, std::string> provider_default_models_;

    mutable std::mutex state_mutex_;
    std::string last_model_;
    std::string last_route_summary_;
    std::string last_guardrail_summary_;
    bool last_should_estimate_cost_ = true;
};

} // namespace core::llm::providers
