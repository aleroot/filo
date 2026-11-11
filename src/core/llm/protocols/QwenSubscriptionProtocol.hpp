#pragma once

/**
 * @file QwenSubscriptionProtocol.hpp
 * @brief OpenAI Chat Completions adapter for the Qwen Subscription gateway.
 *
 * Qwen Code sends subscription traffic through its plain OpenAI-compatible
 * provider (not the DashScope one), so no X-DashScope-* headers, cache
 * anchors or DashScope-only body fields are added here. What the gateway
 * does need is Qwen's thinking switch: `enable_thinking` plus a
 * `thinking_budget` of 4096 / 16384 / 32768 tokens for low / medium / high
 * (xhigh and max map to high, off sends `enable_thinking:false`, and an
 * unset effort is medium), with no competing `reasoning_effort`.
 */

#include "OpenAIProtocol.hpp"

#include <string>

namespace core::llm::protocols {

class QwenSubscriptionProtocol final : public OpenAIProtocol {
public:
    explicit QwenSubscriptionProtocol(std::string default_effort = {})
        : OpenAIProtocol(/*stream_usage=*/true)
        , default_effort_(std::move(default_effort)) {}

    [[nodiscard]] std::string_view name() const noexcept override { return "qwen_subscription"; }
    [[nodiscard]] ReasoningCapabilities reasoning_capabilities(
        std::string_view model) const noexcept override;
    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;
    [[nodiscard]] cpr::Header build_headers(const core::auth::AuthInfo& auth) const override;
    [[nodiscard]] std::string format_error_message(const HttpResponse& response) const override;
    [[nodiscard]] std::unique_ptr<ModelCatalogProvider> model_catalog(
        std::string_view provider_name) const override;
    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<QwenSubscriptionProtocol>(default_effort_);
    }

private:
    std::string default_effort_;
};

} // namespace core::llm::protocols
