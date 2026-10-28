#pragma once
#include "AnthropicProtocol.hpp"
#include "../MiniMaxUsageClient.hpp"
#include "../MiniMaxModelTraits.hpp"

namespace core::llm::protocols {
/// MiniMax business code from any MiniMax JSON body (`error.code`,
/// `base_resp.status_code` or a trailing "(code)"), 0 when there is none.
[[nodiscard]] int minimax_error_code(std::string_view body);
/// One readable failure with actionable guidance, shared by every MiniMax
/// endpoint (Messages, quota, web search).
[[nodiscard]] std::string describe_minimax_error(int http_status, std::string_view body);

/// MiniMax policy over the vendor-neutral Messages encoder/parser.
class MiniMaxProtocol final : public AnthropicProtocol {
public:
    explicit MiniMaxProtocol(bool subscription = false, std::string default_effort = {},
                             std::shared_ptr<IMiniMaxUsageSource> usage = {},
                             int context_window = kMiniMaxContextWindow);
    std::string model_id(std::string_view model) const override;
    void prepare_request(ChatRequest& req) override;
    std::string serialize(const ChatRequest& req) const override;
    void prepare_context(ChatRequest& req, std::string_view base_url,
                         const core::auth::AuthInfo& auth,
                         const std::function<bool()>& cancelled,
                         const ProtocolRequester& requester) override;
    cpr::Header build_headers(const core::auth::AuthInfo& auth) const override;
    void prepare_headers(cpr::Header& headers, const ChatRequest& request,
                         std::string_view base_url) override;
    std::string_view name() const noexcept override { return "minimax"; }
    transport::RetryPolicy stream_retry_policy() const noexcept override;
    ReasoningCapabilities reasoning_capabilities(std::string_view model) const noexcept override;
    std::unique_ptr<ModelCatalogProvider> model_catalog(std::string_view provider) const override;
    ParseResult parse_event(std::string_view raw_event) override;
    std::unique_ptr<ApiProtocolBase> clone() const override;
    void on_response(const HttpResponse& response) override;
    void enrich_rate_limit(std::string_view base_url, const cpr::Header& headers,
                           const HttpResponse& response) override;
    RateLimitInfo last_rate_limit() const noexcept override { return rate_limit_; }
    std::string format_error_message(const HttpResponse& response) const override;
    bool is_retryable(const HttpResponse& response) const noexcept override;
    void reset_state() override;
private:
    int context_window_;
    bool subscription_;
    std::string default_effort_;
    std::shared_ptr<IMiniMaxUsageSource> usage_;
    RateLimitInfo rate_limit_;
};
} // namespace core::llm::protocols
