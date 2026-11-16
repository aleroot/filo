#pragma once
#include "AnthropicProtocol.hpp"
#include "../MiniMaxUsageClient.hpp"
#include "../MiniMaxModelTraits.hpp"
#include "../transport/StreamResilience.hpp"

#include <chrono>

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
                             int context_window = 0);
    std::string model_id(std::string_view model) const override;
    void prepare_request(ChatRequest& req) override;
    /// Images go through the MiniMax user-message image policy.
    ToolResultMediaPlacement tool_result_media() const noexcept override {
        return ToolResultMediaPlacement::follow_up_user;
    }
    std::string serialize(const ChatRequest& req) const override;
    void prepare_context(ChatRequest& req, std::string_view base_url,
                         const core::auth::AuthInfo& auth,
                         const std::function<bool()>& cancelled,
                         const ProtocolRequester& requester) override;
    cpr::Header build_headers(const core::auth::AuthInfo& auth) const override;
    void prepare_headers(cpr::Header& headers, const ChatRequest& request,
                         std::string_view base_url) override;
    std::string_view name() const noexcept override { return "minimax"; }
    /// api.minimax.io negotiates HTTP/2 when the client offers it. MiniMax
    /// Code's Anthropic SDK uses Node fetch, which stays on HTTP/1.1.
    bool stream_over_http1() const noexcept override { return true; }
    transport::RetryPolicy stream_retry_policy() const noexcept override;
    /// Matches the 600s timeout MiniMax Code's Anthropic SDK advertises
    /// (`X-Stainless-Timeout`). The shared 120s/240s budgets cut off a
    /// max-effort thinking stream and then report it as a missing
    /// `message_stop`.
    transport::StreamTimeoutPolicy stream_timeouts() const noexcept override {
        return {
            .response_start = std::chrono::seconds(600),
            .inactivity = std::chrono::seconds(600),
        };
    }
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
    /// Set once this stream's `message_start` arrives. A later `message_stop`
    /// with no `stop_reason` is a truncated reply, not a success.
    bool saw_message_start_ = false;
};
} // namespace core::llm::protocols
