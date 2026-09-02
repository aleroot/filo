#pragma once

/**
 * @file ZaiProtocol.hpp
 * @brief Z.ai GLM extensions of the OpenAI Chat Completions protocol.
 *
 * Z.ai uses the OpenAI wire shape, but owns distinct thinking controls,
 * streaming usage placement, business error codes, Coding Plan validation,
 * and account quota reporting. Keeping those concerns in this protocol leaves
 * OpenAIProtocol vendor-neutral and lets HttpLLMProvider remain a generic
 * protocol driver.
 */

#include "OpenAIProtocol.hpp"

namespace core::llm::protocols {

/**
 * @brief Parse the subscription end/renewal date from the Z.ai Coding Plan
 *        `GET /api/biz/subscription/list` response.
 *
 * The response carries an array of subscriptions under `data`, each with
 * fields such as `productName`, `status`, `purchaseTime` and `valid`
 * (the date the current term is valid until). The first parseable expiry
 * timestamp of the first entry is returned.
 *
 * @return Unix timestamp in seconds, or 0 when the payload carries no
 *         parseable end date.
 *
 * Exposed for unit tests.
 */
[[nodiscard]] int64_t parse_zai_subscription_end(std::string_view payload) noexcept;

class ZaiProtocol : public OpenAIProtocol {
public:
    explicit ZaiProtocol(bool stream_usage = false)
        : OpenAIProtocol(stream_usage) {}

    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;
    void on_response(const HttpResponse& response) override;
    [[nodiscard]] bool is_retryable(
        const HttpResponse& response) const noexcept override;
    [[nodiscard]] std::string
    format_error_message(const HttpResponse& response) const override;
    void enrich_rate_limit(std::string_view base_url,
                           const cpr::Header& request_headers,
                           const HttpResponse& response) override;
    [[nodiscard]] RateLimitInfo last_rate_limit() const noexcept override {
        return last_rate_limit_;
    }

    [[nodiscard]] std::string_view name() const noexcept override { return "zai"; }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<ZaiProtocol>(stream_usage_);
    }

protected:
    void append_extra_fields(std::string& payload,
                             const ChatRequest& req) const override;

private:
    RateLimitInfo last_rate_limit_;
};

class ZaiCodingProtocol final : public ZaiProtocol {
public:
    explicit ZaiCodingProtocol(bool stream_usage = false)
        : ZaiProtocol(stream_usage) {}

    void prepare_request(ChatRequest& req) override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "zai_coding";
    }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<ZaiCodingProtocol>(stream_usage_);
    }
};

} // namespace core::llm::protocols
