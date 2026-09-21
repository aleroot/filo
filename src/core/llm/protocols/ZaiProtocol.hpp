#pragma once

/**
 * @file ZaiProtocol.hpp
 * @brief Z.ai GLM extensions of the OpenAI Chat Completions and Anthropic
 *        Messages protocols.
 *
 * General API (`zai`) stays on OpenAI Chat Completions at /api/paas/v4.
 * Coding Plan (`zai-coding`) uses Anthropic Messages at /api/anthropic, which
 * is the official ZCode / GLM Coding Plan wire. Vendor-specific thinking
 * maps, business error codes, quota reporting, and stream idle budgets live
 * here so OpenAIProtocol and AnthropicProtocol stay vendor-neutral.
 */

#include "AnthropicProtocol.hpp"
#include "OpenAIProtocol.hpp"

#include "../transport/StreamResilience.hpp"

#include <chrono>
#include <cstdint>

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

/// Drops every recorded thinking-signature rejection. Exposed for unit
/// tests so the process-global registry can be reset between cases.
void zai_clear_signature_rejection_cache_for_testing();

[[nodiscard]] inline transport::StreamTimeoutPolicy zai_stream_timeouts() noexcept {
    return {
        .response_start = std::chrono::seconds(180),
        .inactivity = std::chrono::seconds(600),
    };
}

class ZaiProtocol : public OpenAIProtocol {
public:
    explicit ZaiProtocol(bool stream_usage = false)
        : OpenAIProtocol(stream_usage) {}

    [[nodiscard]] transport::RetryPolicy stream_retry_policy()
        const noexcept override;

    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;
    [[nodiscard]] ReasoningCapabilities reasoning_capabilities(
        std::string_view model) const noexcept override;
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

    [[nodiscard]] transport::StreamTimeoutPolicy stream_timeouts()
        const noexcept override {
        return zai_stream_timeouts();
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

class ZaiCodingProtocol final : public AnthropicProtocol {
public:
    ZaiCodingProtocol() = default;

    void prepare_request(ChatRequest& req) override;
    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;
    [[nodiscard]] cpr::Header build_headers(
        const core::auth::AuthInfo& auth) const override;
    void prepare_headers(cpr::Header& headers,
                         const ChatRequest& request,
                         std::string_view base_url) override;
    [[nodiscard]] ReasoningCapabilities reasoning_capabilities(
        std::string_view model) const noexcept override;
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
    void reset_state() override;

    [[nodiscard]] transport::StreamTimeoutPolicy stream_timeouts()
        const noexcept override {
        return zai_stream_timeouts();
    }

    [[nodiscard]] transport::RetryPolicy stream_retry_policy()
        const noexcept override;

    [[nodiscard]] std::string_view name() const noexcept override {
        return "zai_coding";
    }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<ZaiCodingProtocol>();
    }

protected:
    /// GLM Coding Plan reasoning contract (thinking.type +
    /// output_config.effort), replacing the built-in Claude envelope for
    /// serialized requests. This is the only Z.ai-specific serialization
    /// logic, and it lives here — never in the generic base.
    [[nodiscard]] AnthropicReasoningEmitter reasoning_emitter() const override;

private:
    RateLimitInfo last_rate_limit_;

    // Model bound by prepare_request for the request currently being
    // streamed. Continuation items produced by parse_event are stamped with
    // it so serialize() can refuse to replay signed thinking across models
    // (GLM rejects foreign signatures with a 400).
    std::string current_model_;

    // Fingerprint of the request most recently serialized by this instance.
    // on_response() records it when the backend rejects the request's
    // thinking-block signatures; the next serialize() of the same history
    // strips the rejected blocks (ZCode-style repair, one shot). serialize()
    // is const, so the bookkeeping is mutable.
    mutable std::uint64_t last_request_fingerprint_ = 0;
    mutable bool has_request_fingerprint_ = false;
};

} // namespace core::llm::protocols
