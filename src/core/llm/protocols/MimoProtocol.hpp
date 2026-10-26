#pragma once

/**
 * @file MimoProtocol.hpp
 * @brief Xiaomi MiMo Chat Completions contract on top of the OpenAI wire format.
 *
 * MiMo is OpenAI-compatible, but the MiMo model gateway adds vendor behavior
 * that must stay on this subclass:
 *   - every MiMo model runs at temperature 1 (the reference client pins it)
 *   - thinking is on by default and toggled with `thinking:{type}`; there is
 *     no effort scale, and reasoning tokens count against
 *     `max_completion_tokens`
 *   - prefix caching is automatic, so the stable prompt leads and the
 *     per-turn system state trails the conversation (a cache hit costs
 *     1/120 of a miss)
 *   - `X-Mimo-Source` identifies the client to the Token Plan gateway
 *   - moderation (421) and risk-control (441) blocks arrive under HTTP 400,
 *     with the real reason in `error.param` rather than `error.message`
 *   - cold-path time-to-first-token after a context rebuild can approach five
 *     minutes, so the idle budget is far wider than the OpenAI default
 */

#include "OpenAIProtocol.hpp"

#include <chrono>

namespace core::llm::protocols {

/**
 * MiMo Code's DEFAULT_OPENAI_HEADER_TIMEOUT (300s) and DEFAULT_CHUNK_TIMEOUT
 * (480s). The wide inactivity budget is deliberate: it bounds a single-attempt
 * SSE stall on a gateway whose cold path can go minutes without a chunk.
 */
[[nodiscard]] inline transport::StreamTimeoutPolicy
mimo_stream_timeouts() noexcept {
    return {
        .response_start = std::chrono::seconds(300),
        .inactivity = std::chrono::seconds(480),
    };
}

class MimoProtocol final : public OpenAIProtocol {
public:
    /// Stream usage is always requested so Token Plan consumption is visible.
    /// @param default_effort  Provider-configured effort for requests that name none.
    explicit MimoProtocol(std::string default_effort = {})
        : OpenAIProtocol(/*stream_usage=*/true),
          default_effort_(std::move(default_effort)) {}

    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;
    [[nodiscard]] std::string_view name() const noexcept override { return "mimo"; }

    /// Thinking is an on/off switch for MiMo models: `/effort off` disables
    /// it and every other level maps to on.
    [[nodiscard]] ReasoningCapabilities reasoning_capabilities(
        std::string_view model) const noexcept override;

    [[nodiscard]] transport::StreamTimeoutPolicy stream_timeouts()
        const noexcept override {
        return mimo_stream_timeouts();
    }

    [[nodiscard]] cpr::Header build_headers(
        const core::auth::AuthInfo& auth) const override;
    [[nodiscard]] std::string format_error_message(
        const HttpResponse& response) const override;
    /// An `insufficient_quota` error is final; retrying it only burns time.
    [[nodiscard]] bool is_retryable(
        const HttpResponse& response) const noexcept override;

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<MimoProtocol>(*this);
    }

private:
    std::string default_effort_;
};

} // namespace core::llm::protocols
