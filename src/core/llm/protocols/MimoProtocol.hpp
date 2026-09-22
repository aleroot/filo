#pragma once

/**
 * @file MimoProtocol.hpp
 * @brief Xiaomi MiMo Chat Completions contract on top of the OpenAI wire format.
 *
 * MiMo is OpenAI-compatible, but the MiMo model gateway adds vendor behavior
 * that must stay on this subclass:
 *   - every MiMo model runs at temperature 1 (the reference client pins it)
 *   - reasoning is always on and streams back as `delta.reasoning_content`;
 *     there is no `reasoning_effort` knob to send
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
    /// Always request stream usage so Token Plan consumption is visible.
    explicit MimoProtocol(bool stream_usage = true)
        : OpenAIProtocol(stream_usage) {}

    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;
    [[nodiscard]] std::string_view name() const noexcept override { return "mimo"; }

    /**
     * MiMo models think by default and expose no effort control, so the
     * request carries no reasoning field. `delta.reasoning_content` is already
     * captured by OpenAIProtocol::parse_event.
     */
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

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<MimoProtocol>(*this);
    }
};

} // namespace core::llm::protocols
