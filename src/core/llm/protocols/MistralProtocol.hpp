#pragma once

/**
 * @file MistralProtocol.hpp
 * @brief Mistral Chat Completions contract on top of the OpenAI wire format.
 *
 * Mistral's API is OpenAI-compatible, but mistral-vibe's backend uses a
 * distinct reasoning contract that must stay on this subclass:
 *   - `reasoning_effort` is only `"none"` or `"high"`
 *   - reasoning turns use temperature 1
 *   - assistant thinking is replayed as typed content blocks, not a sibling field
 *   - streams can think for several minutes (720s idle budget, matching Vibe)
 */

#include "OpenAIProtocol.hpp"

#include <chrono>

namespace core::llm::protocols {

/// Vibe's DEFAULT_API_TIMEOUT is 720s; keep a slightly shorter response-start
/// budget than that overall ceiling so a hung handshake fails fast.
[[nodiscard]] inline transport::StreamTimeoutPolicy
mistral_stream_timeouts() noexcept {
    return {
        .response_start = std::chrono::seconds(180),
        .inactivity = std::chrono::seconds(720),
    };
}

class MistralProtocol final : public OpenAIProtocol {
public:
    // Always request usage in the stream so subscription cache hits surface.
    explicit MistralProtocol(bool stream_usage = true)
        : OpenAIProtocol(stream_usage) {}

    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;
    [[nodiscard]] std::string_view name() const noexcept override { return "mistral"; }
    [[nodiscard]] ReasoningCapabilities reasoning_capabilities(
        std::string_view model) const noexcept override;
    [[nodiscard]] transport::StreamTimeoutPolicy stream_timeouts()
        const noexcept override {
        return mistral_stream_timeouts();
    }

    [[nodiscard]] cpr::Header build_headers(
        const core::auth::AuthInfo& auth) const override;
    [[nodiscard]] std::string format_error_message(
        const HttpResponse& response) const override;

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<MistralProtocol>(*this);
    }

protected:
    void append_extra_fields(std::string& payload,
                             const ChatRequest& req) const override;
};

} // namespace core::llm::protocols
