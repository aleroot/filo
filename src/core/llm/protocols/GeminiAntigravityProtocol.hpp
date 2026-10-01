#pragma once

#include "GeminiCodeAssistProtocol.hpp"

namespace core::llm::protocols {

/// Cloud Code Assist hosts the real Antigravity client uses, in rotation order.
inline constexpr std::string_view kAntigravityEndpoint =
    "https://daily-cloudcode-pa.googleapis.com";
inline constexpr std::string_view kAntigravitySandboxEndpoint =
    "https://daily-cloudcode-pa.sandbox.googleapis.com";

/// Latest Antigravity client release (update manifest, 2026-10-01). The backend
/// gates newer models on this, so bump it when Google ships a new build.
inline constexpr std::string_view kAntigravityClientVersion = "2.18.1";

/**
 * @brief Cloud Code Assist protocol variant used by the unofficial
 *        "Antigravity" OAuth strategy (see core::auth::GoogleAntigravityOAuthFlow).
 *
 * Same endpoint and response format as gemini_code_assist, but the request
 * mirrors the real `antigravity/hub` client:
 *  - envelope: `userAgent:"antigravity"`, `requestType:"agent"`, a structured
 *    `requestId` (`agent/<agentId>/<ms>/<trajectoryId>/<step>`), plus
 *    request-level `sessionId` and `labels`. The ids are derived from the
 *    conversation (session id, else first user message) so they stay stable
 *    across turns without per-conversation protocol state;
 *  - `systemInstruction.role = "user"`, `toolConfig` mode `VALIDATED`;
 *  - the thought-signature bypass on the first function call of Gemini 3+
 *    turns, and Claude's fixed 64000 output-token ceiling;
 *  - `User-Agent` / `Client-Metadata` headers identifying the Antigravity hub;
 *  - host rotation: each failed attempt (transport error, 408, 429, 5xx) moves
 *    the next retry to the following host of daily -> sandbox -> daily ...
 *    `build_url` reflects that, so HttpLLMProvider needs no endpoint logic;
 *    a user-supplied CODE_ASSIST_ENDPOINT never rotates.
 *
 * This is an unofficial, ToS-violating integration path. See
 * GoogleAntigravityOAuthFlow.hpp for details.
 */
class GeminiAntigravityProtocol : public GeminiCodeAssistProtocol {
public:
    GeminiAntigravityProtocol() = default;

    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;
    [[nodiscard]] cpr::Header build_headers(const core::auth::AuthInfo& auth) const override;
    [[nodiscard]] std::string build_url(std::string_view base_url,
                                        std::string_view model) const override;
    void on_response(const HttpResponse& response) override;
    [[nodiscard]] bool is_retryable(const HttpResponse& response) const noexcept override;
    [[nodiscard]] std::string_view name() const noexcept override { return "gemini_antigravity"; }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<GeminiAntigravityProtocol>(*this);
    }

private:
    /// Failed attempts seen by this per-stream clone; selects the host.
    int failed_attempts_ = 0;
};

} // namespace core::llm::protocols
