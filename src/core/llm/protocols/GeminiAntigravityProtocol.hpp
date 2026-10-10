#pragma once

#include "GeminiCodeAssistProtocol.hpp"

#include "../../auth/GoogleAntigravityIdentity.hpp"

namespace core::llm::protocols {

/// Cloud Code Assist hosts the real Antigravity client uses, in rotation
/// order; single source of truth lives in core::auth::antigravity.
using core::auth::antigravity::kAntigravityEndpoint;
using core::auth::antigravity::kAntigravitySandboxEndpoint;

/// Latest Antigravity client release; single source of truth lives in
/// core::auth::antigravity so the control plane reports the same identity.
using core::auth::antigravity::kAntigravityClientVersion;

/// Per-wire-id Antigravity request constants, captured from the real
/// `antigravity/hub` client against `daily-cloudcode-pa`. `model_enum` is the
/// opaque `labels.model_enum` telemetry token the client tags each request
/// with (empty when the id carries none — Anthropic-backed wire ids such as
/// `claude-sonnet-4-6` are accepted without one). `max_output_tokens` is the
/// fixed `generationConfig.maxOutputTokens` the client sends independent of
/// the thinking budget; the backend rejects Claude above 64000 and enforces
/// the Gemini caps.
struct AntigravityModelWireProfile {
    std::string_view model_enum;
    int max_output_tokens;
};

/// Wire profile for a routed Antigravity model id, or `nullopt` when the id is
/// not one the hub client sends (in that case no telemetry label and no fixed
/// output cap are emitted).
[[nodiscard]] std::optional<AntigravityModelWireProfile>
antigravity_wire_profile(std::string_view model);

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
 *    across turns without per-conversation protocol state; `labels` carries
 *    `last_step_index`, `trajectory_id`, `used_claude`, and — when known —
 *    `model_enum` plus the previous response's `last_execution_id`
 *    (AntigravityConversationState);
 *  - `systemInstruction.role = "user"`, `toolConfig` mode `VALIDATED`;
 *  - the thought-signature bypass on the first function call of Gemini 3+
 *    turns, and the per-model fixed output cap (Claude falls back to the
 *    64000 ceiling);
 *  - `User-Agent` identifying the Antigravity hub and nothing else: the real
 *    client sends no `X-Goog-Api-Client` / `Client-Metadata` on these calls,
 *    so neither does Filo;
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

    void prepare_request(ChatRequest& request) override;
    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;
    [[nodiscard]] cpr::Header build_headers(const core::auth::AuthInfo& auth) const override;
    [[nodiscard]] std::string build_url(std::string_view base_url,
                                        std::string_view model) const override;
    ParseResult parse_event(std::string_view raw_event) override;
    void on_response(const HttpResponse& response) override;
    [[nodiscard]] bool is_retryable(const HttpResponse& response) const noexcept override;
    [[nodiscard]] std::string_view name() const noexcept override { return "gemini_antigravity"; }
    /// The hub serves `v1internal:fetchAvailableModels`, so discovery works.
    [[nodiscard]] bool supports_model_catalog() const noexcept override { return true; }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<GeminiAntigravityProtocol>(*this);
    }

private:
    /// Failed attempts seen by this per-stream clone; selects the host.
    int failed_attempts_ = 0;
    /// Conversation anchor of the request this clone is streaming. Latched by
    /// prepare_request()/serialize() so parse_event() can attribute the
    /// response's `responseId` to the right trajectory (copied by clone()).
    std::string stream_anchor_;
};

} // namespace core::llm::protocols
