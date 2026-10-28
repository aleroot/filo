#pragma once

/**
 * @file KimiProtocol.hpp
 * @brief Moonshot AI Kimi extension of the OpenAI wire protocol.
 *
 * KimiProtocol inherits the full OpenAI Chat Completions wire format and
 * overrides header generation to inject the X-Msh-* headers required by
 * the Kimi API (api.moonshot.ai) and Kimi Code (api.kimi.ai / api.kimi.com).
 *
 * ## Why This Is Necessary
 *
 * The official Kimi CLI sends special X-Msh-* headers with EVERY request
 * (both OAuth token operations AND API calls). These headers are used for
 * device identification and are REQUIRED for the API to accept requests
 * when using OAuth authentication.
 *
 * The required headers are:
 *   - X-Msh-Platform: "kimi_code_cli"
 *   - X-Msh-Version: Kimi client version
 *   - X-Msh-Device-Name: hostname
 *   - X-Msh-Device-Model: OS and architecture info
 *   - X-Msh-Os-Version: OS version
 *   - X-Msh-Device-Id: unique device identifier (persisted)
 *
 * Without these headers, API calls with OAuth tokens will fail with 401
 * "Invalid Authentication" errors even though the OAuth login succeeded.
 *
 * ## Kimi-Specific Features
 *
 * ### Rate Limit Handling
 * Kimi returns standard OpenAI-compatible rate limit headers:
 *   - x-ratelimit-limit-requests / x-ratelimit-remaining-requests
 *   - x-ratelimit-limit-tokens / x-ratelimit-remaining-tokens
 *
 * ### Authentication
 * The Kimi API supports two authentication methods:
 *   1. OAuth (obtained via KimiOAuthFlow) - requires X-Msh-* headers
 *   2. API Key (from platform.kimi.ai) - standard Bearer token
 *
 * When using OAuth, the X-Msh-* headers are REQUIRED in addition to the
 * Authorization header.
 *
 * ## Usage Example
 * @code{.cpp}
 * // OAuth-authenticated Kimi provider
 * auto auth_manager = core::auth::AuthenticationManager::create_with_defaults(
 *     config_dir);
 * auto cred_source = auth_manager.create_credential_source("kimi", provider_config);
 *
 * auto provider = std::make_shared<HttpLLMProvider>(
 *     "https://api.moonshot.ai/v1",
 *     std::move(cred_source),
 *     "kimi-k2-5",
 *     std::make_unique<KimiProtocol>()
 * );
 * @endcode
 */

#include "OpenAIProtocol.hpp"
#include <chrono>
#include <string_view>

namespace core::llm::protocols {

/**
 * @brief Moonshot AI Kimi protocol — OpenAI format + Kimi-specific headers.
 *
 * ## Features
 * - Inherits full OpenAI wire compatibility
 * - Automatically injects X-Msh-* headers for device identification
 * - Supports both OAuth and API key authentication
 * - Rate limit header extraction
 *
 * ## Headers Added
 * Every request includes these additional headers:
 * - X-Msh-Platform: "kimi_code_cli"
 * - X-Msh-Version: Kimi client version
 * - X-Msh-Device-Name: system hostname
 * - X-Msh-Device-Model: OS and architecture (e.g., "Linux 5.15 x86_64")
 * - X-Msh-Os-Version: kernel/OS version
 * - X-Msh-Device-Id: OAuth token device id when available
 */
/// OpenAI Python SDK default timeout is 600s; keep a shorter response-start
/// budget so a hung handshake fails before the idle ceiling.
[[nodiscard]] inline transport::StreamTimeoutPolicy
kimi_stream_timeouts() noexcept {
    return {
        .response_start = std::chrono::seconds(180),
        .inactivity = std::chrono::seconds(600),
    };
}

class KimiProtocol : public OpenAIProtocol {
public:
    // Match official kimi-cli behavior: always request usage in streaming mode.
    explicit KimiProtocol(
        PromptCacheTtl cache_ttl = PromptCacheTtl::Standard)
        : OpenAIProtocol(/*stream_usage=*/true), cache_ttl_(cache_ttl) {}

    [[nodiscard]] std::string_view name() const noexcept override { return "kimi"; }
    [[nodiscard]] ReasoningCapabilities reasoning_capabilities(
        std::string_view model) const noexcept override;

    /**
     * @brief Moonshot's prompt cache economics.
     *
     * A hit is billed at 0.1x input and renews the entry for free under its
     * original TTL; the write is billed separately at 1x input (5-minute TTL)
     * or 2x (1h) for the K3 series. The TTL is requested on the wire via
     * `prompt_cache_options` (omitted for the documented 5-minute default),
     * and a replay of the last request is a hit, so it can keep a cache warm.
     * The profile uses the conservative 5-minute lifetime and write price:
     * existing prefixes retain their original tier, which the API omits.
     * Non-K3 Kimi models publish no cache prices, so their cost ratios stay
     * zero and only manual renewal is offered.
     */
    [[nodiscard]] PromptCacheProfile prompt_cache_profile(
        std::string_view model) const noexcept override;

    /// K3/K2 thinking can sit silent for minutes. Match the OpenAI SDK's
    /// 10-minute request budget with a faster handshake timeout.
    [[nodiscard]] transport::StreamTimeoutPolicy stream_timeouts()
        const noexcept override {
        return kimi_stream_timeouts();
    }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<KimiProtocol>(*this);
    }

    /**
     * @brief Size the completion budget the way kimi-code does.
     *
     * Moonshot publishes no completion ceiling below the context window:
     * `max_completion_tokens` accepts up to the window and the API lowers it
     * automatically when prompt + completion would overflow. The reference
     * client therefore clamps the budget to the room left after the prompt
     * (`resolveMaxCompletionCap`) and, when nothing was configured, fills
     * exactly that room (`kimiUnsetCompletionTokens`). A small static cap
     * instead truncates long turns — and on an always-thinking model it can
     * spend the whole budget on reasoning and return empty content.
     */
    void prepare_request(ChatRequest& request) override;

    /**
     * @brief Serialize request using Kimi-specific message extensions.
     *
     * Unlike the generic OpenAI serializer, Kimi thinking models may require
     * assistant.reasoning_content to be echoed in subsequent tool turns.
     * This override enables that field only for the Kimi protocol.
     */
    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;

    [[nodiscard]] bool supports_video_upload() const noexcept override { return true; }

    void prepare_media_uploads(ChatRequest& request,
                               std::string_view base_url,
                               const core::auth::AuthInfo& auth,
                               const ProtocolRequester& requester) override;

    /**
     * @brief Build HTTP headers, adding Kimi-specific X-Msh-* headers.
     *
     * This overrides the base OpenAIProtocol to inject the X-Msh-* headers
     * that are required by the Kimi API for device identification.
     *
     * The headers are merged as follows:
     * 1. Base protocol headers (Content-Type, Accept)
     * 2. Kimi X-Msh-* headers (device identification)
     * 3. Auth headers from the credential source (Authorization)
     *
     * @param auth Resolved auth info from ICredentialSource::get_auth().
     * @return Complete set of headers for the HTTP request.
     */
    [[nodiscard]] cpr::Header build_headers(const core::auth::AuthInfo& auth) const override;

    /**
     * @brief Returns a human-readable, Kimi-specific error message.
     *
     * Maps Kimi status codes to actionable guidance:
     * - **401** — Authentication failed (check OAuth headers or API key)
     * - **429** — Rate limit exceeded
     * - **500** — Internal server error
     */
    [[nodiscard]] std::string format_error_message(const HttpResponse& response) const override;

    /**
     * @brief Returns true for status codes that warrant automatic retry.
     *
     * Retryable, matching kimi-code's isRetryableGenerateError: 408, 409, 429,
     * 500, 502, 503, 504 and 529 (provider overloaded).
     * Non-retryable: other 4xx (400, 401, 403, 404, 422), plus a 429 whose body
     * reports exhausted quota/balance — deterministic until the account is
     * recharged.
     */
    [[nodiscard]] bool is_retryable(const HttpResponse& response) const noexcept override;

    /**
     * @brief Extract Kimi/OpenAI-compatible rate limit headers from responses.
     *
     * Captures request/token limits plus unified utilization headers when present.
     */
    void on_response(const HttpResponse& response) override;

    /**
     * @brief Optionally enrich rate-limit state via Kimi Code `/usages`.
     *
     * Mirrors kimi-cli behavior: when running against Kimi Code managed
     * endpoints (`.../coding/...`), fetches package window usage data when
     * header-derived quota info is incomplete.
     */
    void enrich_rate_limit(std::string_view base_url,
                           const cpr::Header& request_headers,
                           const HttpResponse& response) override;

    /**
     * @brief Returns the rate-limit info captured by the most recent response.
     */
    [[nodiscard]] RateLimitInfo last_rate_limit() const noexcept override {
        return last_rate_limit_;
    }

    /**
     * @brief Parse SSE event, handling Kimi-specific reasoning_content.
     *
     * Kimi K2.5 models return reasoning_content for thinking tokens in addition
     * to the standard content field. This override extracts both.
     */
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;

protected:
    /**
     * @brief Append Kimi thinking controls for reasoning-capable Kimi models.
     *
     * Filo carries effort as a provider-neutral request hint. Public kimi-k3
     * uses top-level `reasoning_effort`; managed Kimi Code K3 uses
     * `thinking.effort`; K2.x uses `thinking.type`.
     */
    void append_extra_fields(std::string& payload, const ChatRequest& req) const override;

private:
    RateLimitInfo last_rate_limit_;
    PromptCacheTtl cache_ttl_ = PromptCacheTtl::Standard;
};

} // namespace core::llm::protocols
