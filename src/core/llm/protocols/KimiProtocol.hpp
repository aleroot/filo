#pragma once

/**
 * @file KimiProtocol.hpp
 * @brief Moonshot AI Kimi extension of the OpenAI wire protocol.
 *
 * KimiProtocol inherits the full OpenAI Chat Completions wire format and
 * overrides header generation to inject the X-Msh-* headers required by
 * the Kimi API (api.moonshot.cn).
 *
 * ## Why This Is Necessary
 *
 * The official Kimi CLI sends special X-Msh-* headers with EVERY request
 * (both OAuth token operations AND API calls). These headers are used for
 * device identification and are REQUIRED for the API to accept requests
 * when using OAuth authentication.
 *
 * The required headers are:
 *   - X-Msh-Platform: "kimi_cli"
 *   - X-Msh-Version: CLI version (e.g., "1.25.0")
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
 *   2. API Key (from platform.moonshot.cn) - standard Bearer token
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
 *     "https://api.moonshot.cn/v1",
 *     std::move(cred_source),
 *     "kimi-k2-5",
 *     std::make_unique<KimiProtocol>()
 * );
 * @endcode
 */

#include "OpenAIProtocol.hpp"
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
 * - X-Msh-Platform: "kimi_cli"
 * - X-Msh-Version: "1.25.0" (matching official CLI)
 * - X-Msh-Device-Name: system hostname
 * - X-Msh-Device-Model: OS and architecture (e.g., "Linux 5.15 x86_64")
 * - X-Msh-Os-Version: kernel/OS version
 * - X-Msh-Device-Id: persistent UUID (stored in ~/.config/filo/kimi_device_id)
 */
class KimiProtocol : public OpenAIProtocol {
public:
    // Match official kimi-cli behavior: always request usage in streaming mode.
    KimiProtocol() : OpenAIProtocol(/*stream_usage=*/true) {}

    [[nodiscard]] std::string_view name() const noexcept override { return "kimi"; }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<KimiProtocol>(*this);
    }

    /**
     * @brief Serialize request using Kimi-specific message extensions.
     *
     * Unlike the generic OpenAI serializer, Kimi thinking models may require
     * assistant.reasoning_content to be echoed in subsequent tool turns.
     * This override enables that field only for the Kimi protocol.
     */
    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;

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
     * Retryable: 429 (rate limit), 500, 502, 503, 504.
     * Non-retryable: 400, 401, 403, 404.
     */
    [[nodiscard]] bool is_retryable(const HttpResponse& response) const noexcept override;

    /**
     * @brief Extract Kimi/OpenAI-compatible rate limit headers from responses.
     *
     * Captures request/token limits plus unified utilization headers when present.
     */
    void on_response(const HttpResponse& response) override;

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

private:
    RateLimitInfo last_rate_limit_;
};

} // namespace core::llm::protocols
