#pragma once

/**
 * @file GrokProtocol.hpp
 * @brief xAI Grok extension of the OpenAI wire protocol.
 *
 * GrokProtocol inherits the full OpenAI Chat Completions wire format and
 * overrides serialization to inject the `reasoning_effort` field when the
 * selected model supports it (currently the grok-3-mini family).
 *
 * Grok 4 and later always-reasoning models do **not** accept this field and
 * return HTTP 400 if it is present — `append_extra_fields()` gates on
 * `grok_supports_reasoning_effort()` to prevent this.
 *
 * ## xAI-Specific Features
 *
 * ### Rate Limit Handling
 * xAI returns standard HTTP rate limit headers:
 *   - `x-ratelimit-limit-requests` / `x-ratelimit-remaining-requests`
 *   - `x-ratelimit-limit-tokens` / `x-ratelimit-remaining-tokens`
 *   - `retry-after` (on 429 errors)
 *
 * These are extracted by `on_response()` and available via `last_rate_limit()`.
 *
 * ### Error Codes
 * xAI uses OpenAI-compatible status codes plus xAI-specific extensions:
 *   - `529` - Server overloaded (retryable, distinct from 429)
 *
 * ### JSON Mode
 * Supported on all Grok models via `ChatRequest::response_format`.
 * Set `response_format.type = ResponseFormat::Type::JsonObject` for
 * guaranteed JSON output. Use with `stream: false` for best results.
 *
 * ## Usage Example
 * @code{.cpp}
 * // JSON mode request
 * ChatRequest req;
 * req.model = "grok-code-fast-1";
 * req.response_format.type = ResponseFormat::Type::JsonObject;
 * req.stream = false;  // JSON mode works best with non-streaming
 * req.messages.push_back({"user", "Generate a JSON object with 'name' and 'age'"});
 *
 * auto provider = std::make_shared<HttpLLMProvider>(
 *     "https://api.x.ai/v1",
 *     ApiKeyCredentialSource::as_bearer(key),
 *     "grok-code-fast-1",
 *     std::make_unique<GrokProtocol>()
 * );
 * @endcode
 */

#include "OpenAIProtocol.hpp"
#include "OpenAIResponsesProtocol.hpp"
#include "GrokBillingUsage.hpp"
#include <memory>
#include <string_view>

namespace core::llm::protocols {

/**
 * @brief Reasoning effort level for models that support it (e.g. grok-3-mini).
 *
 * `None` omits the field entirely, letting the API use its own default.
 * `Medium` is kept as a backwards-compatible alias for `High`.
 */
enum class GrokReasoningEffort { None, Low, Medium, High };

/**
 * @brief Return true if @p model accepts the Chat Completions `reasoning_effort`
 *        field.
 *
 * Currently only the `grok-3-mini` model family supports this top-level
 * parameter. Sending it to Grok 4 or `grok-code-fast-1` causes the API to
 * return HTTP 400.
 */
[[nodiscard]] bool grok_supports_reasoning_effort(std::string_view model) noexcept;

/**
 * @brief Return true if @p model accepts the Responses API
 *        `reasoning:{effort:...}` control.
 *
 * Unlike the Chat Completions `reasoning_effort` top-level field (which Grok 4
 * rejects), the nested Responses-API object is supported by Grok 4.6, 4.5,
 * 4.3, and the Grok Build coding model. Other Grok reasoning models are
 * always-on and do not expose a knob.
 */
[[nodiscard]] bool grok_responses_supports_effort(std::string_view model) noexcept;
[[nodiscard]] bool grok_responses_supports_xhigh_effort(
    std::string_view model) noexcept;

/**
 * @brief xAI Grok protocol — OpenAI format + xAI-specific enhancements.
 *
 * ## Features
 * - `reasoning_effort` injection for grok-3-mini models
 * - xAI-specific error message formatting (including 529 overloaded)
 * - Rate limit header extraction
 * - Full OpenAI wire compatibility
 *
 * ## Authentication
 * `Authorization: Bearer <XAI_API_KEY>`
 *
 * ## Base URL
 * `https://api.x.ai/v1`
 */
class GrokProtocol : public OpenAIProtocol {
public:
    /**
     * @param effort        Reasoning effort level.  Pass `None` to omit the field
     *                      entirely (default; suitable for models that don't support it).
     * @param stream_usage  Append `stream_options:{include_usage:true}` to
     *                      streaming requests (mirrors OpenAIProtocol).
     */
    explicit GrokProtocol(GrokReasoningEffort effort = GrokReasoningEffort::None,
                          bool stream_usage = false)
        : OpenAIProtocol(stream_usage)
        , effort_(effort) {}

    [[nodiscard]] std::string_view name() const noexcept override { return "grok"; }
    [[nodiscard]] ReasoningCapabilities reasoning_capabilities(
        std::string_view model) const noexcept override {
        if (!grok_supports_reasoning_effort(model)) return {};
        return ReasoningCapabilities{ReasoningCapability::Effort};
    }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<GrokProtocol>(*this);
    }

    // ── Response lifecycle hooks (xAI-specific) ──────────────────────────────

    /**
     * @brief Extracts xAI rate limit headers from the HTTP response.
     *
     * Parses xAI-specific headers:
     * - `x-ratelimit-limit-requests` / `x-ratelimit-remaining-requests`
     * - `x-ratelimit-limit-tokens` / `x-ratelimit-remaining-tokens`
     * - `retry-after` (present on 429 responses)
     *
     * The cached rate limit info is available via `last_rate_limit()`.
     */
    void on_response(const HttpResponse& response) override;

    /**
     * @brief Returns a human-readable, xAI-specific error message.
     *
     * Maps xAI status codes to actionable guidance:
     * - **400** — Bad request (malformed JSON, invalid parameters, or
     *             unsupported parameter like `reasoning_effort` on Grok 4)
     * - **401** — Invalid API key
     * - **403** — Permission denied (model access restricted)
     * - **429** — Rate limit exceeded (check `last_rate_limit()` for reset time)
     * - **500** — Internal server error (xAI-side issue, retry)
     * - **529** — Server overloaded (xAI-specific, always retryable with backoff)
     *
     * For 400 errors on Grok 4, suggests checking `reasoning_effort` usage.
     */
    [[nodiscard]] std::string format_error_message(const HttpResponse& response) const override;

    /**
     * @brief Returns true for status codes that warrant automatic retry.
     *
     * Retryable: 408, 409, 429, and transient 5xx responses, including xAI
     * overload and Cloudflare edge failures. Cloudflare origin-TLS failures
     * 525/526 are terminal, as is any response carrying
     * `x-should-retry: false`.
     */
    [[nodiscard]] bool is_retryable(const HttpResponse& response) const noexcept override;

    void prepare_headers(cpr::Header& headers,
                         const ChatRequest& request,
                         std::string_view base_url) override;

    /**
     * @brief Returns rate limit info extracted from the most recent response.
     *
     * Populated by `on_response()`. Returns zero-initialized struct if no
     * response has been processed yet or if headers were absent.
     */
    [[nodiscard]] RateLimitInfo last_rate_limit() const noexcept override { return last_rate_limit_; }

protected:
    /// Append `"reasoning_effort"` when the model supports it.
    void append_extra_fields(std::string&       payload,
                              const ChatRequest& req) const override;

private:
    GrokReasoningEffort effort_;
    RateLimitInfo       last_rate_limit_;  ///< Cached from on_response()
    
    // Helper to safely parse integer headers
    [[nodiscard]] static int32_t parse_int_header(const cpr::Header& headers, 
                                                   std::string_view key) noexcept;
    
    // Parse xAI rate limit headers into RateLimitInfo
    [[nodiscard]] static RateLimitInfo parse_rate_limit_headers(
        const cpr::Header& headers) noexcept;
};

/**
 * @brief xAI Responses API variant used by Grok OAuth session models.
 *
 * Extends the base OpenAI Responses protocol with:
 *  - Grok Build session proxy headers (`x-grok-*`)
 *  - Reasoning-effort control (`reasoning:{effort:...}`) for Grok 4.6 / 4.5 /
 *    4.3 / Grok Build. Grok 4.6 additionally exposes the `xhigh` tier.
 *  - Encrypted reasoning replay (`include:["reasoning.encrypted_content"]`) so
 *    prior reasoning can be carried across turns
 *  - Optional xAI hosted server-side tools (real-time `web_search` and
 *    `x_search`), which the proxy resolves internally
 */
class GrokResponsesProtocol final : public OpenAIResponsesProtocol {
public:
    /**
     * @param service_tier       Optional Responses API service tier override.
     * @param enable_hosted_tools When true (default), advertises the xAI hosted
     *                           `web_search` and `x_search` tools so the model
     *                           can use server-side live/X search.
     * @param default_effort     Provider-configured reasoning effort ("low" /
     *                           "medium" / "high") applied when the session
     *                           leaves effort unset. Ignored for models that do
     *                           not expose the Responses effort knob.
     */
    explicit GrokResponsesProtocol(std::string service_tier = {},
                                   bool enable_hosted_tools = true,
                                   std::string default_effort = {},
                                   std::shared_ptr<IGrokBillingUsageSource>
                                       billing_usage_source = {});

    [[nodiscard]] std::string_view name() const noexcept override {
        return "grok_responses";
    }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        auto cloned = std::make_unique<GrokResponsesProtocol>(
            default_service_tier_,
            enable_hosted_tools_,
            default_effort_,
            billing_usage_source_);
        share_continuity_state_with(*cloned);
        return cloned;
    }

    /// Reports effort support for Grok models that expose it on the Responses
    /// API. Grok 4.6 also advertises its extra-high effort tier.
    [[nodiscard]] ReasoningCapabilities reasoning_capabilities(
        std::string_view model) const noexcept override {
        if (grok_responses_supports_effort(model)) {
            ReasoningCapabilities capabilities{ReasoningCapability::Effort};
            if (grok_responses_supports_xhigh_effort(model)) {
                capabilities = capabilities
                    | ReasoningCapability::XHighEffort;
            }
            return capabilities;
        }
        return {};
    }

    /// Injects the xAI hosted server-side tools when enabled.
    [[nodiscard]] std::string serialize(const ChatRequest& request) const override;

    /// xAI reports model-generation failures inside otherwise-successful SSE
    /// responses. Retry transient failures only before output, honoring the
    /// server retry veto and failing fast on deterministic context overflow.
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;

    void reset_state() override;
    void observe_response_headers(const cpr::Header& headers,
                                  const ChatRequest& request) override;
    [[nodiscard]] std::string format_error_message(
        const HttpResponse& response) const override;

    void prepare_headers(cpr::Header& headers,
                         const ChatRequest& request,
                         std::string_view base_url) override;

    void on_response(const HttpResponse& response) override;
    void enrich_rate_limit(std::string_view base_url,
                           const cpr::Header& request_headers,
                           const HttpResponse& response) override;
    [[nodiscard]] bool is_retryable(
        const HttpResponse& response) const noexcept override;
    [[nodiscard]] RateLimitInfo last_rate_limit() const noexcept override {
        return grok_rate_limit_;
    }

private:
    [[nodiscard]] ConversationContextStrategy conversation_context_strategy()
        const noexcept override {
        return ConversationContextStrategy::ReplayInput;
    }

    bool enable_hosted_tools_;
    std::string default_effort_;
    std::shared_ptr<IGrokBillingUsageSource> billing_usage_source_;
    RateLimitInfo grok_rate_limit_;
    bool stream_retry_vetoed_ = false;
};

} // namespace core::llm::protocols
