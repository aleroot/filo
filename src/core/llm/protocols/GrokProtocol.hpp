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
#include <string_view>
#include <optional>

namespace core::llm::protocols {

/**
 * @brief Reasoning effort level for models that support it (e.g. grok-3-mini).
 *
 * `None` omits the field entirely, letting the API use its own default.
 * `Medium` is kept as a backwards-compatible alias for `High`.
 */
enum class GrokReasoningEffort { None, Low, Medium, High };

/**
 * @brief Return true if @p model accepts the `reasoning_effort` field.
 *
 * Currently only the `grok-3-mini` model family supports this parameter.
 * Sending it to Grok 4 or `grok-code-fast-1` causes the API to return HTTP 400.
 */
[[nodiscard]] bool grok_supports_reasoning_effort(std::string_view model) noexcept;

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
     * @param effort  Reasoning effort level.  Pass `None` to omit the field
     *                entirely (default; suitable for models that don't support it).
     */
    explicit GrokProtocol(GrokReasoningEffort effort = GrokReasoningEffort::None)
        : OpenAIProtocol(/*stream_usage=*/false)
        , effort_(effort) {}

    [[nodiscard]] std::string_view name() const noexcept override { return "grok"; }

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
     * Retryable: 429 (rate limit), 500, 502, 503, 504, 529 (overloaded).
     * Non-retryable: 400, 401, 403, 404.
     */
    [[nodiscard]] bool is_retryable(const HttpResponse& response) const noexcept override;

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

} // namespace core::llm::protocols
