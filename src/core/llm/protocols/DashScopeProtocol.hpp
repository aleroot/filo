#pragma once

/**
 * @file DashScopeProtocol.hpp
 * @brief Alibaba Cloud DashScope extension of the OpenAI wire protocol.
 *
 * DashScopeProtocol wraps the OpenAI Chat Completions wire format for the
 * Alibaba Cloud DashScope service, the inference backend for all Qwen model
 * variants.  DashScope exposes an OpenAI-compatible endpoint at:
 *
 *   https://dashscope.aliyuncs.com/compatible-mode/v1
 *
 * and adds several proprietary extensions handled here:
 *
 * ## Prompt Caching
 * Activated server-side via `X-DashScope-CacheControl: enable`.  This header
 * is always included — the server caches the system prompt and any repeated
 * context prefix automatically, with no changes required to the request body.
 *
 * ## Thinking Mode (Qwen3 models)
 * Qwen3 models (qwen3-coder-plus, qwen3-max, etc.) support extended thinking
 * via `enable_thinking: true` + `thinking_budget: <N>` in the request body.
 * The streaming response then carries `delta.reasoning_content` chunks before
 * the regular `delta.content` chunks — identical to Kimi K2.5 thinking tokens.
 * Pass a non-zero `thinking_budget` to the constructor to enable this mode.
 *
 * ## Rate Limit Headers
 * DashScope returns both its own naming convention and the OpenAI-standard
 * naming.  Both variants are tried when extracting rate-limit information.
 *
 * ## Authentication
 * `Authorization: Bearer <DASHSCOPE_API_KEY>`
 *
 * ## Base URL
 * `https://dashscope.aliyuncs.com/compatible-mode/v1`
 *
 * @par Extending
 * Additional Qwen-specific fields (e.g. repetition_penalty) can be injected
 * by further subclassing and overriding `append_extra_fields()`.
 */

#include "OpenAIProtocol.hpp"

namespace core::llm::protocols {

/**
 * @brief Alibaba Cloud DashScope protocol — OpenAI format + DashScope extensions.
 *
 * ## Features
 * - Inherits full OpenAI Chat Completions wire compatibility
 * - Adds `X-DashScope-CacheControl: enable` for transparent prompt caching
 * - Adds `X-DashScope-UserAgent` for client telemetry
 * - Injects `enable_thinking` + `thinking_budget` for Qwen3 thinking mode
 * - Extracts `reasoning_content` from streaming deltas (thinking tokens)
 * - Handles both DashScope and OpenAI-standard rate-limit header names
 * - DashScope-specific error messages with actionable guidance
 *
 * ## Usage
 * @code{.cpp}
 * // Thinking mode: 8 K budget
 * auto protocol = std::make_unique<DashScopeProtocol>(8192);
 *
 * // Plain text / tool calling (no thinking)
 * auto protocol = std::make_unique<DashScopeProtocol>();
 * @endcode
 */
class DashScopeProtocol : public OpenAIProtocol {
public:
    /**
     * @param thinking_budget  Token budget for Qwen3 extended thinking.
     *                         Zero (default) disables thinking mode entirely.
     *                         Only takes effect on models that support it
     *                         (qwen3-coder-plus, qwen3-max, qwen3-plus, etc.).
     *                         The server silently ignores the field on models
     *                         that do not support thinking, so it is safe to
     *                         always pass a non-zero value for Qwen3 providers.
     */
    explicit DashScopeProtocol(int thinking_budget = 0)
        : OpenAIProtocol(/*stream_usage=*/true)  // DashScope supports usage in stream
        , thinking_budget_(thinking_budget) {}

    [[nodiscard]] std::string_view name() const noexcept override { return "dashscope"; }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<DashScopeProtocol>(*this);
    }

    // ── Request building ─────────────────────────────────────────────────────

    /**
     * @brief Build HTTP headers, adding DashScope-specific headers.
     *
     * Extends the base OpenAI headers with:
     * - `X-DashScope-CacheControl: enable`   — activates server-side prompt caching
     * - `X-DashScope-UserAgent: filo/0.1`    — client identification for telemetry
     *
     * @param auth  Resolved auth info (Bearer token from DASHSCOPE_API_KEY).
     */
    [[nodiscard]] cpr::Header build_headers(const core::auth::AuthInfo& auth) const override;

    // ── Response lifecycle ───────────────────────────────────────────────────

    /**
     * @brief Parse SSE event, extracting both content and reasoning_content.
     *
     * Qwen3 thinking models emit `delta.reasoning_content` chunks during the
     * thinking phase, followed by regular `delta.content` chunks.  This
     * override calls the base parser for standard fields, then makes a second
     * pass to capture any `reasoning_content` and append it as a reasoning
     * chunk to the result.
     */
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;

    /**
     * @brief Extract DashScope rate-limit headers from the HTTP response.
     *
     * Tries both DashScope-native names and OpenAI-standard names:
     * - `x-ratelimit-requests-limit`   / `x-ratelimit-limit-requests`
     * - `x-ratelimit-requests-remaining` / `x-ratelimit-remaining-requests`
     * - `x-ratelimit-tokens-limit`     / `x-ratelimit-limit-tokens`
     * - `x-ratelimit-tokens-remaining` / `x-ratelimit-remaining-tokens`
     * - `retry-after`
     */
    void on_response(const HttpResponse& response) override;

    /**
     * @brief Returns a human-readable, DashScope-specific error message.
     *
     * Parses DashScope's JSON error body (`code` + `message` fields) and maps
     * HTTP status codes to actionable guidance.
     */
    [[nodiscard]] std::string format_error_message(const HttpResponse& response) const override;

    /**
     * @brief Returns true for status codes that warrant automatic retry.
     *
     * Retryable: 429, 500, 502, 503, 504.
     * Non-retryable: 400, 401, 403, 404.
     */
    [[nodiscard]] bool is_retryable(const HttpResponse& response) const noexcept override;

    [[nodiscard]] RateLimitInfo last_rate_limit() const noexcept override {
        return last_rate_limit_;
    }

protected:
    /**
     * @brief Inject Qwen3 thinking-mode fields when `thinking_budget_ > 0`.
     *
     * Appends `,"enable_thinking":true,"thinking_budget":<N>` when the budget
     * is positive.  Safe to include even for non-thinking models — the server
     * ignores unknown fields on those.
     */
    void append_extra_fields(std::string& payload, const ChatRequest& req) const override;

private:
    int           thinking_budget_;  ///< 0 = disabled; >0 = thinking token budget
    RateLimitInfo last_rate_limit_;  ///< Populated by on_response()
};

} // namespace core::llm::protocols
