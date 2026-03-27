#pragma once

/**
 * @file ApiProtocol.hpp
 * @brief Abstract LLM API wire-protocol interface.
 *
 * ## Architecture
 *
 * A *provider* is a named endpoint + credentials + default model.  A *protocol*
 * is the HTTP wire format that endpoint speaks.  Separating the two means:
 *
 *   - A single protocol class covers every service that shares the same wire
 *     format (e.g. OpenAIProtocol works for OpenAI, Mistral, Groq, Together AI,
 *     llama.cpp server, LM Studio, vLLM, and any other OpenAI-compatible API).
 *   - Users can add new providers in `settings.json` without writing C++ code.
 *   - Vendor quirks (Grok's `reasoning_effort`, Anthropic extended thinking)
 *     are handled by thin subclasses rather than duplicated HTTP boilerplate.
 *
 * ## Protocol hierarchy
 *
 * @code
 * ApiProtocolBase  (this file — pure interface)
 *  ├─ OpenAIProtocol     POST /v1/chat/completions, SSE, OpenAI JSON
 *  │   └─ GrokProtocol  adds reasoning_effort for grok-3-mini family
 *  ├─ AnthropicProtocol  POST /v1/messages, named SSE events, Anthropic JSON
 *  ├─ GeminiProtocol     POST /v1beta/models/{model}:streamGenerateContent
 *  └─ OllamaProtocol     POST /api/chat, NDJSON (newline-delimited JSON)
 * @endcode
 *
 * ## Lifecycle hooks
 *
 * `HttpLLMProvider` drives any concrete protocol through a small set of virtual
 * hooks that fire before serialization and after `session.Post()` completes.
 * This keeps all
 * provider-specific logic inside the protocol subclass and out of the generic
 * HTTP driver:
 *
 *   - `prepare_request()`    — request-side defaults/continuity injection
 *   - `on_response()`        — side-effect hook (rate-limit extraction, metrics)
 *   - `format_error_message()` — human-readable error string for the UI
 *   - `is_retryable()`       — backoff hint for transient failures
 *
 * The base-class implementations are sensible no-op / generic defaults.
 * Override only the hooks your protocol needs.
 *
 * ## Adding a new OpenAI-compatible endpoint (zero C++ required)
 *
 * Add an entry to the `providers` map in `settings.json` with an explicit
 * `api_type` and `base_url`:
 * @code{.json}
 * {
 *   "providers": {
 *     "my-groq": {
 *       "api_type": "openai",
 *       "base_url": "https://api.groq.com/openai/v1",
 *       "api_key": "gsk_...",
 *       "model": "llama-3.3-70b-versatile"
 *     },
 *     "local-lm-studio": {
 *       "api_type": "openai",
 *       "base_url": "http://localhost:1234/v1",
 *       "model": "lmstudio-community/llama-3.2-3b"
 *     }
 *   }
 * }
 * @endcode
 *
 * ## Extending a protocol for a new vendor quirk (one subclass)
 *
 * Override `append_extra_fields()` in OpenAIProtocol to inject extra JSON:
 * @code{.cpp}
 * class MyVendorProtocol : public OpenAIProtocol {
 * protected:
 *     void append_extra_fields(std::string& payload,
 *                              const ChatRequest& req) const override {
 *         // payload has all base fields but no closing '}'
 *         payload += R"(,"my_vendor_param":true)";
 *     }
 * public:
 *     std::string_view name() const noexcept override { return "my_vendor"; }
 *     std::unique_ptr<ApiProtocolBase> clone() const override {
 *         return std::make_unique<MyVendorProtocol>(*this);
 *     }
 * };
 * @endcode
 */

#include "../Models.hpp"
#include "../../auth/ICredentialSource.hpp"
#include <cpr/cpr.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace core::llm::protocols {

// ─────────────────────────────────────────────────────────────────────────────
// UsageWindow — one subscription utilization window
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief A single subscription quota window reported by an OAuth/subscription API.
 *
 * Currently only Anthropic returns these (a 5-hour and a 7-day window).
 * Other providers (OpenAI, Gemini, Grok) do not return window utilization headers.
 * Adding a new window requires no struct changes — only one extra entry in the
 * protocol's header-parsing table.
 */
struct UsageWindow {
    std::string label;       ///< Window identifier shown in the status bar, e.g. "5h", "7d"
    float       utilization; ///< Consumed fraction: 0.0 = nothing used, 1.0+ = over limit
};

// ─────────────────────────────────────────────────────────────────────────────
// RateLimitInfo — populated from HTTP response headers
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Rate limit information extracted from API response headers.
 *
 * All providers:
 *   - requests_limit/remaining : RPM limits (OpenAI, Grok, Kimi)
 *   - tokens_limit/remaining   : TPM limits (OpenAI, Grok, Kimi)
 *   - retry_after              : Seconds to wait on 429 errors
 *
 * Anthropic-specific (API key users):
 *   - anthropic-ratelimit-requests-* / tokens-*
 *
 * Anthropic-specific (OAuth/claude.ai subscription users only):
 *   - usage_windows            : 5-hour and 7-day utilization (0.0–1.0)
 *   - unified_status           : "allowed" or "rate_limited"
 *   - unified_representative_claim : which window is currently authoritative
 *
 * Gemini does not return rate-limit response headers.
 */
struct RateLimitInfo {
    // Request-based limits (RPM)
    int32_t requests_limit     = 0;     ///< Maximum requests per minute allowed
    int32_t requests_remaining = 0;     ///< Requests remaining in current window
    int64_t requests_reset     = 0;     ///< Unix timestamp when request limit resets

    // Token-based limits (TPM)
    int32_t tokens_limit       = 0;     ///< Maximum tokens per minute allowed
    int32_t tokens_remaining   = 0;     ///< Tokens remaining in current window
    int64_t tokens_reset       = 0;     ///< Unix timestamp when token limit resets

    // Retry handling
    int32_t retry_after        = 0;     ///< Seconds to wait before retry (from 429 errors)
    bool    is_rate_limited    = false; ///< True if this response is a 429 rate limit

    // Subscription window utilizations (zero or more, populated by on_response()).
    // Each entry is a named window reported by the provider.  The set is open-ended
    // so new windows can be parsed without changing this struct.
    std::vector<UsageWindow> usage_windows;
    std::string unified_status;               ///< "allowed" or "rate_limited"
    std::string unified_representative_claim; ///< Provider hint, e.g. "five_hour"

    /// Returns the highest utilization across all subscription windows, or 0 if none.
    [[nodiscard]] float max_window_utilization() const noexcept {
        float best = 0.0f;
        for (const auto& w : usage_windows) best = std::max(best, w.utilization);
        return best;
    }

    [[nodiscard]] bool has_data() const noexcept {
        return requests_limit > 0
            || tokens_limit > 0
            || retry_after > 0
            || !usage_windows.empty()
            || !unified_status.empty();
    }

    /// Returns true if quota is below 10% remaining
    [[nodiscard]] bool is_quota_critical() const noexcept {
        if (requests_limit > 0 && requests_remaining < requests_limit * 0.1f) return true;
        if (tokens_limit > 0 && tokens_remaining < tokens_limit * 0.1f) return true;
        return false;
    }

    /// Returns true if quota is below 25% remaining
    [[nodiscard]] bool is_quota_low() const noexcept {
        if (requests_limit > 0 && requests_remaining < requests_limit * 0.25f) return true;
        if (tokens_limit > 0 && tokens_remaining < tokens_limit * 0.25f) return true;
        return false;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// ParseResult
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Result of parsing one discrete streaming event.
 *
 * Returned by ApiProtocolBase::parse_event() for each unit produced by
 * splitting the raw byte stream on ApiProtocolBase::event_delimiter().
 *
 * `chunks` may be empty for no-op events (e.g. a finish_reason SSE line).
 * `done` is set to true exactly once, on the terminal event.
 * `prompt_tokens` / `completion_tokens` are non-zero only on the event that
 * carries usage data (not every event does).
 */
struct ParseResult {
    std::vector<StreamChunk> chunks;        ///< Chunks to forward to the caller.
    bool    done              = false;      ///< True when the stream is complete.
    int32_t prompt_tokens     = 0;          ///< Input token count (non-zero when reported).
    int32_t completion_tokens = 0;          ///< Output token count (non-zero when reported).
    RateLimitInfo rate_limit;               ///< Rate limit info from HTTP headers (if available).
};

// ─────────────────────────────────────────────────────────────────────────────
// HttpResponse — non-owning view passed to response lifecycle hooks
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Read-only, non-owning view of a completed HTTP response.
 *
 * `HttpResponse` is constructed on the stack inside `HttpLLMProvider` from a
 * `cpr::Response` and passed by `const` reference to the virtual lifecycle
 * hooks on `ApiProtocolBase` (`on_response`, `format_error_message`,
 * `is_retryable`).  It is **not** safe to store or copy outside those calls:
 * both `body` and `headers` alias data owned by the `cpr::Response` object.
 *
 * The reference member `headers` intentionally makes this type
 * non-default-constructible and non-copy-assignable, enforcing the
 * non-owning contract at compile time.
 *
 * @par Typical construction in HttpLLMProvider
 * @code{.cpp}
 * cpr::Response r = session.Post();
 * const HttpResponse response{r.status_code, r.text, r.header};
 * protocol->on_response(response);
 * @endcode
 */
struct HttpResponse {
    int                status_code; ///< HTTP status code (200, 429, 500, …).
    std::string_view   body;        ///< Response body; zero-copy view into `cpr::Response::text`.
    const cpr::Header& headers;     ///< Response headers; reference into `cpr::Response::header`.
};

// ─────────────────────────────────────────────────────────────────────────────
// ApiProtocolBase
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Abstract base for all LLM API wire protocols.
 *
 * A concrete subclass encapsulates everything specific to one API family:
 * serialization, HTTP headers, URL construction, and stream-event parsing.
 * HttpLLMProvider drives any HTTP-based protocol through this interface.
 *
 * @par Thread safety
 * `serialize()`, `build_headers()`, and `build_url()` are stateless and safe
 * to call from multiple threads simultaneously.
 *
 * `parse_event()` may be stateful (e.g. AnthropicProtocol accumulates
 * tool-call argument fragments across events).  HttpLLMProvider calls `clone()`
 * once per `stream_response()` invocation so each streaming session receives
 * an independent protocol instance with fresh state.
 */
class ApiProtocolBase {
public:
    virtual ~ApiProtocolBase() = default;

    /**
     * @brief Serialise a ChatRequest into the HTTP request body.
     *
     * Stateless — safe to call concurrently.
     *
     * @param req  The unified chat request (model, messages, tools, options).
     * @return JSON string suitable for use as the POST body.
     */
    [[nodiscard]] virtual std::string
    serialize(const ChatRequest& req) const = 0;

    /**
     * @brief Build HTTP headers, merging protocol headers with auth credentials.
     *
     * Protocol-level headers (Content-Type, API version, Accept) are merged
     * with any headers provided by @p auth (e.g. `Authorization: Bearer …`).
     * Stateless — safe to call concurrently.
     *
     * @param auth  Resolved auth info from ICredentialSource::get_auth().
     */
    [[nodiscard]] virtual cpr::Header
    build_headers(const core::auth::AuthInfo& auth) const = 0;

    /**
     * @brief Construct the full request URL for a streaming call.
     *
     * Most protocols append a fixed path to @p base_url (e.g. `/chat/completions`).
     * GeminiProtocol additionally embeds the model name in the path.
     *
     * @param base_url  Root URL without trailing slash,
     *                  e.g. `"https://api.openai.com/v1"`.
     * @param model     Active model identifier, forwarded to protocols that
     *                  embed it in the URL.
     *
     * Stateless — safe to call concurrently.
     */
    [[nodiscard]] virtual std::string
    build_url(std::string_view base_url, std::string_view model) const = 0;

    /**
     * @brief Byte sequence used to split the raw byte stream into events.
     *
     * - SSE protocols (OpenAI, Anthropic, Gemini): `"\n\n"`
     * - NDJSON protocols (Ollama): `"\n"`
     *
     * Stateless — safe to call concurrently.
     */
    [[nodiscard]] virtual std::string_view
    event_delimiter() const noexcept = 0;

    /**
     * @brief Parse one discrete event into chunks and state signals.
     *
     * Called for each unit produced by splitting the stream on event_delimiter().
     * Strips protocol framing (e.g. `"data: "` prefix), parses JSON, and
     * translates the payload into zero or more StreamChunk values.
     *
     * @param raw_event  One complete event unit from the raw byte stream.
     * @return ParseResult containing any chunks plus done/usage signals.
     *
     * @note May be stateful. See class-level note on `clone()`.
     */
    [[nodiscard]] virtual ParseResult
    parse_event(std::string_view raw_event) = 0;

    /**
     * @brief Short identifier used in logs and diagnostics.
     *
     * Examples: `"openai"`, `"grok"`, `"anthropic"`, `"gemini"`, `"ollama"`.
     */
    [[nodiscard]] virtual std::string_view
    name() const noexcept = 0;

    /**
     * @brief Create an independent copy with fresh per-stream state.
     *
     * HttpLLMProvider calls this once per stream_response() invocation.
     * Stateful protocol members (e.g. AnthropicProtocol's tool-block
     * accumulator) are reset in the clone so concurrent sessions are isolated.
     */
    [[nodiscard]] virtual std::unique_ptr<ApiProtocolBase>
    clone() const = 0;

    // ── Request lifecycle hook ──────────────────────────────────────────────
    //
    // `HttpLLMProvider` resolves the active model, then invokes this hook on
    // the cloned protocol instance before calling `serialize()`.  Protocols
    // can use it to inject protocol-specific defaults or continuity fields
    // into the request while keeping the provider protocol-agnostic.
    //
    // The default implementation is a no-op.

    /**
     * @brief Mutate the request just before serialization.
     *
     * Called once per stream invocation on the per-call cloned protocol
     * instance after model resolution and before `serialize()`.
     *
     * @param request Mutable chat request that will be serialized and sent.
     */
    virtual void prepare_request([[maybe_unused]] ChatRequest& request) {}

    // ── Response lifecycle hooks ─────────────────────────────────────────────
    //
    // These three methods form the *response lifecycle* contract.  Together
    // they let each protocol handle every aspect of a completed HTTP response
    // without any provider-specific logic leaking into HttpLLMProvider.
    //
    // Execution order inside HttpLLMProvider::stream_response():
    //
    //   1. session.Post()          — blocks until the full HTTP response arrives.
    //   2. on_response()           — side-effect hook; always called.
    //   3. format_error_message()  — only when status_code != 200.
    //   4. is_retryable()          — only when status_code != 200.
    //
    // All three methods operate on a `const HttpResponse&` which is valid only
    // for the duration of the call.  Do not store the reference.
    // ────────────────────────────────────────────────────────────────────────

    /**
     * @brief Lifecycle hook called once after the HTTP response is received.
     *
     * Invoked by `HttpLLMProvider` immediately after `session.Post()` returns,
     * regardless of HTTP status code.  For streaming 2xx responses the
     * `WriteCallback` runs concurrently with `Post()`, so the stream may be
     * partially or fully consumed by the time this hook fires.
     *
     * **Intended uses:**
     * - Extract and cache rate-limit metadata from response headers.
     * - Record latency, quota, or circuit-breaker metrics.
     * - Any per-request side effect that needs the full HTTP response context.
     *
     * The **default implementation is a no-op**.  Override only when a protocol
     * needs to act on response-level data that is not available during stream
     * event parsing (i.e. HTTP headers or the final status code).
     *
     * @note `on_response()` is called on the **cloned** instance created by
     *       `clone()` for this streaming session. Most protocols keep state
     *       clone-local, but protocols may intentionally use shared internal
     *       state for cross-request continuity.
     *
     * @param response  Read-only view of the HTTP response.  Valid only for
     *                  the duration of this call — do not store the reference.
     */
    virtual void on_response([[maybe_unused]] const HttpResponse& response) {}

    /**
     * @brief Format a human-readable error string for a non-2xx HTTP response.
     *
     * Called by `HttpLLMProvider` when `status_code != 200`.  The returned
     * string is prepended with `"\n"` and wrapped in
     * `StreamChunk::make_error()` before being forwarded to the streaming
     * callback.
     *
     * The **default** implementation returns a generic
     * `"[HTTP Error: <code> - <body>]"` string suitable for any unknown
     * protocol.  Override to provide provider-specific guidance, for example:
     * - Distinguishing a 529 "server overloaded" from a 429 "rate limited".
     * - Including actionable hints ("check your API key") on a 401.
     * - Surfacing the structured error field from the response JSON body.
     *
     * @param response  The failed HTTP response.
     * @return          Error text **without** a leading newline (the caller
     *                  prepends `"\n"` before forwarding to the UI).
     */
    [[nodiscard]] virtual std::string
    format_error_message(const HttpResponse& response) const {
        return "[HTTP Error: " + std::to_string(response.status_code) +
               " - " + std::string(response.body) + "]";
    }

    /**
     * @brief Indicate whether a failed response is a candidate for automatic retry.
     *
     * Called by `HttpLLMProvider` on non-2xx responses.  A return value of
     * `true` signals that the failure is *transient* and that retrying with
     * exponential backoff is appropriate.  `false` means the error is
     * permanent or unknown — fail immediately.
     *
     * The **default** always returns `false` (fail fast for unknown
     * protocols).  Override when a protocol has well-defined retryable status
     * codes, e.g. 429 / 500 / 502 / 503 / 504 / 529 for Anthropic.
     *
     * @note Declared `noexcept`.  All overrides must guarantee no exceptions;
     *       a throwing override will call `std::terminate`.
     *
     * @param response  The failed HTTP response.
     * @return          `true` if the caller should retry with backoff.
     */
    [[nodiscard]] virtual bool
    is_retryable([[maybe_unused]] const HttpResponse& response) const noexcept {
        return false;
    }
    
    /**
     * @brief Returns the rate-limit info captured by the most recent `on_response()` call.
     *
     * The **default** implementation returns a zero-initialized `RateLimitInfo`.
     * Protocols that track rate limits (e.g., Anthropic) override this to return
     * their cached rate limit state. Useful for UI status bars and adaptive
     * back-off strategies.
     *
     * @return The rate limit info from the last response (or zero-initialized if none).
     */
    [[nodiscard]] virtual RateLimitInfo last_rate_limit() const noexcept {
        return RateLimitInfo{};
    }
};

} // namespace core::llm::protocols
