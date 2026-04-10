#pragma once

/**
 * @file AnthropicProtocol.hpp
 * @brief Anthropic Messages API wire protocol.
 *
 * The Anthropic API differs structurally from OpenAI Chat Completions in
 * several significant ways:
 *   - `system` prompt is a top-level field, not a message with role "system".
 *   - Tool definitions use `input_schema` (not `parameters`).
 *   - Tool call results are `user` messages with `tool_result` content blocks.
 *   - `max_tokens` is **mandatory**.
 *   - SSE events carry a named `event:` line before the `data:` line.
 *   - Tool call arguments are streamed as `input_json_delta` fragments and
 *     must be accumulated until `content_block_stop`.
 *   - Usage is split: input tokens in `message_start`, output tokens in
 *     `message_delta`. With prompt caching enabled, `message_start` usage
 *     may also include cache creation/read token fields that contribute to
 *     effective prompt footprint.
 *   - Extended thinking is enabled via a request-level `thinking` block and
 *     a beta header.
 */

#include "ApiProtocol.hpp"
#include <optional>
#include <string>
#include <vector>

namespace core::llm::protocols {

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Extended thinking configuration for Claude models.
 *
 * When enabled, Claude uses chain-of-thought reasoning before responding.
 * `budget_tokens` caps the thinking token budget (min 1024 per Anthropic docs).
 * Requires a model that supports extended thinking (claude-sonnet-4-5+).
 */
struct AnthropicThinkingConfig {
    bool enabled      = false;
    int  budget_tokens = 10000;
};

/**
 * @brief State for a tool_use content block being reassembled across SSE events.
 *
 * Claude streams tool arguments incrementally via `input_json_delta` events.
 * This struct accumulates the pieces until `content_block_stop`.
 */
struct AnthropicToolBlockState {
    int         index            = -1;
    std::string id;
    std::string name;
    std::string accumulated_args; ///< Built up from `input_json_delta` chunks.
};

/**
 * @brief Stateful SSE event parser for the Anthropic streaming Messages API.
 *
 * Claude's SSE format differs from OpenAI's: each event has an `"event:"` type
 * line plus a `"data:"` JSON line.  Tool calls are spread across multiple events
 * (`content_block_start` → N×`content_block_delta` → `content_block_stop`).
 *
 * This class tracks inter-event state (in-progress tool block) so all parsing
 * logic is unit-testable without network calls.
 *
 * Event type handling:
 *   `content_block_start`  — captures `tool_use` id/name into state; text ignored
 *   `content_block_delta`  — `text_delta` → text output; `input_json_delta` → arg accumulation
 *   `content_block_stop`   — finalises any pending tool call and emits it
 *   `message_stop`         — signals `done = true`
 *   `ping` / `message_start` / `message_delta` / `thinking_delta` — no content output
 */
class AnthropicSSEParser {
public:
    struct Result {
        std::string          text;             ///< Non-empty on `text_delta` events.
        std::vector<ToolCall> completed_tools; ///< Non-empty on `content_block_stop` with a tool.
        bool    done          = false;         ///< True on `message_stop`.
        int32_t input_tokens  = 0;             ///< From `message_start` usage (includes cache token fields when present).
        int32_t output_tokens = 0;             ///< From `message_delta`.
    };

    /**
     * @brief Process a single SSE event.
     * @param event_type  Content of the `"event:"` line.
     * @param json_str    Content of the `"data:"` line.
     */
    Result process_event(std::string_view event_type, std::string_view json_str);

private:
    std::optional<AnthropicToolBlockState> current_tool_;
};

/**
 * @brief Serializer for the Anthropic Messages API.
 *
 * Converts the shared OpenAI-style ChatRequest into the Anthropic format:
 *   - `system` role messages → top-level `"system"` field.
 *   - Assistant messages with tool_calls → `content` blocks (type `"tool_use"`).
 *   - Tool role messages → user messages with `tool_result` content blocks.
 *   - Tools → `"input_schema"` (vs OpenAI `"parameters"`).
 *   - `max_tokens` is always emitted (mandatory in the Anthropic API).
 *
 * Fully testable without network or I/O.
 */
struct AnthropicSerializer {
    static std::string serialize(const ChatRequest& req,
                                 int default_max_tokens          = 8096,
                                 const AnthropicThinkingConfig& thinking = {});
};

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicProtocol
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Anthropic Messages API protocol (used by Claude models).
 *
 * **Request**: `POST {base_url}/v1/messages` with a JSON body serialised by
 * `AnthropicSerializer::serialize()`.  `max_tokens` is always emitted; the
 * default is 8096 tokens.
 *
 * **Response**: named SSE events (`event: {type}\ndata: {json}\n\n`).
 * Tool calls are reassembled across `content_block_start` →
 * N×`content_block_delta` → `content_block_stop` events by `AnthropicSSEParser`.
 *
 * **Auth**: `x-api-key: <key>` header (not Bearer) for API-key auth;
 * supports OAuth via ICredentialSource.
 *
 * **Stateful parsing**: `AnthropicSSEParser` tracks in-progress tool blocks
 * across events.  `HttpLLMProvider` calls `clone()` per stream so each session
 * gets a fresh parser instance.
 */
class AnthropicProtocol : public ApiProtocolBase {
public:
    /**
     * @param thinking           Extended-thinking configuration.  When
     *                           `thinking.enabled` is true the `thinking` block
     *                           and the beta header are added to every request.
     * @param default_max_tokens Fallback when `ChatRequest::max_tokens` is unset.
     */
    explicit AnthropicProtocol(AnthropicThinkingConfig thinking          = {},
                               int                    default_max_tokens = 8096)
        : thinking_(thinking)
        , default_max_tokens_(default_max_tokens) {}

    void prepare_request(ChatRequest& request) override;
    [[nodiscard]] std::string  serialize(const ChatRequest& req) const override;
    [[nodiscard]] cpr::Header  build_headers(const core::auth::AuthInfo& auth) const override;
    [[nodiscard]] std::string  build_url(std::string_view base_url,
                                         std::string_view model) const override;

    [[nodiscard]] std::string_view event_delimiter() const noexcept override { return "\n\n"; }
    [[nodiscard]] ParseResult      parse_event(std::string_view raw_event) override;
    [[nodiscard]] std::string_view name()  const noexcept override { return "anthropic"; }

    /// Clone with a fresh AnthropicSSEParser and cleared rate-limit state.
    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override;

    // ── Response lifecycle hooks ─────────────────────────────────────────────

    /**
     * @brief Extracts and caches Anthropic rate-limit metadata from response headers.
     *
     * Parses the following Anthropic-specific headers into `last_rate_limit_`:
     * - `anthropic-ratelimit-requests-{limit,remaining,reset}`
     * - `anthropic-ratelimit-tokens-{limit,remaining,reset}`
     * - `retry-after` (present on 429 responses)
     * - `anthropic-ratelimit-unified-*` (OAuth / subscription accounts)
     *
     * The cached value is accessible via `last_rate_limit()` and is scoped to
     * the lifetime of this clone (see `ApiProtocolBase::on_response`).
     */
    void on_response(const HttpResponse& response) override;

    /**
     * @brief Returns a provider-specific, actionable error message.
     *
     * Maps Anthropic status codes to descriptive guidance:
     * - **400** — malformed request
     * - **401** — authentication failure (bad API key)
     * - **403** — permission denied (model/feature access)
     * - **429** — rate limit exceeded (back off and retry)
     * - **500** — internal server error
     * - **529** — server overloaded (distinct from 429; always retryable)
     */
    [[nodiscard]] std::string format_error_message(const HttpResponse& response) const override;

    /**
     * @brief Returns `true` for Anthropic status codes that warrant backoff retry.
     *
     * Retryable: 429, 500, 502, 503, 504, 529.
     * Non-retryable: 400, 401, 403, 404, and all other 4xx.
     */
    [[nodiscard]] bool is_retryable(const HttpResponse& response) const noexcept override;

    /**
     * @brief Returns the rate-limit info captured by the most recent `on_response()` call.
     *
     * Always returns a zero-initialised `RateLimitInfo` on a freshly cloned
     * instance.  Populated after the HTTP response arrives.  Useful for
     * quota dashboards and adaptive back-off strategies.
     */
    [[nodiscard]] RateLimitInfo last_rate_limit() const noexcept override { return last_rate_limit_; }

private:
    AnthropicThinkingConfig thinking_;
    int                     default_max_tokens_;
    bool                    request_uses_context_1m_ = false; ///< Set in prepare_request() from `[1m]` suffix.
    std::string             last_requested_model_;    ///< Normalized in prepare_request(); used for better 429 hints.
    AnthropicSSEParser      sse_parser_;          ///< Stateful; reset on clone().
    int32_t                 accumulated_input_  = 0;
    int32_t                 accumulated_output_ = 0;
    RateLimitInfo           last_rate_limit_;     ///< Populated by on_response(); scoped to this clone.
};

} // namespace core::llm::protocols
