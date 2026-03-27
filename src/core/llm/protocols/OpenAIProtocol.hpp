#pragma once

/**
 * @file OpenAIProtocol.hpp
 * @brief OpenAI Chat Completions wire protocol (and base for compatible APIs).
 *
 * OpenAIProtocol covers every service that speaks the OpenAI Chat Completions
 * wire format without modification:
 *   - OpenAI (`api.openai.com`)
 *   - Mistral (`api.mistral.ai`)
 *   - DeepSeek (`api.deepseek.com`)
 *   - Groq, Together AI, Fireworks AI, OpenRouter
 *   - llama.cpp HTTP server, LM Studio, vLLM, Jan
 *   - Any other OpenAI-compatible endpoint
 *
 * Vendor quirks that require extra JSON fields are handled by subclassing
 * and overriding protocol behavior in a dedicated subclass (for example,
 * `GrokProtocol` or `KimiProtocol`).
 */

#include "ApiProtocol.hpp"
#include <string>
#include <vector>

namespace core::llm::protocols {

/**
 * @brief Parse a single SSE data-line JSON chunk from any OpenAI-compatible API.
 *
 * Shared by OpenAI, Grok, Mistral, Kimi, and any other service that speaks the
 * OpenAI Chat Completions wire format.  Pure function — no side effects.
 *
 * @param json_str  The JSON string after stripping the leading `"data: "` prefix.
 * @return `{content_text, tool_calls}`.  Both empty on malformed or no-op input.
 */
[[nodiscard]] std::pair<std::string, std::vector<ToolCall>>
parse_openai_sse_chunk(std::string_view json_str);

/**
 * @brief OpenAI Chat Completions wire protocol.
 *
 * **Request**: `POST {base_url}/chat/completions` with a JSON body produced
 * by `Serializer::serialize()`.  When streaming and `stream_usage` is true,
 * `stream_options: {include_usage: true}` is appended so the final SSE chunk
 * carries token counts.
 *
 * **Response**: SSE stream, `\n\n`-delimited events with `data: {json}` lines.
 * Tool calls are streamed incrementally; the `index` field disambiguates
 * parallel calls.  `data: [DONE]` terminates the stream.
 *
 * **Auth**: credentials injected as-is from AuthInfo (typically
 * `Authorization: Bearer <key>` produced by ApiKeyCredentialSource::as_bearer).
 *
 * @par Extending for a vendor quirk
 * @code{.cpp}
 * class DeepSeekProtocol : public OpenAIProtocol {
 * protected:
 *     void append_extra_fields(std::string& payload,
 *                              const ChatRequest& req) const override {
 *         // DeepSeek supports a "reasoning_content" response field;
 *         // no request-side extension needed for now.
 *     }
 * public:
 *     std::string_view name() const noexcept override { return "deepseek"; }
 *     std::unique_ptr<ApiProtocolBase> clone() const override {
 *         return std::make_unique<DeepSeekProtocol>(*this);
 *     }
 * };
 * @endcode
 */
class OpenAIProtocol : public ApiProtocolBase {
public:
    /**
     * @param stream_usage  Append `stream_options:{include_usage:true}` to
     *                      streaming requests.  Set to `true` for the OpenAI
     *                      built-in endpoint; leave `false` for third-party
     *                      OpenAI-compatible APIs that may not support it.
     */
    explicit OpenAIProtocol(bool stream_usage = false)
        : stream_usage_(stream_usage) {}

    [[nodiscard]] std::string   serialize(const ChatRequest& req) const override;
    [[nodiscard]] cpr::Header   build_headers(const core::auth::AuthInfo& auth) const override;
    [[nodiscard]] std::string   build_url(std::string_view base_url,
                                          std::string_view model) const override;

    [[nodiscard]] std::string_view event_delimiter() const noexcept override { return "\n\n"; }
    [[nodiscard]] ParseResult      parse_event(std::string_view raw_event) override;
    [[nodiscard]] std::string_view name()  const noexcept override { return "openai"; }

    void on_response(const HttpResponse& response) override;
    [[nodiscard]] RateLimitInfo last_rate_limit() const noexcept override { return last_rate_limit_; }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<OpenAIProtocol>(stream_usage_);
    }

protected:
    /**
     * @brief Extension point for vendor-specific top-level JSON fields.
     *
     * Called during `serialize()` after all base fields have been written to
     * @p payload but **before** the closing `}`.  Override to append one or
     * more `,"key":value` pairs.  The default implementation is a no-op.
     *
     * @param payload  JSON string being assembled (no closing `}` yet).
     * @param req      The original ChatRequest.
     */
    virtual void append_extra_fields([[maybe_unused]] std::string&      payload,
                                     [[maybe_unused]] const ChatRequest& req) const {}

    bool stream_usage_ = false; ///< Whether to request usage in the stream.

private:
    RateLimitInfo last_rate_limit_;
};

} // namespace core::llm::protocols
