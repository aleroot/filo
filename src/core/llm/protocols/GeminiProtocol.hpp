#pragma once

/**
 * @file GeminiProtocol.hpp
 * @brief Google Gemini streaming API wire protocol.
 *
 * The Gemini API differs from the OpenAI format in several key ways:
 *   - System prompt → `systemInstruction.parts[{text}]` (top-level).
 *   - Messages array is called `contents`; roles are `"user"` / `"model"`.
 *   - Tool definitions use `functionDeclarations`; schema types are uppercase.
 *   - The model name is embedded in the URL path (not in the request body).
 *   - Tool calls arrive **complete** per event (arguments are not streamed).
 *   - Auth is via a `?key=` query param (API key) or `Authorization: Bearer`
 *     (OAuth).
 *   - The stream sometimes ends without a `[DONE]` sentinel; `HttpLLMProvider`
 *     always emits a final `make_final()` chunk when the HTTP response ends.
 */

#include "ApiProtocol.hpp"
#include <string>
#include <vector>

namespace core::llm::protocols {

struct GeminiUsageMetadata {
    int32_t prompt_tokens = 0;
    int32_t completion_tokens = 0;
};

/**
 * @brief Resolve Gemini convenience aliases to concrete model IDs.
 *
 * This mirrors the official Gemini CLI aliases where practical, but defaults
 * to stable production models unless preview selection is requested
 * explicitly or enabled via environment.
 */
[[nodiscard]] std::string
normalize_requested_gemini_model(std::string_view raw_model);

/**
 * @brief Serialise a ChatRequest into the Gemini request body format.
 *
 * Exposed as a free function so it can be unit-tested independently.
 * The `default_model` parameter is unused when the model name is already
 * embedded in the URL; it is accepted for API symmetry.
 */
[[nodiscard]] std::string
serialize_gemini_request(const ChatRequest& req, const std::string& default_model);

/**
 * @brief Parse a single Gemini SSE data-line JSON chunk.
 *
 * Gemini streaming format:
 *   `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"..."}]}}]}\n\n`
 *
 * Tool calls arrive complete (not streamed), with a synthetic auto-generated id
 * of the form `"call_gemini_N"`.  Pure function — exposed for unit testing.
 *
 * @param json_sv  The JSON after stripping the leading `"data: "` prefix.
 * @return `{content_text, tool_calls}`.  Both empty on malformed or no-op input.
 */
[[nodiscard]] std::pair<std::string, std::vector<ToolCall>>
parse_gemini_sse_chunk(std::string_view json_sv);

/**
 * @brief Extract token usage metadata from a Gemini response object.
 *
 * Accepts the bare Gemini/Vertex-style response JSON containing an optional
 * `usageMetadata` object and returns the prompt/completion counts when present.
 */
[[nodiscard]] GeminiUsageMetadata
extract_gemini_usage_metadata(std::string_view json_sv);

/**
 * @brief Google Gemini streaming API protocol.
 *
 * **Request**: `POST {base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse`
 * (plus `&key=…` if using API key auth).
 *
 * **Response**: `data: {json}\n\n` SSE events.  Each event carries a
 * `candidates[0].content.parts` array with text or `functionCall` parts.
 * Unlike OpenAI, function call arguments are delivered complete in a single event.
 *
 * **Stateless**: Gemini SSE events are self-contained; no cross-event state needed.
 */
class GeminiProtocol : public ApiProtocolBase {
public:
    GeminiProtocol() = default;

    void                        prepare_request(ChatRequest& request) override;
    [[nodiscard]] std::string  serialize(const ChatRequest& req) const override;
    [[nodiscard]] cpr::Header  build_headers(const core::auth::AuthInfo& auth) const override;
    [[nodiscard]] std::string  build_url(std::string_view base_url,
                                          std::string_view model) const override;

    [[nodiscard]] std::string_view event_delimiter() const noexcept override { return "\n\n"; }
    [[nodiscard]] ParseResult      parse_event(std::string_view raw_event) override;
    [[nodiscard]] std::string_view name()  const noexcept override { return "gemini"; }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<GeminiProtocol>(*this);
    }
};

} // namespace core::llm::protocols
