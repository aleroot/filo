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
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::llm::protocols {

struct GeminiUsageMetadata {
    int32_t prompt_tokens = 0;
    int32_t completion_tokens = 0;
    /// `cachedContentTokenCount`: the part of prompt_tokens served from an implicit or explicit context cache.
    int32_t cached_prompt_tokens = 0;
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
/**
 * @brief Optional wire additions for Cloud Code Assist variants that mirror a
 *        first-party client (Antigravity). Plain Gemini passes none.
 */
struct GeminiRequestExtras {
    /// Role stamped on `systemInstruction` (empty = omit the field).
    std::string system_instruction_role;
    /// Tag the first `functionCall` of every model turn with the bypass
    /// signature (Gemini 3+ rejects unsigned first calls; Filo does not
    /// round-trip thought signatures).
    bool skip_thought_signature_on_first_call = false;
    /// Tag *every* replayed `functionCall` part with the bypass signature.
    /// The Antigravity hub carries one `thoughtSignature` per part, so a turn
    /// that issued several calls has all of them signed.
    bool skip_thought_signature_on_all_calls = false;
    /// Emit one `{"functionDeclarations":[…]}` element per tool instead of a
    /// single element holding every declaration (the Antigravity hub shape).
    bool tool_declaration_per_element = false;
    /// Request-level `sessionId` (empty = omit).
    std::string session_id;
    /// Raw JSON object emitted as request-level `labels` (empty = omit).
    std::string labels_json;
    /// Fixed `maxOutputTokens` ceiling: clamps a caller value, and is used
    /// verbatim when the caller sent none.
    std::optional<int> max_output_tokens_cap;
    /// Ask the backend to stream thought summaries (`thinkingConfig.
    /// includeThoughts`). Without it Gemini 3+ still thinks, but the thoughts
    /// are never returned, so the UI shows no reasoning.
    bool include_thoughts = false;
    /// `thinkingConfig.thinkingLevel`: "LOW" / "MEDIUM" / "HIGH" (empty = omit,
    /// i.e. let the backend apply the model default).
    std::string thinking_level;
    /// `thinkingConfig.thinkingBudget`; -1 asks for dynamic budgeting (the
    /// `*-tiered` SKUs). Mutually exclusive with `thinking_level` in practice.
    std::optional<int> thinking_budget;
};

/// Value of `thoughtSignature` that makes Cloud Code Assist skip validation.
inline constexpr std::string_view kSkipThoughtSignatureValidator =
    "skip_thought_signature_validator";

[[nodiscard]] std::string
serialize_gemini_request(const ChatRequest& req, const std::string& default_model,
                         const GeminiRequestExtras* extras = nullptr);

/// Content, thought-summary and tool-call parts of one Gemini response object.
struct GeminiChunkParts {
    /// `parts[].text` of ordinary (non-thought) parts.
    std::string content;
    /// `parts[].text` of parts flagged `"thought": true` — the model's
    /// thinking summary, which must never be mixed into the answer.
    std::string reasoning;
    std::vector<ToolCall> tools;
};

/**
 * @brief Parse a single Gemini SSE data-line JSON chunk.
 *
 * Gemini streaming format:
 *   `data: {"candidates":[{"content":{"role":"model","parts":[{"text":"..."}]}}]}\n\n`
 *
 * Tool calls arrive complete (not streamed), with a synthetic auto-generated id
 * of the form `"call_gemini_N"`.  Pure function — exposed for unit testing.
 * Also accepts an already-extracted response object (Cloud Code Assist wraps
 * the same shape under `response`); callers pass `simdjson::minify(...)` then.
 *
 * @param json_sv  The JSON after stripping the leading `"data: "` prefix.
 * @return Parsed parts.  All empty on malformed or no-op input.
 */
[[nodiscard]] GeminiChunkParts parse_gemini_sse_chunk(std::string_view json_sv);

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
    [[nodiscard]] std::string model_id(std::string_view model) const override;

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<GeminiProtocol>(*this);
    }
};

} // namespace core::llm::protocols
