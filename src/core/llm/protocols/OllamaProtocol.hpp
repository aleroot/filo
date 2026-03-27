#pragma once

/**
 * @file OllamaProtocol.hpp
 * @brief Ollama NDJSON streaming protocol.
 *
 * Ollama's wire format differs from SSE in two important ways:
 *   - The stream is newline-delimited JSON (NDJSON), one object per line,
 *     with `\n` as the separator (not `\n\n`).
 *   - Tool call arguments arrive as a JSON **object** (not a JSON-encoded
 *     string as in OpenAI), and are stringified by the parser.
 *   - Completion is signalled by `"done": true` inside the JSON object
 *     rather than by a `[DONE]` sentinel or named SSE event.
 *   - No authentication is required for the default local deployment.
 */

#include "ApiProtocol.hpp"
#include <string>
#include <vector>

namespace core::llm::protocols {

/**
 * @brief Result of parsing one NDJSON line from an Ollama streaming response.
 */
struct OllamaNdJsonResult {
    std::string          content;
    std::vector<ToolCall> tools;
    bool                 done = false;
};

/**
 * @brief Parse a single NDJSON line from an Ollama streaming response.
 *
 * Ollama sends one JSON object per line.  Each object has:
 *   - `message.content`    — text chunk (may be empty)
 *   - `message.tool_calls` — tool calls (arguments as JSON objects, not strings)
 *   - `done`               — true when the stream is complete
 *
 * Tool call arguments are converted from JSON objects to JSON-encoded strings
 * (the format the rest of the codebase expects) via `simdjson::minify()`.
 *
 * Pure function — exposed for unit testing.
 */
[[nodiscard]] OllamaNdJsonResult parse_ollama_ndjson_line(std::string_view json_str);

/**
 * @brief Ollama local inference protocol (NDJSON streaming).
 *
 * **Request**: `POST {base_url}/api/chat` with a JSON body produced by
 * `Serializer::serialize()`.
 *
 * **Response**: newline-delimited stream of JSON objects.  Each object has
 * `message.content` (text) and/or `message.tool_calls` (tools), plus a
 * boolean `done` field.  The stream ends when `done: true` is received.
 *
 * **Auth**: none by default; `auth.headers` are forwarded if present for
 * deployments that add a reverse-proxy authentication layer.
 *
 * **Stateless**: each NDJSON line is self-contained.
 */
class OllamaProtocol : public ApiProtocolBase {
public:
    OllamaProtocol() = default;

    [[nodiscard]] std::string  serialize(const ChatRequest& req) const override;
    [[nodiscard]] cpr::Header  build_headers(const core::auth::AuthInfo& auth) const override;
    [[nodiscard]] std::string  build_url(std::string_view base_url,
                                          std::string_view model) const override;

    /// Ollama streams NDJSON — one JSON object per line, not SSE blocks.
    [[nodiscard]] std::string_view event_delimiter() const noexcept override { return "\n"; }
    [[nodiscard]] ParseResult      parse_event(std::string_view raw_event) override;
    [[nodiscard]] std::string_view name()  const noexcept override { return "ollama"; }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        return std::make_unique<OllamaProtocol>(*this);
    }
};

} // namespace core::llm::protocols
