#pragma once
#include "ApiProtocol.hpp"
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace core::llm::protocols {
struct AnthropicToolBlockState {
    int index = -1;
    std::string id, name, accumulated_args, initial_args;
};
struct AnthropicContinuationBlockState {
    int index = -1;
    std::string type, thinking, signature, initial_payload;
};
/**
 * @brief Stateful SSE event parser for the Anthropic streaming Messages API.
 *
 * Messages SSE differs from OpenAI's: each event has an `"event:"` type
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
 *   `message_stop`         — signals `done = true`; marks incomplete tool_use if still open
 *   `message_delta`        — captures stop_reason and usage
 *   `ping` / `message_start` / `thinking_delta` — no content output
 */
class AnthropicSSEParser {
public:
    struct Result {
        std::string          text;             ///< Non-empty on `text_delta` events.
        /// Display-only thinking text streamed on `thinking_delta` events. This
        /// is a copy of the chain-of-thought surfaced for the UI; the signed,
        /// authoritative thinking block is preserved separately in
        /// `continuation_items` for upstream replay and is never derived from it.
        std::string          reasoning_delta;
        std::vector<ToolCall> completed_tools; ///< Non-empty on `content_block_stop` with a tool.
        std::vector<ContinuationItem> continuation_items; ///< Completed signed thinking blocks.
        bool    done          = false;         ///< True on `message_stop`.
        bool    stream_started = false;        ///< True on `message_start`.
        bool    stream_error   = false;        ///< True on `event: error`.
        bool    retryable_stream_error = false; ///< True for transient stream errors.
        int32_t input_tokens  = 0;             ///< From `message_start` usage (includes cache token fields when present).
        int32_t output_tokens = 0;             ///< From `message_delta`.
        int32_t cached_input_tokens = 0;
        int32_t cache_creation_input_tokens = 0;
        bool input_usage_reported = false;
        std::string error_type;
        std::string error_message;
        std::string stop_reason;               ///< From `message_delta.delta.stop_reason`.
        std::size_t prefix_binding_mismatches = 0; ///< Reasoning dropped after context edits.
        std::size_t model_binding_mismatches = 0;  ///< Reasoning dropped after a model switch.
        bool incomplete_tool_call = false;      ///< True if message_stop arrives before content_block_stop for tool_use.
    };

    /**
     * @brief Process a single SSE event.
     * @param event_type  Content of the `"event:"` line.
     * @param json_str    Content of the `"data:"` line.
     */
    Result process_event(std::string_view event_type, std::string_view json_str);

private:
    std::optional<AnthropicToolBlockState> current_tool_;
    std::optional<AnthropicContinuationBlockState> current_continuation_;
    int text_index_ = -1;
    std::optional<std::string> current_text_;
    std::vector<std::string> replay_blocks_;
    bool replay_complete_ = true;
    int64_t uncached_input_ = 0;
    int64_t cached_input_ = 0;
    int64_t cache_creation_input_ = 0;
};

using AnthropicReasoningEmitter = std::function<void(std::string&, const ChatRequest&)>;

/// `max_tokens` is mandatory on the Messages API; used when neither the
/// request nor the model registry supplies one.
inline constexpr int kAnthropicDefaultMaxTokens = 8096;

/// Opt-in endpoint policy. The Messages encoder makes no vendor/model decisions.
struct AnthropicSerializationOptions {
    std::string system_prefix;
    bool system_as_text = false;
    bool cache_control = false;
    /// Requested cache lifetime for every `cache_control` marker. Extended
    /// asks for Anthropic's one-hour cache; all markers share one lifetime
    /// because the API requires longer lifetimes to precede shorter ones.
    PromptCacheTtl cache_ttl = PromptCacheTtl::Standard;
    bool image_urls = false;
    std::optional<core::tools::schema::StrictDialect> tool_schema_dialect;
    std::function<bool(std::string_view)> admit_strict_tool;
    bool omit_historical_thinking = false;
    AnthropicReasoningEmitter extra_fields;
};
struct AnthropicSerializer {
    static int max_tokens(const ChatRequest& req, int fallback);
    static std::string serialize(const ChatRequest& req, int default_max_tokens = kAnthropicDefaultMaxTokens,
                                const AnthropicSerializationOptions& options = {});
};

/// Anthropic Messages wire format: encoding, SSE assembly and usage accounting.
/// Vendor policy belongs in a concrete adapter, never in this class.
class AnthropicProtocol : public ApiProtocolBase {
public:
    explicit AnthropicProtocol(int default_max_tokens = kAnthropicDefaultMaxTokens) : default_max_tokens_(default_max_tokens) {}
    std::string serialize(const ChatRequest& req) const override;
    cpr::Header build_headers(const core::auth::AuthInfo& auth) const override;
    std::string build_url(std::string_view base_url, std::string_view model) const override;
    std::string_view event_delimiter() const noexcept override { return "\n\n"; }
    bool requires_terminal_event() const noexcept override { return true; }
    ParseResult parse_event(std::string_view raw_event) override;
    std::string_view name() const noexcept override { return "anthropic"; }
    std::unique_ptr<ApiProtocolBase> clone() const override;
    void reset_state() override;
    bool accepts_legacy_continuation() const noexcept override { return false; }
protected:
    virtual AnthropicReasoningEmitter reasoning_emitter() const;
    virtual bool preserves_native_assistant_content() const noexcept { return true; }
    int default_max_tokens_;
private:
    AnthropicSSEParser sse_parser_;
    int32_t accumulated_input_ = 0, accumulated_output_ = 0;
    int32_t accumulated_cached_input_ = 0, accumulated_cache_creation_ = 0;
    std::string last_stop_reason_;
};
} // namespace core::llm::protocols
