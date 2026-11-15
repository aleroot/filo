#pragma once
#include "AnthropicAdvisor.hpp"
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
        /// A provider-side advisor consultation the provider reported.
        std::optional<AdvisorEvent> advisor_event;
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
    /// The provider's advisor declaration, already rendered as one JSON tool
    /// entry. It is declared after the client tools, so the cache breakpoint
    /// stays on the last client tool.
    std::string advisor_tool_json;
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
    /// @p advisor_support, when given, is the provider's advisor capability.
    /// The default is Anthropic's, which every Anthropic-wire provider speaks.
    explicit AnthropicProtocol(int default_max_tokens = kAnthropicDefaultMaxTokens,
                               std::shared_ptr<const core::llm::IAdvisorSupport> advisor_support =
                                   std::make_shared<anthropic::AnthropicAdvisorSupport>())
        : default_max_tokens_(default_max_tokens),
          advisor_support_(std::move(advisor_support)) {}
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
    /// The provider's advisor capability.
    [[nodiscard]] std::shared_ptr<const core::llm::IAdvisorSupport>
    advisor_support() const override {
        return advisor_support_;
    }
    /// Records the request's advisor, so the headers and the stream know which
    /// consultation they belong to.
    void prepare_request(ChatRequest& request) override;
    /// Drops the advisor when the provider refuses it, and rebuilds the
    /// request without it. The turn continues; the client is told.
    bool rebuilds_after_rejection(const HttpResponse& response) override;
    /// The refusal, as one advisor event.
    [[nodiscard]] std::vector<StreamChunk> rejection_notice() const override;
    /// A `tool_result` block may hold image blocks.
    ToolResultMediaPlacement tool_result_media() const noexcept override {
        return ToolResultMediaPlacement::native;
    }
protected:
    virtual AnthropicReasoningEmitter reasoning_emitter() const;
    virtual bool preserves_native_assistant_content() const noexcept { return true; }
    /// The advisor declaration for @p req, or empty when it asks for none.
    [[nodiscard]] std::string advisor_tool_json(const ChatRequest& req) const;
    /// The feature flags this request's advisor needs.
    [[nodiscard]] std::vector<std::string_view> advisor_feature_flags_for_request() const;
    int default_max_tokens_;
private:
    AnthropicSSEParser sse_parser_;
    int32_t accumulated_input_ = 0, accumulated_output_ = 0;
    int32_t accumulated_cached_input_ = 0, accumulated_cache_creation_ = 0;
    std::string last_stop_reason_;
    std::shared_ptr<const core::llm::IAdvisorSupport> advisor_support_;
    /// The advisor this request asked for.
    std::optional<AdvisorSpec> advisor_;
    /// True when the request's history carries advisor result blocks: the
    /// provider must keep accepting them even when no advisor runs now.
    bool advisor_history_present_ = false;
    /// True when the provider refused this request's advisor. The request
    /// goes again without it.
    bool advisor_refused_ = false;
    /// The provider's own reason for the refusal.
    std::string advisor_refusal_detail_ = {};
};
} // namespace core::llm::protocols
