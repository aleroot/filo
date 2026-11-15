#pragma once

#include "ApiProtocol.hpp"
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_map>

namespace core::llm::codex {
class IUsageSource;
}

namespace core::llm {
class IProviderClientIdentitySource;
}

namespace core::llm::protocols {

/**
 * @brief OpenAI Responses API wire protocol.
 *
 * Request shape:
 *   POST {base_url}/responses
 *
 * Stream shape:
 *   SSE events such as:
 *   - response.output_text.delta
 *   - response.output_item.done
 *   - response.completed
 */
class OpenAIResponsesProtocol : public ApiProtocolBase {
public:
    explicit OpenAIResponsesProtocol(bool include_reasoning_encrypted = false,
                                     std::string default_service_tier = {})
        : include_reasoning_encrypted_(include_reasoning_encrypted)
        , default_service_tier_(std::move(default_service_tier)) {}

    [[nodiscard]] std::string serialize(const ChatRequest& req) const override;
    [[nodiscard]] cpr::Header build_headers(const core::auth::AuthInfo& auth) const override;
    [[nodiscard]] std::string build_url(std::string_view base_url,
                                        std::string_view model) const override;

    void prepare_request(ChatRequest& request) override;
    void reset_state() override;
    void reset_attempt_state() override;

    [[nodiscard]] std::string_view event_delimiter() const noexcept override { return "\n\n"; }
    [[nodiscard]] bool requires_terminal_event() const noexcept override { return true; }
    [[nodiscard]] bool uses_provider_reasoning_catalog() const noexcept override { return true; }
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;
    [[nodiscard]] std::string_view name() const noexcept override { return "openai_responses"; }
    [[nodiscard]] ReasoningCapabilities reasoning_capabilities(
        std::string_view model) const noexcept override;

    void on_response(const HttpResponse& response) override;
    void observe_response_headers(const cpr::Header& headers,
                                  const ChatRequest& request) override;
    [[nodiscard]] bool is_retryable(
        const HttpResponse& response) const noexcept override;
    [[nodiscard]] RateLimitInfo last_rate_limit() const noexcept override { return last_rate_limit_; }
    [[nodiscard]] const std::string& last_response_id() const noexcept { return last_response_id_; }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        auto cloned = std::make_unique<OpenAIResponsesProtocol>(
            include_reasoning_encrypted_, default_service_tier_);
        share_continuity_state_with(*cloned);
        return cloned;
    }

protected:
    [[nodiscard]] virtual bool preserves_output_messages() const noexcept { return true; }

    /// How a failed response is classified, and when the request may run again.
    ///
    /// A provider names the class it failed with, and only the classes a
    /// retry cannot fix are terminal. codex-rs states the classes for the
    /// ChatGPT backend (codex-api/src/sse/responses_error.rs); the generic
    /// Responses API keeps to the classes it knows.
    struct FailureVerdict {
        bool retryable = false;
        /// Seconds the server asks the client to wait, when it says so.
        std::optional<int32_t> retry_after_seconds = {};
    };

    /// Classify a failure event by its error class.
    ///
    /// @param code    The error class the event names.
    /// @param message The error message, which may state when to come back.
    /// @param advice  The headers the event carries, which may state it too.
    [[nodiscard]] virtual FailureVerdict classify_failure(
        std::string_view code,
        std::string_view message,
        const cpr::Header& advice) const;
    /// Defines how a Responses API implementation carries conversation context.
    /// The default preserves state through previous_response_id; replay-only
    /// implementations can instead place the complete conversation in input.
    enum class ConversationContextStrategy {
        StatefulPreviousResponse,
        ReplayInput,
    };

    [[nodiscard]] virtual ConversationContextStrategy conversation_context_strategy()
        const noexcept {
        return ConversationContextStrategy::StatefulPreviousResponse;
    }

    /// Replay-only APIs carry system messages inside `input` instead of the
    /// top-level `instructions` field.
    [[nodiscard]] virtual bool replay_system_messages_in_input() const noexcept {
        return true;
    }

    /// `service_tier` to send, empty to omit: the request's, else the
    /// endpoint's configured tier.
    [[nodiscard]] virtual std::string request_service_tier(const ChatRequest& request) const {
        return request.service_tier.empty() ? default_service_tier_ : request.service_tier;
    }

    /// The reasoning controls a request resolves to: what the model accepts
    /// and the effort to send. The wire value is resolved once, so the
    /// request parameter and a durable effort update always agree.
    struct ResolvedReasoning {
        ReasoningCapabilities capabilities;
        std::string effort;
    };

    /// Resolve the reasoning controls for a request. `effort_override` is the
    /// provider's own forced value, which the catalog never narrows.
    [[nodiscard]] ResolvedReasoning resolve_request_reasoning(
        const ChatRequest& request,
        std::optional<std::string_view> effort_override = std::nullopt) const;

    struct SerializationOptions {
        bool include_store = true;
        bool include_prompt_cache_key = true;
        bool include_response_include = true;
        // Codex defaults to a summary, subject to its model catalog.
        bool include_reasoning_summary = false;
        std::string reasoning_summary = "auto";
        // Codex sends its catalog's `default_verbosity` as `text.verbosity`.
        bool include_catalog_verbosity = false;
        bool include_sampling_controls = true;
        bool tools_in_input = false;
        std::optional<std::string_view> reasoning_effort_override;
        std::span<const std::string_view> hosted_tool_types;
    };

    struct SharedState {
        struct SessionState {
            std::string previous_response_id;
            std::string prompt_cache_key;
        };

        std::mutex  mutex;
        std::unordered_map<std::string, SessionState> sessions;
    };

    void share_continuity_state_with(OpenAIResponsesProtocol& cloned) const {
        cloned.shared_state_ = shared_state_;
        cloned.active_conversation_ = active_conversation_;
    }

    [[nodiscard]] std::string serialize_with_input_items(
        const ChatRequest& request,
        const std::vector<std::string>& input_items,
        std::optional<std::string_view> previous_response_id_override = std::nullopt) const;
    [[nodiscard]] std::string serialize_with_input_items(
        const ChatRequest& request,
        const std::vector<std::string>& input_items,
        std::optional<std::string_view> previous_response_id_override,
        const SerializationOptions& options) const;
    [[nodiscard]] std::string serialize_with_options(
        const ChatRequest& request,
        const SerializationOptions& options) const;
    bool include_reasoning_encrypted_ = false;
    std::string default_service_tier_;
    std::shared_ptr<SharedState> shared_state_ = std::make_shared<SharedState>();

private:
    std::string active_conversation_;
    bool saw_text_delta_ = false;
    bool stream_retry_vetoed_ = false;
    std::string in_progress_response_id_;
    std::string last_response_id_;
    RateLimitInfo last_rate_limit_;
};

/// Codex `DEFAULT_STREAM_IDLE_TIMEOUT_MS` is 300s. GPT-5.x reasoning on the
/// ChatGPT subscription backend routinely thinks longer than Filo's generic
/// 120s/240s HTTP watchdog, which would abort a healthy stream.
[[nodiscard]] inline transport::StreamTimeoutPolicy
codex_stream_timeouts() noexcept {
    return {
        .response_start = std::chrono::seconds(300),
        .inactivity = std::chrono::seconds(300),
    };
}

/// Codex `DEFAULT_STREAM_MAX_RETRIES` is 5. Responses `response.failed` can
/// fire after streamed reasoning, so retry even after output — same contract
/// as grok-build, handled by HttpLLMProvider's reset_attempt path.
[[nodiscard]] inline transport::RetryPolicy
codex_stream_retry_policy() noexcept {
    transport::RetryPolicy policy;
    policy.max_retries = 5;
    policy.retry_only_before_output = false;
    return policy;
}

class CodexResponsesProtocol final : public OpenAIResponsesProtocol {
public:
    /// What a request is for (codex-rs CodexResponsesRequestKind): a turn of
    /// the conversation, the warm-up that opens the socket, or a compaction.
    enum class RequestKind { Turn, Prewarm, Compaction };

    explicit CodexResponsesProtocol(bool include_reasoning_encrypted = true,
                                    std::string default_service_tier = {},
                                    std::shared_ptr<IProviderClientIdentitySource>
                                        client_identity_source = {},
                                    std::shared_ptr<codex::IUsageSource> usage = {});

    [[nodiscard]] transport::StreamTimeoutPolicy stream_timeouts()
        const noexcept override {
        return codex_stream_timeouts();
    }
    [[nodiscard]] transport::RetryPolicy stream_retry_policy()
        const noexcept override {
        return codex_stream_retry_policy();
    }

    [[nodiscard]] cpr::Header build_headers(
        const core::auth::AuthInfo& auth) const override;
    void on_response(const HttpResponse& response) override;
    /// Adds the account's `/wham/usage` reading: workspaces metered by credits
    /// stream no quota headers or windows, only this endpoint reports them.
    void enrich_rate_limit(std::string_view base_url,
                           const cpr::Header& request_headers,
                           const HttpResponse& response) override;
    [[nodiscard]] RateLimitInfo last_rate_limit() const noexcept override;
    /// ChatGPT plan limits (`error.type` usage_limit_reached /
    /// usage_not_included) in the official client's wording.
    [[nodiscard]] std::string format_error_message(
        const HttpResponse& response) const override;

    [[nodiscard]] std::string serialize(const ChatRequest& request) const override;
    void prepare_headers(cpr::Header& headers,
                         const ChatRequest& request,
                         std::string_view base_url) override;
    void observe_response_headers(const cpr::Header& headers,
                                  const ChatRequest& request) override;
    /// codex-rs `x-codex-turn-state`: the endpoint names the routing token of
    /// the turn in a metadata event, and the next request of the turn carries
    /// it back.
    void observe_turn_state(const ChatRequest& request,
                            std::string_view turn_state) override;
    [[nodiscard]] bool supports_websocket_transport() const noexcept override { return true; }
    [[nodiscard]] bool supports_mid_turn_compaction() const noexcept override { return true; }
    /// codex-rs remote compaction v2 (core/src/compact_remote_v2*.rs): the
    /// conversation plus a `compaction_trigger` answers with one encrypted
    /// `compaction` item, the ChatGPT backend's own checkpoint.
    [[nodiscard]] bool supports_native_compaction() const noexcept override { return true; }
    /// codex-rs core/src/session/reasoning_effort.rs: a model that accepts
    /// reasoning-effort updates records the change in the conversation as a
    /// `configuration_update` item, so it survives in the history and keeps
    /// the prompt cache intact.
    [[nodiscard]] std::vector<ContinuationItem>
    conversation_control_items(const ChatRequest& request) const override;
    /// codex-rs codex-api/src/endpoint/responses_websocket.rs: the frame that
    /// asks the endpoint to stop the response it is serving.
    [[nodiscard]] std::optional<std::string>
    interrupt_request(std::string_view response_id) const override;
    [[nodiscard]] std::string build_websocket_url(std::string_view base_url,
                                                  std::string_view model) const override;
    void prepare_websocket_headers(cpr::Header& headers,
                                   const ChatRequest& request,
                                   std::string_view base_url) override;
    [[nodiscard]] std::string serialize_websocket_request(
        const ChatRequest& request) const override;
    [[nodiscard]] WebSocketRequestFrame initial_websocket_request_frame(
        const ChatRequest& request) const override;
    [[nodiscard]] WebSocketRequestFrame next_websocket_request_frame(
        const ChatRequest& request,
        const WebSocketRequestFrame& completed_frame) const override;
    void abandon_websocket_request(const ChatRequest& request) override;
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;
    [[nodiscard]] std::string websocket_connection_key(
        std::string_view url,
        const cpr::Header& headers,
        const ChatRequest& request) const override;
    void reset_state() override;

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override;
    [[nodiscard]] std::unique_ptr<core::llm::ModelCatalogProvider>
    model_catalog(std::string_view provider_name) const override;

protected:
    [[nodiscard]] ConversationContextStrategy conversation_context_strategy()
        const noexcept override {
        // HTTP responses on the ChatGPT backend are not guaranteed to remain
        // addressable by previous_response_id. Replay is the durable fallback;
        // the WebSocket path explicitly opts into incremental continuation.
        return ConversationContextStrategy::ReplayInput;
    }
    [[nodiscard]] bool replay_system_messages_in_input() const noexcept override {
        // The ChatGPT backend answers a top-level `instructions` field with
        // HTTP 400, so the stable prompt travels in `input`.
        return true;
    }
    /// codex-rs tui service_tier_resolution.rs effective_service_tier and
    /// ModelInfo::service_tier_for_request: a tier the live catalog does not
    /// offer for this model is dropped, `default` opts out of the catalog's
    /// default tier, and `flex` is always accepted.
    [[nodiscard]] std::string request_service_tier(const ChatRequest& request) const override;
    /// codex-rs codex-api/src/sse/responses_error.rs: the ChatGPT backend
    /// names the class it failed with, and a class it does not know is
    /// transient. The classes a retry cannot fix stay terminal.
    [[nodiscard]] FailureVerdict classify_failure(
        std::string_view code,
        std::string_view message,
        const cpr::Header& advice) const override;

private:
    class RateLimitState;

    struct TransportState {
        struct LastWebSocketRequest {
            std::string response_id;
            std::string request_signature;
            std::vector<std::string> request_input_items;
            std::vector<std::string> response_items_added;
            bool from_prewarm = false;
        };

        struct PendingWebSocketRequest {
            std::string request_signature;
            std::vector<std::string> request_input_items;
            bool prewarm = false;
        };

        std::mutex mutex;
        std::string installation_id;
        std::unordered_map<std::string, std::string> turn_states;
        std::unordered_map<std::string, LastWebSocketRequest> last_ws_requests;
        std::unordered_map<std::string, PendingWebSocketRequest> pending_ws_requests;
    };

    [[nodiscard]] std::string installation_id() const;
    /// Keeps the routing token of a turn, so the next request of the same
    /// turn carries it back.
    void remember_turn_state(const ChatRequest& request, std::string_view turn_state);
    /// The request body without the client metadata: the model's controls and
    /// the given input items. The shape of a request, free of the fields that
    /// name the turn it belongs to.
    [[nodiscard]] std::string codex_request_body(
        const ChatRequest& request,
        const std::vector<std::string>& input_items,
        std::optional<std::string_view> previous_response_id_override = std::nullopt) const;
    /// The client metadata member codex-rs sends: the conversation the
    /// request belongs to, the worker that sent it, the thread that spawned
    /// it and the turn it serves.
    [[nodiscard]] std::string client_metadata_member(
        const ChatRequest& request,
        bool websocket,
        RequestKind request_kind) const;
    /// What the turn is, for the backend (codex-rs
    /// CodexTurnMetadataPayload): the conversation, the turn and the tree it
    /// belongs to.
    [[nodiscard]] std::string turn_metadata(
        const ChatRequest& request,
        RequestKind request_kind) const;
    [[nodiscard]] std::string serialize_codex_with_input_items(
        const ChatRequest& request,
        const std::vector<std::string>& input_items,
        std::optional<std::string_view> previous_response_id_override = std::nullopt,
        bool websocket = false,
        RequestKind request_kind = RequestKind::Turn) const;
    [[nodiscard]] std::string serialize_websocket_prewarm_request(
        const ChatRequest& request) const;
    [[nodiscard]] std::string serialize_websocket_request_impl(
        const ChatRequest& request,
        bool prewarm) const;

    std::shared_ptr<IProviderClientIdentitySource> client_identity_source_;
    std::shared_ptr<codex::IUsageSource> usage_;
    std::shared_ptr<TransportState> transport_state_ = std::make_shared<TransportState>();
    std::shared_ptr<RateLimitState> rate_limit_state_;
    mutable std::vector<std::string> active_response_items_;
    mutable std::string active_continuation_;
};

} // namespace core::llm::protocols
