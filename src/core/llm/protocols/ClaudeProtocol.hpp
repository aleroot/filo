#pragma once
#include "AnthropicProtocol.hpp"

namespace core::llm::protocols {

struct ClaudeThinkingConfig {
    bool enabled = false;
    int budget_tokens = 10000;
};

struct ClaudeWirePolicy {
    ReasoningCapabilities effort;             ///< Effort levels the model accepts.
    bool known = false;                       ///< False when no card or generation matched.
    bool adaptive_thinking = false;           ///< `thinking.type=adaptive` is the wire mode.
    bool manual_thinking = false;             ///< `thinking.type=enabled` with a budget is accepted.
    bool thinking_always_on = false;          ///< Reasoning cannot be disabled; effort is the control.
    bool reasoning_text_hidden = false;       ///< Reasoning text is empty unless `display` is requested.
    bool reasoning_bound_to_prefix = false;   ///< Replayed reasoning is validated against the prefix.
    bool fixed_sampling = false;              ///< Non-default sampling parameters are rejected.
    bool forced_tool_choice_rejected = false; ///< `tool_choice` `any`/`tool` are rejected.
    bool between_tools_thinking = false;
    bool disabled_thinking = false;

    /// Manual budgets are refused by every known adaptive-only model, but an
    /// unrecognized id keeps the permissive historical behavior.
    [[nodiscard]] constexpr bool rejects_manual_thinking() const noexcept {
        return known && !manual_thinking;
    }
};

/// Resolve the wire policy for a model id, alias, or `[1m]`-suffixed variant.
/// @p live carries the provider catalog's reasoning profile when it reported
/// one, and wins over the registry card because it describes the endpoint
/// actually serving the request.
[[nodiscard]] ClaudeWirePolicy claude_wire_policy(
    std::string_view model,
    const ModelReasoningProfile* live = nullptr);

/// Claude's model, reasoning, attribution and caching policy over Messages.
struct ClaudeSerializer {
    static std::string serialize(const ChatRequest& req, int default_max_tokens = 8096,
                                 const ClaudeThinkingConfig& thinking = {});
};

class ClaudeProtocol : public AnthropicProtocol {
public:
    explicit ClaudeProtocol(ClaudeThinkingConfig thinking = {}, int default_max_tokens = 8096)
        : AnthropicProtocol(default_max_tokens), thinking_(thinking) {}
    void prepare_request(ChatRequest& request) override;
    std::string model_id(std::string_view model) const override;
    bool accepts_legacy_continuation() const noexcept override { return true; }
    std::string serialize(const ChatRequest& req) const override;
    cpr::Header build_headers(const core::auth::AuthInfo& auth) const override;
    ReasoningCapabilities reasoning_capabilities(std::string_view model) const noexcept override;
    std::unique_ptr<ApiProtocolBase> clone() const override;
    void reset_state() override;
    void on_response(const HttpResponse& response) override;
    void enrich_rate_limit(std::string_view base_url, const cpr::Header& headers,
                           const HttpResponse& response) override;
    RateLimitInfo last_rate_limit() const noexcept override { return last_rate_limit_; }
    std::string format_error_message(const HttpResponse& response) const override;
    bool is_retryable(const HttpResponse& response) const noexcept override;
private:
    ClaudeThinkingConfig thinking_;
    bool request_uses_context_1m_ = false;
    std::string last_requested_model_;
    RateLimitInfo last_rate_limit_;
};
} // namespace core::llm::protocols
