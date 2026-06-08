#pragma once

#include "ApiProtocol.hpp"
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <filesystem>
#include <utility>
#include <unordered_map>

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

    [[nodiscard]] std::string_view event_delimiter() const noexcept override { return "\n\n"; }
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;
    [[nodiscard]] std::string_view name() const noexcept override { return "openai_responses"; }

    void on_response(const HttpResponse& response) override;
    [[nodiscard]] RateLimitInfo last_rate_limit() const noexcept override { return last_rate_limit_; }
    [[nodiscard]] const std::string& last_response_id() const noexcept { return last_response_id_; }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        auto cloned = std::make_unique<OpenAIResponsesProtocol>(
            include_reasoning_encrypted_, default_service_tier_);
        share_continuity_state_with(*cloned);
        return cloned;
    }

protected:
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
        cloned.active_session_id_ = active_session_id_;
    }

    [[nodiscard]] std::string serialize_with_input_items(
        const ChatRequest& request,
        const std::vector<std::string>& input_items,
        std::optional<std::string_view> previous_response_id_override = std::nullopt) const;

    bool include_reasoning_encrypted_ = false;
    std::string default_service_tier_;
    std::shared_ptr<SharedState> shared_state_ = std::make_shared<SharedState>();

private:
    std::string active_session_id_;
    bool saw_text_delta_ = false;
    std::string in_progress_response_id_;
    std::string last_response_id_;
    RateLimitInfo last_rate_limit_;
};

class CodexResponsesProtocol final : public OpenAIResponsesProtocol {
public:
    explicit CodexResponsesProtocol(bool include_reasoning_encrypted = false,
                                    std::string default_service_tier = {},
                                    std::filesystem::path config_dir = {});

    [[nodiscard]] std::string serialize(const ChatRequest& request) const override;
    void prepare_headers(cpr::Header& headers,
                         const ChatRequest& request,
                         std::string_view base_url) override;
    void observe_response_headers(const cpr::Header& headers,
                                  const ChatRequest& request) override;
    [[nodiscard]] bool supports_websocket_transport() const noexcept override { return true; }
    [[nodiscard]] std::string build_websocket_url(std::string_view base_url,
                                                  std::string_view model) const override;
    void prepare_websocket_headers(cpr::Header& headers,
                                   const ChatRequest& request,
                                   std::string_view base_url) override;
    [[nodiscard]] std::string serialize_websocket_request(
        const ChatRequest& request) const override;
    void abandon_websocket_request(const ChatRequest& request) override;
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;
    [[nodiscard]] std::string websocket_connection_key(
        std::string_view url,
        const cpr::Header& headers,
        const ChatRequest& request) const override;
    void reset_state() override;

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override;

private:
    struct TransportState {
        struct LastWebSocketRequest {
            std::string response_id;
            std::string request_signature;
            std::vector<std::string> request_input_items;
            std::vector<std::string> response_items_added;
        };

        struct PendingWebSocketRequest {
            std::string request_signature;
            std::vector<std::string> request_input_items;
        };

        std::mutex mutex;
        std::string installation_id;
        std::unordered_map<std::string, std::string> turn_states;
        std::unordered_map<std::string, LastWebSocketRequest> last_ws_requests;
        std::unordered_map<std::string, PendingWebSocketRequest> pending_ws_requests;
    };

    [[nodiscard]] std::string installation_id() const;
    [[nodiscard]] std::string serialize_codex_with_input_items(
        const ChatRequest& request,
        const std::vector<std::string>& input_items,
        std::optional<std::string_view> previous_response_id_override = std::nullopt) const;

    std::filesystem::path config_dir_;
    std::shared_ptr<TransportState> transport_state_ = std::make_shared<TransportState>();
    mutable std::vector<std::string> active_response_items_;
    mutable std::string active_transport_turn_id_;
};

} // namespace core::llm::protocols
