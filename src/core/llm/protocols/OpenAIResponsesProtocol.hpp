#pragma once

#include "ApiProtocol.hpp"
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>

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

    [[nodiscard]] std::string_view event_delimiter() const noexcept override { return "\n\n"; }
    [[nodiscard]] ParseResult parse_event(std::string_view raw_event) override;
    [[nodiscard]] std::string_view name() const noexcept override { return "openai_responses"; }

    void on_response(const HttpResponse& response) override;
    [[nodiscard]] RateLimitInfo last_rate_limit() const noexcept override { return last_rate_limit_; }
    [[nodiscard]] const std::string& last_response_id() const noexcept { return last_response_id_; }

    [[nodiscard]] std::unique_ptr<ApiProtocolBase> clone() const override {
        auto cloned = std::make_unique<OpenAIResponsesProtocol>(
            include_reasoning_encrypted_, default_service_tier_);
        cloned->shared_state_ = shared_state_;
        return cloned;
    }

private:
    struct SharedState {
        std::mutex  mutex;
        std::string previous_response_id;
        std::string prompt_cache_key;
    };

    bool include_reasoning_encrypted_ = false;
    std::string default_service_tier_;
    std::shared_ptr<SharedState> shared_state_ = std::make_shared<SharedState>();
    bool saw_text_delta_ = false;
    std::string in_progress_response_id_;
    std::string last_response_id_;
    RateLimitInfo last_rate_limit_;
};

} // namespace core::llm::protocols
