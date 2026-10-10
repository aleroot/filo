#pragma once

#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/ContextTokenEstimator.hpp"

#include <unordered_set>

namespace core::llm::providers {

/// Qwen request policy layered over the existing, unmodified HTTP transport.
/// Owns budgeting and one pre-output repair; protocols own wire semantics.
class QwenProvider final : public HttpLLMProvider, public ContextTokenEstimator {
public:
    struct Options {
        std::string base_url;
        std::shared_ptr<core::auth::ICredentialSource> credentials;
        std::string model;
        std::string provider_name;
        std::string service_id = {};
        std::shared_ptr<const IModelCatalogSelector> model_selector = {};
        int thinking_budget = 0;
        std::string default_effort = {};
        bool token_plan = false;
        bool responses = false;
        HttpStreamTransportOptions transport = {};
    };

    explicit QwenProvider(Options options);
    void stream_response(const ChatRequest& request,
                         std::function<void(const StreamChunk&)> callback) override;
    void cancel() override;
    void reset_conversation_state() override;
    [[nodiscard]] int max_context_size() const noexcept override;
    [[nodiscard]] std::size_t estimate_context_tokens(
        const std::vector<Message>& messages) const noexcept override;
    [[nodiscard]] ReasoningCapabilities reasoning_capabilities(std::string_view model) const noexcept override;
    [[nodiscard]] std::shared_ptr<LLMProvider> fork_for_parallel_request() const override;

private:
    struct State {
        std::mutex mutex;
        std::unordered_set<std::string> required_thinking_models;
        std::size_t last_context_estimate = 0;
        int32_t last_prompt_tokens = 0;
        // Per-attempt values; parallel callers receive independent forks.
        std::optional<int> requested_max_tokens;
        std::string model;
        std::size_t context_estimate = 0;
        int status = 0;
        std::string error_message;
        int32_t prompt_tokens = 0;
    };

    QwenProvider(Options options, std::shared_ptr<State> state);
    [[nodiscard]] static std::unique_ptr<protocols::ApiProtocolBase> make_protocol(
        const Options& options, const std::shared_ptr<State>& state);
    [[nodiscard]] static std::size_t calibrated_estimate(
        const State& state, std::size_t estimate) noexcept;
    [[nodiscard]] bool repair_request(ChatRequest& request);
    void record_context_usage();

    Options options_;
    std::shared_ptr<State> state_;
    std::atomic_bool cancelled_{false};
};
} // namespace core::llm::providers
