#pragma once
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/ContextTokenEstimator.hpp"
#include "core/llm/MiniMaxModelTraits.hpp"

#include <atomic>
#include <string_view>

namespace core::llm::providers {

/// True when `text` contains MiniMax M3's tool-call namespace marker
/// (`]<]minimax[>[`). The model prefixes every structural tool-call tag with
/// that marker, and the serving stack is supposed to turn the result into a
/// structured tool_use before the client sees it. A clean text final that
/// still contains the marker is the gateway leak, including a truncated leak
/// that never reached `<invoke`. The marker is not ordinary prose.
[[nodiscard]] bool minimax_leaked_tool_call_markup(std::string_view text) noexcept;

/// Supplies the same context estimate to compaction that request admission uses.
/// Streaming wraps the shared HTTP transport so a leaked tool call is retracted
/// and reported as a failure. The leaked text is never executed, and the
/// request is not sent again. Authentication and the wire stay in the base.
class MiniMaxProvider final : public HttpLLMProvider, public ContextTokenEstimator {
public:
    struct Options {
        std::string base_url;
        std::shared_ptr<core::auth::ICredentialSource> credentials;
        std::string model = std::string(kMiniMaxDefaultModel);
        std::string provider_name = "minimax-m-plan";
        bool subscription = true;
        std::string default_effort;
        /// 0 keeps the model working default (Flash 1M, M3 512k).
        int context_window = 0;
    };
    explicit MiniMaxProvider(Options options);
    void stream_response(const ChatRequest& request,
                         std::function<void(const StreamChunk&)> callback) override;
    void cancel() override;
    std::size_t estimate_context_tokens(const std::vector<Message>& messages) const noexcept override;
    int max_context_size() const noexcept override;
    int max_context_size_for(std::string_view model) const noexcept override;
    std::shared_ptr<LLMProvider> fork_for_parallel_request() const override;
private:
    Options options_;
    std::atomic_bool cancelled_{false};
};
} // namespace core::llm::providers
