#pragma once
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/ContextTokenEstimator.hpp"
#include "core/llm/MiniMaxModelTraits.hpp"

namespace core::llm::providers {
/// Supplies the same context estimate to compaction that request admission uses.
/// All streaming, authentication and cancellation remain in HttpLLMProvider.
class MiniMaxProvider final : public HttpLLMProvider, public ContextTokenEstimator {
public:
    struct Options {
        std::string base_url;
        std::shared_ptr<core::auth::ICredentialSource> credentials;
        std::string model = std::string(kMiniMaxDefaultModel);
        std::string provider_name = "minimax-m-plan";
        bool subscription = true;
        std::string default_effort;
        int context_window = kMiniMaxContextWindow;
    };
    explicit MiniMaxProvider(Options options);
    std::size_t estimate_context_tokens(const std::vector<Message>& messages) const noexcept override;
    int max_context_size() const noexcept override;
    std::shared_ptr<LLMProvider> fork_for_parallel_request() const override;
private:
    Options options_;
};
} // namespace core::llm::providers
