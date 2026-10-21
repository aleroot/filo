#include "MiniMaxProvider.hpp"
#include "MiniMaxRequestBudget.hpp"
#include "core/llm/protocols/MiniMaxProtocol.hpp"

#include <algorithm>

namespace core::llm::providers {

MiniMaxProvider::MiniMaxProvider(Options options)
    : HttpLLMProvider(options.base_url, options.credentials, options.model,
        std::make_unique<protocols::MiniMaxProtocol>(
            options.subscription, options.default_effort, nullptr, options.context_window),
        core::config::ApiType::Anthropic, options.provider_name), options_(std::move(options)) {}

std::size_t MiniMaxProvider::estimate_context_tokens(
    const std::vector<Message>& messages) const noexcept {
    return MiniMaxRequestBudget::estimate_history(messages);
}

int MiniMaxProvider::max_context_size() const noexcept {
    if (const auto live = provider_model_info(canonical_minimax_model(options_.model));
        live && live->context_window > 0) {
        return std::min(options_.context_window, live->context_window);
    }
    return options_.context_window;
}

std::shared_ptr<LLMProvider> MiniMaxProvider::fork_for_parallel_request() const {
    return std::make_shared<MiniMaxProvider>(options_);
}

} // namespace core::llm::providers
