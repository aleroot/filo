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

int MiniMaxProvider::max_context_size_for(std::string_view model) const noexcept {
    const auto id = canonical_minimax_model(model.empty() ? std::string_view(options_.model) : model);
    const int selected = minimax_working_context_window(id, options_.context_window);
    if (const auto live = provider_model_info(id); live && live->context_window > 0) {
        return std::min(selected, live->context_window);
    }
    return selected;
}

int MiniMaxProvider::max_context_size() const noexcept {
    return max_context_size_for(options_.model);
}

std::shared_ptr<LLMProvider> MiniMaxProvider::fork_for_parallel_request() const {
    return std::make_shared<MiniMaxProvider>(options_);
}

} // namespace core::llm::providers
