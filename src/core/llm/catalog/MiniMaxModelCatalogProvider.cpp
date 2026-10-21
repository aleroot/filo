#include "MiniMaxModelCatalogProvider.hpp"

#include <algorithm>

namespace core::llm {

MiniMaxModelCatalogProvider::MiniMaxModelCatalogProvider(
    std::string provider, bool subscription, int context_window)
    : messages_(provider), provider_(std::move(provider)), subscription_(subscription), context_window_(context_window) {}

std::string_view MiniMaxModelCatalogProvider::provider_name() const noexcept {
    return provider_;
}

std::string MiniMaxModelCatalogProvider::model_list_path(std::string_view cursor) const {
    return messages_.model_list_path(cursor);
}

ModelCatalogResult MiniMaxModelCatalogProvider::parse_models_response(std::string_view body) const {
    auto result = messages_.parse_models_response(body);
    std::vector<ModelInfo> models;
    for (const auto& remote : result.models) {
        if (subscription_ && !is_minimax_plan_model(remote.canonical_id)) continue;
        if (auto curated = minimax_model_info(remote.canonical_id)) {
            curated->provider = provider_;
            curated->context_window = remote.context_window > 0
                ? std::min(context_window_, remote.context_window) : context_window_;
            if (remote.max_output_tokens > 0) {
                curated->max_output_tokens = std::min(
                    curated->max_output_tokens, remote.max_output_tokens);
            }
            if (std::ranges::none_of(models, [&](const auto& model) { return model.canonical_id == curated->canonical_id; })) {
                models.push_back(std::move(*curated));
            }
        }
    }
    result.models = std::move(models);
    return result;
}

} // namespace core::llm
