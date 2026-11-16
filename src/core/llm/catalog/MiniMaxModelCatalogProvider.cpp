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
            // The API's reported input limit, when present, is a hard ceiling.
            // An explicit provider `context_window` replaces the model working
            // default; otherwise M3 stays at 512k and Flash at 1M.
            const int ceiling = remote.context_window > 0
                ? std::min(kMiniMaxContextWindow, remote.context_window)
                : kMiniMaxContextWindow;
            curated->context_window = std::min(
                minimax_working_context_window(curated->canonical_id, context_window_), ceiling);
            curated->max_context_window = ceiling;
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
