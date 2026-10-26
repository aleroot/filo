#include "QwenModelCatalogProvider.hpp"
#include "../QwenModelTraits.hpp"
#include "../ModelMetadata.hpp"

#include <utility>

namespace core::llm {

QwenModelCatalogProvider::QwenModelCatalogProvider(std::string provider_name)
    : compatible_(std::move(provider_name)) {}

std::string_view QwenModelCatalogProvider::provider_name() const noexcept {
    return compatible_.provider_name();
}

std::string QwenModelCatalogProvider::model_list_path(std::string_view page_token) const {
    return compatible_.model_list_path(page_token);
}

ModelCatalogResult QwenModelCatalogProvider::parse_models_response(std::string_view body) const {
    auto result = compatible_.parse_models_response(body);
    for (auto& model : result.models) {
        if (auto card = qwen_hosted_model_info(model.canonical_id)) {
            card->provider = model.provider;
            model = merge_model_metadata(std::move(*card), std::move(model));
        }
    }
    return result;
}

} // namespace core::llm
