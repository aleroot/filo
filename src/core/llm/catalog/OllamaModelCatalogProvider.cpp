#include "../ModelCatalogProvider.hpp"

#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"

#include <simdjson.h>

#include <utility>

namespace core::llm {

OllamaModelCatalogProvider::OllamaModelCatalogProvider(
    std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) {
        provider_name_ = "ollama";
    }
}

std::string_view OllamaModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string OllamaModelCatalogProvider::model_list_path(
    std::string_view page_token) const {
    (void)page_token;
    return "/api/tags";
}

ModelCatalogResult OllamaModelCatalogProvider::parse_models_response(
    std::string_view body) const {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    ModelCatalogResult result =
        catalog::parse_catalog_json(body, document, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (document["models"].get(models) != simdjson::SUCCESS) {
        result.error = "Ollama model catalog response is missing models[]";
        return result;
    }

    for (simdjson::dom::element element : models) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) continue;
        const catalog::JsonObjectView model(object);

        ModelInfo info;
        if (!model.string("name", info.canonical_id)) {
            (void)model.string("model", info.canonical_id);
        }
        if (info.canonical_id.empty()) continue;

        info.provider = provider_name_;
        info.display_name = info.canonical_id;
        if (catalog::is_embedding_model(info.canonical_id)) {
            info.capabilities =
                static_cast<uint32_t>(ModelCapability::Embeddings);
        } else {
            // /api/tags is authoritative for installed identities, but it does
            // not advertise model-level tools, vision, thinking, or limits.
            info.capabilities =
                catalog::kTextCapabilities |
                static_cast<uint32_t>(ModelCapability::JsonMode);
        }
        info.tier = catalog::infer_tier(info.canonical_id);
        info.constraints = catalog::standard_constraints();
        result.models.push_back(std::move(info));
    }
    return result;
}

} // namespace core::llm
