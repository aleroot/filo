#include "../ModelCatalogProvider.hpp"

#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"

#include <simdjson.h>

#include <utility>

namespace core::llm {
namespace {

[[nodiscard]] ModelInfo decode_mistral_model(
    const catalog::JsonObjectView& model,
    std::string_view provider_name) {
    ModelInfo info;
    if (!model.string("id", info.canonical_id)
        || info.canonical_id.empty()) {
        return {};
    }
    if (model.boolean("archived")) {
        return {};
    }

    info.provider = catalog::copy_string(provider_name);
    info.display_name = info.canonical_id;
    info.aliases = model.string_array("aliases");
    info.context_window = model.integer("max_context_length");

    if (catalog::is_embedding_model(info.canonical_id)
        || model.capability_supported("embeddings")) {
        info.capabilities =
            static_cast<uint32_t>(ModelCapability::Embeddings);
        info.constraints = catalog::standard_constraints();
        return info;
    }

    const bool has_catalog = model.has_capability_catalog();
    const bool chat = !has_catalog
        || model.capability_supported("completion_chat");
    if (!chat) {
        return {};
    }

    // Streaming, system messages, and JSON response mode are API-level
    // features. The remaining switches come from Mistral's per-model matrix.
    info.capabilities =
        catalog::kTextCapabilities |
        static_cast<uint32_t>(ModelCapability::JsonMode);
    if (model.capability_supported("function_calling")) {
        info.capabilities |=
            static_cast<uint32_t>(ModelCapability::FunctionCalling) |
            static_cast<uint32_t>(ModelCapability::ParallelToolCalls);
    }
    if (model.capability_supported("vision")) {
        info.capabilities |=
            static_cast<uint32_t>(ModelCapability::Vision);
    }
    if (model.capability_supported("reasoning")) {
        info.capabilities |=
            static_cast<uint32_t>(ModelCapability::Reasoning);
    }

    // Mistral's matrix covers task-specific switches, but omits several Filo
    // capabilities (for example reasoning and API-level behavior). Treat it as
    // an enrichment, not an exhaustive replacement of a built-in model card.
    info.capabilities_complete = false;
    info.tier = catalog::infer_tier(
        info.canonical_id,
        info.supports(ModelCapability::Reasoning));
    info.constraints = catalog::standard_constraints();
    return info;
}

} // namespace

MistralModelCatalogProvider::MistralModelCatalogProvider(
    std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) {
        provider_name_ = "mistral";
    }
}

std::string_view MistralModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string MistralModelCatalogProvider::model_list_path(
    std::string_view page_token) const {
    (void)page_token;
    return "/models";
}

ModelCatalogResult MistralModelCatalogProvider::parse_models_response(
    std::string_view body) const {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    ModelCatalogResult result =
        catalog::parse_catalog_json(body, document, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (document["data"].get(models) != simdjson::SUCCESS
        && document.get(models) != simdjson::SUCCESS) {
        result.error =
            "Mistral model catalog response is missing data[]";
        return result;
    }

    for (simdjson::dom::element element : models) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) continue;
        ModelInfo info = decode_mistral_model(
            catalog::JsonObjectView(object),
            provider_name_);
        if (!info.canonical_id.empty()) {
            result.models.push_back(std::move(info));
        }
    }
    return result;
}

} // namespace core::llm
