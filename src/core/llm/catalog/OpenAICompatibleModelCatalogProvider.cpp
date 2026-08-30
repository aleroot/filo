#include "../ModelCatalogProvider.hpp"

#include "CompatibleModelMetadata.hpp"
#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"

#include <simdjson.h>

#include <utility>

namespace core::llm {
namespace {

[[nodiscard]] ModelInfo decode_model(
    const catalog::JsonObjectView& model,
    std::string_view provider_name) {
    ModelInfo info;
    (void)model.string("id", info.canonical_id);
    if (info.canonical_id.empty()) return info;

    info.provider = catalog::copy_string(provider_name);
    if (!model.string("display_name", info.display_name)) {
        info.display_name = info.canonical_id;
    }
    info.capabilities = catalog::compatible_advertised_capabilities(
        info.canonical_id,
        model);
    info.context_window = model.first_integer(
        {"context_window", "max_context_window", "context_length"});
    info.max_output_tokens = model.first_integer(
        {"max_output_tokens", "max_tokens"});
    info.max_reasoning_tokens = model.first_integer(
        {"max_reasoning_tokens", "reasoning_max_tokens"});
    info.reasoning = catalog::compatible_reasoning_profile(model);
    info.tier = catalog::infer_tier(
        info.canonical_id,
        info.supports(ModelCapability::Reasoning));
    info.constraints = catalog::standard_constraints();
    return info;
}

} // namespace

OpenAICompatibleModelCatalogProvider::
OpenAICompatibleModelCatalogProvider(
    std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) {
        provider_name_ = "openai";
    }
}

std::string_view
OpenAICompatibleModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string OpenAICompatibleModelCatalogProvider::model_list_path(
    std::string_view page_token) const {
    (void)page_token;
    return "/models";
}

ModelCatalogResult
OpenAICompatibleModelCatalogProvider::parse_models_response(
    std::string_view body) const {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    ModelCatalogResult result =
        catalog::parse_catalog_json(body, document, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (document["data"].get(models) == simdjson::SUCCESS) {
        for (simdjson::dom::element element : models) {
            simdjson::dom::object object;
            if (element.get(object) != simdjson::SUCCESS) continue;
            ModelInfo info = decode_model(
                catalog::JsonObjectView(object),
                provider_name_);
            if (!info.canonical_id.empty()) {
                result.models.push_back(std::move(info));
            }
        }
        return result;
    }

    result.error = "OpenAI-compatible model catalog response is missing data[]";
    return result;
}

} // namespace core::llm
