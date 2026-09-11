#include "../ModelCatalogProvider.hpp"

#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"
#include "../../utils/UriUtils.hpp"
#include "../../utils/JsonUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <utility>

namespace core::llm {
namespace {

[[nodiscard]] ModelCapabilities gemini_capabilities(
    bool generate_content,
    bool count_tokens,
    bool embed_content,
    bool create_cached_content,
    bool batch_generate_content,
    bool thinking) {
    if (embed_content && !generate_content) {
        return static_cast<uint32_t>(ModelCapability::Embeddings);
    }

    ModelCapabilities capabilities = generate_content
        ? catalog::kTextCapabilities
        : ModelCapabilities{};
    if (embed_content) {
        capabilities |= static_cast<uint32_t>(ModelCapability::Embeddings);
    }
    if (count_tokens) {
        capabilities |= static_cast<uint32_t>(
            ModelCapability::TokenCounting);
    }
    if (create_cached_content) {
        capabilities |= static_cast<uint32_t>(
            ModelCapability::PromptCaching);
    }
    if (batch_generate_content) {
        capabilities |= static_cast<uint32_t>(ModelCapability::Batch);
    }
    if (thinking) {
        capabilities |= static_cast<uint32_t>(ModelCapability::Reasoning);
    }
    return capabilities;
}

} // namespace

GeminiModelCatalogProvider::GeminiModelCatalogProvider(
    std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) {
        provider_name_ = "gemini";
    }
}

std::string_view GeminiModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string GeminiModelCatalogProvider::model_list_path(
    std::string_view page_token) const {
    std::string path = "/v1beta/models?pageSize=1000";
    if (!page_token.empty()) {
        path += "&pageToken=";
        path += core::utils::uri::percent_encode_uri_query_component(page_token);
    }
    return path;
}

ModelCatalogResult GeminiModelCatalogProvider::parse_models_response(
    std::string_view body) const {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    ModelCatalogResult result =
        catalog::parse_catalog_json(body, document, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (document["models"].get(models) != simdjson::SUCCESS) {
        result.error = "Gemini model catalog response is missing models[]";
        return result;
    }
    (void)catalog::JsonObjectView(document.get_object())
        .string("nextPageToken", result.next_page_token);

    for (simdjson::dom::element element : models) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) continue;
        const catalog::JsonObjectView model(object);

        std::string name;
        if (!model.string("name", name)) continue;

        // Google documents baseModelId as the value to pass to generation
        // requests. Preserve the versioned resource name as an alias.
        std::string id;
        if (!model.string("baseModelId", id) || id.empty()) {
            id = catalog::strip_prefix(name, "models/");
        }
        if (id.empty()) continue;

        const bool generate_content = model.string_array_contains(
            "supportedGenerationMethods", "generateContent");
        const bool count_tokens = model.string_array_contains(
            "supportedGenerationMethods", "countTokens");
        const bool embed_content = model.string_array_contains(
            "supportedGenerationMethods", "embedContent")
            || model.string_array_contains(
                "supportedGenerationMethods", "batchEmbedContents");
        const bool create_cached_content = model.string_array_contains(
            "supportedGenerationMethods", "createCachedContent");
        const bool batch_generate_content = model.string_array_contains(
            "supportedGenerationMethods", "batchGenerateContent");
        const bool thinking = model.boolean("thinking")
            || catalog::contains_ascii(id, "thinking");

        ModelInfo info;
        info.canonical_id = id;
        info.provider = provider_name_;
        if (!model.string("displayName", info.display_name)) {
            info.display_name = id;
        }
        const std::string resource_id =
            catalog::strip_prefix(name, "models/");
        if (!resource_id.empty() && resource_id != id) {
            info.aliases.push_back(resource_id);
        }
        info.context_window = model.integer("inputTokenLimit");
        info.max_output_tokens = model.integer("outputTokenLimit");
        info.capabilities = gemini_capabilities(
            generate_content,
            count_tokens,
            embed_content,
            create_cached_content,
            batch_generate_content,
            thinking);
        info.tier = catalog::infer_tier(id, thinking);

        double max_temperature = 2.0;
        core::utils::json::ignore_error(object["maxTemperature"].get(max_temperature));
        info.constraints =
            catalog::standard_constraints(std::max(0.0, max_temperature));
        result.models.push_back(std::move(info));
    }
    return result;
}

} // namespace core::llm
