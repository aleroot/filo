#include "../ModelCatalogProvider.hpp"

#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"

#include <simdjson.h>

#include <optional>
#include <string>
#include <utility>

namespace core::llm {
namespace {

// Ids the real hub client keeps out of its model picker: retired preview
// chat builds and the superseded Gemini 2.5 Pro route.
constexpr std::string_view kDeniedModelIds[] = {
    "chat_20706", "chat_23310", "gemini-2.5-pro"};

[[nodiscard]] bool is_denied_model_id(std::string_view id) {
    for (const auto denied : kDeniedModelIds) {
        if (denied == id) return true;
    }
    return false;
}

} // namespace

CodeAssistModelCatalogProvider::CodeAssistModelCatalogProvider(
    std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) {
        provider_name_ = "gemini-antigravity";
    }
}

std::string_view CodeAssistModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string CodeAssistModelCatalogProvider::model_list_path(
    std::string_view /*page_token*/) const {
    // Single unpaginated endpoint; `fetchAvailableModels` ignores any token.
    return "/v1internal:fetchAvailableModels";
}

ModelCatalogRequest
CodeAssistModelCatalogProvider::catalog_request(std::string_view /*page_token*/) const {
    // The real hub client posts an empty JSON object.
    return {.method = "POST", .body = "{}"};
}

ModelCatalogResult CodeAssistModelCatalogProvider::parse_models_response(
    std::string_view body) const {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    ModelCatalogResult result = catalog::parse_catalog_json(body, document, parser);
    if (!result.ok()) return result;

    simdjson::dom::object models;
    if (document["models"].get(models) != simdjson::SUCCESS) {
        result.error = "Cloud Code Assist model catalog response is missing models{}";
        return result;
    }

    for (auto field : models) {
        simdjson::dom::object model_object;
        if (field.value.get(model_object) != simdjson::SUCCESS) continue;

        const std::string_view id = field.key;
        if (id.empty()) continue;

        const catalog::JsonObjectView model(model_object);
        if (model.optional_boolean("isInternal").value_or(false)) continue;
        if (is_denied_model_id(id)) continue;

        ModelInfo info;
        info.canonical_id = std::string(id);
        info.provider = provider_name_;
        if (!model.string("displayName", info.display_name)
            || info.display_name.empty()) {
            info.display_name = info.canonical_id;
        }
        info.context_window = model.first_integer({"maxTokens", "inputTokenLimit"});
        info.max_output_tokens =
            model.first_integer({"maxOutputTokens", "outputTokenLimit"});

        ModelCapabilities capabilities = catalog::kToolCapabilities;
        if (model.boolean("supportsImages")) {
            capabilities |= static_cast<uint32_t>(ModelCapability::Vision);
        }
        if (model.boolean("supportsVideo")) {
            capabilities |= static_cast<uint32_t>(ModelCapability::VideoInput);
        }
        if (model.boolean("supportsThinking")) {
            capabilities |= static_cast<uint32_t>(ModelCapability::Reasoning);
        }
        info.capabilities = capabilities;

        result.models.push_back(std::move(info));
    }

    return result;
}

} // namespace core::llm
