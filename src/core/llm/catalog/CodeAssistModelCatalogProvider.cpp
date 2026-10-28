#include "../ModelCatalogProvider.hpp"

#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"
#include "../AntigravityModelTraits.hpp"

#include <simdjson.h>

#include <algorithm>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

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

/// Folds the per-level SKUs of every family sold at more than one thinking
/// level into a single model, so the picker lists `claude-opus-5-5` once and
/// the session's reasoning effort selects the SKU (see AntigravityModelTraits).
/// The family takes the place of its first SKU; the SKU ids stay reachable as
/// aliases, so a configured `claude-opus-5-5-high` still resolves to it. Limits
/// and capabilities are the ones every level shares.
[[nodiscard]] std::vector<ModelInfo> fold_thinking_variants(std::vector<ModelInfo> skus) {
    std::map<std::string, std::vector<std::size_t>, std::less<>> families;
    for (std::size_t i = 0; i < skus.size(); ++i) {
        if (const auto variant = antigravity::split_thinking_variant(skus[i].canonical_id)) {
            families[std::string(variant->family)].push_back(i);
        }
    }

    std::vector<ModelInfo> models;
    models.reserve(skus.size());
    std::vector<bool> folded(skus.size(), false);
    for (std::size_t i = 0; i < skus.size(); ++i) {
        if (folded[i]) continue;
        const auto variant = antigravity::split_thinking_variant(skus[i].canonical_id);
        const auto family = variant ? families.find(variant->family) : families.end();
        if (family == families.end() || family->second.size() < 2) {
            models.push_back(std::move(skus[i]));
            continue;
        }

        ModelInfo merged;
        merged.canonical_id = family->first;
        merged.provider = skus[i].provider;
        merged.display_name = skus[i].display_name == skus[i].canonical_id
            ? family->first
            : antigravity::family_display_name(skus[i].display_name);
        merged.capabilities = skus[i].capabilities;
        const auto shared_limit = [](int32_t current, int32_t sku) {
            if (sku <= 0) return current;
            return current <= 0 ? sku : std::min(current, sku);
        };
        std::vector<std::string_view> levels;
        for (const std::size_t index : family->second) {
            ModelInfo& sku = skus[index];
            folded[index] = true;
            levels.push_back(antigravity::split_thinking_variant(sku.canonical_id)->level);
            merged.capabilities &= sku.capabilities;
            merged.context_window = shared_limit(merged.context_window, sku.context_window);
            merged.max_output_tokens =
                shared_limit(merged.max_output_tokens, sku.max_output_tokens);
            merged.aliases.push_back(std::move(sku.canonical_id));
        }
        merged.reasoning = antigravity::thinking_family_profile(levels);
        models.push_back(std::move(merged));
    }
    return models;
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

    result.models = fold_thinking_variants(std::move(result.models));
    return result;
}

} // namespace core::llm
