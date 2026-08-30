#include "../ModelCatalogProvider.hpp"

#include "CompatibleModelMetadata.hpp"
#include "CodexModelCatalog.hpp"
#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"
#include "../../utils/StringUtils.hpp"
#include "core/version/Version.hpp"

#include <simdjson.h>

#include <utility>

namespace core::llm {
namespace {

[[nodiscard]] bool is_reasoning_model(std::string_view model_id) {
    const std::string lower =
        core::utils::str::to_lower_ascii_copy(model_id);
    return lower.starts_with("o1")
        || lower.starts_with("o3")
        || lower.starts_with("o4")
        || lower.starts_with("gpt-5")
        || catalog::contains_ascii(model_id, "reasoning")
        || catalog::contains_ascii(model_id, "thinking");
}

[[nodiscard]] ModelCapabilities codex_capabilities(
    std::string_view model_id,
    const catalog::JsonObjectView& model) {
    if (catalog::is_embedding_model(model_id)) {
        return static_cast<uint32_t>(ModelCapability::Embeddings);
    }

    ModelCapabilities capabilities =
        catalog::kTextCapabilities
        | static_cast<uint32_t>(ModelCapability::FunctionCalling)
        | static_cast<uint32_t>(ModelCapability::JsonMode);
    if (model.boolean("supports_parallel_tool_calls")) {
        capabilities |= static_cast<uint32_t>(
            ModelCapability::ParallelToolCalls);
    }
    if (model.boolean("supports_image_detail_original")
        || model.string_array_contains("input_modalities", "image")
        || catalog::contains_ascii(model_id, "gpt-4")
        || catalog::contains_ascii(model_id, "gpt-5")
        || catalog::contains_ascii(model_id, "4o")
        || catalog::contains_ascii(model_id, "vision")) {
        capabilities |= static_cast<uint32_t>(ModelCapability::Vision);
    }

    std::string default_reasoning_level;
    if (is_reasoning_model(model_id)
        || model.string("default_reasoning_level", default_reasoning_level)
        || catalog::compatible_reasoning_profile(model).effort.supports_effort()) {
        capabilities |= static_cast<uint32_t>(ModelCapability::Reasoning);
    }
    return capabilities;
}

[[nodiscard]] ModelInfo decode_codex_model(
    const catalog::JsonObjectView& model,
    std::string_view provider_name) {
    ModelInfo info;
    if (!model.string("slug", info.canonical_id)) {
        (void)model.string("id", info.canonical_id);
    }
    if (info.canonical_id.empty()) return info;

    info.provider = catalog::copy_string(provider_name);
    if (!model.string("display_name", info.display_name)) {
        info.display_name = info.canonical_id;
    }
    info.capabilities =
        catalog::compatible_advertised_capabilities(info.canonical_id, model)
        | codex_capabilities(info.canonical_id, model);
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

CodexModelCatalogProvider::CodexModelCatalogProvider(std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) provider_name_ = "openai-codex";
}

std::string_view CodexModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string CodexModelCatalogProvider::model_list_path(
    std::string_view page_token) const {
    (void)page_token;
    return "/models?client_version=" + std::string(core::version::value);
}

ModelCatalogResult catalog::parse_codex_style_model_catalog(
    std::string_view body,
    std::string_view provider_name) {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    ModelCatalogResult result =
        catalog::parse_catalog_json(body, document, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (document["models"].get(models) != simdjson::SUCCESS) {
        result.error = "Codex model catalog response is missing models[]";
        return result;
    }

    for (simdjson::dom::element element : models) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) continue;
        ModelInfo info = decode_codex_model(
            catalog::JsonObjectView(object), provider_name);
        if (!info.canonical_id.empty()) {
            result.models.push_back(std::move(info));
        }
    }
    return result;
}

ModelCatalogResult CodexModelCatalogProvider::parse_models_response(
    std::string_view body) const {
    return catalog::parse_codex_style_model_catalog(body, provider_name_);
}

} // namespace core::llm
