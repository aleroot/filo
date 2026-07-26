#include "../ModelCatalogProvider.hpp"

#include "CompatibleModelMetadata.hpp"
#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"

#include <simdjson.h>

#include <optional>
#include <utility>

namespace core::llm {
namespace {

[[nodiscard]] int32_t infer_context_window(std::string_view model_id) {
    if (model_id == "k3" || catalog::contains_ascii(model_id, "kimi-k3")) {
        return 1'048'576;
    }
    if (catalog::contains_ascii(model_id, "moonshot-v1-8k")) return 8'192;
    if (catalog::contains_ascii(model_id, "moonshot-v1-32k")) return 32'768;
    if (catalog::contains_ascii(model_id, "moonshot-v1-128k")) return 128'000;
    if (catalog::contains_ascii(model_id, "kimi-k2.7-code")
        || catalog::contains_ascii(model_id, "kimi-k2.6")
        || catalog::contains_ascii(model_id, "kimi-k2-6")
        || catalog::contains_ascii(model_id, "kimi-k2.5")
        || catalog::contains_ascii(model_id, "kimi-k2-5")
        || catalog::contains_ascii(model_id, "kimi-for-coding")) {
        return 256'000;
    }
    return 128'000;
}

[[nodiscard]] int32_t infer_max_output_tokens(std::string_view model_id) {
    return model_id == "k3"
            || catalog::contains_ascii(model_id, "kimi-k3")
        ? 1'048'576
        : 8'192;
}

[[nodiscard]] ModelCapabilities kimi_capabilities(
    std::string_view model_id,
    std::optional<bool> supports_reasoning,
    std::optional<bool> supports_image,
    std::optional<bool> supports_video) {
    ModelCapabilities capabilities =
        catalog::kToolCapabilities |
        static_cast<uint32_t>(ModelCapability::JsonMode);

    const bool inferred_reasoning =
        model_id == "k3"
        || catalog::contains_ascii(model_id, "kimi-k3")
        || catalog::contains_ascii(model_id, "thinking")
        || catalog::contains_ascii(model_id, "reason")
        || catalog::contains_ascii(model_id, "kimi-for-coding")
        || catalog::contains_ascii(model_id, "kimi-code");
    if (supports_reasoning.value_or(inferred_reasoning)) {
        capabilities |= static_cast<uint32_t>(ModelCapability::Reasoning);
    }

    const bool inferred_image =
        model_id == "k3"
        || catalog::contains_ascii(model_id, "kimi-k3")
        || catalog::contains_ascii(model_id, "vl")
        || catalog::contains_ascii(model_id, "vision")
        || catalog::contains_ascii(model_id, "kimi-k2")
        || catalog::contains_ascii(model_id, "kimi-for-coding")
        || catalog::contains_ascii(model_id, "kimi-code");
    if (supports_image.value_or(inferred_image)
        || supports_video.value_or(false)) {
        capabilities |= static_cast<uint32_t>(ModelCapability::Vision);
    }

    const bool inferred_video =
        model_id == "k3"
        || catalog::contains_ascii(model_id, "kimi-k3")
        || catalog::contains_ascii(model_id, "kimi-k2")
        || catalog::contains_ascii(model_id, "kimi-for-coding")
        || catalog::contains_ascii(model_id, "kimi-code");
    if (supports_video.value_or(inferred_video)) {
        capabilities |= static_cast<uint32_t>(ModelCapability::VideoInput);
    }
    return capabilities;
}

} // namespace

KimiModelCatalogProvider::KimiModelCatalogProvider(std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) {
        provider_name_ = "kimi";
    }
}

std::string_view KimiModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string KimiModelCatalogProvider::model_list_path(
    std::string_view page_token) const {
    (void)page_token;
    return "/models";
}

ModelCatalogResult KimiModelCatalogProvider::parse_models_response(
    std::string_view body) const {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    ModelCatalogResult result =
        catalog::parse_catalog_json(body, document, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (document["data"].get(models) != simdjson::SUCCESS) {
        result.error = "Kimi model catalog response is missing data[]";
        return result;
    }

    for (simdjson::dom::element element : models) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) continue;
        const catalog::JsonObjectView model(object);

        ModelInfo info;
        if (!model.string("id", info.canonical_id)
            || info.canonical_id.empty()) {
            continue;
        }
        info.provider = provider_name_;
        if (!model.string("display_name", info.display_name)) {
            info.display_name = info.canonical_id;
        }
        info.context_window = model.first_integer(
            {"context_length", "context_window", "max_input_tokens"},
            infer_context_window(info.canonical_id));
        info.max_output_tokens = model.first_integer(
            {"max_output_tokens", "max_tokens"},
            infer_max_output_tokens(info.canonical_id));
        info.capabilities = kimi_capabilities(
            info.canonical_id,
            model.optional_boolean("supports_reasoning"),
            model.optional_boolean("supports_image_in"),
            model.optional_boolean("supports_video_in"))
            | catalog::compatible_advertised_capabilities(
                info.canonical_id,
                model);
        info.tier = catalog::infer_tier(
            info.canonical_id,
            info.supports(ModelCapability::Reasoning));
        info.constraints = catalog::standard_constraints();
        result.models.push_back(std::move(info));
    }
    return result;
}

} // namespace core::llm
