#include "../ModelCatalogProvider.hpp"

#include "CompatibleModelMetadata.hpp"
#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"
#include "../KimiModelTraits.hpp"

#include <simdjson.h>

#include <optional>
#include <utility>

namespace core::llm {
namespace {

[[nodiscard]] int32_t infer_context_window(std::string_view model_id) {
    if (const int32_t known = kimi_model_context_window(model_id);
        known > 0) {
        return known;
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
    return kimi_model_max_output_tokens(model_id);
}

[[nodiscard]] ModelReasoningProfile kimi_reasoning_profile(
    const catalog::JsonObjectView& model,
    bool supports_reasoning) {
    ModelReasoningProfile profile;
    if (!supports_reasoning) {
        profile.complete =
            model.optional_boolean("supports_reasoning").has_value();
        return profile;
    }

    profile.effort = ReasoningCapabilities{
        ReasoningCapability::Effort};
    profile.manual_thinking = true;

    std::string thinking_type;
    const bool has_thinking_type =
        model.string("supports_thinking_type", thinking_type);
    if (has_thinking_type && catalog::contains_ascii(
            thinking_type, "only")) {
        profile.effort =
            profile.effort | ReasoningCapability::Required;
    }

    simdjson::dom::object effort_object;
    const bool has_effort_object =
        model.raw()["think_efforts"].get(effort_object)
        == simdjson::SUCCESS;
    if (has_effort_object) {
        const catalog::JsonObjectView efforts(effort_object);
        if (efforts.boolean("support")) {
            const auto valid_efforts =
                efforts.string_array("valid_efforts");
            for (const auto& effort : valid_efforts) {
                if (effort == "max") {
                    profile.effort =
                        profile.effort | ReasoningCapability::MaxEffort;
                } else if (effort == "xhigh") {
                    profile.effort =
                        profile.effort | ReasoningCapability::XHighEffort;
                }
            }
        }
    }

    profile.complete = has_thinking_type || has_effort_object;
    return profile;
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
        is_kimi_k3_model(model_id)
        || catalog::contains_ascii(model_id, "thinking")
        || catalog::contains_ascii(model_id, "reason")
        || catalog::contains_ascii(model_id, "kimi-for-coding")
        || catalog::contains_ascii(model_id, "kimi-code");
    if (supports_reasoning.value_or(inferred_reasoning)) {
        capabilities |= static_cast<uint32_t>(ModelCapability::Reasoning);
    }

    const bool inferred_image =
        is_kimi_k3_model(model_id)
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
        (is_kimi_k3_model(model_id) && !is_kimi_k3_256k_model(model_id))
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
        if (info.context_window > 0
            && info.max_output_tokens > info.context_window) {
            info.max_output_tokens = info.context_window;
        }
        info.capabilities = kimi_capabilities(
            info.canonical_id,
            model.optional_boolean("supports_reasoning"),
            model.optional_boolean("supports_image_in"),
            model.optional_boolean("supports_video_in"))
            | catalog::compatible_advertised_capabilities(
                info.canonical_id,
                model);
        info.reasoning = kimi_reasoning_profile(
            model,
            info.supports(ModelCapability::Reasoning));
        info.tier = catalog::infer_tier(
            info.canonical_id,
            info.supports(ModelCapability::Reasoning));
        info.constraints = catalog::standard_constraints();
        result.models.push_back(std::move(info));
    }
    return result;
}

} // namespace core::llm
