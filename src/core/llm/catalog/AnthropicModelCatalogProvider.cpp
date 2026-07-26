#include "../ModelCatalogProvider.hpp"

#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"
#include "../../utils/UriUtils.hpp"

#include <simdjson.h>

#include <utility>

namespace core::llm {
namespace {

[[nodiscard]] ModelReasoningProfile anthropic_reasoning_profile(
    const catalog::JsonObjectView& model) {
    ModelReasoningProfile profile;
    profile.complete = model.has_capability_catalog();
    if (!profile.complete || !model.capability_supported("effort")) {
        profile.adaptive_thinking = model.nested_capability_supported(
            "thinking", {"types", "adaptive"});
        profile.manual_thinking = model.nested_capability_supported(
            "thinking", {"types", "enabled"});
        return profile;
    }

    profile.effort =
        ReasoningCapabilities{ReasoningCapability::Effort};
    if (model.nested_capability_supported("effort", {"max"})) {
        profile.effort =
            profile.effort | ReasoningCapability::MaxEffort;
    }
    if (model.nested_capability_supported("effort", {"xhigh"})) {
        profile.effort =
            profile.effort | ReasoningCapability::XHighEffort;
    }
    profile.adaptive_thinking = model.nested_capability_supported(
        "thinking", {"types", "adaptive"});
    profile.manual_thinking = model.nested_capability_supported(
        "thinking", {"types", "enabled"});
    return profile;
}

[[nodiscard]] ModelCapabilities anthropic_capabilities(
    const catalog::JsonObjectView& model) {
    if (!model.has_capability_catalog()) {
        return 0;
    }

    // These are Messages API features rather than per-model switches.
    ModelCapabilities capabilities =
        catalog::kToolCapabilities |
        static_cast<uint32_t>(ModelCapability::PromptCaching) |
        static_cast<uint32_t>(ModelCapability::TokenCounting);
    const auto add = [&](ModelCapability capability, bool supported) {
        if (supported) {
            capabilities |= static_cast<uint32_t>(capability);
        }
    };

    add(ModelCapability::Vision, model.capability_supported("image_input"));
    add(ModelCapability::PdfInput, model.capability_supported("pdf_input"));
    add(
        ModelCapability::JsonMode,
        model.capability_supported("structured_outputs"));
    add(
        ModelCapability::Reasoning,
        model.capability_supported("thinking")
            || model.capability_supported("effort"));
    add(ModelCapability::Citations, model.capability_supported("citations"));
    add(
        ModelCapability::CodeExecution,
        model.capability_supported("code_execution"));
    add(ModelCapability::Batch, model.capability_supported("batch"));
    add(
        ModelCapability::ContextManagement,
        model.capability_supported("context_management"));
    return capabilities;
}

} // namespace

AnthropicModelCatalogProvider::AnthropicModelCatalogProvider(
    std::string provider_name)
    : provider_name_(std::move(provider_name)) {
    if (provider_name_.empty()) {
        provider_name_ = "anthropic";
    }
}

std::string_view AnthropicModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string AnthropicModelCatalogProvider::model_list_path(
    std::string_view page_token) const {
    std::string path = "/v1/models?limit=1000";
    if (!page_token.empty()) {
        path += "&after_id=";
        path += core::utils::uri::percent_encode_uri_query_component(page_token);
    }
    return path;
}

ModelCatalogResult AnthropicModelCatalogProvider::parse_models_response(
    std::string_view body) const {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    ModelCatalogResult result =
        catalog::parse_catalog_json(body, document, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (document["data"].get(models) != simdjson::SUCCESS) {
        result.error = "Anthropic model catalog response is missing data[]";
        return result;
    }

    const catalog::JsonObjectView root(document.get_object());
    if (root.boolean("has_more")) {
        (void)root.string("last_id", result.next_page_token);
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
        info.context_window = model.integer("max_input_tokens");
        info.max_output_tokens = model.integer("max_tokens");
        info.capabilities = anthropic_capabilities(model);
        info.capabilities_complete = model.has_capability_catalog();
        info.reasoning = anthropic_reasoning_profile(model);
        info.tier = catalog::infer_tier(
            info.canonical_id,
            info.supports(ModelCapability::Reasoning));
        info.constraints = catalog::standard_constraints(1.0);
        result.models.push_back(std::move(info));
    }
    return result;
}

} // namespace core::llm
