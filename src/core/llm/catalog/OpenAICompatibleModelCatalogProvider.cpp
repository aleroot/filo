#include "../ModelCatalogProvider.hpp"

#include "CompatibleModelMetadata.hpp"
#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"
#include "../../utils/StringUtils.hpp"

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
        catalog::kTextCapabilities |
        static_cast<uint32_t>(ModelCapability::FunctionCalling) |
        static_cast<uint32_t>(ModelCapability::JsonMode);
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
        || model.string(
            "default_reasoning_level",
            default_reasoning_level)
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "minimal")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "low")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "medium")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "high")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "xhigh")) {
        capabilities |= static_cast<uint32_t>(ModelCapability::Reasoning);
    }
    return capabilities;
}

[[nodiscard]] ModelReasoningProfile compatible_reasoning_profile(
    const catalog::JsonObjectView& model) {
    ModelReasoningProfile profile;
    const bool has_effort =
        model.capability_supported("effort")
        || model.capability_supported("reasoning")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "minimal")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "low")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "medium")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "high")
        || model.object_array_contains(
            "supported_reasoning_levels", "effort", "xhigh");
    if (has_effort) {
        profile.effort =
            ReasoningCapabilities{ReasoningCapability::Effort};
    }
    if (model.object_array_contains(
            "supported_reasoning_levels", "effort", "max")) {
        profile.effort =
            profile.effort | ReasoningCapability::MaxEffort;
    }
    if (model.object_array_contains(
            "supported_reasoning_levels", "effort", "xhigh")) {
        profile.effort =
            profile.effort | ReasoningCapability::XHighEffort;
    }
    return profile;
}

[[nodiscard]] ModelInfo decode_model(
    const catalog::JsonObjectView& model,
    std::string_view provider_name,
    bool codex_shape) {
    ModelInfo info;
    if (codex_shape) {
        if (!model.string("slug", info.canonical_id)) {
            (void)model.string("id", info.canonical_id);
        }
    } else {
        (void)model.string("id", info.canonical_id);
    }
    if (info.canonical_id.empty()) return info;

    info.provider = catalog::copy_string(provider_name);
    if (!model.string("display_name", info.display_name)) {
        info.display_name = info.canonical_id;
    }
    info.capabilities = catalog::compatible_advertised_capabilities(
        info.canonical_id,
        model);
    if (codex_shape) {
        info.capabilities |= codex_capabilities(info.canonical_id, model);
    }
    info.context_window = model.first_integer(
        {"context_window", "max_context_window", "context_length"});
    info.max_output_tokens = model.first_integer(
        {"max_output_tokens", "max_tokens"});
    info.max_reasoning_tokens = model.first_integer(
        {"max_reasoning_tokens", "reasoning_max_tokens"});
    info.reasoning = compatible_reasoning_profile(model);
    info.tier = catalog::infer_tier(
        info.canonical_id,
        info.supports(ModelCapability::Reasoning));
    info.constraints = catalog::standard_constraints();
    return info;
}

} // namespace

OpenAICompatibleModelCatalogProvider::
OpenAICompatibleModelCatalogProvider(
    std::string provider_name,
    bool include_session_only_models)
    : provider_name_(std::move(provider_name))
    , include_session_only_models_(include_session_only_models) {
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
                provider_name_,
                false);
            if (!info.canonical_id.empty()) {
                result.models.push_back(std::move(info));
            }
        }
        return result;
    }

    if (document["models"].get(models) != simdjson::SUCCESS) {
        result.error =
            "OpenAI-compatible model catalog response is missing data[] or models[]";
        return result;
    }

    for (simdjson::dom::element element : models) {
        simdjson::dom::object object;
        if (element.get(object) != simdjson::SUCCESS) continue;
        const catalog::JsonObjectView model(object);
        if (!include_session_only_models_
            && !model.boolean("supported_in_api", true)) {
            continue;
        }

        ModelInfo info = decode_model(model, provider_name_, true);
        if (!info.canonical_id.empty()) {
            result.models.push_back(std::move(info));
        }
    }
    return result;
}

} // namespace core::llm
