#include "../ModelCatalogProvider.hpp"

#include "CodexModelCatalog.hpp"

#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"
#include "../../utils/JsonUtils.hpp"

#include <simdjson.h>

#include <cstdint>
#include <utility>

namespace core::llm {
namespace {

[[nodiscard]] bool xai_reasoning_model(std::string_view model_id) {
    return catalog::contains_ascii(model_id, "reasoning")
        || catalog::contains_ascii(model_id, "thinking");
}

[[nodiscard]] double xai_price_per_million(int64_t cents_per_100m) {
    // xAI publishes integer USD cents per 100 million tokens.
    return cents_per_100m > 0
        ? static_cast<double>(cents_per_100m) / 10'000.0
        : 0.0;
}

// ─────────────────────────────────────────────────────────────────────────────
// Remote catalog rows.
//
// grok-build (xai-org/grok-build, remote/model_source/oai.rs +
// remote/client.rs parse_remote_model_value) reads an OpenAI-shaped
// `{"data":[...]}` list from BOTH endpoints:
//   - cli-chat-proxy.grok.com/v1/models (account sessions): rich camelCase
//     "models-v2" rows — model/modelId, contextWindow, maxCompletionTokens,
//     reasoningEfforts[{value,label,default}], supportsReasoningEffort,
//     supportsBackendSearch, hidden, supportedInApi;
//   - api.x.ai/v1/models (API keys): id/aliases/context_length plus xAI
//     pricing fields, with the effort menu nested under
//     `capabilities.reasoning_effort` (bare strings) and
//     `capabilities.default_reasoning_effort`.
// The rows below decode both dialects, so new models (grok-4.7 and later)
// surface with their server-published context window and effort menu as soon
// as the catalog serves them.
// ─────────────────────────────────────────────────────────────────────────────

[[nodiscard]] ModelReasoningProfile xai_reasoning_profile(
    const catalog::JsonObjectView& model) {
    ModelReasoningProfile profile;
    ReasoningCapabilities effort{ReasoningCapability::Effort};

    // Session-proxy shape: reasoningEfforts[{value,...}] object rows.
    const auto object_menu_has = [&](std::string_view level) {
        return model.object_array_contains("reasoningEfforts", "value", level)
            || model.object_array_contains("reasoning_efforts", "value",
                                           level);
    };

    // Public-API shape: capabilities.reasoning_effort bare-string menu.
    const auto capability_menu_has = [&](std::string_view level) {
        simdjson::dom::object capabilities;
        simdjson::dom::array levels;
        return model.raw()["capabilities"].get(capabilities)
                == simdjson::SUCCESS
            && capabilities["reasoning_effort"].get(levels)
                == simdjson::SUCCESS
            && catalog::JsonObjectView(capabilities)
                   .string_array_contains("reasoning_effort", level);
    };

    const auto menu_has = [&](std::string_view level) {
        return object_menu_has(level) || capability_menu_has(level);
    };

    if (menu_has("low") || menu_has("medium") || menu_has("high")
        || menu_has("xhigh")
        || model.boolean("supportsReasoningEffort")
        || model.boolean("supports_reasoning_effort")) {
        if (menu_has("xhigh")) {
            effort = effort | ReasoningCapability::XHighEffort;
        }
        profile.effort = effort;
    }
    return profile;
}

[[nodiscard]] ModelInfo decode_xai_catalog_row(
    const catalog::JsonObjectView& model,
    std::string_view provider_name,
    bool use_session_catalog) {
    ModelInfo info;
    // Proxy rows carry the wire id in `model`/`modelId`; public rows in `id`.
    (void)model.string("id", info.canonical_id);
    if (info.canonical_id.empty()) {
        (void)model.string("model", info.canonical_id);
    }
    if (info.canonical_id.empty()) {
        (void)model.string("modelId", info.canonical_id);
    }
    if (info.canonical_id.empty()
        || catalog::is_embedding_model(info.canonical_id)) {
        return info;
    }

    // Catalog-side visibility flags: hidden or API-unavailable rows are not
    // selectable and must not leak into the model picker.
    if (model.boolean("hidden")) {
        info.canonical_id.clear();
        return info;
    }

    // /models also contains image-generation-only models. Filo's provider is a
    // language/chat transport, so on the public API expose only entries with a
    // text completion price field (including explicitly free entries). Session
    // rows never carry pricing and must all survive.
    int64_t completion_price = 0;
    const bool has_completion_price =
        model.raw()["completion_text_token_price"].get(completion_price)
        == simdjson::SUCCESS;
    if (!use_session_catalog && !has_completion_price) {
        info.canonical_id.clear();
        return info;
    }

    info.provider = catalog::copy_string(provider_name);
    if (!model.string("name", info.display_name)
        && !model.string("display_name", info.display_name)) {
        info.display_name = info.canonical_id;
    }
    info.aliases = model.string_array("aliases");
    info.context_window = model.first_integer(
        {"contextWindow", "context_window", "max_context_window",
         "context_length"});
    info.max_output_tokens = model.first_integer(
        {"maxCompletionTokens", "max_completion_tokens", "max_output_tokens",
         "max_tokens"});

    info.capabilities =
        catalog::kToolCapabilities |
        static_cast<uint32_t>(ModelCapability::JsonMode) |
        static_cast<uint32_t>(ModelCapability::PromptCaching);

    int64_t image_price = 0;
    if (model.raw()["prompt_image_token_price"].get(image_price)
            == simdjson::SUCCESS
        && image_price > 0) {
        info.capabilities |=
            static_cast<uint32_t>(ModelCapability::Vision);
    }

    info.reasoning = xai_reasoning_profile(model);
    if (xai_reasoning_model(info.canonical_id)
        || !info.reasoning.effort.empty()) {
        info.capabilities |=
            static_cast<uint32_t>(ModelCapability::Reasoning);
    }

    int64_t input_price = 0;
    int64_t cached_price = 0;
    core::utils::json::ignore_error(
        model.raw()["prompt_text_token_price"].get(input_price));
    const bool has_cached_price =
        model.raw()["cached_prompt_text_token_price"].get(cached_price)
        == simdjson::SUCCESS;
    info.pricing.input_per_mtok = xai_price_per_million(input_price);
    info.pricing.output_per_mtok = xai_price_per_million(completion_price);
    info.pricing.cached_input_per_mtok = has_cached_price
        ? xai_price_per_million(cached_price)
        : -1.0;

    info.tier = catalog::infer_tier(
        info.canonical_id,
        info.supports(ModelCapability::Reasoning));
    info.constraints = catalog::standard_constraints();
    return info;
}

} // namespace

XaiModelCatalogProvider::XaiModelCatalogProvider(
    std::string provider_name,
    bool use_session_catalog)
    : provider_name_(std::move(provider_name))
    , use_session_catalog_(use_session_catalog) {
    if (provider_name_.empty()) {
        provider_name_ = "grok";
    }
}

std::string_view XaiModelCatalogProvider::provider_name() const noexcept {
    return provider_name_;
}

std::string XaiModelCatalogProvider::model_list_path(
    std::string_view page_token) const {
    (void)page_token;
    // Both the public xAI API and the Grok account-session proxy expose the
    // OpenAI-shaped endpoint at /v1/models.
    return "/models";
}

ModelCatalogResult XaiModelCatalogProvider::parse_models_response(
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
            const catalog::JsonObjectView model(object);
            if (model.boolean("supported_in_api", /*fallback=*/true)
                && model.boolean("supportedInApi", /*fallback=*/true)) {
                ModelInfo info =
                    decode_xai_catalog_row(model, provider_name_,
                                           use_session_catalog_);
                if (!info.canonical_id.empty()) {
                    result.models.push_back(std::move(info));
                }
            }
        }
        return result;
    }

    if (use_session_catalog_) {
        // Older proxy builds served the Codex-style {models:[{slug}]} schema;
        // keep decoding it so a catalog refresh against one is not a failure.
        return catalog::parse_codex_style_model_catalog(body, provider_name_);
    }

    result.error = "xAI model catalog response is missing data[]";
    return result;
}

} // namespace core::llm
