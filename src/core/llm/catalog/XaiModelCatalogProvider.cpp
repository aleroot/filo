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
    // The Grok account-session proxy exposes the Codex-style catalog. Public
    // xAI API keys use the documented OpenAI-shaped endpoint.
    return "/models";
}

ModelCatalogResult XaiModelCatalogProvider::parse_models_response(
    std::string_view body) const {
    if (use_session_catalog_) {
        return catalog::parse_codex_style_model_catalog(body, provider_name_);
    }

    simdjson::dom::parser parser;
    simdjson::dom::element document;
    ModelCatalogResult result =
        catalog::parse_catalog_json(body, document, parser);
    if (!result.ok()) return result;

    simdjson::dom::array models;
    if (document["data"].get(models) != simdjson::SUCCESS) {
        result.error = "xAI model catalog response is missing data[]";
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

        // /models also contains image-generation-only models. Filo's provider
        // is a language/chat transport, so expose only entries with a text
        // completion price field (including explicitly free entries).
        int64_t completion_price = 0;
        if (object["completion_text_token_price"].get(completion_price)
            != simdjson::SUCCESS) {
            continue;
        }

        info.provider = provider_name_;
        info.display_name = info.canonical_id;
        info.aliases = model.string_array("aliases");
        info.context_window = model.integer("context_length");

        info.capabilities =
            catalog::kToolCapabilities |
            static_cast<uint32_t>(ModelCapability::JsonMode) |
            static_cast<uint32_t>(ModelCapability::PromptCaching);

        int64_t image_price = 0;
        if (object["prompt_image_token_price"].get(image_price)
                == simdjson::SUCCESS
            && image_price > 0) {
            info.capabilities |=
                static_cast<uint32_t>(ModelCapability::Vision);
        }
        if (xai_reasoning_model(info.canonical_id)) {
            info.capabilities |=
                static_cast<uint32_t>(ModelCapability::Reasoning);
        }

        int64_t input_price = 0;
        int64_t cached_price = 0;
        core::utils::json::ignore_error(object["prompt_text_token_price"].get(input_price));
        const bool has_cached_price =
            object["cached_prompt_text_token_price"].get(cached_price)
            == simdjson::SUCCESS;
        info.pricing.input_per_mtok =
            xai_price_per_million(input_price);
        info.pricing.output_per_mtok =
            xai_price_per_million(completion_price);
        info.pricing.cached_input_per_mtok = has_cached_price
            ? xai_price_per_million(cached_price)
            : -1.0;

        info.tier = catalog::infer_tier(
            info.canonical_id,
            info.supports(ModelCapability::Reasoning));
        info.constraints = catalog::standard_constraints();
        result.models.push_back(std::move(info));
    }
    return result;
}

} // namespace core::llm
