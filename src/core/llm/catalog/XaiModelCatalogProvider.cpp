#include "../ModelCatalogProvider.hpp"

#include "CodexModelCatalog.hpp"

#include "ModelCatalogJson.hpp"
#include "ModelCatalogTraits.hpp"
#include "../../utils/JsonUtils.hpp"
#include "core/utils/StringUtils.hpp"

#include <simdjson.h>

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

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

// Both endpoints return {"data":[...]}. Session rows are camelCase
// (model, contextWindow, reasoningEfforts); public rows use the OpenAI
// shape, with the effort menu under capabilities.

struct GrokCatalogEffortOption {
    std::string value;
    bool is_default = false;
};

[[nodiscard]] std::optional<std::string> canonical_grok_catalog_effort(
    std::string_view raw) {
    // Lowercase, and only the effort tokens Grok accepts.
    const std::string effort = core::utils::str::to_lower_ascii_copy(raw);
    if (effort == "none" || effort == "minimal" || effort == "low"
        || effort == "medium" || effort == "high" || effort == "xhigh"
        || effort == "max") {
        return effort;
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string> effort_field(
    simdjson::dom::object object,
    std::string_view key) {
    std::string_view value;
    if (object[key].get(value) != simdjson::SUCCESS) return std::nullopt;
    return canonical_grok_catalog_effort(value);
}

// Bare strings and {value, default?} objects. Unknown tokens are skipped.
[[nodiscard]] std::vector<GrokCatalogEffortOption> parse_effort_options(
    simdjson::dom::array array) {
    std::vector<GrokCatalogEffortOption> options;
    for (simdjson::dom::element item : array) {
        std::string_view bare;
        if (item.get(bare) == simdjson::SUCCESS) {
            if (auto value = canonical_grok_catalog_effort(bare)) {
                options.push_back({std::move(*value), false});
            }
            continue;
        }
        simdjson::dom::object object;
        if (item.get(object) != simdjson::SUCCESS) continue;
        auto value = effort_field(object, "value");
        if (!value) continue;
        bool is_default = false;
        core::utils::json::ignore_error(object["default"].get(is_default));
        options.push_back({std::move(*value), is_default});
    }
    return options;
}

[[nodiscard]] std::optional<simdjson::dom::array> array_field(
    simdjson::dom::object object,
    std::string_view key) {
    simdjson::dom::array array;
    if (object[key].get(array) != simdjson::SUCCESS) return std::nullopt;
    return array;
}

// First present key wins. A non-bool value is false; a missing key falls through.
[[nodiscard]] bool first_bool(
    simdjson::dom::object object,
    const simdjson::dom::object* meta,
    std::initializer_list<std::string_view> object_keys,
    std::string_view meta_key) {
    const auto read = [](simdjson::dom::object source, std::string_view key)
        -> std::optional<bool> {
        simdjson::dom::element element;
        if (source[key].get(element) != simdjson::SUCCESS) return std::nullopt;
        bool value = false;
        if (element.get(value) != simdjson::SUCCESS) return false;
        return value;
    };
    for (const std::string_view key : object_keys) {
        if (auto value = read(object, key)) return *value;
    }
    if (meta) {
        if (auto value = read(*meta, meta_key)) return *value;
    }
    return false;
}

[[nodiscard]] std::vector<GrokCatalogEffortOption> first_nonempty_menu(
    simdjson::dom::object object,
    const simdjson::dom::object* meta) {
    for (const std::string_view key :
         {"reasoningEfforts", "reasoning_efforts"}) {
        if (auto array = array_field(object, key)) {
            auto options = parse_effort_options(*array);
            if (!options.empty()) return options;
        }
    }
    if (meta) {
        if (auto array = array_field(*meta, "reasoningEfforts")) {
            return parse_effort_options(*array);
        }
    }
    return {};
}

struct GrokCatalogEffortMenu {
    std::vector<GrokCatalogEffortOption> options;
    bool server_default = false;
};

// Public rows keep the menu under capabilities.reasoning_effort.
[[nodiscard]] std::optional<GrokCatalogEffortMenu> capabilities_effort_menu(
    simdjson::dom::object model) {
    simdjson::dom::object capabilities;
    if (model["capabilities"].get(capabilities) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    auto levels = array_field(capabilities, "reasoning_effort");
    if (!levels) return std::nullopt;
    GrokCatalogEffortMenu menu;
    menu.options = parse_effort_options(*levels);
    if (menu.options.empty()) return std::nullopt;

    simdjson::dom::element default_effort;
    const bool has_default =
        capabilities["default_reasoning_effort"].get(default_effort)
        == simdjson::SUCCESS
        && !default_effort.is_null();
    bool marked = false;
    if (has_default) {
        std::string_view raw;
        const auto level = default_effort.get(raw) == simdjson::SUCCESS
            ? canonical_grok_catalog_effort(raw)
            : std::nullopt;
        if (level) {
            for (auto& option : menu.options) {
                option.is_default = option.value == *level;
                marked = marked || option.is_default;
            }
        }
    }
    menu.server_default = !marked;
    return menu;
}

[[nodiscard]] ModelReasoningProfile xai_reasoning_profile(
    const catalog::JsonObjectView& model) {
    ModelReasoningProfile profile;

    simdjson::dom::object meta_object;
    const bool has_meta =
        model.raw()["meta"].get(meta_object) == simdjson::SUCCESS;
    const simdjson::dom::object* meta = has_meta ? &meta_object : nullptr;

    const bool supports_flag = first_bool(
        model.raw(), meta,
        {"supportsReasoningEffort", "supports_reasoning_effort"},
        "supportsReasoningEffort");

    std::optional<std::string> explicit_effort =
        effort_field(model.raw(), "reasoningEffort");
    if (!explicit_effort) {
        explicit_effort = effort_field(model.raw(), "reasoning_effort");
    }
    if (!explicit_effort && meta) {
        explicit_effort = effort_field(*meta, "reasoningEffort");
    }

    std::optional<GrokCatalogEffortMenu> menu;
    if (auto object_menu = first_nonempty_menu(model.raw(), meta);
        !object_menu.empty()) {
        menu = GrokCatalogEffortMenu{
            std::move(object_menu),
            /*server_default=*/false,
        };
    } else {
        menu = capabilities_effort_menu(model.raw());
    }

    // Explicit field, else the marked entry, else the first session-menu entry.
    std::optional<std::string> chosen = explicit_effort;
    if (!chosen && menu && !menu->options.empty()) {
        const auto marked = std::ranges::find_if(
            menu->options,
            [](const GrokCatalogEffortOption& option) {
                return option.is_default;
            });
        if (marked != menu->options.end()) {
            chosen = marked->value;
        } else if (!menu->server_default) {
            chosen = menu->options.front().value;
        }
    }

    const bool menu_supports = menu && !menu->options.empty();
    const bool supports =
        supports_flag || menu_supports || explicit_effort.has_value();
    if (!supports) return profile;

    ReasoningCapabilities effort{ReasoningCapability::Effort};
    bool xhigh = chosen && *chosen == "xhigh";
    if (menu) {
        for (const auto& option : menu->options) {
            if (option.value == "xhigh") xhigh = true;
        }
    }
    if (xhigh) {
        effort = effort | ReasoningCapability::XHighEffort;
    }
    profile.effort = effort;
    profile.catalog_default_effort = chosen.value_or(std::string{});
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
