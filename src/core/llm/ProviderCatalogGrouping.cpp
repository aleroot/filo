#include "ProviderCatalogGrouping.hpp"

#include "KimiModelTraits.hpp"
#include "ProviderDefinition.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <array>
#include <unordered_set>

namespace core::llm {
namespace {

constexpr std::array<std::string_view, 7> kZaiPlanSeparatedModels{{
    "glm-5.3",
    "glm-5.3-flash",
    "glm-5.3-flashx",
    "glm-5.2",
    "glm-5-turbo",
    "glm-4.7",
    "glm-4.5-air",
}};

// The live /models response is shared by Z.ai's General and Coding hosts, but
// FlashX is not currently included with the Coding Plan. Keep it reserved from
// General to avoid routing an aliased Coding Plan key there, while omitting it
// from the plan source itself.
constexpr std::array<std::string_view, 6> kZaiCodingPlanModels{{
    "glm-5.3",
    "glm-5.3-flash",
    "glm-5.2",
    "glm-5-turbo",
    "glm-4.7",
    "glm-4.5-air",
}};

// The Qwen family is served by three mutually exclusive endpoints that share a
// single registry provider key ("qwen"). Without disjoint model sets every
// endpoint would offer every Qwen model, so the picker would list the same ID
// under providers that cannot serve it -- and the first, usually
// credential-less, copy wins.
constexpr std::array<std::string_view, 6> kQwenTokenPlanTextModels{{
    "qwen3.8-max",
    "qwen3.8-flash",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen3.6-flash",
}};

constexpr std::array<std::string_view, 3> kQwenCodingPlanModels{{
    "qwen3-coder-plus",
    "qwen3-coder-flash",
    "coder-model",
}};

[[nodiscard]] std::string normalized(std::string_view value) {
    return core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(value));
}

[[nodiscard]] bool is_zai_coding_source(std::string_view provider_name) {
    return normalized(provider_name).starts_with("zai-coding");
}

[[nodiscard]] bool is_kimi_code_source(std::string_view provider_name) {
    return is_kimi_code_provider_name(provider_name);
}

[[nodiscard]] bool is_qwen_token_plan_source(std::string_view provider_name) {
    return normalized(provider_name).starts_with("qwen-token-plan");
}

[[nodiscard]] bool is_qwen_coding_source(std::string_view provider_name) {
    return normalized(provider_name).starts_with("qwen-coding");
}

[[nodiscard]] ProviderCatalogModelFilter model_filter(
    ProviderCatalogModelRule rule,
    std::span<const std::string_view> model_ids) {
    ProviderCatalogModelFilter filter{.rule = rule};
    filter.model_ids.reserve(model_ids.size());
    for (const auto model_id : model_ids) {
        filter.model_ids.emplace_back(model_id);
    }
    return filter;
}

[[nodiscard]] ProviderCatalogModelFilter zai_regular_filter() {
    return model_filter(
        ProviderCatalogModelRule::Exclude, kZaiPlanSeparatedModels);
}

[[nodiscard]] ProviderCatalogModelFilter zai_coding_filter() {
    return model_filter(ProviderCatalogModelRule::Include, kZaiCodingPlanModels);
}

[[nodiscard]] ProviderCatalogModelFilter kimi_regular_filter() {
    return model_filter(
        ProviderCatalogModelRule::Exclude,
        kimi_code_model_ids());
}

[[nodiscard]] ProviderCatalogModelFilter kimi_code_filter() {
    // Kimi Code model availability is controlled by the account/subscription.
    // Keep the configured default as a safety net, but do not supplement the
    // authenticated /models response with guesses from the static registry.
    return ProviderCatalogModelFilter{.rule = ProviderCatalogModelRule::Include};
}

[[nodiscard]] ProviderCatalogModelFilter qwen_token_plan_filter() {
    // The authenticated /models response remains authoritative. These exact
    // text-model IDs provide a conservative fallback when the catalog request
    // is still in flight or temporarily unavailable; unrelated DashScope
    // models must never leak into the subscription picker.
    return model_filter(
        ProviderCatalogModelRule::Include,
        kQwenTokenPlanTextModels);
}

[[nodiscard]] ProviderCatalogModelFilter qwen_coding_plan_filter() {
    // coding.dashscope.aliyuncs.com only serves the coder line; "coder-model"
    // is the endpoint's own subscription alias.
    return model_filter(
        ProviderCatalogModelRule::Include,
        kQwenCodingPlanModels);
}

[[nodiscard]] ProviderCatalogModelFilter qwen_public_dashscope_filter() {
    // Public DashScope keys are rejected by the Token Plan and Coding Plan
    // hosts and vice versa, so the pay-as-you-go source claims exactly the
    // models the other two do not.
    std::vector<std::string_view> reserved;
    reserved.reserve(
        kQwenTokenPlanTextModels.size() + kQwenCodingPlanModels.size());
    reserved.insert(
        reserved.end(),
        kQwenTokenPlanTextModels.begin(),
        kQwenTokenPlanTextModels.end());
    reserved.insert(
        reserved.end(),
        kQwenCodingPlanModels.begin(),
        kQwenCodingPlanModels.end());
    return model_filter(ProviderCatalogModelRule::Exclude, reserved);
}

[[nodiscard]] ProviderCatalogSource source_for_provider(std::string_view provider_name,
                                                        std::string_view group_name) {
    ProviderCatalogSource source{
        .provider_name = std::string(provider_name),
        .service_id = std::string(provider_name),
        .category_label = {},
        .registry_model_filter = {},
        .api_model_filter = {},
        .api_model_policy = ProviderCatalogApiModelPolicy::All,
    };

    // Z.ai's General and Coding hosts publish identical /models catalogs.
    // Membership has to be claimed on both origins or glm-5.3 is offered
    // under General API and a Coding Plan key returns HTTP 429 / error 1113.
    if (group_name == "zai" && is_zai_coding_source(provider_name)) {
        source.category_label = "GLM Coding Plan.";
        source.registry_model_filter = zai_coding_filter();
        source.api_model_filter = source.registry_model_filter;
        source.api_model_policy = ProviderCatalogApiModelPolicy::TextGeneration;
    } else if (group_name == "zai") {
        source.category_label = "General API.";
        source.registry_model_filter = zai_regular_filter();
        source.api_model_filter = source.registry_model_filter;
        source.api_model_policy = ProviderCatalogApiModelPolicy::TextGeneration;
    } else if (group_name == "kimi" && is_kimi_code_source(provider_name)) {
        source.service_id = std::string(kimi_service_id(KimiService::Code));
        source.category_label = "Kimi Code subscription.";
        source.registry_model_filter = kimi_code_filter();
    } else if (group_name == "kimi") {
        source.service_id =
            std::string(kimi_service_id(KimiService::PublicApi));
        source.category_label = "Kimi API.";
        source.registry_model_filter = kimi_regular_filter();
    } else if (group_name == "qwen" && is_qwen_token_plan_source(provider_name)) {
        source.category_label = "Token Plan endpoint.";
        source.registry_model_filter = qwen_token_plan_filter();
        source.api_model_policy = ProviderCatalogApiModelPolicy::TextGeneration;
    } else if (group_name == "qwen" && is_qwen_coding_source(provider_name)) {
        source.category_label = "Coding Plan endpoint.";
        source.registry_model_filter = qwen_coding_plan_filter();
        source.api_model_policy = ProviderCatalogApiModelPolicy::TextGeneration;
    } else if (group_name == "qwen") {
        source.category_label = "DashScope API.";
        source.registry_model_filter = qwen_public_dashscope_filter();
    }

    return source;
}

[[nodiscard]] bool contains_name(std::span<const std::string> names,
                                 std::string_view name) {
    return std::ranges::any_of(names, [&](const std::string& candidate) {
        return candidate == name;
    });
}

template <typename Predicate>
void append_first_matching_source(
    ProviderCatalogGroup& group,
    std::string_view group_name,
    std::span<const std::string> configured_provider_names,
    Predicate predicate,
    std::string_view preferred) {
    if (!preferred.empty() && contains_name(configured_provider_names, preferred)) {
        group.sources.push_back(source_for_provider(preferred, group_name));
        return;
    }
    const auto it = std::ranges::find_if(
        configured_provider_names,
        [&](const std::string& configured) {
            return provider_catalog_group_name(configured) == group_name
                && predicate(configured);
        });
    if (it != configured_provider_names.end()) {
        group.sources.push_back(source_for_provider(*it, group_name));
    }
}

} // namespace

bool ProviderCatalogModelFilter::matches(std::string_view model_id) const {
    if (rule == ProviderCatalogModelRule::All) {
        return true;
    }

    const std::string needle = normalized(model_id);
    const bool listed = std::ranges::any_of(model_ids, [&](const std::string& candidate) {
        return normalized(candidate) == needle;
    });
    return rule == ProviderCatalogModelRule::Include ? listed : !listed;
}

bool ProviderCatalogSource::includes_registry_model(std::string_view model_id) const {
    return registry_model_filter.matches(model_id);
}

bool ProviderCatalogSource::includes_api_model(std::string_view model_id) const {
    if (!api_model_filter.matches(model_id)) {
        return false;
    }

    if (api_model_policy == ProviderCatalogApiModelPolicy::All) {
        return true;
    }

    // Some provider catalogs mix chat models with image, audio, embedding, and
    // ranking endpoints but expose no modality metadata. Keep future text and
    // multimodal-chat IDs, rejecting only explicit non-generation markers.
    static constexpr std::array<std::string_view, 9> kNonTextMarkers{{
        "embedding",
        "rerank",
        "image",
        "video",
        "audio",
        "speech",
        "tts",
        "asr",
        "moderation",
    }};
    const std::string id = normalized(model_id);
    return std::ranges::none_of(kNonTextMarkers, [&](std::string_view marker) {
        return id.find(marker) != std::string::npos;
    });
}

bool ProviderCatalogGroup::contains_source_provider(std::string_view provider) const {
    // A family may deliberately hide compatibility presets from the picker.
    // They still belong to the visible group for active-selection highlighting.
    return find_source(provider) != nullptr
        || ((provider_name == "grok" || provider_name == "kimi")
            && provider_catalog_group_name(provider) == provider_name);
}

const ProviderCatalogSource*
ProviderCatalogGroup::find_source(std::string_view provider) const {
    const auto it = std::ranges::find_if(sources, [&](const ProviderCatalogSource& source) {
        return source.provider_name == provider;
    });
    return it == sources.end() ? nullptr : &*it;
}

const ProviderCatalogSource*
ProviderCatalogGroup::find_source_by_service_id(std::string_view service) const {
    const auto it = std::ranges::find_if(
        sources,
        [&](const ProviderCatalogSource& source) {
            return source.service_id == service;
        });
    return it == sources.end() ? nullptr : &*it;
}

std::string provider_catalog_group_name(std::string_view provider_name) {
    const std::string lowered = normalized(provider_name);
    if (const auto* definition =
            find_builtin_provider_definition(lowered);
        definition && !definition->catalog_group.empty()) {
        return std::string(definition->catalog_group);
    }
    return std::string(provider_name);
}

std::string provider_catalog_selection_key(
    std::string_view service_id,
    std::string_view model_id) {
    std::string key = normalized(service_id);
    key.push_back('\x1f');
    key += normalized(model_id);
    return key;
}

ProviderCatalogGroup provider_catalog_group_for(
    std::string_view provider_name,
    std::span<const std::string> configured_provider_names) {
    const std::string group_name = provider_catalog_group_name(provider_name);
    ProviderCatalogGroup group{
        .provider_name = group_name,
        .sources = {},
    };

    if (group_name == "grok") {
        // Every built-in Grok preset is a routing alias for the same xAI
        // service and returns the same authenticated model catalog. Querying
        // and rendering every preset repeats that catalog once per alias.
        // Prefer the canonical provider so model selections retain the active
        // credentials/profile; fall back to the first configured Grok alias
        // for custom configurations that omit it.
        if (contains_name(configured_provider_names, group_name)) {
            group.sources.push_back(source_for_provider(group_name, group_name));
            return group;
        }
        const auto it = std::ranges::find_if(
            configured_provider_names,
            [&](const std::string& configured) {
                return provider_catalog_group_name(configured) == group_name;
            });
        if (it != configured_provider_names.end()) {
            group.sources.push_back(source_for_provider(*it, group_name));
        }
        return group;
    }

    if (group_name == "zai") {
        // Coding Plan is the subscribed product. List it first so its models
        // sit at the top even when a historical login overlay copied the same
        // key onto General API (those keys cannot bill pay-as-you-go).
        append_first_matching_source(
            group,
            group_name,
            configured_provider_names,
            [](std::string_view configured) { return is_zai_coding_source(configured); },
            "zai-coding");
        append_first_matching_source(
            group,
            group_name,
            configured_provider_names,
            [](std::string_view configured) { return !is_zai_coding_source(configured); },
            "zai");
        return group;
    }

    if (group_name == "kimi") {
        // Filo keeps several Kimi presets for direct selectors and backwards
        // compatibility. They are routing aliases, not user-facing services.
        // Present one public API source and one managed Kimi Code source, just
        // like the official client models Kimi Code as a single provider whose
        // available models come from its authenticated /models endpoint.
        append_first_matching_source(
            group,
            group_name,
            configured_provider_names,
            [](std::string_view configured) { return !is_kimi_code_source(configured); },
            "kimi");
        append_first_matching_source(
            group,
            group_name,
            configured_provider_names,
            [](std::string_view configured) { return is_kimi_code_source(configured); },
            "kimi-code");
        return group;
    }

    if (contains_name(configured_provider_names, group_name)) {
        group.sources.push_back(source_for_provider(group_name, group_name));
    }

    for (const auto& configured_provider : configured_provider_names) {
        if (configured_provider == group_name) {
            continue;
        }
        if (provider_catalog_group_name(configured_provider) == group_name) {
            group.sources.push_back(source_for_provider(configured_provider, group_name));
        }
    }

    if (group.sources.empty() && contains_name(configured_provider_names, provider_name)) {
        group.sources.push_back(source_for_provider(provider_name, group_name));
    }

    return group;
}

std::vector<ProviderCatalogGroup> provider_catalog_groups(
    std::span<const std::string> configured_provider_names) {
    std::vector<ProviderCatalogGroup> groups;
    std::unordered_set<std::string> seen;

    for (const auto& provider_name : configured_provider_names) {
        const std::string group_name = provider_catalog_group_name(provider_name);
        if (!seen.insert(group_name).second) {
            continue;
        }
        auto group = provider_catalog_group_for(group_name, configured_provider_names);
        if (!group.sources.empty()) {
            groups.push_back(std::move(group));
        }
    }

    return groups;
}

} // namespace core::llm
