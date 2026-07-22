#include "ProviderCatalogGrouping.hpp"

#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <array>
#include <unordered_set>

namespace core::llm {
namespace {

struct ProviderCatalogFamily {
    std::string_view group_name;
};

// Built-in provider presets are separate routing targets, but the model picker
// presents them as one provider family. Keep this list centralized so adding a
// preset never creates another top-level /model entry by accident.
constexpr std::array kProviderCatalogFamilies{
    ProviderCatalogFamily{"grok"},
    ProviderCatalogFamily{"kimi"},
    ProviderCatalogFamily{"qwen"},
    ProviderCatalogFamily{"zai"},
};

constexpr std::array<std::string_view, 4> kZaiCodingModels{{
    "glm-5.2",
    "glm-5-turbo",
    "glm-4.7",
    "glm-4.5-air",
}};

[[nodiscard]] std::string normalized(std::string_view value) {
    return core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(value));
}

[[nodiscard]] bool is_family_member(std::string_view provider_name,
                                    std::string_view group_name) {
    const std::string lowered = normalized(provider_name);
    return lowered == group_name
        || (lowered.starts_with(group_name)
            && lowered.size() > group_name.size()
            && lowered[group_name.size()] == '-');
}

[[nodiscard]] const ProviderCatalogFamily*
find_family(std::string_view provider_name) {
    const auto it = std::ranges::find_if(
        kProviderCatalogFamilies,
        [&](const ProviderCatalogFamily& family) {
            return is_family_member(provider_name, family.group_name);
        });
    return it == kProviderCatalogFamilies.end() ? nullptr : &*it;
}

[[nodiscard]] bool is_zai_coding_source(std::string_view provider_name) {
    return normalized(provider_name).starts_with("zai-coding");
}

[[nodiscard]] bool is_kimi_code_source(std::string_view provider_name) {
    const std::string lowered = normalized(provider_name);
    return lowered == "kimi-code"
        || lowered.starts_with("kimi-code-")
        || lowered == "kimi-for-coding"
        || lowered.starts_with("kimi-for-coding-");
}

[[nodiscard]] bool is_qwen_token_plan_source(std::string_view provider_name) {
    return normalized(provider_name).starts_with("qwen-token-plan");
}

template <std::size_t N>
[[nodiscard]] ProviderCatalogModelFilter model_filter(
    ProviderCatalogModelRule rule,
    const std::array<std::string_view, N>& model_ids) {
    ProviderCatalogModelFilter filter{.rule = rule};
    filter.model_ids.reserve(model_ids.size());
    for (const auto model_id : model_ids) {
        filter.model_ids.emplace_back(model_id);
    }
    return filter;
}

[[nodiscard]] ProviderCatalogModelFilter zai_regular_filter() {
    return model_filter(ProviderCatalogModelRule::Exclude, kZaiCodingModels);
}

[[nodiscard]] ProviderCatalogModelFilter zai_coding_filter() {
    return model_filter(ProviderCatalogModelRule::Include, kZaiCodingModels);
}

[[nodiscard]] ProviderCatalogModelFilter kimi_regular_filter() {
    constexpr std::array<std::string_view, 3> code_models{{
        "k3",
        "kimi-for-coding",
        "kimi-for-coding-highspeed",
    }};
    return model_filter(ProviderCatalogModelRule::Exclude, code_models);
}

[[nodiscard]] ProviderCatalogModelFilter kimi_code_filter() {
    // Kimi Code model availability is controlled by the account/subscription.
    // Keep the configured default as a safety net, but do not supplement the
    // authenticated /models response with guesses from the static registry.
    return ProviderCatalogModelFilter{.rule = ProviderCatalogModelRule::Include};
}

[[nodiscard]] ProviderCatalogModelFilter qwen_token_plan_filter() {
    // Token Plan availability is account- and subscription-specific. An empty
    // include filter intentionally disables static-registry fallback; the live
    // /models response is the source of truth for this endpoint.
    return ProviderCatalogModelFilter{.rule = ProviderCatalogModelRule::Include};
}

[[nodiscard]] ProviderCatalogSource source_for_provider(std::string_view provider_name,
                                                        std::string_view group_name) {
    ProviderCatalogSource source{
        .provider_name = std::string(provider_name),
        .category_label = {},
        .registry_model_filter = {},
    };

    if (group_name == "zai" && is_zai_coding_source(provider_name)) {
        source.category_label = "Coding endpoint.";
        source.registry_model_filter = zai_coding_filter();
    } else if (group_name == "zai") {
        source.registry_model_filter = zai_regular_filter();
    } else if (group_name == "kimi" && is_kimi_code_source(provider_name)) {
        source.category_label = "Kimi Code subscription.";
        source.registry_model_filter = kimi_code_filter();
    } else if (group_name == "kimi") {
        source.category_label = "Kimi API.";
        source.registry_model_filter = kimi_regular_filter();
    } else if (group_name == "qwen" && is_qwen_token_plan_source(provider_name)) {
        source.category_label = "Token Plan endpoint.";
        source.registry_model_filter = qwen_token_plan_filter();
    }

    return source;
}

[[nodiscard]] bool contains_name(std::span<const std::string> names,
                                 std::string_view name) {
    return std::ranges::any_of(names, [&](const std::string& candidate) {
        return candidate == name;
    });
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

bool ProviderCatalogGroup::contains_source_provider(std::string_view provider) const {
    // A family may deliberately hide compatibility presets from the picker.
    // They still belong to the visible group for active-selection highlighting.
    return find_source(provider) != nullptr
        || (provider_name == "kimi"
            && provider_catalog_group_name(provider) == provider_name);
}

const ProviderCatalogSource*
ProviderCatalogGroup::find_source(std::string_view provider) const {
    const auto it = std::ranges::find_if(sources, [&](const ProviderCatalogSource& source) {
        return source.provider_name == provider;
    });
    return it == sources.end() ? nullptr : &*it;
}

std::string provider_catalog_group_name(std::string_view provider_name) {
    if (const auto* family = find_family(provider_name)) {
        return std::string(family->group_name);
    }
    return std::string(provider_name);
}

ProviderCatalogGroup provider_catalog_group_for(
    std::string_view provider_name,
    std::span<const std::string> configured_provider_names) {
    const std::string group_name = provider_catalog_group_name(provider_name);
    ProviderCatalogGroup group{
        .provider_name = group_name,
        .sources = {},
    };

    if (group_name == "kimi") {
        // Filo keeps several Kimi presets for direct selectors and backwards
        // compatibility. They are routing aliases, not user-facing services.
        // Present one public API source and one managed Kimi Code source, just
        // like the official client models Kimi Code as a single provider whose
        // available models come from its authenticated /models endpoint.
        const auto add_first_matching = [&](auto&& predicate,
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
        };

        add_first_matching(
            [](std::string_view configured) { return !is_kimi_code_source(configured); },
            "kimi");
        add_first_matching(
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
