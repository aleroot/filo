#include "ProviderCatalogGrouping.hpp"

#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <array>
#include <unordered_set>

namespace core::llm {
namespace {

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

[[nodiscard]] bool is_zai_group_member(std::string_view provider_name) {
    const std::string lowered = normalized(provider_name);
    return lowered == "zai" || lowered.starts_with("zai-");
}

[[nodiscard]] bool is_zai_coding_source(std::string_view provider_name) {
    return normalized(provider_name).starts_with("zai-coding");
}

[[nodiscard]] ProviderCatalogModelFilter zai_regular_filter() {
    return ProviderCatalogModelFilter{
        .rule = ProviderCatalogModelRule::Exclude,
        .model_ids = {
            std::string(kZaiCodingModels[0]),
            std::string(kZaiCodingModels[1]),
            std::string(kZaiCodingModels[2]),
            std::string(kZaiCodingModels[3]),
        },
    };
}

[[nodiscard]] ProviderCatalogModelFilter zai_coding_filter() {
    return ProviderCatalogModelFilter{
        .rule = ProviderCatalogModelRule::Include,
        .model_ids = {
            std::string(kZaiCodingModels[0]),
            std::string(kZaiCodingModels[1]),
            std::string(kZaiCodingModels[2]),
            std::string(kZaiCodingModels[3]),
        },
    };
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
    } else if (group_name == "zai" && is_zai_group_member(provider_name)) {
        source.registry_model_filter = zai_regular_filter();
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
    return find_source(provider) != nullptr;
}

const ProviderCatalogSource*
ProviderCatalogGroup::find_source(std::string_view provider) const {
    const auto it = std::ranges::find_if(sources, [&](const ProviderCatalogSource& source) {
        return source.provider_name == provider;
    });
    return it == sources.end() ? nullptr : &*it;
}

std::string provider_catalog_group_name(std::string_view provider_name) {
    if (is_zai_group_member(provider_name)) {
        return "zai";
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
