#pragma once

#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::llm {

enum class ProviderCatalogModelRule {
    All,
    Include,
    Exclude,
};

enum class ProviderCatalogApiModelPolicy {
    All,
    TextGeneration,
};

struct ProviderCatalogModelFilter {
    ProviderCatalogModelRule rule = ProviderCatalogModelRule::All;
    std::vector<std::string> model_ids;

    [[nodiscard]] bool matches(std::string_view model_id) const;
};

struct ProviderCatalogSource {
    std::string provider_name;
    std::string service_id;
    std::string category_label;
    ProviderCatalogModelFilter registry_model_filter;
    ProviderCatalogApiModelPolicy api_model_policy =
        ProviderCatalogApiModelPolicy::All;

    [[nodiscard]] bool includes_registry_model(std::string_view model_id) const;
    [[nodiscard]] bool includes_api_model(std::string_view model_id) const;
};

struct ProviderCatalogGroup {
    std::string provider_name;
    std::vector<ProviderCatalogSource> sources;

    [[nodiscard]] bool contains_source_provider(std::string_view provider_name) const;
    [[nodiscard]] const ProviderCatalogSource*
    find_source(std::string_view provider_name) const;
    [[nodiscard]] const ProviderCatalogSource*
    find_source_by_service_id(std::string_view service_id) const;
};

[[nodiscard]] std::string provider_catalog_group_name(std::string_view provider_name);

/**
 * Stable identity for a selectable model on one provider service.
 *
 * Model IDs are not globally unique across services: the same ID can be
 * offered by both a pay-as-you-go API and a subscription endpoint with
 * different billing and entitlement semantics. Picker deduplication must
 * therefore preserve the service dimension.
 */
[[nodiscard]] std::string provider_catalog_selection_key(
    std::string_view service_id,
    std::string_view model_id);

[[nodiscard]] ProviderCatalogGroup provider_catalog_group_for(
    std::string_view provider_name,
    std::span<const std::string> configured_provider_names);

[[nodiscard]] std::vector<ProviderCatalogGroup> provider_catalog_groups(
    std::span<const std::string> configured_provider_names);

} // namespace core::llm
