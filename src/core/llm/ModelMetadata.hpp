#pragma once

#include "ModelRegistry.hpp"

#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

namespace core::config {
enum class ApiType;
}

namespace core::llm {

enum class ModelMetadataOrigin {
    None,
    ProviderApi,
    InternalRegistry,
};

/**
 * A non-owning view of the selected catalog tier.
 *
 * Provider API results are the primary source. The internal registry is used
 * only when the API tier contains no models. Callers retain ownership of both
 * input ranges and must not outlive them.
 */
struct ResolvedModelCatalog {
    std::span<const ModelInfo> models;
    ModelMetadataOrigin origin = ModelMetadataOrigin::None;

    [[nodiscard]] bool empty() const noexcept { return models.empty(); }
    [[nodiscard]] bool uses_registry_fallback() const noexcept {
        return origin == ModelMetadataOrigin::InternalRegistry;
    }
};

/**
 * An owning model-metadata result suitable for request validation and wire
 * serialization.
 */
struct ResolvedModelMetadata {
    std::optional<ModelInfo> model;
    ModelMetadataOrigin origin = ModelMetadataOrigin::None;

    [[nodiscard]] explicit operator bool() const noexcept {
        return model.has_value();
    }
};

/**
 * Merge live provider metadata into an existing model card.
 *
 * This pure policy function deliberately knows nothing about registry locking,
 * aliases maps, or publication. Complete provider sections replace stale
 * values; partial sections enrich the baseline without erasing known data.
 */
[[nodiscard]] ModelInfo merge_model_metadata(
    ModelInfo baseline,
    ModelInfo discovered);

/**
 * Select a provider catalog using an explicit two-tier fallback policy.
 *
 * Tier 1: models returned by the provider API, including a retained stale
 * catalog after a transient refresh failure.
 * Tier 2: built-in/internal registry cards.
 */
[[nodiscard]] ResolvedModelCatalog resolve_model_catalog(
    std::span<const ModelInfo> provider_api_models,
    std::span<const ModelInfo> registry_models) noexcept;

/**
 * Resolve one model API-first. Partial API metadata enriches the registry card;
 * complete API sections replace their registry counterparts.
 */
[[nodiscard]] ResolvedModelMetadata resolve_model_metadata(
    std::string_view model_id,
    std::span<const ModelInfo> provider_api_models,
    std::shared_ptr<const ModelInfo> registry_model);

/**
 * Map a configured provider instance to the built-in registry family used as
 * its fallback tier. Custom provider names are preserved.
 */
[[nodiscard]] std::string model_registry_provider_key(
    std::string_view provider_name,
    config::ApiType api_type);

} // namespace core::llm
