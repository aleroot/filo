#pragma once

#include "ModelCatalogJson.hpp"

namespace core::llm::catalog {

/**
 * Decode the optional enriched metadata used by OpenAI-compatible catalogs.
 *
 * The standard OpenAI data[] response is intentionally sparse. Compatible
 * services may add any of these fields without requiring a new provider type.
 */
[[nodiscard]] ModelCapabilities compatible_advertised_capabilities(
    std::string_view model_id,
    const JsonObjectView& model);

} // namespace core::llm::catalog
