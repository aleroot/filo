#pragma once

#include "../ModelCatalogProvider.hpp"

#include <string_view>

namespace core::llm::catalog {

/** Decode the model schema shared by Codex-derived subscription proxies. */
[[nodiscard]] ModelCatalogResult parse_codex_style_model_catalog(
    std::string_view body,
    std::string_view provider_name);

} // namespace core::llm::catalog
