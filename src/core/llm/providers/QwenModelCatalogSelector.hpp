#pragma once

#include "../ModelCatalogSelector.hpp"

#include <memory>

namespace core::llm::providers {

/** Selects the highest Qwen text-model generation from server catalog data. */
[[nodiscard]] std::shared_ptr<const IModelCatalogSelector>
make_qwen_model_catalog_selector();

} // namespace core::llm::providers
