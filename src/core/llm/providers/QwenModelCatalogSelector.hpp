#pragma once

#include "../ModelCatalogSelector.hpp"

#include <memory>

namespace core::llm::providers {

/** Selects the highest Qwen text-model generation from server catalog data. */
[[nodiscard]] std::shared_ptr<const IModelCatalogSelector>
make_qwen_model_catalog_selector();

/**
 * Selects the first model the Qwen Subscription gateway lists. The gateway
 * orders its catalog by product priority, and Qwen Code signs in to its first
 * entry; nothing is invented when the list could not be loaded.
 */
[[nodiscard]] std::shared_ptr<const IModelCatalogSelector>
make_qwen_subscription_model_catalog_selector();

} // namespace core::llm::providers
