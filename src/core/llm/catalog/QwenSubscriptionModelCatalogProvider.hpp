#pragma once

#include "../ModelCatalogProvider.hpp"

namespace core::llm {

/**
 * Model list of the Qwen Subscription gateway (`GET /models`).
 *
 * The gateway's order is the product order (Qwen Code lists it first and
 * signs in to the first entry), so it is preserved and duplicates are dropped.
 * Each entry may carry `input_modalities` and `context_window_tokens`; an
 * absent or invalid field falls back to the model-name rules, and an explicit
 * `["text"]` makes the model text-only even if its name suggests otherwise.
 */
class QwenSubscriptionModelCatalogProvider final : public ModelCatalogProvider {
public:
    explicit QwenSubscriptionModelCatalogProvider(std::string provider_name);
    [[nodiscard]] std::string_view provider_name() const noexcept override;
    [[nodiscard]] std::string model_list_path(std::string_view page_token = {}) const override;
    [[nodiscard]] ModelCatalogResult parse_models_response(std::string_view body) const override;

private:
    std::string provider_name_;
};

} // namespace core::llm
