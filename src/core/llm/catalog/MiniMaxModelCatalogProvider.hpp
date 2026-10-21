#pragma once
#include "../ModelCatalogProvider.hpp"
#include "../MiniMaxModelTraits.hpp"

namespace core::llm {
/// MiniMax's sparse Messages catalog enriched with endpoint-owned model traits.
class MiniMaxModelCatalogProvider final : public ModelCatalogProvider {
public:
    MiniMaxModelCatalogProvider(std::string provider, bool subscription,
                               int context_window = kMiniMaxContextWindow);
    std::string_view provider_name() const noexcept override;
    std::string model_list_path(std::string_view cursor = {}) const override;
    ModelCatalogResult parse_models_response(std::string_view body) const override;
private:
    AnthropicModelCatalogProvider messages_;
    std::string provider_;
    bool subscription_;
    int context_window_;
};
} // namespace core::llm
