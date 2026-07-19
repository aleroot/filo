#include "QwenModelCatalogSelector.hpp"

#include "core/llm/QwenModelTraits.hpp"

#include <vector>

namespace core::llm::providers {
namespace {

class QwenModelCatalogSelector final : public IModelCatalogSelector {
public:
    [[nodiscard]] ModelCatalogSelection select(
        std::span<const ModelInfo> models) const override {
        const ModelInfo* selected = nullptr;
        std::vector<int> selected_generation;

        for (const auto& model : models) {
            if (!core::llm::is_qwen_text_model(model.canonical_id)) continue;
            const auto generation = core::llm::qwen_model_generation(model.canonical_id);
            if (generation.empty()) continue;
            if (!selected || generation > selected_generation) {
                selected = &model;
                selected_generation = generation;
            }
        }

        if (!selected) {
            return {
                .error = "Qwen Token Plan returned no Qwen text models from /models; "
                         "refresh /model or select an explicit model.",
            };
        }
        return {.model = selected->canonical_id};
    }
};

} // namespace

std::shared_ptr<const IModelCatalogSelector>
make_qwen_model_catalog_selector() {
    return std::make_shared<QwenModelCatalogSelector>();
}

} // namespace core::llm::providers
