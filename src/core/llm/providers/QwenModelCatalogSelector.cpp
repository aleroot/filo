#include "QwenModelCatalogSelector.hpp"

#include "core/llm/QwenModelTraits.hpp"

#include <vector>
#include <tuple>

namespace core::llm::providers {
namespace {

class QwenModelCatalogSelector final : public IModelCatalogSelector {
public:
    [[nodiscard]] ModelCatalogSelection select(
        std::span<const ModelInfo> models) const override {
        const ModelInfo* selected = nullptr;
        std::tuple<std::vector<int>, int, int> selected_rank;

        for (const auto& model : models) {
            if (!core::llm::is_qwen_text_model(model.canonical_id)) continue;
            const auto generation = core::llm::qwen_model_generation(model.canonical_id);
            if (generation.empty()) continue;
            const auto& id = model.canonical_id;
            if (id.find("-vl") != std::string::npos || id.find("-omni") != std::string::npos) continue;
            const int quality = id.find("-max") != std::string::npos ? 3
                : id.find("-plus") != std::string::npos ? 2 : 1;
            const int stable = id.find("preview") == std::string::npos ? 1 : 0;
            const auto rank = std::make_tuple(generation, quality, stable);
            if (!selected || rank > selected_rank) {
                selected = &model;
                selected_rank = rank;
            }
        }

        if (!selected) {
            return {
                .error = "Neither Qwen Token Plan /models nor the internal "
                         "registry provided a Qwen text model; refresh /model "
                         "or select an explicit model.",
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
