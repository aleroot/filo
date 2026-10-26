#pragma once

#include "../ModelCatalogProvider.hpp"

namespace core::llm {

/// Qwen's OpenAI-compatible catalog plus endpoint-local hosted model cards.
/// Live membership and explicit server metadata always win.
class QwenModelCatalogProvider final : public ModelCatalogProvider {
public:
    explicit QwenModelCatalogProvider(std::string provider_name);
    [[nodiscard]] std::string_view provider_name() const noexcept override;
    [[nodiscard]] std::string model_list_path(std::string_view page_token = {}) const override;
    [[nodiscard]] ModelCatalogResult parse_models_response(std::string_view body) const override;

private:
    OpenAICompatibleModelCatalogProvider compatible_;
};

} // namespace core::llm
