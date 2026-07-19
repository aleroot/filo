#pragma once

#include "ModelRegistry.hpp"

#include <span>
#include <string>

namespace core::llm {

struct ModelCatalogSelection {
    std::string model;
    std::string error;

    [[nodiscard]] bool ok() const noexcept {
        return !model.empty() && error.empty();
    }
};

/**
 * Provider-owned policy for choosing an implicit default from a live catalog.
 *
 * HttpLLMProvider owns catalog transport and lifecycle only. Ranking, filtering,
 * and provider-specific diagnostics are supplied through this interface.
 */
class IModelCatalogSelector {
public:
    virtual ~IModelCatalogSelector() = default;

    [[nodiscard]] virtual ModelCatalogSelection select(
        std::span<const ModelInfo> models) const = 0;
};

} // namespace core::llm
