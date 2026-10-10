#pragma once

#include <cstddef>
#include <vector>

namespace core::llm {
struct Message;

/// Optional provider capability; transport implementations need not estimate
/// model context. Consumers discover it like ModelCatalogDiscoverable.
class ContextTokenEstimator {
public:
    virtual ~ContextTokenEstimator() = default;
    [[nodiscard]] virtual std::size_t estimate_context_tokens(
        const std::vector<Message>& messages) const noexcept = 0;
};
} // namespace core::llm
