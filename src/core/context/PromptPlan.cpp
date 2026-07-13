#include "PromptPlan.hpp"

#include <stdexcept>
#include <utility>

namespace core::context {

void PromptPlan::append(ContextLayer layer) {
    if (layer.content.empty()) return;
    if (!layers_.empty()
        && static_cast<int>(layer.stability)
            < static_cast<int>(layers_.back().stability)) {
        throw std::logic_error("prompt layers must be ordered from stable to dynamic");
    }
    layers_.push_back(std::move(layer));
}

std::string PromptPlan::render() const {
    std::string result;
    for (const auto& layer : layers_) result += layer.content;
    return result;
}

std::string PromptPlan::render_through(PromptStability stability) const {
    std::string result;
    for (const auto& layer : layers_) {
        if (static_cast<int>(layer.stability) > static_cast<int>(stability)) break;
        result += layer.content;
    }
    return result;
}

} // namespace core::context
