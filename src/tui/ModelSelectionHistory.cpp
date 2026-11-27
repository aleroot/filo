#include "ModelSelectionHistory.hpp"

#include <algorithm>

namespace tui {

bool same_selection_target(const ModelSelectionSnapshot& a,
                           const ModelSelectionSnapshot& b) {
    if (a.mode != b.mode) {
        return false;
    }
    if (a.mode == ModelSelectionMode::Manual) {
        return a.manual_provider_name == b.manual_provider_name
            && a.manual_model_name == b.manual_model_name;
    }
    return a.router_policy == b.router_policy;
}

void record_model_switch(std::vector<ModelSelectionSnapshot>& recent,
                         const ModelSelectionSnapshot& before,
                         const ModelSelectionSnapshot& after) {
    if (same_selection_target(before, after)) {
        return;
    }
    std::erase_if(recent, [&](const ModelSelectionSnapshot& entry) {
        return same_selection_target(entry, before)
            || same_selection_target(entry, after);
    });
    recent.insert(recent.begin(), before);
    if (recent.size() > kMaxRecentModelSelections) {
        recent.resize(kMaxRecentModelSelections);
    }
}

std::optional<std::size_t> older_model_choice(
    std::span<const ModelSelectionSnapshot> recent,
    std::optional<std::size_t> position) {
    for (std::size_t i = position ? *position + 1 : 0; i < recent.size(); ++i) {
        if (is_model_selection(recent[i])) {
            return i;
        }
    }
    return std::nullopt;
}

std::optional<std::size_t> newer_model_choice(
    std::span<const ModelSelectionSnapshot> recent,
    std::size_t position) {
    for (std::size_t i = std::min(position, recent.size()); i > 0; --i) {
        if (is_model_selection(recent[i - 1])) {
            return i - 1;
        }
    }
    return std::nullopt;
}

} // namespace tui
