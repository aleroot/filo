#pragma once

#include <cstddef>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace tui {

enum class ModelSelectionMode {
    Manual,
    Router,
    Auto,
};

struct ModelSelectionSnapshot {
    ModelSelectionMode mode = ModelSelectionMode::Manual;
    std::string manual_provider_name = {};
    std::string manual_model_name = {};
    std::string router_policy = {};

    bool operator==(const ModelSelectionSnapshot&) const = default;
};

/// True for a selection that names one model (manual mode); router and
/// auto selections pick a model per request and have no picker row.
[[nodiscard]] inline bool is_model_selection(const ModelSelectionSnapshot& selection) {
    return selection.mode == ModelSelectionMode::Manual;
}

/// How many earlier selections a thread remembers for `/model -` and the
/// model picker's Ctrl+P walk.
inline constexpr std::size_t kMaxRecentModelSelections = 8;

/// True when restoring either snapshot lands on the same target. A manual
/// selection carries the router policy along but restoring it never reads it,
/// so two manual snapshots differing only there are one entry, not two.
[[nodiscard]] bool same_selection_target(const ModelSelectionSnapshot& a,
                                         const ModelSelectionSnapshot& b);

/// Records a switch from `before` to `after` in a most-recent-first list.
/// The list never holds `after` (it is the current selection), holds each
/// target once, and is capped at kMaxRecentModelSelections.
void record_model_switch(std::vector<ModelSelectionSnapshot>& recent,
                         const ModelSelectionSnapshot& before,
                         const ModelSelectionSnapshot& after);

/// The model picker lists models, so its history walk visits manual entries
/// only; router and auto selections stay reachable through `/model -`.
/// `position` is the entry the picker shows, nullopt for the current model.
/// Returns the next older model entry, or nullopt when there is none.
[[nodiscard]] std::optional<std::size_t> older_model_choice(
    std::span<const ModelSelectionSnapshot> recent,
    std::optional<std::size_t> position);

/// The next newer model entry than `position`; nullopt means the walk is
/// back on the current model.
[[nodiscard]] std::optional<std::size_t> newer_model_choice(
    std::span<const ModelSelectionSnapshot> recent,
    std::size_t position);

} // namespace tui
