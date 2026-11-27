#pragma once

#include "ModelSelectionHistory.hpp"
#include "PromptComponents.hpp"

#include <cstddef>
#include <functional>
#include <span>
#include <string_view>
#include <vector>

namespace tui {

using ModelIdEquivalence =
    std::function<bool(std::string_view lhs, std::string_view rhs)>;

/// True when choosing `row` would switch to `selection`.
[[nodiscard]] bool row_selects(const ModelPickerRow& row,
                               const ModelSelectionSnapshot& selection,
                               const ModelIdEquivalence& equivalent);

/// Index of the row that switches to the manual `selection`. A model the
/// catalogue does not list (set by `/model <provider> <id>`, or dropped by
/// the endpoint since) gets a row prepended, so the picker can always land
/// on what the user is running or ran before.
[[nodiscard]] std::size_t focus_model_row(std::vector<ModelPickerRow>& rows,
                                          const ModelSelectionSnapshot& selection,
                                          const ModelIdEquivalence& equivalent);

/// Marks the row of the current selection as active and the row Ctrl+P
/// would land on first as previous.
void mark_model_rows(std::vector<ModelPickerRow>& rows,
                     const ModelSelectionSnapshot& current,
                     std::span<const ModelSelectionSnapshot> recent,
                     const ModelIdEquivalence& equivalent);

} // namespace tui
