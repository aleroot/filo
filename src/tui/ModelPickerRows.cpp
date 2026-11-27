#include "ModelPickerRows.hpp"

#include <algorithm>
#include <iterator>

namespace tui {

bool row_selects(const ModelPickerRow& row,
                 const ModelSelectionSnapshot& selection,
                 const ModelIdEquivalence& equivalent) {
    if (!is_model_selection(selection) || row.source_provider != selection.manual_provider_name) {
        return false;
    }
    return row.selector.empty()
        ? selection.manual_model_name.empty()
        : equivalent(selection.manual_model_name, row.selector);
}

std::size_t focus_model_row(std::vector<ModelPickerRow>& rows,
                            const ModelSelectionSnapshot& selection,
                            const ModelIdEquivalence& equivalent) {
    const auto it = std::ranges::find_if(rows, [&](const ModelPickerRow& row) {
        return row_selects(row, selection, equivalent);
    });
    if (it != rows.end()) {
        return static_cast<std::size_t>(std::distance(rows.begin(), it));
    }
    rows.insert(rows.begin(), ModelPickerRow{
        .id = selection.manual_model_name.empty()
            ? std::string("<provider default>")
            : selection.manual_model_name,
        .selector = selection.manual_model_name,
        .source_provider = selection.manual_provider_name,
        .description = "Not listed in this provider's catalogue.",
    });
    return 0;
}

void mark_model_rows(std::vector<ModelPickerRow>& rows,
                     const ModelSelectionSnapshot& current,
                     std::span<const ModelSelectionSnapshot> recent,
                     const ModelIdEquivalence& equivalent) {
    const auto previous = older_model_choice(recent, std::nullopt);
    for (auto& row : rows) {
        row.active = row_selects(row, current, equivalent);
        row.previous = !row.active
            && previous.has_value()
            && row_selects(row, recent[*previous], equivalent);
    }
}

} // namespace tui
