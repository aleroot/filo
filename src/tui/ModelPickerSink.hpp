#pragma once

#include "ModelSelectionHistory.hpp"

#include <functional>
#include <string>

namespace tui {

/// Where a model chosen in the model picker goes when it is not the
/// conversation's own model. Without a sink the picker switches the
/// conversation, as /model does; with one it hands the pick back, so any
/// caller that needs a model (a subagent, a setting) reuses the same picker.
struct ModelPickerSink {
    /// Shown in the picker: what the choice is for.
    std::string purpose;
    /// What the target runs now; its row is marked and focused.
    ModelSelectionSnapshot current;
    /// `model` is empty for the provider's configured default.
    std::function<void(std::string provider, std::string model)> choose;
    /// The picker closed without a choice.
    std::function<void()> cancel;
};

} // namespace tui
