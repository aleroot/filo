#pragma once

#include "ModelSelectionHistory.hpp"
#include "core/config/ConfigManager.hpp"

#include <string>
#include <string_view>
#include <vector>

namespace tui {

/// The /settings category that sets which model each subagent runs on.
inline constexpr std::string_view kAgentsSettingsCategory = "Agents";

/// One row of the Agents page.
struct SubagentSettingsRow {
    std::string name;
    /// "openai · gpt-6.1-terra", or what applies when nothing is set.
    std::string value;
    std::string description;
    /// Nothing is set in the scope being edited; the value comes from below.
    bool inherited = true;
};

/// The provider and model `name` resolves to in `config` (either may be
/// empty, meaning the conversation's own).
[[nodiscard]] core::config::SubagentModelChoice
effective_subagent_choice(const core::config::AppConfig& config, std::string_view name);

/// How a choice reads in the panel.
[[nodiscard]] std::string describe_subagent_choice(const core::config::SubagentModelChoice& choice);

/// The model `choice` runs on, as a picker selection: an unset provider or
/// model is the conversation's, an unset model with a provider is that
/// provider's configured one.
[[nodiscard]] ModelSelectionSnapshot
subagent_model_selection(const core::config::AppConfig& config,
                         const core::config::SubagentModelChoice& choice,
                         const ModelSelectionSnapshot& conversation);

/// Built-in profiles first (in their usual order), then the reader worker,
/// then any other configured profile by name. `scoped` is what the edited
/// scope's subagents.json sets.
[[nodiscard]] std::vector<SubagentSettingsRow>
subagent_settings_rows(const core::config::AppConfig& config,
                       const core::config::SubagentModelChoices& scoped);

} // namespace tui
