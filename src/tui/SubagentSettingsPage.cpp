#include "SubagentSettingsPage.hpp"

#include "SettingsInput.hpp"

#include <algorithm>
#include <format>
#include <utility>

namespace tui {

ConfigSubagentModelStore::ConfigSubagentModelStore(
    core::config::ConfigManager& config,
    std::function<std::filesystem::path()> workspace)
    : config_(config), workspace_(std::move(workspace)) {}

core::config::AppConfig ConfigSubagentModelStore::config() const {
    return config_.get_config();
}

std::expected<core::config::SubagentModelChoices, std::string>
ConfigSubagentModelStore::scoped(core::config::SettingsScope scope) const {
    return core::config::read_subagent_overlay(
        config_.get_subagent_overlay_path(scope, workspace_()));
}

bool ConfigSubagentModelStore::save(core::config::SettingsScope scope,
                                    std::string_view subagent,
                                    const std::optional<core::config::SubagentModelChoice>& choice,
                                    std::string* error) {
    return config_.persist_subagent_model(scope, subagent, choice, workspace_(), error);
}

SubagentSettingsPage::SubagentSettingsPage(SubagentModelStore& store,
                                           std::function<void()> on_saved)
    : store_(store), on_saved_(std::move(on_saved)) {}

void SubagentSettingsPage::open() {
    selected_ = 0;
    views_.clear();
}

const SubagentSettingsPage::ScopeView&
SubagentSettingsPage::view(core::config::SettingsScope scope) {
    if (const auto it = views_.find(scope); it != views_.end()) return it->second;
    auto scoped = store_.scoped(scope);
    auto& view = views_[scope];
    if (!scoped) {
        view.note = std::format("Could not read {} agent settings: {}",
                                settings_scope_label(scope), scoped.error());
    }
    view.rows = subagent_settings_rows(
        store_.config(), scoped.value_or(core::config::SubagentModelChoices{}));
    selected_ = std::clamp(selected_, 0, std::max(0, static_cast<int>(view.rows.size()) - 1));
    return view;
}

const std::vector<SubagentSettingsRow>&
SubagentSettingsPage::rows(core::config::SettingsScope scope) {
    return view(scope).rows;
}

const std::string& SubagentSettingsPage::note(core::config::SettingsScope scope) {
    return view(scope).note;
}

std::vector<SettingsPanelRow> SubagentSettingsPage::panel_rows(core::config::SettingsScope scope) {
    std::vector<SettingsPanelRow> panel;
    for (const auto& row : rows(scope)) {
        panel.push_back(SettingsPanelRow{
            .label = row.name,
            .value = row.value,
            .description = row.description,
            .inherited = row.inherited,
            .chooser = "model",
        });
    }
    return panel;
}

std::optional<SubagentSettingsPage::Action>
SubagentSettingsPage::handle(const ftxui::Event& event, core::config::SettingsScope scope) {
    using ftxui::Event;
    const auto& list = rows(scope);
    const int count = static_cast<int>(list.size());
    if (event == Event::Escape || event == Event::ArrowLeft) return Action{.kind = Action::Kind::Back};
    if (count == 0) return std::nullopt;
    const auto& row = list[static_cast<std::size_t>(selected_)];

    if (event == Event::ArrowUp || event == Event::ArrowDown) {
        selected_ = (selected_ + (event == Event::ArrowUp ? count - 1 : 1)) % count;
        return Action{.kind = Action::Kind::Notice};
    }
    if (event == Event::Return || event == Event::ArrowRight) {
        return Action{
            .kind = Action::Kind::Choose,
            .subagent = row.name,
            .current = effective_subagent_choice(store_.config(), row.name),
        };
    }
    if (event == Event::Backspace || event == Event::Delete) {
        if (row.inherited) {
            return Action{.kind = Action::Kind::Notice,
                          .message = "This scope is already inheriting the effective value."};
        }
        return Action{.kind = Action::Kind::Clear, .subagent = row.name};
    }
    for (int n = 1; n <= std::min(count, 9); ++n) {
        if (event == Event::Character(static_cast<char>('0' + n))) {
            selected_ = n - 1;
            return Action{.kind = Action::Kind::Notice};
        }
    }
    return std::nullopt;
}

std::string SubagentSettingsPage::save(
    core::config::SettingsScope scope,
    std::string_view subagent,
    const std::optional<core::config::SubagentModelChoice>& choice) {
    std::string error;
    if (!store_.save(scope, subagent, choice, &error)) {
        return std::format("Could not save {} setting: {}", settings_scope_label(scope), error);
    }
    views_.clear();
    if (on_saved_) on_saved_();
    return std::format("{} setting saved: {} · {} -> {}",
                       settings_scope_label(scope), kAgentsSettingsCategory, subagent,
                       choice ? describe_subagent_choice(*choice) : std::string("inherit"));
}

} // namespace tui
