#pragma once

#include "SubagentSettings.hpp"
#include "PromptComponents.hpp"

#include <ftxui/component/event.hpp>

#include <expected>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tui {

/// Where subagent model choices are read and saved.
class SubagentModelStore {
public:
    virtual ~SubagentModelStore() = default;
    /// The configuration in effect now.
    [[nodiscard]] virtual core::config::AppConfig config() const = 0;
    /// What `scope` itself sets.
    [[nodiscard]] virtual std::expected<core::config::SubagentModelChoices, std::string>
    scoped(core::config::SettingsScope scope) const = 0;
    /// Sets, or with nullopt clears, one subagent's choice in `scope`.
    virtual bool save(core::config::SettingsScope scope,
                      std::string_view subagent,
                      const std::optional<core::config::SubagentModelChoice>& choice,
                      std::string* error) = 0;
};

/// The store backed by the loaded configuration; `workspace` names the
/// directory workspace-scoped files belong to.
class ConfigSubagentModelStore final : public SubagentModelStore {
public:
    ConfigSubagentModelStore(core::config::ConfigManager& config,
                             std::function<std::filesystem::path()> workspace);
    [[nodiscard]] core::config::AppConfig config() const override;
    [[nodiscard]] std::expected<core::config::SubagentModelChoices, std::string>
    scoped(core::config::SettingsScope scope) const override;
    bool save(core::config::SettingsScope scope,
              std::string_view subagent,
              const std::optional<core::config::SubagentModelChoice>& choice,
              std::string* error) override;

private:
    core::config::ConfigManager& config_;
    std::function<std::filesystem::path()> workspace_;
};

/// The drilled-in Agents page of /settings: which subagent is selected and
/// what a key asks for. Picking the model itself is the model picker's job;
/// the host opens it on Choose and hands the pick back through save().
/// UI thread only.
class SubagentSettingsPage {
public:
    struct Action {
        enum class Kind {
            Choose,  ///< Open the model picker for `subagent`, focused on `current`.
            Clear,   ///< The scope's choice for `subagent` should be removed.
            Back,    ///< Return to the category overview.
            Notice,  ///< Nothing to do but tell the user `message`.
        };
        Kind kind;
        std::string subagent = {};
        core::config::SubagentModelChoice current = {};
        std::string message = {};
    };

    /// `on_saved` runs after every successful save, e.g. to reload the
    /// profiles of running agents.
    SubagentSettingsPage(SubagentModelStore& store, std::function<void()> on_saved);

    /// Entering the page: first row selected, files read again.
    void open();
    [[nodiscard]] const std::vector<SubagentSettingsRow>& rows(core::config::SettingsScope scope);
    /// rows() as the /settings panel draws them.
    [[nodiscard]] std::vector<SettingsPanelRow> panel_rows(core::config::SettingsScope scope);
    [[nodiscard]] int selected() const noexcept { return selected_; }
    /// Set when the scope's subagents.json could not be read.
    [[nodiscard]] const std::string& note(core::config::SettingsScope scope);

    /// Nullopt for keys the page does not use (the host handles those, Tab
    /// for instance).
    [[nodiscard]] std::optional<Action> handle(const ftxui::Event& event,
                                               core::config::SettingsScope scope);

    /// Saves (or clears) a choice and returns the status line to show.
    std::string save(core::config::SettingsScope scope,
                     std::string_view subagent,
                     const std::optional<core::config::SubagentModelChoice>& choice);

private:
    struct ScopeView {
        std::vector<SubagentSettingsRow> rows;
        /// Set when the scope's subagents.json could not be read.
        std::string note;
    };
    [[nodiscard]] const ScopeView& view(core::config::SettingsScope scope);

    SubagentModelStore& store_;
    std::function<void()> on_saved_;
    std::map<core::config::SettingsScope, ScopeView> views_;
    int selected_ = 0;
};

} // namespace tui
