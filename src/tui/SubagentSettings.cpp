#include "SubagentSettings.hpp"

#include "core/config/BuiltinSubagents.hpp"
#include "core/review/Reviewer.hpp"
#include "core/tools/read/ReadTypes.hpp"
#include "core/tools/read/ReaderProfile.hpp"

#include <algorithm>
#include <format>

namespace tui {
namespace {

[[nodiscard]] std::string extra_note(std::string_view name) {
    if (name == core::review::kReviewerProfile) {
        return "Its model also runs /review.";
    }
    if (name == core::tools::read::kReaderProfile) {
        return "Answers the read tool's questions; it needs a provider and a model.";
    }
    return {};
}

/// The reader does not fall back to the conversation: unset, it takes a
/// routing policy's fast model, or question answering is off.
[[nodiscard]] std::string describe_reader(const core::config::AppConfig& config) {
    const auto it = config.subagents.find(std::string(core::tools::read::kReaderProfile));
    const bool configured = it != config.subagents.end()
        && !it->second.provider.empty() && !it->second.model.empty();
    const auto resolved = core::tools::read::resolve_reader_profile(config);
    if (!resolved) return "Off";
    const std::string choice = describe_subagent_choice({resolved->provider, resolved->model});
    return configured ? choice : choice + " (routing fast model)";
}

} // namespace

core::config::SubagentModelChoice
effective_subagent_choice(const core::config::AppConfig& config, std::string_view name) {
    const auto it = config.subagents.find(std::string(name));
    if (it == config.subagents.end()) return {};
    return {it->second.provider, it->second.model};
}

std::string describe_subagent_choice(const core::config::SubagentModelChoice& choice) {
    if (choice.provider.empty() && choice.model.empty()) return "Conversation's model";
    if (choice.provider.empty()) return choice.model;
    if (choice.model.empty()) return std::format("{} · its default model", choice.provider);
    return std::format("{} · {}", choice.provider, choice.model);
}

ModelSelectionSnapshot
subagent_model_selection(const core::config::AppConfig& config,
                         const core::config::SubagentModelChoice& choice,
                         const ModelSelectionSnapshot& conversation) {
    if (choice.provider.empty() && choice.model.empty()) return conversation;
    ModelSelectionSnapshot selection{.mode = ModelSelectionMode::Manual};
    selection.manual_provider_name =
        choice.provider.empty() ? conversation.manual_provider_name : choice.provider;
    selection.manual_model_name = choice.model;
    if (selection.manual_model_name.empty()) {
        if (const auto it = config.providers.find(selection.manual_provider_name);
            it != config.providers.end()) {
            selection.manual_model_name = it->second.model;
        }
    }
    return selection;
}

std::vector<SubagentSettingsRow>
subagent_settings_rows(const core::config::AppConfig& config,
                       const core::config::SubagentModelChoices& scoped) {
    std::vector<std::string> names;
    for (const auto& builtin : core::config::builtin_subagents::kAll) {
        names.emplace_back(builtin.name);
    }
    names.emplace_back(core::tools::read::kReaderProfile);
    std::vector<std::string> configured;
    for (const auto& [name, profile] : config.subagents) {
        if (std::ranges::find(names, name) == names.end()) configured.push_back(name);
    }
    std::ranges::sort(configured);
    names.insert(names.end(), configured.begin(), configured.end());

    std::vector<SubagentSettingsRow> rows;
    rows.reserve(names.size());
    for (auto& name : names) {
        std::string description;
        if (const auto it = config.subagents.find(name); it != config.subagents.end()) {
            description = it->second.description;
        }
        if (auto note = extra_note(name); !note.empty()) {
            description += description.empty() ? note : " " + note;
        }
        rows.push_back(SubagentSettingsRow{
            .name = name,
            .value = name == core::tools::read::kReaderProfile
                ? describe_reader(config)
                : describe_subagent_choice(effective_subagent_choice(config, name)),
            .description = std::move(description),
            .inherited = !scoped.contains(name),
        });
    }
    return rows;
}

} // namespace tui
