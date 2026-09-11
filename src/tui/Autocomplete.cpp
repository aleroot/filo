#include "Autocomplete.hpp"
#include "core/utils/StringUtils.hpp"

#include <algorithm>
#include <ranges>

namespace tui {

namespace {

std::string join_aliases(const std::vector<std::string>& aliases) {
    if (aliases.empty()) {
        return {};
    }
    return std::ranges::fold_left(
        aliases | std::views::drop(1),
        aliases.front(),
        [](std::string acc, const std::string& s) {
            return std::move(acc) + ", " + s;
        });
}

} // namespace

std::vector<CommandSuggestion> search_command_index(
    const std::vector<core::commands::CommandDescriptor>& index,
    std::string_view token,
    std::size_t max_results) {
    if (index.empty() || max_results == 0) {
        return {};
    }

    std::string_view query = token;
    if (!query.empty() && query.front() == '/') {
        query.remove_prefix(1);
    }

    if (query.empty()) {
        std::vector<CommandSuggestion> out;
        const auto count = std::min(max_results, index.size());
        out.reserve(count);
        for (std::size_t i = 0; i < count; ++i) {
            out.push_back(CommandSuggestion{
                .display_name = index[i].name,
                .insertion_text = index[i].name,
                .description = index[i].description,
                .aliases_label = join_aliases(index[i].aliases),
                .accepts_arguments = index[i].accepts_arguments,
            });
        }
        return out;
    }

    struct ScoredEntry {
        int score = 0;
        std::size_t order = 0;
        const core::commands::CommandDescriptor* command = nullptr;
    };

    const std::string lower_query = core::utils::str::to_lower_ascii_copy(query);
    std::vector<ScoredEntry> matches;
    matches.reserve(index.size());

    for (std::size_t i = 0; i < index.size(); ++i) {
        const auto& command = index[i];
        std::string lower_name = core::utils::str::to_lower_ascii_copy(command.name);
        if (!lower_name.empty() && lower_name.front() == '/') {
            lower_name.erase(lower_name.begin());
        }

        int score = 5;
        if (lower_name.starts_with(lower_query)) {
            score = 0;
        } else if (lower_name.find(lower_query) != std::string::npos) {
            score = 2;
        }

        for (const auto& alias : command.aliases) {
            std::string lower_alias = core::utils::str::to_lower_ascii_copy(alias);
            if (!lower_alias.empty() && lower_alias.front() == '/') {
                lower_alias.erase(lower_alias.begin());
            }

            if (lower_alias.starts_with(lower_query)) {
                score = std::min(score, 1);
            } else if (lower_alias.find(lower_query) != std::string::npos) {
                score = std::min(score, 3);
            }
        }

        if (score == 5) {
            const std::string lower_description = core::utils::str::to_lower_ascii_copy(command.description);
            if (lower_description.find(lower_query) != std::string::npos) {
                score = 4;
            }
        }

        if (score == 5) {
            continue;
        }

        matches.push_back(ScoredEntry{
            .score = score,
            .order = i,
            .command = &command,
        });
    }

    std::ranges::sort(matches, [](const ScoredEntry& lhs, const ScoredEntry& rhs) {
        if (lhs.score != rhs.score) return lhs.score < rhs.score;
        if (lhs.order != rhs.order) return lhs.order < rhs.order;
        return lhs.command->name < rhs.command->name;
    });

    std::vector<CommandSuggestion> out;
    const auto count = std::min(max_results, matches.size());
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        const auto& command = *matches[i].command;
        out.push_back(CommandSuggestion{
            .display_name = command.name,
            .insertion_text = command.name,
            .description = command.description,
            .aliases_label = join_aliases(command.aliases),
            .accepts_arguments = command.accepts_arguments,
        });
    }
    return out;
}

} // namespace tui
