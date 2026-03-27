#include "Autocomplete.hpp"
#include "Constants.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <ranges>

namespace tui {

namespace {

constexpr std::array kSkippedDirectories = {
    std::string_view{".git"},
    std::string_view{"build"},
    std::string_view{".idea"},
    std::string_view{".vscode"},
    std::string_view{"node_modules"},
    std::string_view{".venv"},
    std::string_view{"venv"},
    std::string_view{"dist"},
    std::string_view{"target"},
};

std::string to_lower_copy(std::string_view text) {
    std::string out(text);
    std::ranges::transform(out, out.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return out;
}

std::string_view basename_view(std::string_view path) {
    std::size_t end = path.size();
    if (end > 0 && path[end - 1] == '/') {
        --end;
    }
    const std::size_t slash = path.substr(0, end).find_last_of('/');
    const std::size_t start = slash == std::string_view::npos ? 0 : slash + 1;
    return path.substr(start, end - start);
}

bool should_skip_mention_directory(std::string_view name) {
    return std::ranges::contains(kSkippedDirectories, name);
}

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

std::vector<MentionSuggestion> build_mention_index(const std::filesystem::path& root) {
    std::vector<MentionSuggestion> entries;
    std::error_code ec;

    for (std::filesystem::recursive_directory_iterator it(root, ec), end; it != end; it.increment(ec)) {
        if (ec) {
            ec.clear();
            continue;
        }

        const auto filename = it->path().filename().string();
        const bool is_dir = it->is_directory(ec);
        if (ec) {
            ec.clear();
            continue;
        }

        if (is_dir && should_skip_mention_directory(filename)) {
            it.disable_recursion_pending();
            continue;
        }

        const auto rel = std::filesystem::relative(it->path(), root, ec);
        if (ec) {
            ec.clear();
            continue;
        }

        std::string rel_path = rel.generic_string();
        if (rel_path.empty()) {
            continue;
        }

        if (is_dir) {
            rel_path += "/";
        }

        entries.push_back(MentionSuggestion{
            .display_path = rel_path,
            .insertion_text = rel_path,
            .is_directory = is_dir,
        });
    }

    std::ranges::sort(entries, {}, &MentionSuggestion::display_path);
    return entries;
}

std::vector<MentionSuggestion> search_mention_index(const std::vector<MentionSuggestion>& index,
                                                    std::string_view query,
                                                    std::size_t max_results) {
    if (index.empty() || max_results == 0) {
        return {};
    }

    if (query.empty()) {
        return std::vector<MentionSuggestion>(
            index.begin(),
            index.begin() + std::min(max_results, index.size()));
    }

    struct ScoredEntry {
        int score = 0;
        int depth = 0;
        std::size_t length = 0;
        const MentionSuggestion* suggestion = nullptr;
    };

    const std::string lower_query = to_lower_copy(query);
    std::vector<ScoredEntry> matches;
    matches.reserve(std::min<std::size_t>(index.size(), max_results * 8));

    for (const auto& suggestion : index) {
        const std::string lower_path = to_lower_copy(suggestion.display_path);
        const std::string lower_base = to_lower_copy(basename_view(suggestion.display_path));

        int score = 4;
        if (lower_path.starts_with(lower_query)) {
            score = 0;
        } else if (lower_base.starts_with(lower_query)) {
            score = 1;
        } else if (lower_path.find("/" + lower_query) != std::string::npos) {
            score = 2;
        } else if (lower_path.find(lower_query) != std::string::npos) {
            score = 3;
        } else {
            continue;
        }

        matches.push_back(ScoredEntry{
            .score = score,
            .depth = static_cast<int>(std::ranges::count(suggestion.display_path, '/')),
            .length = suggestion.display_path.size(),
            .suggestion = &suggestion,
        });
    }

    std::ranges::sort(matches, [](const ScoredEntry& lhs, const ScoredEntry& rhs) {
        if (lhs.score != rhs.score) return lhs.score < rhs.score;
        if (lhs.depth != rhs.depth) return lhs.depth < rhs.depth;
        if (lhs.length != rhs.length) return lhs.length < rhs.length;
        return lhs.suggestion->display_path < rhs.suggestion->display_path;
    });

    std::vector<MentionSuggestion> out;
    const auto count = std::min(max_results, matches.size());
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        out.push_back(*matches[i].suggestion);
    }
    return out;
}

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

    const std::string lower_query = to_lower_copy(query);
    std::vector<ScoredEntry> matches;
    matches.reserve(index.size());

    for (std::size_t i = 0; i < index.size(); ++i) {
        const auto& command = index[i];
        std::string lower_name = to_lower_copy(command.name);
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
            std::string lower_alias = to_lower_copy(alias);
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
            const std::string lower_description = to_lower_copy(command.description);
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
