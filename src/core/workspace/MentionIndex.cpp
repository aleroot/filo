#include "MentionIndex.hpp"

#include "core/utils/StringUtils.hpp"
#include "core/workspace/PathVisibility.hpp"

#include <algorithm>
#include <array>
#include <optional>
#include <ranges>
#include <system_error>

namespace core::workspace {

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

std::string_view basename_view(std::string_view path) {
    std::size_t end = path.size();
    if (end > 0 && path[end - 1] == '/') {
        --end;
    }
    const std::size_t slash = path.substr(0, end).find_last_of('/');
    const std::size_t start = slash == std::string_view::npos ? 0 : slash + 1;
    return path.substr(start, end - start);
}

int path_depth(std::string_view path) {
    return static_cast<int>(std::ranges::count(path, '/'));
}

bool should_skip_mention_directory(std::string_view name) {
    return std::ranges::contains(kSkippedDirectories, name);
}

} // namespace

std::vector<MentionSuggestion> build_mention_index(const std::filesystem::path& root,
                                                   std::stop_token stop_token) {
    std::vector<MentionSuggestion> entries;
    std::error_code ec;
    const AgentIgnorePathVisibilityFactory visibility_factory;
    const auto visibility = visibility_factory.for_root(root);

    for (std::filesystem::recursive_directory_iterator it(root, ec), end; it != end; it.increment(ec)) {
        if (stop_token.stop_requested()) {
            return {};
        }

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
        if (!visibility.is_visible(*it)) {
            if (is_dir && visibility.should_prune_directory(*it)) {
                it.disable_recursion_pending();
            }
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

        const std::string search_path = core::utils::str::to_lower_ascii_copy(rel_path);

        entries.push_back(MentionSuggestion{
            .display_path = rel_path,
            .insertion_text = rel_path,
            .search_path = search_path,
            .search_basename = core::utils::str::to_lower_ascii_copy(basename_view(search_path)),
            .depth = path_depth(rel_path),
            .is_directory = is_dir,
        });
    }

    if (stop_token.stop_requested()) {
        return {};
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

    const std::string lower_query = core::utils::str::to_lower_ascii_copy(query);
    std::vector<ScoredEntry> matches;
    matches.reserve(max_results);

    auto better = [](const ScoredEntry& lhs, const ScoredEntry& rhs) {
        if (lhs.score != rhs.score) return lhs.score < rhs.score;
        if (lhs.depth != rhs.depth) return lhs.depth < rhs.depth;
        if (lhs.length != rhs.length) return lhs.length < rhs.length;
        return lhs.suggestion->display_path < rhs.suggestion->display_path;
    };

    for (const auto& suggestion : index) {
        const std::optional<std::string> fallback_path = suggestion.search_path.empty()
            ? std::optional<std::string>{core::utils::str::to_lower_ascii_copy(suggestion.display_path)}
            : std::nullopt;
        const std::string_view lower_path = suggestion.search_path.empty()
            ? std::string_view{*fallback_path}
            : std::string_view{suggestion.search_path};

        const std::optional<std::string> fallback_base = suggestion.search_basename.empty()
            ? std::optional<std::string>{core::utils::str::to_lower_ascii_copy(basename_view(suggestion.display_path))}
            : std::nullopt;
        const std::string_view lower_base = suggestion.search_basename.empty()
            ? std::string_view{*fallback_base}
            : std::string_view{suggestion.search_basename};

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

        ScoredEntry entry{
            .score = score,
            .depth = suggestion.depth != 0 ? suggestion.depth : path_depth(suggestion.display_path),
            .length = suggestion.display_path.size(),
            .suggestion = &suggestion,
        };

        if (matches.size() < max_results) {
            matches.push_back(entry);
            continue;
        }

        auto worst = std::ranges::max_element(matches, better);
        if (worst != matches.end() && better(entry, *worst)) {
            *worst = entry;
        }
    }

    std::ranges::sort(matches, better);

    std::vector<MentionSuggestion> out;
    out.reserve(matches.size());
    for (std::size_t i = 0; i < matches.size(); ++i) {
        out.push_back(*matches[i].suggestion);
    }
    return out;
}

std::shared_ptr<const MentionIndexCache::Index> MentionIndexCache::index_for(
    const std::filesystem::path& root) {
    const std::string key = root.generic_string();
    const auto now = Clock::now();

    // The walk runs under the lock: a concurrent burst for the same root then
    // costs one walk instead of several, and roots are few in practice.
    const std::scoped_lock guard(mutex_);

    if (const auto it = entries_.find(key); it != entries_.end() && it->second.expires_at > now) {
        return it->second.index;
    }

    auto index = std::make_shared<const Index>(build_mention_index(root));
    entries_.insert_or_assign(key, Entry{.index = index, .expires_at = now + ttl_});
    return index;
}

} // namespace core::workspace
