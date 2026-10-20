#include "AgentIgnore.hpp"

#include "SessionWorkspace.hpp"
#include "../context/SessionContext.hpp"
#include "../utils/JsonUtils.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <format>
#include <optional>
#include <ranges>
#include <span>
#include <string_view>

namespace core::workspace {

namespace {

[[nodiscard]] bool is_space(char ch) noexcept {
    return std::isspace(static_cast<unsigned char>(ch)) != 0;
}

[[nodiscard]] std::string strip_unescaped_trailing_space(std::string line) {
    while (!line.empty() && is_space(line.back())) {
        std::size_t slash_count = 0;
        for (std::size_t i = line.size() - 1; i > 0 && line[i - 1] == '\\'; --i) {
            ++slash_count;
        }
        if ((slash_count % 2) == 1) {
            break;
        }
        line.pop_back();
    }
    return line;
}

bool unescape_leading_marker(std::string& line) {
    if (line.size() >= 2
        && line.front() == '\\'
        && (line[1] == '#' || line[1] == '!')) {
        line.erase(line.begin());
        return true;
    }
    return false;
}

[[nodiscard]] std::vector<std::string> split_segments(std::string_view path) {
    std::vector<std::string> segments;
    std::size_t start = 0;
    while (start <= path.size()) {
        const std::size_t slash = path.find('/', start);
        const std::size_t end = slash == std::string_view::npos ? path.size() : slash;
        if (end > start) {
            segments.emplace_back(path.substr(start, end - start));
        }
        if (slash == std::string_view::npos) {
            break;
        }
        start = slash + 1;
    }
    return segments;
}

[[nodiscard]] std::optional<detail::AgentIgnoreRule> parse_rule(std::string line) {
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    line = strip_unescaped_trailing_space(std::move(line));
    if (line.empty()) {
        return std::nullopt;
    }

    const bool escaped_leading_marker = unescape_leading_marker(line);
    if (line.empty() || (!escaped_leading_marker && line.front() == '#')) {
        return std::nullopt;
    }

    detail::AgentIgnoreRule rule;
    if (line.front() == '!') {
        rule.negated = true;
        line.erase(line.begin());
        unescape_leading_marker(line);
    }
    if (line.empty()) {
        return std::nullopt;
    }

    const bool had_leading_slash = line.front() == '/';
    while (!line.empty() && line.front() == '/') {
        line.erase(line.begin());
    }
    if (line.empty()) {
        return std::nullopt;
    }

    if (line.back() == '/') {
        rule.directory_only = true;
        while (!line.empty() && line.back() == '/') {
            line.pop_back();
        }
    }
    if (line.empty()) {
        return std::nullopt;
    }

    std::replace(line.begin(), line.end(), '\\', '/');
    rule.pattern = std::move(line);
    rule.basename_only = !had_leading_slash
        && rule.pattern.find('/') == std::string::npos;
    rule.segments = split_segments(rule.pattern);
    return rule;
}

[[nodiscard]] bool wildcard_match_segment(std::string_view pattern,
                                          std::string_view value) {
    std::size_t p = 0;
    std::size_t v = 0;
    std::size_t star = std::string_view::npos;
    std::size_t star_value = 0;

    while (v < value.size()) {
        if (p < pattern.size() && pattern[p] == '\\' && p + 1 < pattern.size()) {
            ++p;
        }

        if (p < pattern.size()
            && (pattern[p] == '?' || pattern[p] == value[v])) {
            ++p;
            ++v;
            continue;
        }

        if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            star_value = v;
            continue;
        }

        if (star != std::string_view::npos) {
            p = star + 1;
            v = ++star_value;
            continue;
        }

        return false;
    }

    while (p < pattern.size()) {
        if (pattern[p] == '*') {
            ++p;
            continue;
        }
        if (pattern[p] == '\\' && p + 1 < pattern.size()) {
            ++p;
        }
        return false;
    }
    return true;
}

/// Segments of a path, viewing the string they were split from.
using Segments = std::span<const std::string_view>;

/// Splits once; empty segments (doubled or edge slashes) are dropped.
[[nodiscard]] std::vector<std::string_view> split_views(std::string_view path) {
    return path | std::views::split('/')
        | std::views::filter([](auto segment) { return !segment.empty(); })
        | std::views::transform([](auto segment) { return std::string_view(segment); })
        | std::ranges::to<std::vector>();
}

[[nodiscard]] bool match_segments(const std::vector<std::string>& pattern,
                                  std::size_t pattern_index,
                                  Segments path,
                                  std::size_t path_index) {
    if (pattern_index == pattern.size()) {
        return path_index == path.size();
    }

    if (pattern[pattern_index] == "**") {
        if (pattern_index + 1 == pattern.size()) {
            return true;
        }
        for (std::size_t next = path_index; next <= path.size(); ++next) {
            if (match_segments(pattern, pattern_index + 1, path, next)) {
                return true;
            }
        }
        return false;
    }

    if (path_index == path.size()) {
        return false;
    }
    return wildcard_match_segment(pattern[pattern_index], path[path_index])
        && match_segments(pattern, pattern_index + 1, path, path_index + 1);
}

[[nodiscard]] bool match_rule_against_candidate(
    const detail::AgentIgnoreRule& rule,
    Segments candidate_segments,
    bool candidate_is_directory) {
    if (rule.directory_only && !candidate_is_directory) {
        return false;
    }

    if (candidate_segments.empty()) {
        return false;
    }

    if (rule.basename_only) {
        return wildcard_match_segment(rule.pattern, candidate_segments.back());
    }
    return match_segments(rule.segments, 0, candidate_segments, 0);
}

[[nodiscard]] bool pattern_may_match_descendant_under(
    const std::vector<std::string>& pattern,
    std::size_t pattern_index,
    const std::vector<std::string>& directory,
    std::size_t directory_index) {
    if (directory_index == directory.size()) {
        if (pattern_index == pattern.size()) {
            return false;
        }
        return true;
    }

    if (pattern_index == pattern.size()) {
        return false;
    }

    if (pattern[pattern_index] == "**") {
        return pattern_may_match_descendant_under(
                   pattern,
                   pattern_index + 1,
                   directory,
                   directory_index)
            || pattern_may_match_descendant_under(
                   pattern,
                   pattern_index,
                   directory,
                   directory_index + 1);
    }

    return wildcard_match_segment(pattern[pattern_index], directory[directory_index])
        && pattern_may_match_descendant_under(
            pattern,
            pattern_index + 1,
            directory,
            directory_index + 1);
}

[[nodiscard]] bool negated_rule_may_reinclude_descendant(
    const detail::AgentIgnoreRule& rule,
    std::string_view directory) {
    if (!rule.negated) {
        return false;
    }
    if (rule.basename_only) {
        return true;
    }
    return pattern_may_match_descendant_under(
        rule.segments,
        0,
        split_segments(directory),
        0);
}

/// Whether `rule` matches the path or any of its ancestors: candidate n is
/// the first n segments, a directory unless it is the path itself.
[[nodiscard]] bool rule_matches(const detail::AgentIgnoreRule& rule,
                                Segments segments,
                                AgentIgnorePathKind kind) {
    for (std::size_t n = 1; n <= segments.size(); ++n) {
        const bool directory = n < segments.size() || kind == AgentIgnorePathKind::Directory;
        if (match_rule_against_candidate(rule, segments.first(n), directory)) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool path_has_prefix(std::filesystem::path root,
                                   std::filesystem::path target) {
    root = root.lexically_normal();
    target = target.lexically_normal();

    auto root_it = root.begin();
    auto target_it = target.begin();

    for (; root_it != root.end(); ++root_it, ++target_it) {
        if (target_it == target.end() || *root_it != *target_it) {
            return false;
        }
    }

    return true;
}

/// Both paths are normalized already, so this is lexical: resolving them
/// again would cost a realpath per checked path.
[[nodiscard]] std::string relative_generic_path(const std::filesystem::path& file,
                                                const std::filesystem::path& root) {
    return file.lexically_normal().lexically_relative(root.lexically_normal()).generic_string();
}

[[nodiscard]] std::vector<detail::AgentIgnoreRule> load_rules(
    const std::filesystem::path& ignore_file) {
    std::ifstream input(ignore_file);
    if (!input) {
        return {};
    }

    std::vector<detail::AgentIgnoreRule> rules;
    std::string line;
    while (std::getline(input, line)) {
        if (auto rule = parse_rule(std::move(line))) {
            rules.push_back(std::move(*rule));
        }
    }
    return rules;
}

} // namespace

AgentIgnoreMatcher::AgentIgnoreMatcher(std::vector<detail::AgentIgnoreRootPolicy> policies)
    : policies_(std::move(policies)) {}

AgentIgnoreMatcher AgentIgnoreMatcher::load_for_root(
    const std::filesystem::path& root) {
    if (root.empty()) {
        return AgentIgnoreMatcher({});
    }
    return load_for_roots({root});
}

AgentIgnoreMatcher AgentIgnoreMatcher::load_for_roots(
    const std::vector<std::filesystem::path>& roots) {
    return load_for_roots(roots, {".agentignore"});
}

AgentIgnoreMatcher AgentIgnoreMatcher::load_for_roots(
    const std::vector<std::filesystem::path>& roots,
    std::initializer_list<std::string_view> ignore_files) {
    std::vector<detail::AgentIgnoreRootPolicy> policies;
    policies.reserve(roots.size());

    std::vector<std::filesystem::path> seen;
    for (const auto& root : roots) {
        if (root.empty()) {
            continue;
        }
        const auto normalized_root = SessionWorkspace::normalize_path(root);
        if (std::ranges::find(seen, normalized_root) != seen.end()) {
            continue;
        }
        seen.push_back(normalized_root);

        std::vector<detail::AgentIgnoreRule> rules;
        for (auto name : ignore_files) {
            auto loaded = load_rules(normalized_root / name);
            rules.insert(rules.end(), std::make_move_iterator(loaded.begin()), std::make_move_iterator(loaded.end()));
        }
        if (!rules.empty()) {
            policies.push_back(detail::AgentIgnoreRootPolicy{
                .root = normalized_root,
                .rules = std::move(rules),
            });
        }
    }

    std::ranges::sort(policies, [](const detail::AgentIgnoreRootPolicy& lhs,
                                   const detail::AgentIgnoreRootPolicy& rhs) {
        return lhs.root.generic_string().size() > rhs.root.generic_string().size();
    });
    return AgentIgnoreMatcher(std::move(policies));
}

AgentIgnoreMatcher AgentIgnoreMatcher::load_for_context(
    const core::context::SessionContext& context) {
    std::vector<std::filesystem::path> roots;
    const auto& workspace = context.workspace_view();
    if (!workspace.primary().empty()) {
        roots.push_back(workspace.primary());
    }
    for (const auto& additional : workspace.additional()) {
        roots.push_back(additional);
    }
    return load_for_roots(roots);
}

bool AgentIgnoreMatcher::empty() const noexcept {
    return policies_.empty();
}

bool AgentIgnoreMatcher::is_ignored(const std::filesystem::path& path,
                                    AgentIgnorePathKind kind) const {
    if (policies_.empty() || path.empty()) {
        return false;
    }
    return is_ignored_normalized(SessionWorkspace::normalize_path(path), kind);
}

bool AgentIgnoreMatcher::is_ignored_normalized(const std::filesystem::path& normalized,
                                               AgentIgnorePathKind kind) const {
    if (policies_.empty() || normalized.empty()) {
        return false;
    }

    for (const auto& policy : policies_) {
        if (!path_has_prefix(policy.root, normalized)) {
            continue;
        }

        const std::string relative_path = relative_generic_path(normalized, policy.root);
        if (relative_path.empty() || relative_path == ".") {
            return false;
        }

        bool ignored = false;
        const auto segments = split_views(relative_path);
        for (const auto& rule : policy.rules) {
            if (rule_matches(rule, segments, kind)) {
                ignored = !rule.negated;
            }
        }
        return ignored;
    }

    return false;
}

bool AgentIgnoreMatcher::can_prune_directory(
    const std::filesystem::path& path) const {
    if (policies_.empty() || path.empty()) {
        return false;
    }
    return can_prune_normalized_directory(SessionWorkspace::normalize_path(path));
}

bool AgentIgnoreMatcher::can_prune_normalized_directory(
    const std::filesystem::path& normalized) const {
    if (policies_.empty() || normalized.empty()) {
        return false;
    }
    for (const auto& policy : policies_) {
        if (!path_has_prefix(policy.root, normalized)) {
            continue;
        }

        const std::string relative_path = relative_generic_path(normalized, policy.root);
        if (relative_path.empty() || relative_path == ".") {
            return false;
        }

        bool ignored = false;
        std::optional<std::size_t> last_match;
        const auto segments = split_views(relative_path);
        for (std::size_t i = 0; i < policy.rules.size(); ++i) {
            const auto& rule = policy.rules[i];
            if (rule_matches(rule, segments, AgentIgnorePathKind::Directory)) {
                ignored = !rule.negated;
                last_match = i;
            }
        }
        if (!ignored || !last_match.has_value()) {
            return false;
        }

        for (std::size_t i = *last_match + 1; i < policy.rules.size(); ++i) {
            if (negated_rule_may_reinclude_descendant(policy.rules[i], relative_path)) {
                return false;
            }
        }
        return true;
    }

    return false;
}

std::string format_agent_ignore_error(std::string_view path,
                                      std::string_view source) {
    return std::format(
        "Path '{}' is excluded by {}",
        core::utils::escape_json_string(path),
        core::utils::escape_json_string(source));
}

} // namespace core::workspace
