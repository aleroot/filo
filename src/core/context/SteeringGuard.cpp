#include "SteeringGuard.hpp"

#include "SessionContext.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <format>
#include <ranges>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace core::context {

namespace {

namespace fs = std::filesystem;

/// Bound on brace expansion, so a pathological token cannot blow up.
constexpr std::size_t kMaxBraceExpansions = 64;

/// Labels listed in the prompt notice before it degenerates into "…".
constexpr std::size_t kMaxNoticeLabels = 12;

[[nodiscard]] bool is_path_token_char(char ch) noexcept {
    if (std::isalnum(static_cast<unsigned char>(ch)) != 0) {
        return true;
    }
    switch (ch) {
        // Separators are deliberately absent: whitespace, quotes, brackets,
        // ':', '=', '<', '>', '|', '&', ';' and '(' all terminate a reference,
        // which is what lets one tokenizer serve shell commands and Python
        // source alike (`cat AGENTS.md`, `< AGENTS.md`, `open('AGENTS.md')`,
        // `FOO=AGENTS.md` all yield the same candidate).
        case '_': case '.': case '/': case '\\': case '-': case '+':
        case '~': case '$': case '%': case '@': case '*': case '?':
        case '{': case '}': case ',':
            return true;
        default:
            return false;
    }
}

[[nodiscard]] bool has_wildcard(std::string_view token) noexcept {
    return token.find_first_of("*?") != std::string_view::npos;
}

[[nodiscard]] bool contains_separator(std::string_view token) noexcept {
    return token.find_first_of("/\\") != std::string_view::npos;
}

[[nodiscard]] std::string_view last_path_segment(std::string_view token) noexcept {
    const auto separator = token.find_last_of("/\\");
    return separator == std::string_view::npos ? token : token.substr(separator + 1);
}

[[nodiscard]] bool ascii_iequals(char lhs, char rhs) noexcept {
    return core::utils::ascii::to_lower(lhs) == core::utils::ascii::to_lower(rhs);
}

/// `*` / `?` matching, case-insensitively: on the case-insensitive filesystems
/// Filo commonly runs on, `cat agents.md` reads `AGENTS.md`.
[[nodiscard]] bool wildcard_match(std::string_view pattern, std::string_view subject) noexcept {
    std::size_t p = 0;
    std::size_t s = 0;
    std::size_t star = std::string_view::npos;
    std::size_t resume = 0;

    while (s < subject.size()) {
        if (p < pattern.size()
            && (pattern[p] == '?' || ascii_iequals(pattern[p], subject[s]))) {
            ++p;
            ++s;
            continue;
        }
        if (p < pattern.size() && pattern[p] == '*') {
            star = p++;
            resume = s;
            continue;
        }
        if (star == std::string_view::npos) {
            return false;
        }
        p = star + 1;
        s = ++resume;
    }

    while (p < pattern.size() && pattern[p] == '*') {
        ++p;
    }
    return p == pattern.size();
}

/// Non-nested brace alternation, which is the only kind worth defending
/// against: `{AGENTS,CLAUDE}.md` names both files while spelling neither.
void expand_braces(std::string_view token, std::vector<std::string>& out) {
    if (out.size() >= kMaxBraceExpansions) {
        return;
    }

    const auto open = token.find('{');
    const auto close = open == std::string_view::npos ? open : token.find('}', open);
    if (open == std::string_view::npos || close == std::string_view::npos) {
        out.emplace_back(token);
        return;
    }

    const std::string prefix(token.substr(0, open));
    const std::string_view body = token.substr(open + 1, close - open - 1);
    const std::string suffix(token.substr(close + 1));

    if (body.find(',') == std::string_view::npos) {
        expand_braces(prefix + std::string(body) + suffix, out);
        return;
    }

    for (std::size_t start = 0; start <= body.size();) {
        const auto comma = body.find(',', start);
        const auto alternative = comma == std::string_view::npos
            ? body.substr(start)
            : body.substr(start, comma - start);
        expand_braces(prefix + std::string(alternative) + suffix, out);
        if (comma == std::string_view::npos) {
            break;
        }
        start = comma + 1;
    }
}

/// Cheap necessary condition, so the tokenizer's output stays small: a shell
/// command is mostly flags and identifiers that cannot name a steering file.
[[nodiscard]] bool may_reference_steering(std::string_view token) noexcept {
    if (core::utils::ascii::icontains(token, "steering")) {
        return true;
    }
    const auto names_it = [&](std::string_view name) {
        return core::utils::ascii::icontains(token, name);
    };
    if (std::ranges::any_of(hierarchical_steering_names(), names_it)
        || std::ranges::any_of(root_steering_names(), names_it)) {
        return true;
    }
    // Any markdown reference, wildcard or not: a symlink with an innocent name
    // (`NOTES.md -> AGENTS.md`) is the same file, and only resolving it can tell.
    return core::utils::ascii::icontains(token, ".md");
}

/// Path-like references inside a command or snippet, brace-expanded and
/// pre-filtered down to the ones that could name a steering file.
[[nodiscard]] std::vector<std::string> path_reference_candidates(std::string_view text) {
    std::vector<std::string> candidates;

    for (std::size_t index = 0; index < text.size();) {
        if (!is_path_token_char(text[index])) {
            ++index;
            continue;
        }
        std::size_t end = index;
        while (end < text.size() && is_path_token_char(text[end])) {
            ++end;
        }
        const auto token = text.substr(index, end - index);
        index = end;

        std::vector<std::string> alternatives;
        expand_braces(token, alternatives);
        for (auto& alternative : alternatives) {
            if (may_reference_steering(alternative)) {
                candidates.push_back(std::move(alternative));
            }
        }
    }

    return candidates;
}

[[nodiscard]] fs::path home_directory() {
    const char* const home = std::getenv("HOME");
    return home != nullptr ? fs::path(home) : fs::path{};
}

[[nodiscard]] bool contains_expansion(std::string_view token) noexcept {
    return token.find_first_of("$%`") != std::string_view::npos;
}

/// Resolves a concrete (wildcard-free, expansion-free) reference against the
/// directory the command runs in. nullopt when it cannot be resolved here.
[[nodiscard]] std::optional<fs::path> resolve_reference(std::string_view token,
                                                        const fs::path& base_dir) {
    if (token.empty() || contains_expansion(token)) {
        return std::nullopt;
    }
    if (token.front() == '~') {
        const auto home = home_directory();
        if (home.empty()) {
            return std::nullopt;
        }
        auto rest = token.substr(1);
        if (!rest.empty() && (rest.front() == '/' || rest.front() == '\\')) {
            rest.remove_prefix(1);
        }
        return (home / fs::path(rest)).lexically_normal();
    }

    const fs::path candidate(token);
    if (candidate.is_absolute()) {
        return candidate.lexically_normal();
    }
    if (base_dir.empty()) {
        return std::nullopt;
    }
    return (base_dir / candidate).lexically_normal();
}

} // namespace

bool SteeringGuard::may_block(const SteeringPolicy& policy) noexcept {
    switch (policy.mode) {
        case SteeringMode::None:
        case SteeringMode::CustomFile:
        case SteeringMode::CustomDir:
            return true;
        case SteeringMode::Default:
        case SteeringMode::Fallback:
            // Automatic prompt selection is not a read restriction. In
            // particular, additional workspace roots must not activate a
            // sandbox or disable embedded Python without an explicit opt-out.
            return !policy.disabled_sources.empty();
    }
    return true;
}

SteeringGuard SteeringGuard::for_roots(const std::vector<fs::path>& roots,
                                       const SteeringPolicy& policy) {
    SteeringGuard guard;
    if (!may_block(policy)) {
        return guard;
    }

    // The whole rule: every steering file that *exists*, minus what the policy
    // loads. The presence set rather than the loader's, because a file the loader
    // skips — AGENTS.md shadowed by a sibling AGENTS.override.md — is still a
    // project instruction file on disk and still has to be withheld.
    const auto present = enumerate_steering_files(roots);
    if (present.empty()) {
        return guard;
    }

    const auto selection = select_steering_files(roots, policy);
    std::vector<fs::path> loaded;
    std::vector<fs::path> loaded_parents;
    loaded.reserve(selection.files.size());
    for (const auto& file : selection.files) {
        if (!file.enabled) {
            continue;
        }
        const auto identity = steering_file_identity(file.path);
        loaded_parents.push_back(identity.parent_path());
        loaded.push_back(std::move(identity));
    }

    guard.policy_description_ = policy.format();
    for (const auto& file : present) {
        const auto identity = steering_file_identity(file.path);
        if (std::ranges::find(loaded, identity) != loaded.end()) {
            continue;
        }
        guard.blocked_.push_back(
            Entry{.identity = identity,
                  .parent = identity.parent_path(),
                  .label = file.label,
                  .directory = false});
    }

    // Hide a steering directory whose whole content is blocked, so listing it
    // cannot reveal what the agent is forbidden to read.
    std::vector<Entry> hidden_directories;
    for (const auto& entry : guard.blocked_) {
        if (!is_steering_directory(entry.parent)
            || std::ranges::find(loaded_parents, entry.parent) != loaded_parents.end()
            || std::ranges::any_of(hidden_directories,
                                   [&](const auto& hidden) { return hidden.identity == entry.parent; })) {
            continue;
        }
        hidden_directories.push_back(
            Entry{.identity = entry.parent,
                  .parent = entry.parent.parent_path(),
                  .label = fs::path(entry.label).parent_path().generic_string(),
                  .directory = true});
    }
    for (auto& hidden : hidden_directories) {
        guard.blocked_.push_back(std::move(hidden));
    }

    return guard;
}

SteeringGuard SteeringGuard::for_context(const SessionContext& context) {
    const auto& workspace = context.workspace_view();
    // Test the cheap condition before materializing the root list: the default
    // configuration blocks nothing, and this runs on every tool call.
    if (!may_block(context.steering_policy)) {
        return {};
    }
    // The same ordered roots ContextBuilder feeds the loader, so the guard can
    // never disagree with the prompt about what "the workspace" is.
    return for_roots(workspace.ordered_roots(), context.steering_policy);
}

const SteeringGuard::Entry* SteeringGuard::match_identity(const fs::path& identity) const {
    for (const auto& entry : blocked_) {
        if (entry.identity == identity) {
            return &entry;
        }
        // canonical() keeps the spelling it was handed, so on a case-insensitive
        // filesystem `agents.md` and `AGENTS.md` compare unequal as strings yet
        // are the same file. Ask the filesystem instead of guessing the platform.
        std::error_code ec;
        if (std::filesystem::equivalent(entry.identity, identity, ec) && !ec) {
            return &entry;
        }
    }
    return nullptr;
}

const SteeringGuard::Entry* SteeringGuard::match_filename(std::string_view name) const {
    if (name.empty() || has_wildcard(name)) {
        return nullptr;
    }
    const auto lower = core::utils::str::to_lower_ascii_copy(name);
    const auto found = std::ranges::find_if(blocked_, [&](const auto& entry) {
        return !entry.directory
            && core::utils::str::to_lower_ascii_copy(entry.identity.filename().string()) == lower;
    });
    return found == blocked_.end() ? nullptr : &*found;
}

const SteeringGuard::Entry* SteeringGuard::match_glob(std::string_view token,
                                                     const fs::path& base_dir) const {
    const auto segment = last_path_segment(token);
    // `**/AGENTS.md` names the file even though the directories it walks are
    // unknown here.
    if (const auto* named = match_filename(segment)) {
        return named;
    }

    const auto matches_name = [&](std::string_view pattern) {
        const auto found = std::ranges::find_if(blocked_, [&](const auto& entry) {
            return !entry.directory
                && wildcard_match(pattern, entry.identity.filename().string());
        });
        return found == blocked_.end() ? nullptr : &*found;
    };

    if (!contains_separator(token)) {
        // A bare pattern is relative to wherever the command runs, so match it
        // against names: `*.md` at the workspace root still reaches AGENTS.md.
        return matches_name(token);
    }

    const auto pattern = resolve_reference(token, base_dir);
    if (!pattern.has_value()) {
        return matches_name(segment);
    }

    const auto normalized = pattern->generic_string();
    const auto found = std::ranges::find_if(blocked_, [&](const auto& entry) {
        return wildcard_match(normalized, entry.identity.generic_string());
    });
    return found == blocked_.end() ? nullptr : &*found;
}

const SteeringGuard::Entry* SteeringGuard::match_reference(std::string_view token,
                                                          const fs::path& base_dir) const {
    if (has_wildcard(token)) {
        return match_glob(token, base_dir);
    }

    if (const auto resolved = resolve_reference(token, base_dir)) {
        if (const auto* hit = match_identity(steering_file_identity(*resolved))) {
            return hit;
        }
        std::error_code ec;
        if (std::filesystem::exists(*resolved, ec) && !ec) {
            // A real, different file that merely shares a steering name — the
            // secondary root's AGENTS.md while Default mode loads the primary's
            // — stays readable.
            return nullptr;
        }
    }

    // Unresolvable (`$PWD/AGENTS.md`) or pointing at nothing the guard can see:
    // the reference is ambiguous, so the bare name decides. This is also what
    // catches a relative reference from a working directory the guard cannot
    // know about, such as a persistent shell that has since `cd`'d away.
    return match_filename(last_path_segment(token));
}

std::optional<std::string> SteeringGuard::blocked_reason(const fs::path& resolved_path) const {
    if (blocked_.empty() || !is_steering_candidate(resolved_path)) {
        return std::nullopt;
    }
    const auto* hit = match_identity(steering_file_identity(resolved_path));
    if (hit == nullptr) {
        return std::nullopt;
    }
    return describe(*hit, /*referenced_by_command=*/false);
}

std::optional<std::string> SteeringGuard::blocked_text_reason(
    std::string_view text,
    const std::filesystem::path& base_dir) const {
    if (blocked_.empty() || text.empty()) {
        return std::nullopt;
    }
    // Blocked identities are symlink-resolved, so the directory relative
    // references resolve against has to be normalized the same way (on macOS a
    // /var/... temp path only compares equal as /private/var/...).
    const auto base = core::workspace::SessionWorkspace::normalize_path(base_dir);
    for (const auto& candidate : path_reference_candidates(text)) {
        if (const auto* hit = match_reference(candidate, base)) {
            return describe(*hit, /*referenced_by_command=*/true);
        }
    }
    return std::nullopt;
}

std::string SteeringGuard::describe(const Entry& entry, bool referenced_by_command) const {
    const auto name = entry.label.empty() ? entry.identity.generic_string() : entry.label;
    const auto* const kind = entry.directory ? "directory" : "file";
    const auto subject = referenced_by_command
        ? std::format("this command references '{}', a project steering {}", name, kind)
        : std::format("'{}' is a project steering {}", name, kind);
    return std::format(
        "Access denied: {}, and this session's steering policy is '{}', which does not load it. "
        "Steering that is excluded from the context cannot be read, searched, or modified through "
        "any tool; ask the user to re-enable it with /steering if it is needed.",
        subject,
        policy_description_);
}

std::vector<fs::path> SteeringGuard::blocked_paths() const {
    std::vector<fs::path> paths;
    paths.reserve(blocked_.size());
    for (const auto& entry : blocked_) {
        // Every entry, directories included: a hidden steering directory is
        // the only place its blocked content can be reached from, and the
        // consumers subtracting these paths expect a hierarchy, not a file.
        if (!entry.identity.empty()) {
            paths.push_back(entry.identity);
        }
    }
    return paths;
}

std::vector<std::string> SteeringGuard::blocked_labels() const {
    std::vector<std::string> labels;
    labels.reserve(blocked_.size());
    for (const auto& entry : blocked_) {
        if (entry.directory) {
            continue;
        }
        labels.push_back(entry.label.empty() ? entry.identity.generic_string() : entry.label);
    }
    return labels;
}

std::string SteeringGuard::prompt_notice() const {
    const auto labels = blocked_labels();
    if (labels.empty()) {
        return {};
    }

    std::string files;
    const auto shown = std::min(labels.size(), kMaxNoticeLabels);
    for (std::size_t index = 0; index < shown; ++index) {
        if (index != 0) {
            files += ", ";
        }
        files += labels[index];
    }
    if (labels.size() > shown) {
        files += std::format(", … ({} more)", labels.size() - shown);
    }

    return std::format(
        "Steering policy: {}.\n"
        "These project instruction files are NOT part of your context and are NOT accessible through "
        "any tool: {}.\n"
        "Do not open, cat, print, grep, or patch them, directly or through a command; those calls are "
        "rejected. Work from the user's request and the code itself.\n",
        policy_description_,
        files);
}

} // namespace core::context
