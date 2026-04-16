#include "ToolPolicy.hpp"
#include "ToolNames.hpp"

#include "../config/ConfigManager.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../workspace/SessionWorkspace.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <format>
#include <ranges>
#include <utility>

namespace core::tools::policy {

namespace {

struct HttpUrlView {
    std::string_view original;
    std::string_view scheme;
    std::string_view host;
    std::string_view port;
    std::string_view path;
};

[[nodiscard]] std::string trim_copy(std::string_view value) {
    const auto start = value.find_first_not_of(" \t\r\n");
    if (start == std::string_view::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return std::string(value.substr(start, end - start + 1));
}

void apply_overlay(core::config::ToolPolicyConfig& target,
                   const core::config::ToolPolicyConfig& overlay) {
    if (overlay.allowed_paths.has_value()) {
        target.allowed_paths = overlay.allowed_paths;
    }
    if (overlay.denied_paths.has_value()) {
        target.denied_paths = overlay.denied_paths;
    }
    if (overlay.allowed_commands.has_value()) {
        target.allowed_commands = overlay.allowed_commands;
    }
    if (overlay.trusted_urls.has_value()) {
        target.trusted_urls = overlay.trusted_urls;
    }
}

[[nodiscard]] core::config::ToolPolicyConfig merged_policy_for(std::string_view tool_name) {
    const auto& config = core::config::ConfigManager::get_instance().get_config();
    const auto canonical_name = canonical_tool_name(tool_name);

    core::config::ToolPolicyConfig merged;
    // Apply broad defaults first, then aliases in stable order, and finally
    // the exact canonical tool key so specificity always wins deterministically.
    if (const auto wildcard = config.tool_policies.find("*");
        wildcard != config.tool_policies.end()) {
        apply_overlay(merged, wildcard->second);
    }

    std::vector<std::pair<std::string_view, const core::config::ToolPolicyConfig*>> aliases;
    aliases.reserve(config.tool_policies.size());
    const core::config::ToolPolicyConfig* exact_match = nullptr;

    for (const auto& [key, policy] : config.tool_policies) {
        if (key == "*") {
            continue;
        }
        if (canonical_tool_name(key) != canonical_name) {
            continue;
        }
        if (key == canonical_name) {
            exact_match = &policy;
            continue;
        }
        aliases.emplace_back(key, &policy);
    }

    std::ranges::sort(aliases, {}, &std::pair<std::string_view,
                                              const core::config::ToolPolicyConfig*>::first);
    for (const auto& [_, policy] : aliases) {
        apply_overlay(merged, *policy);
    }

    if (exact_match != nullptr) {
        apply_overlay(merged, *exact_match);
    }

    return merged;
}

[[nodiscard]] bool is_subpath(const std::filesystem::path& root,
                              const std::filesystem::path& target) {
    const auto normalized_root =
        core::workspace::SessionWorkspace::normalize_path(root).lexically_normal();
    const auto normalized_target =
        core::workspace::SessionWorkspace::normalize_path(target).lexically_normal();

    auto root_it = normalized_root.begin();
    auto target_it = normalized_target.begin();
    while (root_it != normalized_root.end() && target_it != normalized_target.end()) {
        if (*root_it != *target_it) {
            return false;
        }
        ++root_it;
        ++target_it;
    }
    return root_it == normalized_root.end();
}

[[nodiscard]] bool matches_any_path(const std::vector<std::string>& patterns,
                                    const std::filesystem::path& resolved_path,
                                    const core::context::SessionContext& context) {
    return std::ranges::any_of(patterns, [&](const std::string& pattern) {
        if (pattern.empty()) {
            return false;
        }
        const auto resolved_pattern = context.resolve_path(pattern);
        return is_subpath(resolved_pattern, resolved_path);
    });
}

[[nodiscard]] bool starts_with_command_prefix(std::string_view command,
                                              std::string_view prefix) {
    const auto trimmed_command = trim_copy(command);
    const auto trimmed_prefix = trim_copy(prefix);
    if (trimmed_prefix.empty()) {
        return false;
    }
    if (!trimmed_command.starts_with(trimmed_prefix)) {
        return false;
    }
    if (trimmed_command.size() == trimmed_prefix.size()) {
        return true;
    }

    const char next = trimmed_command[trimmed_prefix.size()];
    return std::isspace(static_cast<unsigned char>(next));
}

[[nodiscard]] bool contains_shell_control_operators(std::string_view command) noexcept {
    bool in_single_quotes = false;
    bool in_double_quotes = false;
    bool escaped = false;

    for (std::size_t i = 0; i < command.size(); ++i) {
        const char ch = command[i];

        if (escaped) {
            escaped = false;
            continue;
        }

        if (in_single_quotes) {
            if (ch == '\'') {
                in_single_quotes = false;
            }
            continue;
        }

        if (ch == '\\') {
            escaped = true;
            continue;
        }

        if (in_double_quotes) {
            if (ch == '"') {
                in_double_quotes = false;
                continue;
            }
            if (ch == '`') {
                return true;
            }
            if (ch == '$' && i + 1 < command.size() && command[i + 1] == '(') {
                return true;
            }
            continue;
        }

        if (ch == '\'') {
            in_single_quotes = true;
            continue;
        }
        if (ch == '"') {
            in_double_quotes = true;
            continue;
        }
        if (ch == '$' && i + 1 < command.size() && command[i + 1] == '(') {
            return true;
        }

        switch (ch) {
            case ';':
            case '|':
            case '&':
            case '(':
            case ')':
            case '<':
            case '>':
            case '`':
            case '\n':
            case '\r':
                return true;
            default:
                break;
        }
    }

    return false;
}

[[nodiscard]] std::optional<HttpUrlView> parse_http_url(std::string_view value) noexcept {
    std::size_t scheme_len = 0;
    std::string_view scheme;
    if (core::utils::ascii::istarts_with(value, "http://")) {
        scheme = "http";
        scheme_len = 7;
    } else if (core::utils::ascii::istarts_with(value, "https://")) {
        scheme = "https";
        scheme_len = 8;
    } else {
        return std::nullopt;
    }

    const std::string_view remainder = value.substr(scheme_len);
    if (remainder.empty()) {
        return std::nullopt;
    }

    const std::size_t authority_end = remainder.find_first_of("/?#");
    const std::string_view authority = authority_end == std::string_view::npos
        ? remainder
        : remainder.substr(0, authority_end);
    if (authority.empty() || authority.find('@') != std::string_view::npos) {
        return std::nullopt;
    }

    std::string_view host;
    std::string_view port;
    if (authority.front() == '[') {
        const std::size_t closing_bracket = authority.find(']');
        if (closing_bracket == std::string_view::npos || closing_bracket == 1) {
            return std::nullopt;
        }

        host = authority.substr(1, closing_bracket - 1);
        if (closing_bracket + 1 < authority.size()) {
            if (authority[closing_bracket + 1] != ':'
                || closing_bracket + 2 >= authority.size()) {
                return std::nullopt;
            }
            port = authority.substr(closing_bracket + 2);
        }
    } else {
        const std::size_t colon = authority.find(':');
        host = colon == std::string_view::npos ? authority : authority.substr(0, colon);
        if (host.empty()) {
            return std::nullopt;
        }
        if (colon != std::string_view::npos) {
            if (colon + 1 >= authority.size()) {
                return std::nullopt;
            }
            port = authority.substr(colon + 1);
        }
    }

    const std::string_view path = authority_end == std::string_view::npos
        ? std::string_view("/")
        : remainder.substr(authority_end).substr(
              0,
              remainder.substr(authority_end).find_first_of("?#"));

    return HttpUrlView{
        .original = value,
        .scheme = scheme,
        .host = host,
        .port = port,
        .path = path.empty() ? std::string_view("/") : path,
    };
}

[[nodiscard]] std::string effective_port(const HttpUrlView& url) {
    if (!url.port.empty()) {
        return std::string(url.port);
    }
    return core::utils::ascii::iequals(url.scheme, "https") ? "443" : "80";
}

[[nodiscard]] bool path_prefix_matches(std::string_view trusted_path,
                                       std::string_view candidate_path) noexcept {
    if (trusted_path.empty() || trusted_path == "/") {
        return true;
    }
    if (candidate_path == trusted_path) {
        return true;
    }
    if (trusted_path.ends_with('/')) {
        return candidate_path.starts_with(trusted_path);
    }
    return candidate_path.size() > trusted_path.size()
        && candidate_path.starts_with(trusted_path)
        && candidate_path[trusted_path.size()] == '/';
}

[[nodiscard]] bool url_matches_trusted_pattern(std::string_view candidate_url,
                                               std::string_view trusted_url) {
    const auto candidate = parse_http_url(candidate_url);
    const auto trusted = parse_http_url(trusted_url);
    if (!candidate.has_value() || !trusted.has_value()) {
        return false;
    }

    return core::utils::ascii::iequals(candidate->scheme, trusted->scheme)
        && core::utils::ascii::iequals(candidate->host, trusted->host)
        && effective_port(*candidate) == effective_port(*trusted)
        && path_prefix_matches(trusted->path, candidate->path);
}

[[nodiscard]] bool is_url_terminator(char ch) noexcept {
    switch (ch) {
        case '"':
        case '\'':
        case '<':
        case '>':
        case '|':
        case '&':
        case ';':
        case '(':
        case ')':
        case '[':
        case ']':
        case '{':
        case '}':
            return true;
        default:
            return std::isspace(static_cast<unsigned char>(ch));
    }
}

[[nodiscard]] std::vector<std::string_view> extract_http_urls(std::string_view text) {
    std::vector<std::string_view> urls;
    std::size_t cursor = 0;
    while (cursor < text.size()) {
        std::size_t start = std::string_view::npos;
        for (std::size_t i = cursor; i < text.size(); ++i) {
            const std::string_view candidate = text.substr(i);
            if (core::utils::ascii::istarts_with(candidate, "http://")
                || core::utils::ascii::istarts_with(candidate, "https://")) {
                start = i;
                break;
            }
        }

        if (start == std::string_view::npos) {
            break;
        }

        std::size_t end = start;
        while (end < text.size() && !is_url_terminator(text[end])) {
            ++end;
        }

        urls.push_back(text.substr(start, end - start));
        cursor = end;
    }
    return urls;
}

} // namespace

std::string canonical_tool_name(std::string_view name) {
    std::string normalized = trim_copy(name);
    std::ranges::transform(normalized, normalized.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });

    static constexpr std::array kAliases = {
        std::pair{"shell", names::kRunTerminalCommand},
        std::pair{"terminal", names::kRunTerminalCommand},
        std::pair{"run_shell", names::kRunTerminalCommand},
        std::pair{"read", names::kReadFile},
        std::pair{"write", names::kWriteFile},
        std::pair{"patch", names::kApplyPatch},
        std::pair{"grep", names::kGrepSearch},
        std::pair{"search", names::kFileSearch},
        std::pair{"ls", names::kListDirectory},
        std::pair{"mkdir", names::kCreateDirectory},
    };

    for (const auto& [alias, canonical] : kAliases) {
        if (normalized == alias) {
            return std::string(canonical);
        }
    }
    return normalized;
}

bool is_tool_allowed(std::string_view tool_name,
                     const std::vector<std::string>& allowed_tools) {
    const auto canonical_name = canonical_tool_name(tool_name);
    return std::ranges::any_of(allowed_tools, [&](std::string_view entry) {
        return canonical_tool_name(entry) == canonical_name;
    });
}

std::optional<std::string> enforce_path_policy(
    std::string_view tool_name,
    const std::filesystem::path& resolved_path,
    const core::context::SessionContext& context) {
    const auto policy = merged_policy_for(tool_name);

    if (policy.denied_paths.has_value()
        && matches_any_path(*policy.denied_paths, resolved_path, context)) {
        return std::format(
            "path '{}' is blocked by denied_paths for tool '{}'",
            resolved_path.string(),
            canonical_tool_name(tool_name));
    }

    if (policy.allowed_paths.has_value()
        && !policy.allowed_paths->empty()
        && !matches_any_path(*policy.allowed_paths, resolved_path, context)) {
        return std::format(
            "path '{}' is outside allowed_paths for tool '{}'",
            resolved_path.string(),
            canonical_tool_name(tool_name));
    }

    return std::nullopt;
}

std::optional<std::string> enforce_command_policy(std::string_view tool_name,
                                                  std::string_view command) {
    const auto policy = merged_policy_for(tool_name);
    if (!policy.allowed_commands.has_value() || policy.allowed_commands->empty()) {
        return std::nullopt;
    }

    const bool allowed = std::ranges::any_of(
        *policy.allowed_commands,
        [&](const std::string& entry) {
            return starts_with_command_prefix(command, entry);
        });
    if (allowed) {
        if (contains_shell_control_operators(command)) {
            return std::format(
                "command '{}' contains shell control operators and is not allowed by tool '{}' (allowed_commands policy)",
                trim_copy(command),
                canonical_tool_name(tool_name));
        }
        return std::nullopt;
    }

    return std::format(
        "command '{}' is not allowed for tool '{}' (allowed_commands policy)",
        trim_copy(command),
        canonical_tool_name(tool_name));
}

std::optional<std::string> enforce_url_policy(std::string_view tool_name,
                                              std::string_view text) {
    const auto policy = merged_policy_for(tool_name);
    if (!policy.trusted_urls.has_value()) {
        return std::nullopt;
    }

    for (const auto url : extract_http_urls(text)) {
        const bool trusted = std::ranges::any_of(
            *policy.trusted_urls,
            [&](const std::string& trusted_url) {
                return url_matches_trusted_pattern(url, trusted_url);
            });
        if (!trusted) {
            return std::format(
                "url '{}' is not listed in trusted_urls for tool '{}'",
                std::string(url),
                canonical_tool_name(tool_name));
        }
    }

    return std::nullopt;
}

} // namespace core::tools::policy
