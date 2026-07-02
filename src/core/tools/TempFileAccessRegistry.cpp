#include "TempFileAccessRegistry.hpp"
#include "../workspace/SessionWorkspace.hpp"

#include <algorithm>
#include <system_error>
#include <utility>

namespace core::tools {

namespace {

constexpr auto kGrantTtl = std::chrono::hours{6};
constexpr std::size_t kMaxGrantsPerSession = 256;
constexpr std::string_view kDefaultCliSession = "__filo_default_cli_session__";

[[nodiscard]] std::filesystem::path normalize(const std::filesystem::path& path) {
    return core::workspace::SessionWorkspace::normalize_path(path);
}

[[nodiscard]] bool path_has_prefix(
    const std::filesystem::path& root,
    const std::filesystem::path& target)
{
    const auto normalized_root = root.lexically_normal();
    const auto normalized_target = target.lexically_normal();

    auto root_it = normalized_root.begin();
    auto target_it = normalized_target.begin();
    while (root_it != normalized_root.end() && target_it != normalized_target.end()) {
        if (*root_it != *target_it) return false;
        ++root_it;
        ++target_it;
    }
    return root_it == normalized_root.end();
}

void append_unique_path(
    std::vector<std::filesystem::path>& paths,
    const std::filesystem::path& path)
{
    if (path.empty()) return;
    if (std::ranges::find(paths, path) == paths.end()) {
        paths.push_back(path);
    }
}

void append_unique_prefix(std::vector<std::string>& prefixes, std::string prefix) {
    if (prefix.empty()) return;
    if (!prefix.ends_with('/')) {
        prefix.push_back('/');
    }
    if (std::ranges::find(prefixes, prefix) == prefixes.end()) {
        prefixes.push_back(std::move(prefix));
    }
}

[[nodiscard]] std::vector<std::filesystem::path> lexical_temp_roots() {
    std::vector<std::filesystem::path> roots;
    std::error_code ec;
    append_unique_path(roots, std::filesystem::temp_directory_path(ec).lexically_normal());
    append_unique_path(roots, std::filesystem::path("/tmp").lexically_normal());
    append_unique_path(roots, std::filesystem::path("/private/tmp").lexically_normal());
    append_unique_path(roots, std::filesystem::path("/var/tmp").lexically_normal());
    append_unique_path(roots, std::filesystem::path("/private/var/tmp").lexically_normal());
    return roots;
}

} // namespace

std::string TempFileAccessRegistry::session_key(std::string_view session_id) {
    return session_id.empty() ? std::string(kDefaultCliSession) : std::string(session_id);
}

std::vector<std::filesystem::path> TempFileAccessRegistry::temp_roots() {
    std::vector<std::filesystem::path> normalized;
    for (const auto& root : lexical_temp_roots()) {
        append_unique_path(normalized, normalize(root));
    }
    return normalized;
}

std::vector<std::string> TempFileAccessRegistry::temp_path_prefixes() {
    std::vector<std::string> prefixes;
    for (const auto& root : lexical_temp_roots()) {
        append_unique_prefix(prefixes, root.string());
    }
    for (const auto& root : temp_roots()) {
        append_unique_prefix(prefixes, root.string());
    }
    return prefixes;
}

bool TempFileAccessRegistry::is_lexically_temp_path(const std::filesystem::path& path) {
    if (path.empty() || !path.is_absolute()) return false;

    const auto normalized = path.lexically_normal();
    for (const auto& root : lexical_temp_roots()) {
        if (path_has_prefix(root, normalized)) {
            return true;
        }
    }
    return false;
}

bool TempFileAccessRegistry::is_temp_path(const std::filesystem::path& path) {
    if (path.empty() || !path.is_absolute()) return false;

    const auto normalized = normalize(path);
    for (const auto& root : TempFileAccessRegistry::temp_roots()) {
        if (path_has_prefix(root, normalized)) {
            return true;
        }
    }
    return false;
}

void TempFileAccessRegistry::grant_read(
    std::string_view session_id,
    const std::filesystem::path& path)
{
    const auto normalized = normalize(path);
    const auto now = std::chrono::steady_clock::now();
    const auto key = session_key(session_id);

    std::lock_guard lock(mutex_);
    prune_locked(key, now);
    auto& grants = grants_[key];

    if (auto it = std::ranges::find_if(grants, [&](const Grant& grant) {
            return grant.path == normalized;
        });
        it != grants.end()) {
        it->expires_at = now + kGrantTtl;
        return;
    }

    grants.push_back(Grant{
        .path = normalized,
        .expires_at = now + kGrantTtl,
    });

    if (grants.size() > kMaxGrantsPerSession) {
        const auto overflow = grants.size() - kMaxGrantsPerSession;
        grants.erase(grants.begin(), grants.begin() + static_cast<std::ptrdiff_t>(overflow));
    }
}

bool TempFileAccessRegistry::can_read(
    std::string_view session_id,
    const std::filesystem::path& path)
{
    const auto normalized = normalize(path);
    const auto now = std::chrono::steady_clock::now();
    const auto key = session_key(session_id);

    std::lock_guard lock(mutex_);
    prune_locked(key, now);

    const auto it = grants_.find(key);
    if (it == grants_.end()) return false;

    return std::ranges::any_of(it->second, [&](const Grant& grant) {
        return grant.path == normalized;
    });
}

void TempFileAccessRegistry::clear_session(std::string_view session_id) {
    std::lock_guard lock(mutex_);
    grants_.erase(session_key(session_id));
}

void TempFileAccessRegistry::prune_locked(
    const std::string& key,
    std::chrono::steady_clock::time_point now)
{
    auto it = grants_.find(key);
    if (it == grants_.end()) return;

    auto& grants = it->second;
    std::erase_if(grants, [&](const Grant& grant) {
        return grant.expires_at <= now;
    });

    if (grants.empty()) {
        grants_.erase(it);
    }
}

} // namespace core::tools
