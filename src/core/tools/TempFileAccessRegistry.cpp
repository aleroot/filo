#include "TempFileAccessRegistry.hpp"
#include "../workspace/SessionWorkspace.hpp"

#include <algorithm>

namespace core::tools {

namespace {

constexpr auto kGrantTtl = std::chrono::hours{6};
constexpr std::size_t kMaxGrantsPerSession = 256;
constexpr std::string_view kDefaultCliSession = "__filo_default_cli_session__";

[[nodiscard]] std::filesystem::path normalize(const std::filesystem::path& path) {
    return core::workspace::SessionWorkspace::normalize_path(path);
}

} // namespace

TempFileAccessRegistry& TempFileAccessRegistry::instance() {
    static TempFileAccessRegistry registry;
    return registry;
}

std::string TempFileAccessRegistry::session_key(std::string_view session_id) {
    return session_id.empty() ? std::string(kDefaultCliSession) : std::string(session_id);
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
