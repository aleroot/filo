#include "KimiSessionStore.hpp"
#include "core/utils/InterprocessFile.hpp"
#include "core/utils/Uuid.hpp"

#include <cctype>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace core::auth {
namespace {
namespace fs = std::filesystem;

fs::path default_directory() {
    // Retain Kimi's existing default-location rules, including legacy callers
    // that supply an empty XDG_CONFIG_HOME or run without HOME.
    if (const char* xdg = std::getenv("XDG_CONFIG_HOME")) return fs::path(xdg) / "filo";
    if (const char* home = std::getenv("HOME")) return fs::path(home) / ".config/filo";
    return fs::path("/tmp") / "filo";
}

std::string read_line(const fs::path& path) {
    std::ifstream file(path);
    std::string value;
    if (!std::getline(file, value)) return {};
    return value;
}

std::string normalize_host(std::string_view host) {
    while (!host.empty() && (host.back() == '/' || host.back() == ' ' || host.back() == '\n'
                            || host.back() == '\r' || host.back() == '\t')) host.remove_suffix(1);
    while (!host.empty() && (host.front() == ' ' || host.front() == '\t'
                            || host.front() == '\n' || host.front() == '\r')) host.remove_prefix(1);
    return std::string(host);
}

bool valid_device_id(std::string_view id) {
    const bool uuid = id.size() == 36;
    if (!uuid && id.size() != 32) return false;
    for (std::size_t i = 0; i < id.size(); ++i) {
        if (uuid && (i == 8 || i == 13 || i == 18 || i == 23)) {
            if (id[i] != '-') return false;
        } else if (!std::isxdigit(static_cast<unsigned char>(id[i]))) return false;
    }
    return true;
}

void write_private(const fs::path& path, std::string_view value) {
    if (fs::is_symlink(fs::symlink_status(path))) {
        throw std::runtime_error("symlink in Kimi session state: " + path.string());
    }
    std::string error;
    if (!core::utils::atomic_write_file(path, value, &error, core::utils::kPrivateFilePermissions)) {
        throw std::runtime_error(error);
    }
}
} // namespace

KimiSessionStore::KimiSessionStore(std::string directory)
    : directory_(directory.empty() ? default_directory() : fs::path(std::move(directory))),
      tokens_(directory_.string()) {}

std::optional<OAuthToken> KimiSessionStore::load(std::string_view provider_id) {
    auto token = tokens_.load(provider_id);
    if (token && provider_id == "kimi") {
        // Older tokens can omit this metadata. Use this session's companions
        // so request headers never fall back to another configuration.
        if (token->device_id.empty()) {
            const auto id = read_line(directory_ / "kimi_device_id");
            if (valid_device_id(id)) token->device_id = id;
        }
        if (token->issuer.empty()) token->issuer = persisted_oauth_host();
    }
    return token;
}

void KimiSessionStore::save(std::string_view provider_id, const OAuthToken& token) {
    if (provider_id == "kimi") {
        const auto host = normalize_host(token.issuer);
        if (!host.empty() && host != persisted_oauth_host()) {
            write_private(directory_ / "kimi_oauth_host", host);
        }
    }
    tokens_.save(provider_id, token);
}

void KimiSessionStore::clear(std::string_view provider_id) { tokens_.clear(provider_id); }

std::unique_ptr<ITokenStoreLock> KimiSessionStore::acquire_refresh_lock(std::string_view provider_id) {
    return tokens_.acquire_refresh_lock(provider_id);
}

KimiSessionIdentity KimiSessionStore::load_identity() {
    const auto host = persisted_oauth_host();
    if (const auto token = load("kimi")) {
        // Retain the existing persisted-host preference. Token metadata also
        // supports older transfers that did not include a host companion.
        return {token->device_id.empty() ? device_id() : token->device_id,
                host.empty() ? normalize_host(token->issuer) : host};
    }
    return {device_id(), host};
}

std::string KimiSessionStore::device_id() {
    const auto path = directory_ / "kimi_device_id";
    if (auto id = read_line(path); valid_device_id(id)) return id;
    std::string error;
    const auto lock_path = core::utils::lock_path_for(path);
    if (fs::is_symlink(fs::symlink_status(lock_path))) {
        throw std::runtime_error("symlink in Kimi device identity lock: " + lock_path.string());
    }
    const auto lock = core::utils::InterprocessFileLock::acquire(lock_path, &error);
    if (!lock) throw std::runtime_error(error);
    if (auto id = read_line(path); valid_device_id(id)) return id;
    const auto id = core::utils::random_uuid_v4();
    write_private(path, id);
    return id;
}

std::string KimiSessionStore::persisted_oauth_host() const {
    return normalize_host(read_line(directory_ / "kimi_oauth_host"));
}

} // namespace core::auth
