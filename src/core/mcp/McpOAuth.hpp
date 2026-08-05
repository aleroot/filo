#pragma once

#include "core/auth/OAuthToken.hpp"
#include "core/auth/ui/AuthUI.hpp"
#include "core/config/ConfigManager.hpp"

#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace core::mcp {

/** Default request timeouts when McpServerConfig::request_timeout_ms == 0. */
inline constexpr std::chrono::milliseconds kDefaultMcpHttpTimeout{std::chrono::seconds(120)};
inline constexpr std::chrono::milliseconds kDefaultMcpStdioTimeout{std::chrono::seconds(60)};

/**
 * Cached discovery + dynamic-client-registration state for one remote MCP server.
 * Persisted under ~/.config/filo/oauth_mcp_<server>_meta.json (owner 0600).
 */
struct McpOAuthMeta {
    std::string resource;                 // canonical resource URL (RFC 8707)
    std::string authorization_server;     // AS issuer / base
    std::string authorization_endpoint;
    std::string token_endpoint;
    std::string registration_endpoint;
    std::string revocation_endpoint;
    std::string client_id;
    std::string client_secret;            // only for confidential DCR clients
    std::vector<std::string> scopes;
    std::vector<std::string> redirect_uris; // registered loopback URIs
};

/** Token-store / meta-file id for a configured MCP server name. */
[[nodiscard]] std::string mcp_oauth_provider_id(std::string_view server_name);

/**
 * Resolve the effective JSON-RPC request timeout for an MCP server config.
 * @p transport_default is used when request_timeout_ms is unset/non-positive.
 */
[[nodiscard]] std::chrono::milliseconds
mcp_request_timeout(const core::config::McpServerConfig& config,
                    std::chrono::milliseconds transport_default);

/**
 * Discover protected-resource + authorization-server metadata for an MCP HTTP URL.
 *
 * Strategy (MCP auth + RFC9728/RFC8414):
 *  1. POST a lightweight initialize to capture WWW-Authenticate resource_metadata
 *  2. Fall back to /.well-known/oauth-protected-resource[/<path>]
 *  3. Resolve authorization_servers via AS metadata / OIDC discovery
 */
[[nodiscard]] McpOAuthMeta discover_mcp_oauth(const std::string& mcp_url);

[[nodiscard]] std::optional<McpOAuthMeta>
load_mcp_oauth_meta(std::string_view server_name, std::string_view config_dir);

void save_mcp_oauth_meta(std::string_view server_name,
                         std::string_view config_dir,
                         const McpOAuthMeta& meta);

void clear_mcp_oauth_meta(std::string_view server_name, std::string_view config_dir);

/** Options for interactive MCP OAuth login (keeps the free function stable). */
struct McpOAuthLoginOptions {
    std::vector<std::string> preferred_scopes;
    std::string client_id;       // optional pre-registered public/confidential client
    std::string client_secret;   // optional confidential client secret
    core::auth::ui::AuthUI* ui = nullptr;
};

/**
 * Interactive OAuth 2.1 authorization-code + PKCE login for a remote MCP server.
 * Performs discovery, dynamic client registration when needed, browser login,
 * and persists tokens under oauth_mcp_<server>.json (0600).
 */
[[nodiscard]] core::auth::OAuthToken login_mcp_oauth(
    const std::string& server_name,
    const std::string& mcp_url,
    const std::string& config_dir,
    McpOAuthLoginOptions options = {});

/** Convenience overload used by existing call sites. */
[[nodiscard]] core::auth::OAuthToken login_mcp_oauth(
    const std::string& server_name,
    const std::string& mcp_url,
    const std::string& config_dir,
    std::vector<std::string> preferred_scopes,
    core::auth::ui::AuthUI* ui);

/**
 * Refresh an access token using persisted metadata + refresh_token.
 * Caller may already hold the token-store refresh lock; this function does not
 * acquire it (single responsibility / avoid nested flock).
 */
[[nodiscard]] core::auth::OAuthToken refresh_mcp_oauth(
    const std::string& server_name,
    const std::string& config_dir,
    std::string_view refresh_token);

/**
 * Load a valid access token, refreshing under the inter-process lock when
 * expired. Returns nullopt when no session exists or refresh fails.
 */
[[nodiscard]] std::optional<core::auth::OAuthToken>
load_valid_mcp_oauth_token(const std::string& server_name,
                           const std::string& config_dir);

/**
 * Force a refresh even when the cached access token is still within the clock
 * skew window. Used after HTTP 401 from the resource server.
 */
[[nodiscard]] std::optional<core::auth::OAuthToken>
force_refresh_mcp_oauth_token(const std::string& server_name,
                              const std::string& config_dir);

void logout_mcp_oauth(const std::string& server_name, const std::string& config_dir);

[[nodiscard]] bool has_mcp_oauth_session(const std::string& server_name,
                                         const std::string& config_dir);

/**
 * Resolve a bearer access token for an MCP server config.
 * Priority:
 *  1. Explicit Authorization header (after ${ENV} expansion) if present
 *  2. Saved OAuth session (refreshing when expired), when auth is not "none"
 */
[[nodiscard]] std::optional<std::string> resolve_mcp_access_token(
    const core::config::McpServerConfig& config,
    const std::string& config_dir);

/**
 * Return a copy of @p config with Authorization injected when OAuth/header
 * credentials are available. All header values are env-expanded.
 */
[[nodiscard]] core::config::McpServerConfig
with_resolved_mcp_auth(const core::config::McpServerConfig& config,
                       const std::string& config_dir);

/** True when this HTTP server should use the interactive OAuth flow. */
[[nodiscard]] bool mcp_server_wants_oauth(const core::config::McpServerConfig& config);

/** Parse resource_metadata URL from a WWW-Authenticate header value. */
[[nodiscard]] std::optional<std::string>
parse_www_authenticate_resource_metadata(std::string_view header_value);

/**
 * Session-scoped HTTP authentication for one MCP server (SRP).
 *
 * Responsibilities:
 *  - Expand static headers (including explicit Authorization)
 *  - Supply a live OAuth bearer token (proactive refresh near expiry)
 *  - Force-refresh after resource-server 401 (mid-session recovery)
 *
 * Thread-safe: all public methods serialize on an internal mutex. Disk token
 * rotation also uses FileTokenStore::acquire_refresh_lock for multi-process
 * safety (same pattern as OAuthTokenManager).
 */
class McpHttpAuth {
public:
    McpHttpAuth(core::config::McpServerConfig config, std::string config_dir);
    ~McpHttpAuth() = default;

    McpHttpAuth(const McpHttpAuth&) = delete;
    McpHttpAuth& operator=(const McpHttpAuth&) = delete;

    /**
     * Headers to merge into every MCP HTTP request (Authorization included when
     * available). May refresh an expired OAuth access token.
     */
    [[nodiscard]] std::vector<std::pair<std::string, std::string>>
    request_headers() const;

    /**
     * After a resource-server 401, attempt a forced OAuth refresh.
     * @return true when Authorization likely changed and the caller should retry.
     */
    [[nodiscard]] bool refresh_after_unauthorized();

    [[nodiscard]] bool can_refresh() const noexcept;
    [[nodiscard]] bool uses_oauth() const noexcept { return oauth_mode_; }

private:
    [[nodiscard]] std::optional<std::string> resolve_authorization_locked(
        bool force_refresh) const;

    core::config::McpServerConfig config_;
    std::string config_dir_;
    std::vector<std::pair<std::string, std::string>> static_headers_;
    bool oauth_mode_ = false;
    mutable std::mutex mutex_;
    mutable std::string cached_authorization_; // full "Bearer …" value
};

[[nodiscard]] std::shared_ptr<McpHttpAuth>
make_mcp_http_auth(const core::config::McpServerConfig& config,
                   std::string_view config_dir);

} // namespace core::mcp
