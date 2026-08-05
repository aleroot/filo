#include "McpOAuth.hpp"

#include "core/auth/AuthBrowserLauncher.hpp"
#include "core/auth/FileTokenStore.hpp"
#include "core/auth/OAuthErrors.hpp"
#include "core/auth/OAuthLoopback.hpp"
#include "core/auth/OAuthPkce.hpp"
#include "core/auth/OAuthTokenEndpoint.hpp"
#include "core/logging/Logger.hpp"
#include "core/mcp/McpClientSession.hpp"
#include "core/utils/JsonWriter.hpp"
#include "core/utils/StringUtils.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace core::mcp {
namespace {

namespace fs = std::filesystem;

// Loopback port window registered with DCR so dynamic port selection still works.
constexpr int kLoopbackPortStart = 17890;
constexpr int kLoopbackPortEnd = 17920;

[[nodiscard]] std::string trim_copy(std::string_view s) {
    return core::utils::str::trim_ascii_copy(s);
}

[[nodiscard]] std::string to_lower(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

[[nodiscard]] std::string sanitize_server_name(std::string_view name) {
    std::string out;
    out.reserve(name.size());
    for (unsigned char c : name) {
        if (std::isalnum(c) || c == '_' || c == '-' || c == '.') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) out = "server";
    return out;
}

[[nodiscard]] std::string origin_of(const std::string& url) {
    const auto scheme_end = url.find("://");
    if (scheme_end == std::string::npos) return url;
    const auto path_start = url.find('/', scheme_end + 3);
    if (path_start == std::string::npos) return url;
    return url.substr(0, path_start);
}

[[nodiscard]] std::string path_of(const std::string& url) {
    const auto scheme_end = url.find("://");
    if (scheme_end == std::string::npos) return "/";
    const auto path_start = url.find('/', scheme_end + 3);
    if (path_start == std::string::npos) return "/";
    auto path = url.substr(path_start);
    const auto q = path.find_first_of("?#");
    if (q != std::string::npos) path.resize(q);
    if (path.empty()) return "/";
    return path;
}

[[nodiscard]] std::string join_url(std::string_view base, std::string_view path) {
    if (path.starts_with("http://") || path.starts_with("https://")) {
        return std::string(path);
    }
    std::string out(base);
    while (!out.empty() && out.back() == '/') out.pop_back();
    if (!path.empty() && path.front() != '/') out.push_back('/');
    out.append(path);
    return out;
}

[[nodiscard]] bool http_ok(long status) {
    return status >= 200 && status < 300;
}

[[nodiscard]] std::optional<std::string> header_value(const cpr::Header& headers,
                                                      std::string_view key) {
    const std::string want = to_lower(key);
    for (const auto& [name, value] : headers) {
        if (to_lower(name) == want) return value;
    }
    return std::nullopt;
}

[[nodiscard]] std::string meta_path(std::string_view server_name,
                                    std::string_view config_dir) {
    return std::string(config_dir) + "/oauth_" + mcp_oauth_provider_id(server_name)
         + "_meta.json";
}

/**
 * Scope selection (DRY for login + DCR):
 *  - preferred wins when non-empty
 *  - otherwise use server-advertised scopes excluding pure OIDC identity scopes
 *  - if nothing remains, omit scopes (empty) so the AS applies its defaults —
 *    never invent "read"/"write" (breaks many MCP ASes)
 */
[[nodiscard]] std::vector<std::string>
select_scopes(const std::vector<std::string>& preferred,
              const std::vector<std::string>& supported) {
    if (!preferred.empty()) return preferred;
    if (supported.empty()) return {};

    std::vector<std::string> chosen;
    auto take_if = [&](std::string_view name) {
        if (std::ranges::find(supported, std::string(name)) != supported.end()) {
            if (std::ranges::find(chosen, std::string(name)) == chosen.end()) {
                chosen.emplace_back(name);
            }
        }
    };
    take_if("read");
    take_if("write");
    take_if("mcp");
    take_if("offline_access");

    if (chosen.empty()) {
        for (const auto& s : supported) {
            if (s != "openid" && s != "email" && s != "profile") {
                chosen.push_back(s);
            }
        }
    }
    return chosen;
}

[[nodiscard]] McpOAuthMeta parse_protected_resource(std::string_view body,
                                                    const std::string& fallback_resource) {
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(body);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) {
        throw std::runtime_error("MCP OAuth: invalid protected resource metadata");
    }

    McpOAuthMeta meta;
    std::string_view resource;
    if (doc["resource"].get(resource) == simdjson::SUCCESS) {
        meta.resource = std::string(resource);
    } else {
        meta.resource = fallback_resource;
    }

    simdjson::dom::array servers;
    if (doc["authorization_servers"].get(servers) == simdjson::SUCCESS) {
        for (simdjson::dom::element s : servers) {
            std::string_view sv;
            if (s.get(sv) == simdjson::SUCCESS && !sv.empty()) {
                meta.authorization_server = std::string(sv);
                break;
            }
        }
    }
    if (meta.authorization_server.empty()) {
        throw std::runtime_error(
            "MCP OAuth: protected resource metadata missing authorization_servers");
    }

    simdjson::dom::array scopes;
    if (doc["scopes_supported"].get(scopes) == simdjson::SUCCESS) {
        for (simdjson::dom::element s : scopes) {
            std::string_view sv;
            if (s.get(sv) == simdjson::SUCCESS && !sv.empty()) {
                meta.scopes.emplace_back(sv);
            }
        }
    }
    return meta;
}

void apply_authorization_server_metadata(McpOAuthMeta& meta, std::string_view body) {
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(body);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) {
        throw std::runtime_error("MCP OAuth: invalid authorization server metadata");
    }

    std::string_view issuer;
    if (doc["issuer"].get(issuer) == simdjson::SUCCESS && !issuer.empty()) {
        meta.authorization_server = std::string(issuer);
    }

    auto read_url = [&](const char* key, std::string& out) {
        std::string_view sv;
        if (doc[key].get(sv) == simdjson::SUCCESS && !sv.empty()) {
            out = std::string(sv);
        }
    };
    read_url("authorization_endpoint", meta.authorization_endpoint);
    read_url("token_endpoint", meta.token_endpoint);
    read_url("registration_endpoint", meta.registration_endpoint);
    read_url("revocation_endpoint", meta.revocation_endpoint);

    if (meta.scopes.empty()) {
        simdjson::dom::array scopes;
        if (doc["scopes_supported"].get(scopes) == simdjson::SUCCESS) {
            for (simdjson::dom::element s : scopes) {
                std::string_view sv;
                if (s.get(sv) == simdjson::SUCCESS && !sv.empty()) {
                    meta.scopes.emplace_back(sv);
                }
            }
        }
    }

    if (meta.authorization_endpoint.empty() || meta.token_endpoint.empty()) {
        throw std::runtime_error(
            "MCP OAuth: authorization server metadata missing authorize/token endpoints");
    }
}

[[nodiscard]] std::optional<std::string> http_get_text(const std::string& url) {
    const auto r = cpr::Get(cpr::Url{url}, cpr::Timeout{15000});
    if (r.error.code != cpr::ErrorCode::OK || !http_ok(r.status_code)) {
        return std::nullopt;
    }
    return r.text;
}

void discover_authorization_server(McpOAuthMeta& meta) {
    const std::string as = meta.authorization_server;
    const std::vector<std::string> candidates = {
        join_url(as, "/.well-known/oauth-authorization-server"),
        join_url(origin_of(as),
                 std::string("/.well-known/oauth-authorization-server") + path_of(as)),
        join_url(as, "/.well-known/openid-configuration"),
        join_url(origin_of(as),
                 std::string("/.well-known/openid-configuration") + path_of(as)),
    };

    for (const auto& url : candidates) {
        if (const auto body = http_get_text(url)) {
            apply_authorization_server_metadata(meta, *body);
            return;
        }
    }

    meta.authorization_endpoint = join_url(origin_of(as), "/authorize");
    meta.token_endpoint = join_url(origin_of(as), "/token");
    meta.registration_endpoint = join_url(origin_of(as), "/register");
}

[[nodiscard]] std::optional<std::string>
probe_resource_metadata_from_401(const std::string& mcp_url) {
    const std::string body =
        R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"filo","version":"0"}}})";
    const auto r = cpr::Post(
        cpr::Url{mcp_url},
        cpr::Header{{"Content-Type", "application/json"},
                    {"Accept", "application/json, text/event-stream"}},
        cpr::Body{body},
        cpr::Timeout{15000});
    if (r.error.code != cpr::ErrorCode::OK) return std::nullopt;
    if (const auto www = header_value(r.header, "WWW-Authenticate")) {
        return parse_www_authenticate_resource_metadata(*www);
    }
    return std::nullopt;
}

[[nodiscard]] std::vector<std::string>
loopback_redirect_uris(std::string_view path = "/callback") {
    std::vector<std::string> uris;
    uris.reserve(static_cast<std::size_t>(kLoopbackPortEnd - kLoopbackPortStart + 1));
    for (int port = kLoopbackPortStart; port <= kLoopbackPortEnd; ++port) {
        uris.push_back(
            std::format("http://127.0.0.1:{}{}", port, path));
    }
    return uris;
}

struct RegisteredClient {
    std::string client_id;
    std::string client_secret;
    std::vector<std::string> redirect_uris;
};

[[nodiscard]] RegisteredClient register_oauth_client(
    const McpOAuthMeta& meta,
    const std::vector<std::string>& redirect_uris,
    const std::string& client_name) {
    if (meta.registration_endpoint.empty()) {
        throw std::runtime_error(
            "MCP OAuth: authorization server does not advertise dynamic client registration. "
            "Set oauth_client_id (and optional oauth_client_secret) or provide an "
            "Authorization header/API key instead.");
    }
    if (redirect_uris.empty()) {
        throw std::runtime_error("MCP OAuth: no redirect_uris for dynamic registration");
    }

    core::utils::JsonWriter writer(1024);
    {
        auto object = writer.object();
        writer.kv_str("client_name", client_name).comma();
        writer.key("redirect_uris");
        {
            auto arr = writer.array();
            for (std::size_t i = 0; i < redirect_uris.size(); ++i) {
                if (i > 0) writer.comma();
                writer.str(redirect_uris[i]);
            }
        }
        writer.comma().key("grant_types");
        {
            auto arr = writer.array();
            writer.str("authorization_code").comma().str("refresh_token");
        }
        writer.comma().key("response_types");
        {
            auto arr = writer.array();
            writer.str("code");
        }
        writer.comma().kv_str("token_endpoint_auth_method", "none");
        if (!meta.scopes.empty()) {
            writer.comma().key("scope").str(core::auth::join_oauth_scopes(meta.scopes));
        }
    }

    const auto r = cpr::Post(
        cpr::Url{meta.registration_endpoint},
        cpr::Header{{"Content-Type", "application/json"},
                    {"Accept", "application/json"}},
        cpr::Body{std::move(writer).take()},
        cpr::Timeout{20000});
    if (r.error.code != cpr::ErrorCode::OK) {
        throw std::runtime_error("MCP OAuth registration failed: " + r.error.message);
    }
    if (!http_ok(r.status_code)) {
        throw std::runtime_error(
            std::format("MCP OAuth registration HTTP {}: {}", r.status_code, r.text));
    }

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(r.text);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) {
        throw std::runtime_error("MCP OAuth: invalid registration response");
    }

    RegisteredClient client;
    std::string_view client_id;
    if (doc["client_id"].get(client_id) != simdjson::SUCCESS || client_id.empty()) {
        throw std::runtime_error("MCP OAuth: registration response missing client_id");
    }
    client.client_id = std::string(client_id);

    std::string_view client_secret;
    if (doc["client_secret"].get(client_secret) == simdjson::SUCCESS) {
        client.client_secret = std::string(client_secret);
    }

    client.redirect_uris = redirect_uris;
    simdjson::dom::array returned_uris;
    if (doc["redirect_uris"].get(returned_uris) == simdjson::SUCCESS) {
        client.redirect_uris.clear();
        for (simdjson::dom::element u : returned_uris) {
            std::string_view sv;
            if (u.get(sv) == simdjson::SUCCESS && !sv.empty()) {
                client.redirect_uris.emplace_back(sv);
            }
        }
        if (client.redirect_uris.empty()) {
            client.redirect_uris = redirect_uris;
        }
    }
    return client;
}

[[nodiscard]] std::string build_authorize_url(const McpOAuthMeta& meta,
                                              const std::string& client_id,
                                              const std::string& redirect_uri,
                                              const std::string& state,
                                              const std::string& challenge,
                                              const std::vector<std::string>& scopes) {
    using core::auth::oauth_pkce::url_encode;
    std::string url = meta.authorization_endpoint;
    url += (url.find('?') == std::string::npos) ? '?' : '&';
    url += "response_type=code";
    url += "&client_id=" + url_encode(client_id);
    url += "&redirect_uri=" + url_encode(redirect_uri);
    url += "&state=" + url_encode(state);
    url += "&code_challenge=" + url_encode(challenge);
    url += "&code_challenge_method=S256";
    if (!scopes.empty()) {
        url += "&scope=" + url_encode(core::auth::join_oauth_scopes(scopes));
    }
    if (!meta.resource.empty()) {
        url += "&resource=" + url_encode(meta.resource);
    }
    return url;
}

[[nodiscard]] core::auth::OAuthTokenEndpointRequest
token_request_from_meta(const McpOAuthMeta& meta) {
    core::auth::OAuthTokenEndpointRequest req;
    req.token_url = meta.token_endpoint;
    req.client_id = meta.client_id;
    req.client_secret = meta.client_secret;
    req.resource = meta.resource;
    req.issuer = meta.authorization_server;
    return req;
}

[[nodiscard]] bool has_authorization_header(
    const std::vector<std::pair<std::string, std::string>>& headers) {
    for (const auto& [name, _] : headers) {
        if (to_lower(name) == "authorization") return true;
    }
    return false;
}

[[nodiscard]] bool is_authorization_header_name(std::string_view name) {
    return to_lower(name) == "authorization";
}

[[nodiscard]] std::string bearer_authorization_value(std::string_view access_token) {
    return std::string("Bearer ") + std::string(access_token);
}

[[nodiscard]] bool token_exchange_looks_like_client_mismatch(std::string_view msg) {
    const auto lower = to_lower(msg);
    return lower.contains("invalid_client")
        || lower.contains("unauthorized_client")
        || lower.contains("redirect_uri")
        || lower.contains("invalid_request");
}

void set_owner_only_permissions(const fs::path& path) {
    std::error_code ec;
    fs::permissions(path, fs::perms::owner_read | fs::perms::owner_write,
                    fs::perm_options::replace, ec);
    if (ec) {
        core::logging::warn(
            "[MCP OAuth] could not set 0600 on {}: {}", path.string(), ec.message());
    }
}

[[nodiscard]] core::auth::OAuthToken
refresh_token_unlocked(const McpOAuthMeta& meta, std::string_view refresh_token) {
    if (meta.token_endpoint.empty() || meta.client_id.empty()) {
        throw core::auth::ReauthenticationRequired(
            "mcp",
            "MCP OAuth metadata missing; run /mcp login again");
    }
    if (refresh_token.empty()) {
        throw core::auth::ReauthenticationRequired(
            "mcp",
            "No MCP OAuth refresh token; run /mcp login again");
    }

    try {
        core::auth::OAuthToken token =
            core::auth::refresh_access_token(token_request_from_meta(meta), refresh_token);
        if (token.scopes.empty()) token.scopes = meta.scopes;
        if (token.client_id.empty()) token.client_id = meta.client_id;
        if (token.issuer.empty()) token.issuer = meta.authorization_server;
        if (token.refresh_token.empty()) {
            token.refresh_token = std::string(refresh_token);
        }
        return token;
    } catch (const core::auth::OAuthRefreshRejected&) {
        throw;
    }
}

void inherit_token_context(core::auth::OAuthToken& refreshed,
                           const core::auth::OAuthToken& previous) {
    if (refreshed.refresh_token.empty()) refreshed.refresh_token = previous.refresh_token;
    if (refreshed.scopes.empty()) refreshed.scopes = previous.scopes;
    if (refreshed.client_id.empty()) refreshed.client_id = previous.client_id;
    if (refreshed.issuer.empty()) refreshed.issuer = previous.issuer;
}

} // namespace

std::chrono::milliseconds
mcp_request_timeout(const core::config::McpServerConfig& config,
                    std::chrono::milliseconds transport_default) {
    if (config.request_timeout_ms > 0) {
        return std::chrono::milliseconds(config.request_timeout_ms);
    }
    if (transport_default.count() > 0) {
        return transport_default;
    }
    return config.transport == "http" ? kDefaultMcpHttpTimeout : kDefaultMcpStdioTimeout;
}

std::string mcp_oauth_provider_id(std::string_view server_name) {
    return "mcp_" + sanitize_server_name(server_name);
}

std::optional<std::string>
parse_www_authenticate_resource_metadata(std::string_view header_value) {
    const auto lower = to_lower(header_value);
    const auto key_pos = lower.find("resource_metadata=");
    if (key_pos == std::string::npos) return std::nullopt;
    std::size_t i = key_pos + std::string_view("resource_metadata=").size();
    if (i >= header_value.size()) return std::nullopt;
    if (header_value[i] == '"') {
        ++i;
        const auto end = header_value.find('"', i);
        if (end == std::string_view::npos) return std::nullopt;
        return std::string(header_value.substr(i, end - i));
    }
    const auto end = header_value.find_first_of(" \t,", i);
    return std::string(header_value.substr(
        i, end == std::string_view::npos ? std::string_view::npos : end - i));
}

McpOAuthMeta discover_mcp_oauth(const std::string& mcp_url) {
    if (mcp_url.empty()) {
        throw std::runtime_error("MCP OAuth: empty server URL");
    }

    std::optional<std::string> metadata_url = probe_resource_metadata_from_401(mcp_url);
    if (!metadata_url) {
        const std::string origin = origin_of(mcp_url);
        const std::string path = path_of(mcp_url);
        const std::vector<std::string> candidates = {
            join_url(origin, std::string("/.well-known/oauth-protected-resource") + path),
            join_url(origin, "/.well-known/oauth-protected-resource"),
        };
        for (const auto& url : candidates) {
            if (http_get_text(url)) {
                metadata_url = url;
                break;
            }
        }
    }
    if (!metadata_url) {
        throw std::runtime_error(
            "MCP OAuth: could not discover protected resource metadata for " + mcp_url
            + ". The server may require a static API key instead.");
    }

    const auto prm_body = http_get_text(*metadata_url);
    if (!prm_body) {
        throw std::runtime_error("MCP OAuth: failed to fetch " + *metadata_url);
    }

    McpOAuthMeta meta = parse_protected_resource(*prm_body, mcp_url);
    if (meta.resource.empty()) meta.resource = mcp_url;
    discover_authorization_server(meta);
    return meta;
}

std::optional<McpOAuthMeta> load_mcp_oauth_meta(std::string_view server_name,
                                                std::string_view config_dir) {
    const auto path = meta_path(server_name, config_dir);
    if (!fs::exists(path)) return std::nullopt;
    try {
        const simdjson::padded_string json = simdjson::padded_string::load(path);
        simdjson::dom::parser parser;
        simdjson::dom::element doc = parser.parse(json);
        McpOAuthMeta meta;
        std::string_view sv;
        if (doc["resource"].get(sv) == simdjson::SUCCESS) meta.resource = std::string(sv);
        if (doc["authorization_server"].get(sv) == simdjson::SUCCESS)
            meta.authorization_server = std::string(sv);
        if (doc["authorization_endpoint"].get(sv) == simdjson::SUCCESS)
            meta.authorization_endpoint = std::string(sv);
        if (doc["token_endpoint"].get(sv) == simdjson::SUCCESS)
            meta.token_endpoint = std::string(sv);
        if (doc["registration_endpoint"].get(sv) == simdjson::SUCCESS)
            meta.registration_endpoint = std::string(sv);
        if (doc["revocation_endpoint"].get(sv) == simdjson::SUCCESS)
            meta.revocation_endpoint = std::string(sv);
        if (doc["client_id"].get(sv) == simdjson::SUCCESS) meta.client_id = std::string(sv);
        if (doc["client_secret"].get(sv) == simdjson::SUCCESS)
            meta.client_secret = std::string(sv);
        simdjson::dom::array scopes;
        if (doc["scopes"].get(scopes) == simdjson::SUCCESS) {
            for (simdjson::dom::element s : scopes) {
                if (s.get(sv) == simdjson::SUCCESS) meta.scopes.emplace_back(sv);
            }
        }
        simdjson::dom::array redirects;
        if (doc["redirect_uris"].get(redirects) == simdjson::SUCCESS) {
            for (simdjson::dom::element u : redirects) {
                if (u.get(sv) == simdjson::SUCCESS) meta.redirect_uris.emplace_back(sv);
            }
        }
        if (meta.token_endpoint.empty() || meta.client_id.empty()) return std::nullopt;
        return meta;
    } catch (...) {
        return std::nullopt;
    }
}

void save_mcp_oauth_meta(std::string_view server_name,
                         std::string_view config_dir,
                         const McpOAuthMeta& meta) {
    fs::create_directories(std::string(config_dir));
    const auto path = meta_path(server_name, config_dir);

    core::utils::JsonWriter writer(1024);
    {
        auto object = writer.object();
        writer.kv_str("resource", meta.resource).comma();
        writer.kv_str("authorization_server", meta.authorization_server).comma();
        writer.kv_str("authorization_endpoint", meta.authorization_endpoint).comma();
        writer.kv_str("token_endpoint", meta.token_endpoint).comma();
        writer.kv_str("registration_endpoint", meta.registration_endpoint).comma();
        writer.kv_str("revocation_endpoint", meta.revocation_endpoint).comma();
        writer.kv_str("client_id", meta.client_id).comma();
        writer.kv_str("client_secret", meta.client_secret).comma();
        writer.key("scopes");
        {
            auto arr = writer.array();
            for (std::size_t i = 0; i < meta.scopes.size(); ++i) {
                if (i > 0) writer.comma();
                writer.str(meta.scopes[i]);
            }
        }
        writer.comma().key("redirect_uris");
        {
            auto arr = writer.array();
            for (std::size_t i = 0; i < meta.redirect_uris.size(); ++i) {
                if (i > 0) writer.comma();
                writer.str(meta.redirect_uris[i]);
            }
        }
    }

    const auto tmp = path + ".tmp";
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        if (!out) throw std::runtime_error("MCP OAuth: cannot write " + tmp);
        out << std::move(writer).take() << '\n';
        if (!out) throw std::runtime_error("MCP OAuth: failed writing " + tmp);
    }
    set_owner_only_permissions(tmp);
    std::error_code ec;
    fs::rename(tmp, path, ec);
    if (ec) {
        fs::remove(tmp, ec);
        throw std::runtime_error("MCP OAuth: cannot publish meta file " + path);
    }
    set_owner_only_permissions(path);
}

void clear_mcp_oauth_meta(std::string_view server_name, std::string_view config_dir) {
    std::error_code ec;
    fs::remove(meta_path(server_name, config_dir), ec);
}

core::auth::OAuthToken login_mcp_oauth(const std::string& server_name,
                                       const std::string& mcp_url,
                                       const std::string& config_dir,
                                       McpOAuthLoginOptions options) {
    McpOAuthMeta meta = discover_mcp_oauth(mcp_url);
    meta.scopes = select_scopes(options.preferred_scopes, meta.scopes);

    if (!options.client_id.empty()) {
        meta.client_id = options.client_id;
        meta.client_secret = options.client_secret;
    } else if (const auto cached = load_mcp_oauth_meta(server_name, config_dir)) {
        if (!cached->client_id.empty()
            && cached->authorization_server == meta.authorization_server) {
            meta.client_id = cached->client_id;
            meta.client_secret = cached->client_secret;
            meta.redirect_uris = cached->redirect_uris;
        }
    }

    const std::string state = core::auth::oauth_pkce::generate_correlation_token();
    const std::string verifier = core::auth::oauth_pkce::generate_code_verifier();
    const std::string challenge = core::auth::oauth_pkce::compute_code_challenge(verifier);

    core::auth::OAuthLoopbackOptions loop_opts;
    loop_opts.bind_host = "127.0.0.1";
    loop_opts.port_start = kLoopbackPortStart;
    loop_opts.port_end = kLoopbackPortEnd;
    loop_opts.callback_path = "/callback";
    loop_opts.extra_paths = {"/oauth/callback"};
    loop_opts.success_html =
        "<html><body style=\"font-family:system-ui;padding:2rem\">"
        "<h2>Filo MCP login successful</h2>"
        "<p>You can close this tab and return to Filo.</p>"
        "</body></html>";

    core::auth::OAuthLoopbackServer loopback(std::move(loop_opts));
    const std::string redirect_uri = loopback.redirect_uri();

    auto ensure_client = [&](bool force_reregister) {
        if (!force_reregister && !meta.client_id.empty()) {
            return;
        }
        const auto redirects = loopback_redirect_uris("/callback");
        const auto registered = register_oauth_client(
            meta,
            redirects,
            std::string("filo-mcp-") + sanitize_server_name(server_name));
        meta.client_id = registered.client_id;
        meta.client_secret = registered.client_secret;
        meta.redirect_uris = registered.redirect_uris;
    };

    ensure_client(/*force_reregister=*/meta.client_id.empty());

    // If the bound redirect is not among registered URIs, re-register the full window.
    if (!meta.redirect_uris.empty()
        && std::ranges::find(meta.redirect_uris, redirect_uri) == meta.redirect_uris.end()) {
        ensure_client(/*force_reregister=*/true);
    }

    const std::string auth_url = build_authorize_url(
        meta, meta.client_id, redirect_uri, state, challenge, meta.scopes);

    auto* ui = options.ui;
    if (ui) {
        ui->show_header("MCP OAuth Login — " + server_name);
        ui->show_instructions(
            "Authorize Filo to access this MCP server in your browser. "
            "This uses the standard MCP OAuth 2.1 + PKCE flow with dynamic client registration.");
        ui->show_url(auth_url, "If the browser does not open, visit:");
    } else {
        core::logging::info("[MCP OAuth] Opening browser for '{}'", server_name);
        core::logging::info("[MCP OAuth] {}", auth_url);
    }

    const bool suppress_browser =
        std::getenv("NO_BROWSER") != nullptr && std::string(std::getenv("NO_BROWSER")) != "0";
    if (!suppress_browser) {
        core::auth::open_browser(auth_url);
    }

    loopback.start();
    auto result = loopback.wait();
    if (!core::auth::complete_oauth_loopback_with_manual_fallback(result, state, ui)) {
        if (!result.error.empty()) {
            throw std::runtime_error("MCP OAuth authorization failed: " + result.error);
        }
        if (result.timed_out) {
            throw std::runtime_error(
                "MCP OAuth: timed out waiting for browser callback (5 minutes)");
        }
        throw std::runtime_error("MCP OAuth: authorization code missing");
    }

    auto exchange = [&]() -> core::auth::OAuthToken {
        auto token_req = token_request_from_meta(meta);
        token_req.redirect_uri = redirect_uri;
        try {
            return core::auth::exchange_authorization_code(
                token_req, result.code, verifier);
        } catch (const std::exception& e) {
            // Pre-registered clients cannot be healed via DCR.
            if (!options.client_id.empty()) throw;
            if (!token_exchange_looks_like_client_mismatch(e.what())) {
                throw;
            }
            // Authorization codes are single-use. Refresh DCR metadata so the
            // next interactive login uses a client the AS still recognizes.
            core::logging::warn(
                "[MCP OAuth] token exchange failed ({}), re-registering client for '{}'",
                e.what(),
                server_name);
            ensure_client(/*force_reregister=*/true);
            save_mcp_oauth_meta(server_name, config_dir, meta);
            throw std::runtime_error(
                std::string("MCP OAuth: client registration was stale (")
                + e.what()
                + "). Client metadata was refreshed — run /mcp login "
                + server_name + " again.");
        }
    };

    core::auth::OAuthToken token = exchange();
    if (token.scopes.empty()) token.scopes = meta.scopes;
    if (token.client_id.empty()) token.client_id = meta.client_id;
    if (token.issuer.empty()) token.issuer = meta.authorization_server;

    save_mcp_oauth_meta(server_name, config_dir, meta);
    core::auth::FileTokenStore store(config_dir);
    auto lock = store.acquire_refresh_lock(mcp_oauth_provider_id(server_name));
    store.save(mcp_oauth_provider_id(server_name), token);

    if (ui) {
        ui->show_success("MCP OAuth login successful for '" + server_name + "'.");
    } else {
        core::logging::info("[MCP OAuth] Login successful for '{}'", server_name);
    }
    return token;
}

core::auth::OAuthToken login_mcp_oauth(const std::string& server_name,
                                       const std::string& mcp_url,
                                       const std::string& config_dir,
                                       std::vector<std::string> preferred_scopes,
                                       core::auth::ui::AuthUI* ui) {
    return login_mcp_oauth(
        server_name,
        mcp_url,
        config_dir,
        McpOAuthLoginOptions{
            .preferred_scopes = std::move(preferred_scopes),
            .ui = ui,
        });
}

core::auth::OAuthToken refresh_mcp_oauth(const std::string& server_name,
                                         const std::string& config_dir,
                                         std::string_view refresh_token) {
    const auto meta = load_mcp_oauth_meta(server_name, config_dir);
    if (!meta) {
        throw core::auth::ReauthenticationRequired(
            mcp_oauth_provider_id(server_name),
            "MCP OAuth metadata missing; run /mcp login " + server_name);
    }

    try {
        core::auth::FileTokenStore store(config_dir);
        const auto provider_id = mcp_oauth_provider_id(server_name);
        auto lock = store.acquire_refresh_lock(provider_id);
        // Prefer a concurrent refresh that may already have landed on disk.
        if (auto current = store.load(provider_id);
            current && current->is_valid() && !current->access_token.empty()) {
            return *current;
        }
        core::auth::OAuthToken token = refresh_token_unlocked(*meta, refresh_token);
        if (auto current = store.load(provider_id)) {
            inherit_token_context(token, *current);
        }
        store.save(provider_id, token);
        return token;
    } catch (const core::auth::OAuthRefreshRejected&) {
        throw core::auth::OAuthRefreshRejected::by_provider(server_name);
    }
}

std::optional<core::auth::OAuthToken>
load_valid_mcp_oauth_token(const std::string& server_name,
                           const std::string& config_dir) {
    if (config_dir.empty()) return std::nullopt;

    core::auth::FileTokenStore store(config_dir);
    const auto provider_id = mcp_oauth_provider_id(server_name);
    auto lock = store.acquire_refresh_lock(provider_id);

    auto token = store.load(provider_id);
    if (!token) return std::nullopt;
    if (token->is_valid()) return token;

    if (!token->has_refresh_token()) return std::nullopt;

    const auto meta = load_mcp_oauth_meta(server_name, config_dir);
    if (!meta) return std::nullopt;

    try {
        core::auth::OAuthToken refreshed =
            refresh_token_unlocked(*meta, token->refresh_token);
        inherit_token_context(refreshed, *token);
        store.save(provider_id, refreshed);
        return refreshed;
    } catch (const std::exception& e) {
        core::logging::warn("[MCP OAuth] proactive refresh for '{}': {}", server_name, e.what());
        return std::nullopt;
    }
}

std::optional<core::auth::OAuthToken>
force_refresh_mcp_oauth_token(const std::string& server_name,
                              const std::string& config_dir) {
    if (config_dir.empty()) return std::nullopt;

    core::auth::FileTokenStore store(config_dir);
    const auto provider_id = mcp_oauth_provider_id(server_name);
    auto lock = store.acquire_refresh_lock(provider_id);

    // Another process may already have rotated the token while we waited.
    auto token = store.load(provider_id);
    if (!token) return std::nullopt;

    // If the on-disk token is already newer and valid, prefer it (no extra AS call).
    // Still force a refresh when the caller asked after a 401 — but only if the
    // token did not change under the lock (compare access token below at call site).
    // Here we always attempt refresh when a refresh_token exists so mid-session
    // 401 recovery actually rotates.
    if (!token->has_refresh_token()) return std::nullopt;

    const auto meta = load_mcp_oauth_meta(server_name, config_dir);
    if (!meta) return std::nullopt;

    try {
        core::auth::OAuthToken refreshed =
            refresh_token_unlocked(*meta, token->refresh_token);
        inherit_token_context(refreshed, *token);
        store.save(provider_id, refreshed);
        return refreshed;
    } catch (const core::auth::OAuthRefreshRejected& e) {
        core::logging::warn("[MCP OAuth] force refresh rejected for '{}': {}",
                            server_name,
                            e.what());
        return std::nullopt;
    } catch (const std::exception& e) {
        core::logging::warn("[MCP OAuth] force refresh for '{}': {}", server_name, e.what());
        return std::nullopt;
    }
}

void logout_mcp_oauth(const std::string& server_name, const std::string& config_dir) {
    core::auth::FileTokenStore store(config_dir);
    auto lock = store.acquire_refresh_lock(mcp_oauth_provider_id(server_name));
    store.clear(mcp_oauth_provider_id(server_name));
    clear_mcp_oauth_meta(server_name, config_dir);
}

bool has_mcp_oauth_session(const std::string& server_name, const std::string& config_dir) {
    core::auth::FileTokenStore store(config_dir);
    return store.load(mcp_oauth_provider_id(server_name)).has_value();
}

std::optional<std::string> resolve_mcp_access_token(
    const core::config::McpServerConfig& config,
    const std::string& config_dir) {
    for (const auto& [name, value] : config.headers) {
        if (!is_authorization_header_name(name)) continue;
        const std::string expanded = expand_mcp_env_placeholders(value);
        if (expanded.empty()) return std::nullopt;
        if (to_lower(expanded).starts_with("bearer ")) {
            return trim_copy(std::string_view(expanded).substr(7));
        }
        return expanded;
    }

    if (to_lower(config.auth) == "none") return std::nullopt;
    if (!mcp_server_wants_oauth(config) && to_lower(config.auth) != "oauth") {
        return std::nullopt;
    }

    if (const auto token = load_valid_mcp_oauth_token(config.name, config_dir)) {
        if (!token->access_token.empty()) return token->access_token;
    }
    return std::nullopt;
}

core::config::McpServerConfig
with_resolved_mcp_auth(const core::config::McpServerConfig& config,
                       const std::string& config_dir) {
    core::config::McpServerConfig resolved = config;
    if (config.transport != "http") return resolved;

    // Expand every header value (Authorization and custom API key headers).
    for (auto& [name, value] : resolved.headers) {
        value = expand_mcp_env_placeholders(value);
    }

    if (has_authorization_header(resolved.headers)) {
        return resolved;
    }

    if (const auto access = resolve_mcp_access_token(config, config_dir)) {
        resolved.headers.emplace_back("Authorization", bearer_authorization_value(*access));
    }
    return resolved;
}

bool mcp_server_wants_oauth(const core::config::McpServerConfig& config) {
    if (config.transport != "http") return false;
    if (has_authorization_header(config.headers)) return false;
    const auto auth = to_lower(config.auth);
    if (auth == "none") return false;
    if (auth == "oauth") return true;
    return config.url.starts_with("https://");
}

// ---------------------------------------------------------------------------
// McpHttpAuth
// ---------------------------------------------------------------------------

McpHttpAuth::McpHttpAuth(core::config::McpServerConfig config, std::string config_dir)
    : config_(std::move(config))
    , config_dir_(std::move(config_dir)) {
    static_headers_.reserve(config_.headers.size());
    bool has_static_authorization = false;
    for (const auto& [name, value] : config_.headers) {
        if (name.empty()) continue;
        const std::string expanded = expand_mcp_env_placeholders(value);
        if (is_authorization_header_name(name)) {
            has_static_authorization = true;
            cached_authorization_ = expanded;
        }
        static_headers_.emplace_back(name, expanded);
    }

    oauth_mode_ = !has_static_authorization
               && mcp_server_wants_oauth(config_)
               && !config_dir_.empty();

    // Strip Authorization from static headers when OAuth owns it so request_headers()
    // does not send a stale static value alongside a refreshed bearer.
    if (oauth_mode_) {
        std::erase_if(static_headers_, [](const auto& pair) {
            return is_authorization_header_name(pair.first);
        });
        cached_authorization_.clear();
    }
}

std::optional<std::string>
McpHttpAuth::resolve_authorization_locked(bool force_refresh) const {
    if (!oauth_mode_) {
        if (cached_authorization_.empty()) return std::nullopt;
        return cached_authorization_;
    }

    std::optional<core::auth::OAuthToken> token;
    if (force_refresh) {
        token = force_refresh_mcp_oauth_token(config_.name, config_dir_);
    } else {
        token = load_valid_mcp_oauth_token(config_.name, config_dir_);
    }
    if (!token || token->access_token.empty()) {
        return cached_authorization_.empty()
            ? std::nullopt
            : std::optional<std::string>{cached_authorization_};
    }

    cached_authorization_ = bearer_authorization_value(token->access_token);
    return cached_authorization_;
}

std::vector<std::pair<std::string, std::string>> McpHttpAuth::request_headers() const {
    std::lock_guard lock(mutex_);
    auto headers = static_headers_;

    if (oauth_mode_) {
        if (const auto auth = resolve_authorization_locked(/*force_refresh=*/false)) {
            headers.emplace_back("Authorization", *auth);
        }
    } else if (!cached_authorization_.empty()
               && !has_authorization_header(headers)) {
        headers.emplace_back("Authorization", cached_authorization_);
    }
    return headers;
}

bool McpHttpAuth::can_refresh() const noexcept {
    return oauth_mode_ && !config_dir_.empty();
}

bool McpHttpAuth::refresh_after_unauthorized() {
    if (!can_refresh()) return false;

    std::lock_guard lock(mutex_);
    const std::string previous = cached_authorization_;
    const auto auth = resolve_authorization_locked(/*force_refresh=*/true);
    if (!auth || auth->empty()) {
        return false;
    }
    // Retry when we obtained a token (even if identical — rare clock skew races).
    // Prefer true when it actually changed.
    if (*auth != previous) {
        core::logging::info(
            "[MCP OAuth] refreshed access token for '{}' after HTTP 401",
            config_.name);
        return true;
    }
    // Token unchanged: still allow one retry in case of transient RS failure with
    // a just-rotated token from another process that matches our previous value.
    return !previous.empty();
}

std::shared_ptr<McpHttpAuth>
make_mcp_http_auth(const core::config::McpServerConfig& config,
                   std::string_view config_dir) {
    if (config.transport != "http") {
        return nullptr;
    }
    return std::make_shared<McpHttpAuth>(config, std::string(config_dir));
}

} // namespace core::mcp
