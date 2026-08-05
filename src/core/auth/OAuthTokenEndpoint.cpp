#include "OAuthTokenEndpoint.hpp"

#include "OAuthErrors.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <chrono>
#include <cstdint>
#include <stdexcept>

namespace core::auth {
namespace {

[[nodiscard]] int64_t now_unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

[[nodiscard]] bool http_ok(long status) {
    return status >= 200 && status < 300;
}

} // namespace

std::vector<std::string> split_oauth_scopes(std::string_view raw) {
    std::vector<std::string> scopes;
    std::size_t i = 0;
    while (i < raw.size()) {
        while (i < raw.size()
               && (raw[i] == ' ' || raw[i] == '\t' || raw[i] == ',')) {
            ++i;
        }
        if (i >= raw.size()) break;
        const std::size_t start = i;
        while (i < raw.size()
               && raw[i] != ' ' && raw[i] != '\t' && raw[i] != ',') {
            ++i;
        }
        scopes.emplace_back(raw.substr(start, i - start));
    }
    return scopes;
}

std::string join_oauth_scopes(const std::vector<std::string>& scopes) {
    std::string out;
    for (std::size_t i = 0; i < scopes.size(); ++i) {
        if (i > 0) out.push_back(' ');
        out += scopes[i];
    }
    return out;
}

OAuthToken parse_oauth_token_response(std::string_view json,
                                      int64_t request_time_unix,
                                      std::string_view client_id,
                                      std::string_view issuer) {
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(json);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) {
        throw std::runtime_error("OAuth: invalid token response JSON");
    }

    std::string_view error;
    if (doc["error"].get(error) == simdjson::SUCCESS) {
        std::string_view desc;
        [[maybe_unused]] const auto ignored = doc["error_description"].get(desc);
        throw std::runtime_error(
            std::string("OAuth token error: ") + std::string(error)
            + (desc.empty() ? "" : (std::string(" — ") + std::string(desc))));
    }

    OAuthToken token;
    std::string_view access;
    if (doc["access_token"].get(access) != simdjson::SUCCESS || access.empty()) {
        throw std::runtime_error("OAuth: token response missing access_token");
    }
    token.access_token = std::string(access);

    std::string_view refresh;
    if (doc["refresh_token"].get(refresh) == simdjson::SUCCESS) {
        token.refresh_token = std::string(refresh);
    }

    std::string_view token_type;
    if (doc["token_type"].get(token_type) == simdjson::SUCCESS && !token_type.empty()) {
        token.token_type = std::string(token_type);
    }

    int64_t expires_in = 3600;
    [[maybe_unused]] const auto exp_err = doc["expires_in"].get(expires_in);
    if (expires_in < 60) expires_in = 60;
    token.expires_at = request_time_unix + expires_in;

    std::string_view scope;
    if (doc["scope"].get(scope) == simdjson::SUCCESS) {
        token.scopes = split_oauth_scopes(scope);
    }

    if (!client_id.empty()) token.client_id = std::string(client_id);
    if (!issuer.empty()) token.issuer = std::string(issuer);
    return token;
}

OAuthToken exchange_authorization_code(const OAuthTokenEndpointRequest& request,
                                       std::string_view code,
                                       std::string_view code_verifier) {
    if (request.token_url.empty()) {
        throw std::runtime_error("OAuth: token_url is empty");
    }
    if (code.empty()) {
        throw std::runtime_error("OAuth: authorization code is empty");
    }

    const int64_t request_time = now_unix_seconds();
    cpr::Payload payload{
        {"grant_type", "authorization_code"},
        {"code", std::string(code)},
        {"redirect_uri", request.redirect_uri},
        {"client_id", request.client_id},
    };
    if (!code_verifier.empty()) {
        payload.Add({"code_verifier", std::string(code_verifier)});
    }
    if (!request.client_secret.empty()) {
        payload.Add({"client_secret", request.client_secret});
    }
    if (!request.resource.empty()) {
        payload.Add({"resource", request.resource});
    }

    const auto r = cpr::Post(
        cpr::Url{request.token_url},
        cpr::Header{{"Accept", "application/json"},
                    {"Content-Type", "application/x-www-form-urlencoded"}},
        payload,
        cpr::Timeout{static_cast<int32_t>(request.timeout.count())});

    if (r.error.code != cpr::ErrorCode::OK) {
        throw std::runtime_error("OAuth token exchange failed: " + r.error.message);
    }
    if (!http_ok(r.status_code)) {
        throw std::runtime_error(
            "OAuth token exchange failed (" + std::to_string(r.status_code) + "): "
            + r.text);
    }

    return parse_oauth_token_response(
        r.text, request_time, request.client_id, request.issuer);
}

OAuthToken refresh_access_token(const OAuthTokenEndpointRequest& request,
                                std::string_view refresh_token) {
    if (request.token_url.empty()) {
        throw std::runtime_error("OAuth: token_url is empty");
    }
    if (refresh_token.empty()) {
        throw std::runtime_error("OAuth: refresh_token is empty");
    }

    const int64_t request_time = now_unix_seconds();
    cpr::Payload payload{
        {"grant_type", "refresh_token"},
        {"refresh_token", std::string(refresh_token)},
        {"client_id", request.client_id},
    };
    if (!request.client_secret.empty()) {
        payload.Add({"client_secret", request.client_secret});
    }
    if (!request.resource.empty()) {
        payload.Add({"resource", request.resource});
    }

    const auto r = cpr::Post(
        cpr::Url{request.token_url},
        cpr::Header{{"Accept", "application/json"},
                    {"Content-Type", "application/x-www-form-urlencoded"}},
        payload,
        cpr::Timeout{static_cast<int32_t>(request.timeout.count())});

    if (r.error.code != cpr::ErrorCode::OK) {
        throw std::runtime_error("OAuth refresh failed: " + r.error.message);
    }
    if (!http_ok(r.status_code)) {
        if (oauth_error_is_invalid_grant(r.text)) {
            throw OAuthRefreshRejected(
                "OAuth provider rejected the refresh token (invalid_grant)");
        }
        throw std::runtime_error(
            "OAuth refresh failed (" + std::to_string(r.status_code) + "): " + r.text);
    }

    OAuthToken token = parse_oauth_token_response(
        r.text, request_time, request.client_id, request.issuer);
    if (token.refresh_token.empty()) {
        token.refresh_token = std::string(refresh_token);
    }
    return token;
}

} // namespace core::auth
