#include "QwenSubscriptionOAuthFlow.hpp"

#include "AuthBrowserLauncher.hpp"
#include "OAuthErrors.hpp"
#include "OAuthTokenEndpoint.hpp"
#include "core/llm/QwenSubscription.hpp"
#include "core/utils/AsciiUtils.hpp"
#include "core/version/Version.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>

namespace core::auth {
namespace {

namespace subscription = core::llm::qwen_subscription;

[[nodiscard]] std::int64_t now_unix_seconds() {
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

/// The `error` / `error_description` pair of an OAuth error body.
struct OAuthFailure {
    long status = 0;
    std::string code;
    std::string description;
    std::chrono::milliseconds retry_after{0};
    bool transport = false;
};

[[nodiscard]] OAuthFailure failure_of(const cpr::Response& response) {
    OAuthFailure failure;
    failure.status = response.status_code;
    if (response.error.code != cpr::ErrorCode::OK) {
        failure.transport = true;
        failure.code = response.error.code == cpr::ErrorCode::OPERATION_TIMEDOUT
            ? "request_timeout" : "network_error";
        return failure;
    }
    simdjson::dom::parser parser;
    simdjson::padded_string padded(response.text);
    simdjson::dom::object body;
    if (parser.parse(padded).get(body) == simdjson::SUCCESS) {
        std::string_view value;
        if (body["error"].get(value) == simdjson::SUCCESS) failure.code = std::string(value);
        if (body["error_description"].get(value) == simdjson::SUCCESS) {
            failure.description = std::string(value);
        }
    }
    if (failure.code.empty()) failure.code = "invalid_response";
    for (const auto& [name, value] : response.header) {
        if (!core::utils::ascii::iequals(name, "retry-after")) continue;
        try {
            failure.retry_after = std::chrono::seconds(std::max(0, std::stoi(value)));
        } catch (...) {
        }
    }
    return failure;
}

[[nodiscard]] std::string describe(std::string_view stage, const OAuthFailure& failure) {
    std::string message = "Qwen Subscription " + std::string(stage) + " failed";
    if (failure.transport) return message + ": the network request failed or timed out.";
    if (failure.status == 429) return message + ": too many requests. Please try again later.";
    message += " (";
    if (failure.status > 0) message += "HTTP " + std::to_string(failure.status) + ", ";
    message += failure.code + ')';
    if (!failure.description.empty()) message += ": " + failure.description;
    return message;
}

[[nodiscard]] bool http_ok(long status) { return status >= 200 && status < 300; }

/// The auth host's firewall answers requests without a client User-Agent
/// (or with curl's) by a 405 / HTML challenge instead of OAuth JSON.
[[nodiscard]] cpr::Header form_headers() {
    return cpr::Header{{"Accept", "application/json"},
                       {"Content-Type", "application/x-www-form-urlencoded"},
                       {"User-Agent", std::string(core::version::user_agent)}};
}

[[nodiscard]] std::string required_string(simdjson::dom::object object, std::string_view key) {
    std::string_view value;
    if (object[key].get(value) != simdjson::SUCCESS || value.empty()) {
        throw std::runtime_error("Qwen Subscription returned an invalid response (missing "
                                 + std::string(key) + ").");
    }
    return std::string(value);
}

[[nodiscard]] std::string optional_string(simdjson::dom::object object, std::string_view key) {
    std::string_view value;
    return object[key].get(value) == simdjson::SUCCESS ? std::string(value) : std::string{};
}

[[nodiscard]] simdjson::dom::object parse_object(simdjson::dom::parser& parser,
                                                 const std::string& text) {
    simdjson::dom::object object;
    if (parser.parse(text).get(object) != simdjson::SUCCESS) {
        throw std::runtime_error("Qwen Subscription returned an invalid response.");
    }
    return object;
}

} // namespace

QwenSubscriptionOAuthFlow::QwenSubscriptionOAuthFlow()
    : QwenSubscriptionOAuthFlow(Options{}) {}

QwenSubscriptionOAuthFlow::QwenSubscriptionOAuthFlow(Options options)
    : options_(std::move(options)) {
    if (options_.auth_origin.empty()) options_.auth_origin = std::string(subscription::kAuthOrigin);
    if (!options_.output) options_.output = &std::cout;
    if (!options_.open_browser) options_.open_browser = [](std::string_view url) { open_browser(url); };
    if (!options_.sleep) {
        options_.sleep = [](std::chrono::milliseconds delay) { std::this_thread::sleep_for(delay); };
    }
}

QwenSubscriptionOAuthFlow::DeviceAuthorization
QwenSubscriptionOAuthFlow::request_device_authorization() const {
    const auto response = cpr::Post(
        cpr::Url{options_.auth_origin + "/api/oauth/device/code"},
        form_headers(),
        cpr::Payload{{"client_id", std::string(subscription::kClientId)},
                     {"scope", std::string(subscription::kScope)}},
        cpr::Timeout{options_.request_timeout}, cpr::Redirect{false});
    if (response.error.code != cpr::ErrorCode::OK || !http_ok(response.status_code)) {
        throw std::runtime_error(describe("authorization request", failure_of(response)));
    }
    simdjson::dom::parser parser;
    const auto body = parse_object(parser, response.text);
    DeviceAuthorization authorization{
        .device_code = required_string(body, "device_code"),
        .user_code = required_string(body, "user_code"),
        .verification_uri_complete = optional_string(body, "verification_uri_complete"),
    };
    if (body["expires_in"].get(authorization.expires_in) != simdjson::SUCCESS
        || authorization.expires_in <= 0) {
        throw std::runtime_error("Qwen Subscription returned an invalid response (expires_in).");
    }
    if (body["interval"].get(authorization.interval) != simdjson::SUCCESS
        || authorization.interval <= 0) {
        authorization.interval = 5;
    }
    return authorization;
}

OAuthToken QwenSubscriptionOAuthFlow::poll_for_token(
    const DeviceAuthorization& authorization) const {
    using std::chrono::milliseconds;
    const auto deadline = std::chrono::steady_clock::now()
        + std::chrono::seconds(authorization.expires_in);
    milliseconds interval = std::chrono::seconds(authorization.interval);
    milliseconds delay = interval;
    while (true) {
        const auto remaining = std::chrono::duration_cast<milliseconds>(
            deadline - std::chrono::steady_clock::now());
        if (remaining <= milliseconds::zero()) {
            throw std::runtime_error(
                "Qwen Subscription authorization timed out. Run the sign-in again.");
        }
        options_.sleep(std::min(delay, remaining));

        const std::int64_t request_time = now_unix_seconds();
        const auto response = cpr::Post(
            cpr::Url{options_.auth_origin + "/api/oauth/token"},
            form_headers(),
            cpr::Payload{{"grant_type", "urn:ietf:params:oauth:grant-type:device_code"},
                         {"client_id", std::string(subscription::kClientId)},
                         {"device_code", authorization.device_code}},
            cpr::Timeout{options_.request_timeout}, cpr::Redirect{false});
        if (response.error.code == cpr::ErrorCode::OK && http_ok(response.status_code)) {
            return parse_oauth_token_response(response.text, request_time,
                                              subscription::kClientId, options_.auth_origin);
        }

        const OAuthFailure failure = failure_of(response);
        if (failure.status == 400 && failure.code == "authorization_pending") {
            delay = interval;
            continue;
        }
        if (failure.status == 400 && failure.code == "slow_down") {
            interval += std::chrono::seconds(5);
            delay = interval;
            continue;
        }
        if (failure.transport || failure.status == 429 || failure.status >= 500) {
            delay = std::max({interval, delay * 2, failure.retry_after});
            continue;
        }
        if (failure.code == "expired_token") {
            throw std::runtime_error(
                "Qwen Subscription authorization timed out. Run the sign-in again.");
        }
        if (failure.code == "access_denied") {
            throw std::runtime_error("Qwen Subscription authorization was denied.");
        }
        throw std::runtime_error(describe("authorization", failure));
    }
}

OAuthToken QwenSubscriptionOAuthFlow::login() {
    const DeviceAuthorization authorization = request_device_authorization();
    const std::string url = subscription::authorization_url(authorization.user_code);
    auto& out = *options_.output;
    out << "\nSign in to your Qwen account in the browser to authorize Filo:\n\n  "
        << url << "\n\nCode: " << authorization.user_code
        << "\n\nWaiting for authorization...\n";
    out.flush();
    options_.open_browser(url);
    return poll_for_token(authorization);
}

OAuthToken QwenSubscriptionOAuthFlow::refresh(std::string_view refresh_token) {
    const std::int64_t request_time = now_unix_seconds();
    const auto response = cpr::Post(
        cpr::Url{options_.auth_origin + "/api/oauth/token"},
        form_headers(),
        cpr::Payload{{"grant_type", "refresh_token"},
                     {"client_id", std::string(subscription::kClientId)},
                     {"refresh_token", std::string(refresh_token)}},
        cpr::Timeout{options_.request_timeout}, cpr::Redirect{false});
    if (response.error.code == cpr::ErrorCode::OK && http_ok(response.status_code)) {
        OAuthToken token = parse_oauth_token_response(
            response.text, request_time, subscription::kClientId, options_.auth_origin);
        if (token.refresh_token.empty()) token.refresh_token = std::string(refresh_token);
        return token;
    }
    const OAuthFailure failure = failure_of(response);
    // Only a refresh-stage invalid_grant means the session itself is gone.
    if (failure.code == "invalid_grant") {
        throw OAuthRefreshRejected(std::string(subscription::kSignInRequiredMessage));
    }
    throw std::runtime_error(describe("session refresh", failure));
}

void QwenSubscriptionOAuthFlow::revoke(const OAuthToken& token) {
    const bool by_refresh = !token.refresh_token.empty();
    const auto response = cpr::Post(
        cpr::Url{options_.auth_origin + "/api/oauth/revoke"},
        form_headers(),
        cpr::Payload{{"client_id", std::string(subscription::kClientId)},
                     {"token", by_refresh ? token.refresh_token : token.access_token},
                     {"token_type_hint", by_refresh ? "refresh_token" : "access_token"}},
        cpr::Timeout{options_.request_timeout}, cpr::Redirect{false});
    if (response.error.code != cpr::ErrorCode::OK || !http_ok(response.status_code)) {
        throw std::runtime_error(describe("revocation", failure_of(response)));
    }
}

QwenSubscriptionOAuthFlow::UserInfo
QwenSubscriptionOAuthFlow::user_info(std::string_view access_token) const {
    const auto response = cpr::Get(
        cpr::Url{options_.auth_origin + "/api/oauth/userinfo"},
        cpr::Header{{"Accept", "application/json"},
                    {"User-Agent", std::string(core::version::user_agent)},
                    {"Authorization", "Bearer " + std::string(access_token)}},
        cpr::Timeout{options_.request_timeout}, cpr::Redirect{false});
    if (response.error.code != cpr::ErrorCode::OK || !http_ok(response.status_code)) {
        const OAuthFailure failure = failure_of(response);
        if (failure.status == 401 || failure.code == "invalid_token") {
            throw std::runtime_error(std::string(subscription::kSignInRequiredMessage));
        }
        throw std::runtime_error(describe("membership check", failure));
    }
    simdjson::dom::parser parser;
    const auto body = parse_object(parser, response.text);
    return UserInfo{
        .subject = required_string(body, "sub"),
        .name = optional_string(body, "name"),
        .email = optional_string(body, "email"),
        .member_type = required_string(body, "member_type"),
    };
}

} // namespace core::auth
