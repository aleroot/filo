#include "KimiOAuthFlow.hpp"
#include "AuthBrowserLauncher.hpp"
#include "core/utils/Base64.hpp"
#include <cpr/cpr.h>
#include <simdjson.h>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <thread>
#if !defined(_WIN32)
#include <sys/utsname.h>
#include <unistd.h>
#endif
#include "../logging/Logger.hpp"

namespace core::auth {

// ── Constants ────────────────────────────────────────────────────────────────

namespace {

constexpr std::string_view KIMI_OAUTH_HOST = "https://auth.kimi.com";
constexpr std::string_view KIMI_CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098";
constexpr std::string_view KIMI_PLATFORM = "kimi_code_cli";
constexpr std::string_view KIMI_USER_AGENT_PRODUCT = "kimi-code-cli";
constexpr std::string_view KIMI_CLIENT_VERSION = "1.49.0";
constexpr int KIMI_OAUTH_TIMEOUT_MS = 30'000;

/**
 * @brief Extract device_id from a Kimi JWT token payload.
 *
 * The Kimi JWT contains a device_id claim that must match the X-Msh-Device-Id
 * header in API requests. This function extracts it from the access_token.
 *
 * JWT format: header.payload.signature (base64url encoded)
 * The payload contains: {"device_id": "...", "user_id": "...", ...}
 */
std::string extractDeviceIdFromJwt(const std::string& jwt) {
    // Find the two dots that separate header.payload.signature
    size_t first_dot = jwt.find('.');
    if (first_dot == std::string::npos) return {};
    
    size_t second_dot = jwt.find('.', first_dot + 1);
    if (second_dot == std::string::npos) return {};
    
    const std::string payload_b64 = jwt.substr(first_dot + 1, second_dot - first_dot - 1);
    const auto payload = core::utils::Base64::decode_url(payload_b64);
    if (!payload.has_value()) {
        return {};
    }
    
    // Parse JSON to extract device_id
    // Use simdjson for robust parsing
    try {
        simdjson::dom::parser parser;
        simdjson::padded_string ps(*payload);
        simdjson::dom::element doc = parser.parse(ps);
        
        std::string_view device_id;
        if (doc["device_id"].get_string().get(device_id) == simdjson::SUCCESS) {
            return std::string(device_id);
        }
    } catch (...) {
        // Parsing failed, return empty
    }
    
    return {};
}

std::filesystem::path getDeviceIdPath() {
    const char* xdg_config = std::getenv("XDG_CONFIG_HOME");
    std::filesystem::path config_dir;
    if (xdg_config) {
        config_dir = std::filesystem::path(xdg_config) / "filo";
    } else {
        const char* home = std::getenv("HOME");
        if (home) {
            config_dir = std::filesystem::path(home) / ".config" / "filo";
        } else {
            config_dir = std::filesystem::path("/tmp") / "filo";
        }
    }
    std::filesystem::create_directories(config_dir);
    return config_dir / "kimi_device_id";
}

[[nodiscard]] std::string ascii_header(std::string_view value,
                                       std::string_view fallback = "unknown") {
    std::string cleaned;
    cleaned.reserve(value.size());
    for (const unsigned char c : value) {
        if (c >= 32 && c < 127) {
            cleaned.push_back(static_cast<char>(c));
        }
    }

    const auto start = cleaned.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return std::string(fallback);
    const auto end = cleaned.find_last_not_of(" \t\r\n");
    return cleaned.substr(start, end - start + 1);
}

[[nodiscard]] bool is_uuid_device_id(std::string_view id) {
    if (id.size() != 36) return false;
    for (std::size_t i = 0; i < id.size(); ++i) {
        const bool dash = (i == 8 || i == 13 || i == 18 || i == 23);
        if (dash) {
            if (id[i] != '-') return false;
            continue;
        }
        if (!std::isxdigit(static_cast<unsigned char>(id[i]))) return false;
    }
    return true;
}

[[nodiscard]] bool is_legacy_device_id(std::string_view id) {
    if (id.size() != 32) return false;
    for (const char c : id) {
        if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
    }
    return true;
}

[[nodiscard]] bool is_retryable_oauth_failure(const cpr::Response& response) {
    if (response.error.code != cpr::ErrorCode::OK) return true;
    return response.status_code == 429
        || response.status_code == 500
        || response.status_code == 502
        || response.status_code == 503
        || response.status_code == 504;
}

} // namespace

// ── Helpers ──────────────────────────────────────────────────────────────────

std::string KimiOAuthFlow::generateDeviceId() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 15);
    const char hex[] = "0123456789abcdef";
    
    std::string id;
    id.reserve(36);
    for (int i = 0; i < 36; ++i) {
        if (i == 8 || i == 13 || i == 18 || i == 23) {
            id += '-';
            continue;
        }
        if (i == 14) {
            id += '4';
            continue;
        }
        if (i == 19) {
            id += hex[8 + (dist(gen) % 4)];
            continue;
        }
        id += hex[dist(gen)];
    }
    return id;
}

std::string KimiOAuthFlow::getDeviceModel() {
    struct utsname buf{};
    if (uname(&buf) == 0) {
        return std::string(buf.sysname) + " " + std::string(buf.release) + " " + std::string(buf.machine);
    }
    return "Unknown";
}

std::unordered_map<std::string, std::string> KimiOAuthFlow::getCommonHeaders() {
    return getCommonHeaders(loadOrCreatePersistentDeviceId());
}

std::unordered_map<std::string, std::string> KimiOAuthFlow::getCommonHeaders(
    std::string_view device_id) {
    std::unordered_map<std::string, std::string> headers;
    
    // Get device name (hostname)
    char hostname[256];
    std::string device_name = "unknown";
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        device_name = hostname;
    }
    
    // Get OS version
    struct utsname buf{};
    std::string os_version = "unknown";
    if (uname(&buf) == 0) {
        os_version = buf.release;
    }
    
    headers["X-Msh-Platform"] = std::string(KIMI_PLATFORM);
    headers["X-Msh-Version"] = std::string(KIMI_CLIENT_VERSION);
    headers["X-Msh-Device-Name"] = ascii_header(device_name);
    headers["X-Msh-Device-Model"] = ascii_header(getDeviceModel());
    headers["X-Msh-Os-Version"] = ascii_header(os_version);
    if (!device_id.empty()) {
        headers["X-Msh-Device-Id"] = std::string(device_id);
    }
    
    return headers;
}

std::string KimiOAuthFlow::getUserAgent() {
    return std::string(KIMI_USER_AGENT_PRODUCT) + "/" + std::string(KIMI_CLIENT_VERSION);
}

std::string KimiOAuthFlow::getDeviceName() const {
    char hostname[256];
    if (gethostname(hostname, sizeof(hostname)) == 0) {
        return std::string(hostname);
    }
    return "unknown";
}

std::string KimiOAuthFlow::loadOrCreatePersistentDeviceId() {
    auto path = getDeviceIdPath();
    if (std::filesystem::exists(path)) {
        std::ifstream file(path);
        std::string id;
        if (std::getline(file, id)) {
            if (is_uuid_device_id(id) || is_legacy_device_id(id)) return id;
        }
    }

    std::string new_id = KimiOAuthFlow::generateDeviceId();
    std::ofstream file(path);
    if (file) {
        file << new_id;
        std::filesystem::permissions(path,
            std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
            std::filesystem::perm_options::replace);
    }
    return new_id;
}

// ── KimiOAuthFlow ─────────────────────────────────────────────────────────────

KimiOAuthFlow::KimiOAuthFlow() {
    // Load or create persistent device_id
    device_id_ = loadOrCreatePersistentDeviceId();
}

OAuthToken KimiOAuthFlow::login() {
    core::logging::info("Starting Kimi OAuth device flow...");
    
    auto auth = requestDeviceAuthorization();
    
    core::logging::info("Please visit the following URL to authorize:");
    core::logging::info("  {}", auth.verification_uri_complete);
    core::logging::info("");
    core::logging::info("User code: {}", auth.user_code);
    core::logging::info("");
    core::logging::info("Waiting for authorization...");
    
    // Try to open browser (best-effort, non-fatal on failure).
    open_browser(auth.verification_uri_complete);

    return pollForToken(auth);
}

KimiOAuthFlow::DeviceAuthorization KimiOAuthFlow::requestDeviceAuthorization() {
    // Build headers using the common function + User-Agent
    auto common_headers = getCommonHeaders(device_id_);
    cpr::Header headers;
    for (const auto& [k, v] : common_headers) {
        headers[k] = v;
    }
    headers["User-Agent"] = getUserAgent();
    
    cpr::Payload payload{
        {"client_id", std::string(KIMI_CLIENT_ID)},
    };
    
    cpr::Response r = cpr::Post(
        cpr::Url{std::string(KIMI_OAUTH_HOST) + "/api/oauth/device_authorization"},
        std::move(headers),
        std::move(payload),
        cpr::Timeout{KIMI_OAUTH_TIMEOUT_MS}
    );
    
    if (r.status_code != 200) {
        throw std::runtime_error("Device authorization failed (" + 
                                 std::to_string(r.status_code) + "): " + r.text);
    }
    
    simdjson::dom::parser parser;
    simdjson::padded_string padded(r.text);
    simdjson::dom::element doc = parser.parse(padded);
    
    DeviceAuthorization auth;
    std::string_view sv;
    int64_t v;
    
    if (doc["user_code"].get_string().get(sv) == simdjson::SUCCESS) {
        auth.user_code = std::string(sv);
    }
    if (doc["device_code"].get_string().get(sv) == simdjson::SUCCESS) {
        auth.device_code = std::string(sv);
    }
    if (doc["verification_uri"].get_string().get(sv) == simdjson::SUCCESS) {
        auth.verification_uri = std::string(sv);
    }
    if (doc["verification_uri_complete"].get_string().get(sv) == simdjson::SUCCESS) {
        auth.verification_uri_complete = std::string(sv);
    }
    if (doc["expires_in"].get_int64().get(v) == simdjson::SUCCESS) {
        auth.expires_in = static_cast<int>(v);
    }
    if (doc["interval"].get_int64().get(v) == simdjson::SUCCESS) {
        auth.interval = static_cast<int>(v);
    }
    
    if (auth.device_code.empty()) {
        throw std::runtime_error("No device_code in authorization response");
    }
    
    return auth;
}

OAuthToken KimiOAuthFlow::pollForToken(const DeviceAuthorization& auth) {
    int max_attempts = auth.expires_in > 0 ? auth.expires_in / auth.interval : 60;
    int interval = std::max(auth.interval, 1);
    
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        std::this_thread::sleep_for(std::chrono::seconds(interval));
        
        // Build headers using the common function + User-Agent
        auto common_headers = getCommonHeaders(device_id_);
        cpr::Header headers;
        for (const auto& [k, v] : common_headers) {
            headers[k] = v;
        }
        headers["User-Agent"] = getUserAgent();
        
        cpr::Payload payload{
            {"client_id", std::string(KIMI_CLIENT_ID)},
            {"device_code", auth.device_code},
            {"grant_type", "urn:ietf:params:oauth:grant-type:device_code"},
        };
        
        cpr::Response r = cpr::Post(
            cpr::Url{std::string(KIMI_OAUTH_HOST) + "/api/oauth/token"},
            std::move(headers),
            std::move(payload),
            cpr::Timeout{KIMI_OAUTH_TIMEOUT_MS}
        );
        
        if (r.status_code == 200) {
            // Success!
            simdjson::dom::parser parser;
            simdjson::padded_string padded(r.text);
            simdjson::dom::element doc = parser.parse(padded);
            
            OAuthToken token;
            std::string_view sv;
            int64_t expires_in = 0;
            
            auto request_time = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
            
            if (doc["access_token"].get_string().get(sv) == simdjson::SUCCESS) {
                token.access_token = std::string(sv);
            }
            if (doc["refresh_token"].get_string().get(sv) == simdjson::SUCCESS) {
                token.refresh_token = std::string(sv);
            }
            if (doc["token_type"].get_string().get(sv) == simdjson::SUCCESS) {
                token.token_type = std::string(sv);
            } else {
                token.token_type = "Bearer";
            }
            if (doc["expires_in"].get_int64().get(expires_in) == simdjson::SUCCESS) {
                token.expires_at = request_time + expires_in;
            }
            
            if (token.access_token.empty()) {
                throw std::runtime_error("No access_token in response");
            }
            
            // Extract device_id from JWT payload - CRITICAL for Kimi OAuth.
            // The X-Msh-Device-Id header must match the device_id claim in the token.
            token.device_id = extractDeviceIdFromJwt(token.access_token);
            if (!token.device_id.empty()) {
                core::logging::info("Kimi OAuth: extracted device_id from token");
            }
            
            core::logging::info("Kimi OAuth authorization successful!");
            return token;
        }
        
        // Parse error to see if we should continue polling
        simdjson::dom::parser parser;
        simdjson::padded_string padded(r.text);
        simdjson::dom::element doc = parser.parse(padded);
        
        std::string_view error;
        if (doc["error"].get_string().get(error) == simdjson::SUCCESS) {
            if (error == "authorization_pending") {
                // Still waiting for user, continue polling
                continue;
            } else if (error == "slow_down") {
                // Server wants us to slow down
                interval += 5;
                continue;
            } else if (error == "expired_token") {
                throw std::runtime_error("Device code expired. Please try again.");
            } else {
                throw std::runtime_error("OAuth error: " + std::string(error));
            }
        }
    }
    
    throw std::runtime_error("Authorization timed out. Please try again.");
}

OAuthToken KimiOAuthFlow::refresh(std::string_view refresh_token) {
    return exchangeRefreshToken(refresh_token);
}

OAuthToken KimiOAuthFlow::exchangeRefreshToken(std::string_view refresh_token) {
    auto request_time = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    
    // Build headers using the common function + User-Agent
    auto common_headers = getCommonHeaders(device_id_);
    cpr::Header headers;
    for (const auto& [k, v] : common_headers) {
        headers[k] = v;
    }
    headers["User-Agent"] = getUserAgent();
    
    cpr::Payload payload{
        {"client_id", std::string(KIMI_CLIENT_ID)},
        {"grant_type", "refresh_token"},
        {"refresh_token", std::string(refresh_token)},
    };
    
    cpr::Response r;
    for (int attempt = 0; attempt < 3; ++attempt) {
        r = cpr::Post(
            cpr::Url{std::string(KIMI_OAUTH_HOST) + "/api/oauth/token"},
            headers,
            payload,
            cpr::Timeout{KIMI_OAUTH_TIMEOUT_MS}
        );

        if (!is_retryable_oauth_failure(r) || attempt == 2) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::seconds(1 << attempt));
    }
    
    if (r.status_code == 401 || r.status_code == 403) {
        throw std::runtime_error("Token refresh unauthorized. Please login again.");
    }
    
    if (r.status_code != 200) {
        throw std::runtime_error("Token refresh failed (" + 
                                 std::to_string(r.status_code) + "): " + r.text);
    }
    
    simdjson::dom::parser parser;
    simdjson::padded_string padded(r.text);
    simdjson::dom::element doc = parser.parse(padded);
    
    OAuthToken token;
    std::string_view sv;
    int64_t expires_in = 0;
    
    if (doc["access_token"].get_string().get(sv) == simdjson::SUCCESS) {
        token.access_token = std::string(sv);
    }
    if (doc["refresh_token"].get_string().get(sv) == simdjson::SUCCESS) {
        token.refresh_token = std::string(sv);
    } else {
        // Refresh response often doesn't include new refresh_token, preserve old one
        token.refresh_token = std::string(refresh_token);
    }
    if (doc["token_type"].get_string().get(sv) == simdjson::SUCCESS) {
        token.token_type = std::string(sv);
    } else {
        token.token_type = "Bearer";
    }
    if (doc["expires_in"].get_int64().get(expires_in) == simdjson::SUCCESS) {
        token.expires_at = request_time + expires_in;
    }
    
    if (token.access_token.empty()) {
        throw std::runtime_error("No access_token in refresh response");
    }
    
    // Extract device_id from JWT payload - CRITICAL for Kimi OAuth.
    // The X-Msh-Device-Id header must match the device_id claim in the token.
    token.device_id = extractDeviceIdFromJwt(token.access_token);
    if (!token.device_id.empty()) {
        core::logging::info("Kimi OAuth refresh: extracted device_id from token");
    }
    
    return token;
}

} // namespace core::auth
