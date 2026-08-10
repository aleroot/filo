#include "GeminiAntigravityProtocol.hpp"
#include "../../utils/JsonUtils.hpp"
#include <atomic>
#include <cstdlib>
#include <string>

namespace core::llm::protocols {

namespace {

[[nodiscard]] std::string antigravity_platform() noexcept {
#if defined(_WIN32)
    return "windows";
#elif defined(__APPLE__)
    return "darwin";
#else
    return "linux";
#endif
}

[[nodiscard]] std::string antigravity_arch() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#else
    return "amd64";
#endif
}

// Antigravity's backend has been observed rejecting requests from
// self-reported client versions it considers too old ("version no longer
// supported"). There is no stable way to discover the "current" version
// short of tracking Google's Antigravity releases, so this is exposed as an
// env var override for users hitting that error without waiting for a Filo
// release.
[[nodiscard]] std::string antigravity_version() {
    if (const char* raw = std::getenv("ANTIGRAVITY_CLI_VERSION"); raw && raw[0] != '\0') {
        return raw;
    }
    return "1.18.3";
}

[[nodiscard]] std::string antigravity_user_agent() {
    if (const char* raw = std::getenv("ANTIGRAVITY_USER_AGENT"); raw && raw[0] != '\0') {
        return raw;
    }
    return "antigravity/" + antigravity_version() + " " + antigravity_platform()
        + "/" + antigravity_arch();
}

[[nodiscard]] std::string next_request_id() {
    static std::atomic_uint64_t counter{0};
    return "filo-antigravity-" + std::to_string(++counter);
}

} // namespace

std::string GeminiAntigravityProtocol::serialize(const ChatRequest& req) const {
    std::string payload = serialize_gemini_code_assist_request(req, req.model);
    // serialize_gemini_code_assist_request always emits a single top-level
    // JSON object terminated by '}'; splice in the extra Antigravity fields
    // rather than re-implementing Gemini request serialization here.
    if (!payload.empty() && payload.back() == '}') {
        payload.pop_back();
        payload += R"(,"userAgent":"antigravity","requestId":")";
        payload += core::utils::escape_json_string(next_request_id());
        payload += "\"}";
    }
    return payload;
}

cpr::Header GeminiAntigravityProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"User-Agent", antigravity_user_agent()},
        {"X-Goog-Api-Client", "google-cloud-sdk vscode_cloudshelleditor/0.1"},
        {"Client-Metadata",
         R"({"ideType":"ANTIGRAVITY","platform":")" + [] {
             const std::string p = antigravity_platform();
             if (p == "darwin") return std::string("MACOS");
             if (p == "windows") return std::string("WINDOWS");
             return std::string("LINUX");
         }() + R"(","pluginType":"GEMINI"})"},
    };
    for (const auto& [k, v] : auth.headers) {
        headers[k] = v;
    }
    return headers;
}

} // namespace core::llm::protocols
