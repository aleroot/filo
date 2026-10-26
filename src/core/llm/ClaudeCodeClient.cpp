#include "ClaudeCodeClient.hpp"

#include "ClaudeModelTraits.hpp"
#include "../utils/AsciiUtils.hpp"
#include "../utils/Crypto.hpp"
#include "../utils/Gzip.hpp"
#include "../utils/StringUtils.hpp"
#include "../utils/Uuid.hpp"

#include <simdjson.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cstdlib>
#include <deque>
#include <format>
#include <mutex>
#include <optional>
#include <random>
#include <regex>
#include <stdexcept>
#include <unordered_map>

namespace core::llm::claude {

namespace {

constexpr std::string_view kClaudeCodeBeta = "claude-code-20250219";
constexpr std::string_view kPromptCachingScopeBeta = "prompt-caching-scope-2026-01-05";
constexpr std::string_view kPerTurnControlBeta = "per-turn-control-2026-07-01";
constexpr std::string_view kCacheDiagnosisBeta = "cache-diagnosis-2026-04-07";

// Claude Code 2.1.302's main-request list for a model without its own row.
constexpr std::array<std::string_view, 14> kMainBetas{
    kClaudeCodeBeta,
    kOAuthBeta,
    "interleaved-thinking-2025-05-14",
    "thinking-token-count-2026-05-13",
    "context-management-2025-06-27",
    kPromptCachingScopeBeta,
    "mid-conversation-tool-changes-2026-07-01",
    "advanced-tool-use-2025-11-20",
    "mid-conversation-system-clear-at-2026-08-21",
    "effort-2025-11-24",
    kThinkingBindingBeta,
    "thinking-display-updates-2026-08-18",
    "extended-cache-ttl-2025-04-11",
    kCacheDiagnosisBeta,
};

// Models whose list carries per-turn-control; Haiku 5.5 also moves
// claude-code-20250219 behind the prompt-caching scope beta.
constexpr std::array<std::string_view, 4> kPerTurnControlModels{
    "fable-5-1", "opus-5-5", "sonnet-5-5", "haiku-5-5"};
constexpr std::string_view kClaudeCodeBetaAfterScopeModel = "haiku-5-5";

constexpr std::string_view kGzipHost = "api.anthropic.com";
constexpr std::string_view kGzipSettingEnv = "CLAUDE_CODE_GZIP_REQUEST_BODIES";
constexpr std::size_t kMaxGzipPadding = 256;
constexpr std::size_t kMaxConversations = 1000;

std::atomic_bool g_gzip_rejected{false};

struct Conversation {
    std::string session_id;
    std::string prompt_id;
    int prompt_index = 0;
    std::string previous_request_id;
    std::string previous_message_id;
};

class LineageRegistry {
public:
    RequestLineage begin(std::string_view key, int prompt_index) {
        std::lock_guard lock(mutex_);
        auto& conversation = find_or_create(key);
        if (prompt_index != conversation.prompt_index || conversation.prompt_id.empty()) {
            conversation.prompt_index = prompt_index;
            conversation.prompt_id = prompt_index > 0 ? core::utils::random_uuid_v4() : std::string{};
        }
        return RequestLineage{
            .session_id = conversation.session_id,
            .billing = {.previous_request_id = conversation.previous_request_id,
                        .prompt_id = conversation.prompt_id,
                        .prompt_index = conversation.prompt_index},
            .previous_message_id = conversation.previous_message_id,
        };
    }

    void record_request(std::string_view key, std::string_view id) {
        std::lock_guard lock(mutex_);
        find_or_create(key).previous_request_id = id;
    }

    void record_message(std::string_view key, std::string_view id) {
        std::lock_guard lock(mutex_);
        find_or_create(key).previous_message_id = id;
    }

private:
    Conversation& find_or_create(std::string_view key) {
        if (const auto it = conversations_.find(std::string(key)); it != conversations_.end()) return it->second;
        if (conversations_.size() >= kMaxConversations) {
            conversations_.erase(order_.front());
            order_.pop_front();
        }
        order_.emplace_back(key);
        auto& conversation = conversations_[std::string(key)];
        conversation.session_id = core::utils::random_uuid_v4();
        return conversation;
    }

    std::mutex mutex_;
    std::unordered_map<std::string, Conversation> conversations_;
    std::deque<std::string> order_;
};

LineageRegistry& registry() {
    static LineageRegistry instance;
    return instance;
}

[[nodiscard]] std::string_view env(const char* name) {
    const char* value = std::getenv(name);
    return value ? std::string_view(value) : std::string_view{};
}

/// `1/true/yes/on` → true, `0/false/no/off` → false, anything else → nullopt.
[[nodiscard]] std::optional<bool> tri_bool(std::string_view value) {
    const auto normalized = core::utils::str::to_lower_ascii_copy(core::utils::str::trim_ascii_copy(value));
    if (normalized == "1" || normalized == "true" || normalized == "yes" || normalized == "on") return true;
    if (normalized == "0" || normalized == "false" || normalized == "no" || normalized == "off") return false;
    return std::nullopt;
}

struct UrlParts {
    std::string host;   ///< Lowercase, without port.
    std::string port;   ///< Explicit or scheme default.
};

[[nodiscard]] std::optional<UrlParts> split_url(std::string_view url) {
    std::string_view port = "443";
    if (core::utils::ascii::istarts_with(url, "https://")) {
        url.remove_prefix(8);
    } else if (core::utils::ascii::istarts_with(url, "http://")) {
        url.remove_prefix(7);
        port = "80";
    } else {
        return std::nullopt;
    }
    auto authority = url.substr(0, url.find_first_of("/?#"));
    if (const auto at = authority.rfind('@'); at != std::string_view::npos) authority.remove_prefix(at + 1);
    UrlParts parts;
    if (const auto colon = authority.rfind(':'); colon != std::string_view::npos) {
        parts.host = core::utils::str::to_lower_ascii_copy(authority.substr(0, colon));
        parts.port = std::string(authority.substr(colon + 1));
    } else {
        parts.host = core::utils::str::to_lower_ascii_copy(authority);
        parts.port = std::string(port);
    }
    return parts;
}

/// Whether NO_PROXY exempts @p url: `*`, an exact host, `host:port`, or a
/// `.suffix` matching the domain and its subdomains.
[[nodiscard]] bool bypasses_proxy(const UrlParts& url) {
    std::string list(env("no_proxy"));
    if (const auto upper = env("NO_PROXY"); !upper.empty() && upper != list) {
        list = list.empty() ? std::string(upper) : list + "," + std::string(upper);
    }
    std::size_t start = 0;
    while (start < list.size()) {
        auto end = list.find_first_of(", \t", start);
        if (end == std::string::npos) end = list.size();
        const auto entry = core::utils::str::to_lower_ascii_copy(
            core::utils::str::trim_ascii_copy(std::string_view(list).substr(start, end - start)));
        start = end + 1;
        if (entry.empty()) continue;
        if (entry == "*") return true;
        if (entry.find(':') != std::string::npos) {
            if (entry == url.host + ":" + url.port) return true;
        } else if (entry.starts_with('.')) {
            if (url.host == entry.substr(1) || url.host.ends_with(entry)) return true;
        } else if (url.host == entry) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool proxy_configured() {
    return std::ranges::any_of(std::array{"HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"},
                               [](const char* name) {
                                   return !core::utils::str::trim_ascii_view(env(name)).empty();
                               });
}

/// JavaScript `string.length` of UTF-8 @p text.
[[nodiscard]] std::size_t utf16_length(std::string_view text) noexcept {
    std::size_t length = 0;
    for (const char ch : text) {
        const auto byte = static_cast<unsigned char>(ch);
        if ((byte & 0xC0) != 0x80) ++length;     // Every lead byte starts a code point.
        if ((byte & 0xF8) == 0xF0) ++length;     // Supplementary planes take two units.
    }
    return length;
}

[[nodiscard]] bool is_invalid_json_report(std::string_view body) {
    static const std::regex bad_json(
        R"(^bad json: (?:invalid character |unexpected EOF$|unexpected end of JSON input$|EOF$))");
    const auto matches_bad_json = [](std::string_view text) {
        return std::regex_search(text.begin(), text.end(), bad_json);
    };
    simdjson::dom::parser parser;
    const simdjson::padded_string padded(body);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        auto trimmed = body;
        while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back()))) trimmed.remove_suffix(1);
        return matches_bad_json(trimmed);
    }
    std::string_view type, error_type, message, error_text, reason;
    if (doc["type"].get(type) == simdjson::SUCCESS && type == "error"
        && doc["error"]["type"].get(error_type) == simdjson::SUCCESS
        && doc["error"]["message"].get(message) == simdjson::SUCCESS) {
        return message.starts_with("The request body is not valid JSON")
            || message.starts_with("Failed to parse request: could not parse request body as JSON");
    }
    return doc["error"].get(error_text) == simdjson::SUCCESS
        && doc["reason"].get(reason) == simdjson::SUCCESS && reason == "bad_json"
        && matches_bad_json(error_text);
}

} // namespace

std::string_view stainless_os() noexcept {
#if defined(__APPLE__)
    return "MacOS";
#elif defined(_WIN32)
    return "Windows";
#elif defined(__linux__)
    return "Linux";
#elif defined(__FreeBSD__)
    return "FreeBSD";
#else
    return "Unknown";
#endif
}

std::string_view stainless_arch() noexcept {
#if defined(__aarch64__) || defined(_M_ARM64)
    return "arm64";
#elif defined(__x86_64__) || defined(_M_X64)
    return "x64";
#elif defined(__i386__) || defined(_M_IX86)
    return "x32";
#else
    return "unknown";
#endif
}

void apply_client_headers(cpr::Header& headers) {
    headers["User-Agent"] = std::string(kUserAgent);
    headers["anthropic-dangerous-direct-browser-access"] = "true";
    headers["x-app"] = "cli";
    headers["x-stainless-arch"] = std::string(stainless_arch());
    headers["x-stainless-lang"] = "js";
    headers["x-stainless-os"] = std::string(stainless_os());
    headers["x-stainless-package-version"] = std::string(kStainlessPackageVersion);
    headers["x-stainless-retry-count"] = "0";
    headers["x-stainless-runtime"] = "node";
    headers["x-stainless-runtime-version"] = std::string(kStainlessRuntimeVersion);
    headers["x-stainless-timeout"] = "600";
}

std::vector<std::string_view> main_request_betas(std::string_view model, bool oauth, bool structured_output) {
    std::vector<std::string_view> betas(kMainBetas.begin(), kMainBetas.end());
    const auto normalized = normalized_claude_id(model);
    const auto is_model = [&](std::string_view token) { return matches_generation(normalized, token); };
    const auto position = [&](std::string_view beta) { return std::ranges::find(betas, beta); };

    if (is_model(kClaudeCodeBetaAfterScopeModel)) {
        betas.erase(position(kClaudeCodeBeta));
        betas.insert(position(kPromptCachingScopeBeta) + 1, kClaudeCodeBeta);
    }
    if (std::ranges::any_of(kPerTurnControlModels, is_model)) {
        auto anchor = position(kPromptCachingScopeBeta) + 1;
        if (anchor != betas.end() && *anchor == kClaudeCodeBeta) ++anchor;
        betas.insert(anchor, kPerTurnControlBeta);
    }
    if (structured_output) betas.insert(position(kCacheDiagnosisBeta), kStructuredOutputsBeta);
    if (!oauth) betas.erase(position(kOAuthBeta));
    return betas;
}

RequestLineage begin_request(std::string_view conversation, int prompt_index) {
    return registry().begin(conversation, prompt_index);
}

void record_request_id(std::string_view conversation, std::string_view request_id) {
    if (is_valid_request_id(request_id)) registry().record_request(conversation, request_id);
}

void record_message_id(std::string_view conversation, std::string_view message_id) {
    if (!message_id.empty()) registry().record_message(conversation, message_id);
}

std::string_view device_id() {
    static const std::string id = [] {
        std::string hex;
        for (const auto byte : core::utils::crypto::random_bytes(32)) hex += std::format("{:02x}", byte);
        return hex;
    }();
    return id;
}

std::string metadata_user_id(std::string_view account_uuid, std::string_view session_id) {
    return std::format(R"({{"device_id":"{}","account_uuid":"{}","session_id":"{}"}})",
                       device_id(), account_uuid, session_id);
}

bool should_gzip_body(std::string_view url, std::string_view body) {
    const auto parts = split_url(url);
    if (!parts || parts->host != kGzipHost) return false;
    const auto setting = tri_bool(env(kGzipSettingEnv.data()));
    if (setting == false) return false;
    if (utf16_length(body) < kGzipMinBodyChars) return false;
    if (g_gzip_rejected.load(std::memory_order_relaxed)) return false;
    // Forcing gzip on skips the transport checks, not an earlier rejection.
    if (setting == true) return true;
    if (proxy_configured() && !bypasses_proxy(*parts)) return false;
    if (!core::utils::str::trim_ascii_view(env("CLAUDE_CODE_CLIENT_CERT")).empty()
        || !core::utils::str::trim_ascii_view(env("CLAUDE_CODE_CLIENT_KEY")).empty()) {
        return false;
    }
    return true;
}

std::string gzip_body(std::string_view signed_body) {
    std::string text(signed_body);
    if (const auto block = text.find(std::string(kBillingMarker) + " cc_version="); block != std::string::npos) {
        if (const auto cch = text.find(" cch=", block);
            cch != std::string::npos && text.substr(block, cch - block).find('"') == std::string::npos
            && cch + 10 < text.size() && text[cch + 10] == ';') {
            text.replace(cch + 5, 5, "00000");
        }
    }
    static thread_local std::mt19937_64 rng(std::random_device{}());
    std::uniform_int_distribution<std::size_t> length(0, kMaxGzipPadding);
    std::bernoulli_distribution space(0.5);
    text += '\n';
    for (std::size_t i = length(rng); i > 0; --i) text += space(rng) ? ' ' : '\t';

    // Node's zlib.gzip defaults, as Claude Code's runtime writes them: level 6,
    // mtime 0, OS 255.
    return core::utils::gzip::compress(
        text, {.level = 6, .operating_system = core::utils::gzip::kUnknownOperatingSystem});
}

bool is_gzip_rejection(int status, const cpr::Header& headers, std::string_view body) {
    if (status != 400 && status != 403 && status != 415) return false;
    const auto request_id = headers.find("request-id");
    const bool from_anthropic = (request_id != headers.end() && request_id->second.starts_with("req_"))
        || headers.contains("cf-ray");
    if (!from_anthropic) return true;
    return status == 400 && is_invalid_json_report(body.substr(0, 8192));
}

void disable_body_gzip() noexcept {
    g_gzip_rejected.store(true, std::memory_order_relaxed);
}

} // namespace core::llm::claude
