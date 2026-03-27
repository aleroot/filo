#include "DashScopeProtocol.hpp"
#include "../Models.hpp"
#include <simdjson.h>
#include <cctype>

namespace core::llm::protocols {

namespace {

[[nodiscard]] int32_t parse_int_header(const cpr::Header& headers,
                                        std::string_view   key) noexcept {
    const auto it = headers.find(std::string(key));
    if (it == headers.end() || it->second.empty()) return 0;
    try {
        return static_cast<int32_t>(std::stoi(it->second));
    } catch (...) {
        return 0;
    }
}

// DashScope uses its own header naming in addition to the OpenAI-standard names.
// This helper tries primary first and falls through to fallback.
[[nodiscard]] int32_t parse_first_int_header(const cpr::Header& headers,
                                              std::string_view   primary,
                                              std::string_view   fallback) noexcept {
    if (const int32_t v = parse_int_header(headers, primary); v != 0) return v;
    return parse_int_header(headers, fallback);
}

[[nodiscard]] RateLimitInfo build_rate_limit_info(const cpr::Header& headers,
                                                   int                status_code) noexcept {
    RateLimitInfo info;

    // Request-based limits — DashScope native naming takes precedence.
    info.requests_limit = parse_first_int_header(
        headers, "x-ratelimit-requests-limit", "x-ratelimit-limit-requests");
    info.requests_remaining = parse_first_int_header(
        headers, "x-ratelimit-requests-remaining", "x-ratelimit-remaining-requests");

    // Token-based limits.
    info.tokens_limit = parse_first_int_header(
        headers, "x-ratelimit-tokens-limit", "x-ratelimit-limit-tokens");
    info.tokens_remaining = parse_first_int_header(
        headers, "x-ratelimit-tokens-remaining", "x-ratelimit-remaining-tokens");

    info.retry_after     = parse_int_header(headers, "retry-after");
    info.is_rate_limited = (status_code == 429 || info.retry_after > 0);

    // Avoid false "0 remaining" alarms when the header is simply absent.
    if (info.requests_limit > 0 && info.requests_remaining == 0 && !info.is_rate_limited)
        info.requests_remaining = info.requests_limit;
    if (info.tokens_limit > 0 && info.tokens_remaining == 0 && !info.is_rate_limited)
        info.tokens_remaining = info.tokens_limit;

    return info;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Request building
// ─────────────────────────────────────────────────────────────────────────────

cpr::Header DashScopeProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    auto headers = OpenAIProtocol::build_headers(auth);
    // Activate server-side prompt caching transparently for all requests.
    headers["X-DashScope-CacheControl"] = "enable";
    // Client identification used by DashScope for debugging and telemetry.
    headers["X-DashScope-UserAgent"]    = "filo/0.1 (compatible-mode)";
    return headers;
}

void DashScopeProtocol::append_extra_fields(std::string&       payload,
                                             const ChatRequest& /*req*/) const {
    if (thinking_budget_ <= 0) return;
    payload += R"(,"enable_thinking":true)";
    payload += R"(,"thinking_budget":)";
    payload += std::to_string(thinking_budget_);
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE parsing — reasoning_content support for Qwen3 thinking models
// ─────────────────────────────────────────────────────────────────────────────

ParseResult DashScopeProtocol::parse_event(std::string_view raw_event) {
    // Base parser handles: content, tool_calls, usage, [DONE].
    ParseResult result = OpenAIProtocol::parse_event(raw_event);
    if (result.done) return result;

    // Strip "data: " prefix to reach the raw JSON.
    if (!raw_event.starts_with("data:")) return result;
    std::string_view json_sv = raw_event.substr(5);
    if (!json_sv.empty() && json_sv[0] == ' ') json_sv = json_sv.substr(1);
    if (json_sv == "[DONE]") return result;

    // Second pass: scan for delta.reasoning_content (Qwen3 thinking tokens).
    // Thinking tokens arrive in separate chunks before regular content chunks.
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(json_sv);
    simdjson::dom::element  doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) return result;

    simdjson::dom::array choices;
    if (doc["choices"].get(choices) != simdjson::SUCCESS) return result;

    for (simdjson::dom::element choice : choices) {
        simdjson::dom::object delta;
        if (choice["delta"].get(delta) != simdjson::SUCCESS) continue;

        std::string_view rc;
        if (delta["reasoning_content"].get(rc) != simdjson::SUCCESS || rc.empty()) continue;

        StreamChunk thinking;
        thinking.reasoning_content = std::string(rc);
        result.chunks.push_back(std::move(thinking));
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Response lifecycle
// ─────────────────────────────────────────────────────────────────────────────

void DashScopeProtocol::on_response(const HttpResponse& response) {
    last_rate_limit_ = build_rate_limit_info(response.headers, response.status_code);
}

// ─────────────────────────────────────────────────────────────────────────────
// Error handling
// ─────────────────────────────────────────────────────────────────────────────

std::string DashScopeProtocol::format_error_message(const HttpResponse& response) const {
    const int code = response.status_code;

    // DashScope error body format: {"code":"<ErrorCode>","message":"<detail>","request_id":"<id>"}
    // Fallback: OpenAI-style {"error":{"message":"<detail>"}}
    std::string dash_code;
    std::string dash_message;
    if (!response.body.empty()) {
        thread_local simdjson::dom::parser parser;
        simdjson::padded_string ps(response.body);
        simdjson::dom::element  doc;
        if (parser.parse(ps).get(doc) == simdjson::SUCCESS) {
            std::string_view sv;
            if (doc["code"].get(sv) == simdjson::SUCCESS)    dash_code    = std::string(sv);
            if (doc["message"].get(sv) == simdjson::SUCCESS) dash_message = std::string(sv);
            if (dash_message.empty()) {
                if (doc["error"]["message"].get(sv) == simdjson::SUCCESS)
                    dash_message = std::string(sv);
            }
        }
    }

    const auto detail = [&]() -> std::string {
        if (!dash_code.empty() && !dash_message.empty())
            return " (" + dash_code + ": " + dash_message + ")";
        if (!dash_message.empty())
            return " " + dash_message;
        return {};
    };

    switch (code) {
        case 400:
            return "[DashScope Error 400: Bad request." + detail() +
                   " Check all parameters are valid for this model. "
                   "Note: 'enable_thinking' requires a Qwen3 model.]";

        case 401:
            return "[DashScope Error 401: Authentication failed." + detail() +
                   " Verify DASHSCOPE_API_KEY is set correctly. "
                   "Manage keys at https://bailian.console.aliyun.com]";

        case 403:
            return "[DashScope Error 403: Access denied." + detail() +
                   " Your account may not have access to this model or tier. "
                   "See https://help.aliyun.com/product/610990.html for details.]";

        case 404:
            return "[DashScope Error 404: Model not found." + detail() +
                   " Valid models: qwen3-coder-plus, qwen3-coder-flash, qwen3-max, "
                   "qwen3-plus, qwen3-turbo, qwen3.5-plus.]";

        case 429: {
            std::string msg = "[DashScope Error 429: Rate limit or quota exceeded.";
            msg += detail();
            if (last_rate_limit_.retry_after > 0)
                msg += " Retry after " + std::to_string(last_rate_limit_.retry_after) + "s.";
            msg += " Monitor usage at https://bailian.console.aliyun.com]";
            return msg;
        }

        case 500:
            return "[DashScope Error 500: Internal server error." + detail() +
                   " Temporary Alibaba Cloud issue — retry with backoff.]";

        case 502:
            return "[DashScope Error 502: Bad gateway." + detail() +
                   " DashScope connectivity issue — retry with backoff.]";

        case 503:
            return "[DashScope Error 503: Service unavailable." + detail() +
                   " DashScope may be under maintenance — retry with backoff.]";

        case 504:
            return "[DashScope Error 504: Gateway timeout." + detail() +
                   " Request timed out. Reduce max_tokens or context size and retry.]";

        default:
            if (code >= 500)
                return "[DashScope Error " + std::to_string(code) + ": Server error." +
                       detail() + " Retry with exponential backoff.]";
            if (code >= 400)
                return "[DashScope Error " + std::to_string(code) + ": Client error." +
                       detail() + " See https://help.aliyun.com/product/610990.html]";
            return "[DashScope Error " + std::to_string(code) + detail() + "]";
    }
}

bool DashScopeProtocol::is_retryable(const HttpResponse& response) const noexcept {
    return response.status_code == 429 ||
           response.status_code == 500 ||
           response.status_code == 502 ||
           response.status_code == 503 ||
           response.status_code == 504;
}

} // namespace core::llm::protocols
