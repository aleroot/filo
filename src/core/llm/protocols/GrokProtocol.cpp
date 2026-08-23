#include "GrokProtocol.hpp"
#include "GrokBuildEndpoint.hpp"
#include "SseUtils.hpp"
#include "core/auth/XaiGrokClientIdentity.hpp"
#include "core/utils/AsciiUtils.hpp"
#include "core/utils/StringUtils.hpp"
#include "core/utils/Uuid.hpp"
#include "../Models.hpp"
#include <simdjson.h>
#include <array>
#include <cctype>
#include <optional>
#include <string>
#include <string_view>

namespace core::llm::protocols {

namespace {

struct GrokResponsesStreamError {
    std::string type;
    std::string message;
};

[[nodiscard]] std::string read_string_field(
    simdjson::dom::object object,
    std::string_view key) {
    std::string_view value;
    if (object[key].get(value) == simdjson::SUCCESS && !value.empty()) {
        return std::string(value);
    }
    return {};
}

[[nodiscard]] std::optional<GrokResponsesStreamError>
parse_grok_responses_stream_error(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    std::string data_scratch;
    if (!sse::parse_event_payload(raw_event, parsed, data_scratch)
        || parsed.is_done) {
        return std::nullopt;
    }

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string padded(parsed.data);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        return std::nullopt;
    }

    std::string event_type(parsed.event);
    if (event_type.empty()) {
        std::string_view payload_type;
        if (doc["type"].get(payload_type) == simdjson::SUCCESS) {
            event_type = std::string(payload_type);
        }
    }
    if (event_type != "response.failed" && event_type != "error") {
        return std::nullopt;
    }

    GrokResponsesStreamError result;
    const auto read_error = [&](simdjson::dom::object error) {
        if (result.type.empty()) result.type = read_string_field(error, "code");
        if (result.type.empty()) result.type = read_string_field(error, "type");
        if (result.message.empty()) result.message = read_string_field(error, "message");
    };

    simdjson::dom::object response;
    if (doc["response"].get(response) == simdjson::SUCCESS) {
        simdjson::dom::object error;
        if (response["error"].get(error) == simdjson::SUCCESS) {
            read_error(error);
        }
        if (result.message.empty()) {
            result.message = read_string_field(response, "message");
        }
    }

    simdjson::dom::object error;
    if (doc["error"].get(error) == simdjson::SUCCESS) {
        read_error(error);
    }
    if (result.type.empty()) result.type = read_string_field(doc, "code");
    if (result.message.empty()) result.message = read_string_field(doc, "message");
    if (result.type.empty()) result.type = event_type;
    if (result.message.empty()) result.message = "Grok response stream failed.";
    return result;
}

[[nodiscard]] bool grok_retry_vetoed(const HttpResponse& response) noexcept {
    for (const auto& [name, value] : response.headers) {
        if (core::utils::ascii::iequals(name, "x-should-retry")
            && core::utils::ascii::iequals(value, "false")) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] bool is_grok_retryable_response(
    const HttpResponse& response) noexcept {
    if (grok_retry_vetoed(response)) return false;
    const int status = response.status_code;
    return status == 408 || status == 409 || status == 429
        || (status >= 500 && status <= 599 && status != 525 && status != 526);
}

[[nodiscard]] bool is_xai_oauth_request(const ChatRequest& request) {
    const auto oauth = request.auth_properties.find("oauth");
    if (oauth == request.auth_properties.end() || oauth->second != "1") return false;
    const auto issuer = request.auth_properties.find("oauth_issuer");
    return issuer == request.auth_properties.end()
        || issuer->second == "https://auth.x.ai";
}

void prepare_grok_session_headers(cpr::Header& headers,
                                  const ChatRequest& request,
                                  std::string_view base_url) {
    if (!is_xai_oauth_request(request)) return;
    if (!grok_build::is_official_proxy(base_url)) {
        // XaiOAuthCredentialSource also serves model discovery, so its generic
        // auth map carries proxy identity headers. Remove them here when the
        // inference destination is not the official Grok CLI proxy.
        core::auth::xai_grok::remove_proxy_identity_headers(headers);
        return;
    }

    const std::string conversation_id = request.session_id.empty()
        ? core::auth::xai_grok::process_agent_id()
        : request.session_id;
    const std::string request_id = request.transport_turn_id.empty()
        ? core::utils::random_uuid_v4()
        : request.transport_turn_id;

    core::auth::xai_grok::apply_proxy_identity_headers(headers);
    headers["x-grok-conv-id"] = conversation_id;
    headers["x-grok-session-id"] = conversation_id;
    headers["x-grok-req-id"] = request_id;
    headers["x-grok-model-override"] = request.model;
    if (const auto user_id = request.auth_properties.find("user_id");
        user_id != request.auth_properties.end() && !user_id->second.empty()) {
        // Chat requests use x-grok-user-id; the billing endpoint expects the
        // same value translated to x-userid by GrokBillingUsageSource.
        headers["x-grok-user-id"] = user_id->second;
    }
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

bool grok_supports_reasoning_effort(std::string_view model) noexcept {
    // Only the grok-3-mini family accepts reasoning_effort.
    // Grok 4 and grok-code models return HTTP 400 if the field is present.
    return model.starts_with("grok-3-mini");
}

bool grok_responses_supports_effort(std::string_view model) noexcept {
    using core::utils::ascii::istarts_with;
    // The Responses-API `reasoning:{effort:...}` object is supported by the
    // Grok 4.6, 4.5, and 4.3 families and by the Grok Build coding model.
    // Grok 4 / 4.1 and the non-reasoning variants are always-on or always-off
    // and reject the control.
    if (istarts_with(model, "grok-4.6") || istarts_with(model, "grok-4-6")) return true;
    if (istarts_with(model, "grok-4.5") || istarts_with(model, "grok-4-5")) return true;
    if (istarts_with(model, "grok-4.3") || istarts_with(model, "grok-4-3")) return true;
    if (istarts_with(model, "grok-build")) return true;
    // grok-composer-* are compatibility aliases of grok-4.5.
    if (istarts_with(model, "grok-composer")) return true;
    return false;
}

bool grok_responses_supports_xhigh_effort(std::string_view model) noexcept {
    using core::utils::ascii::istarts_with;
    return istarts_with(model, "grok-4.6")
        || istarts_with(model, "grok-4-6");
}

namespace {

[[nodiscard]] std::string normalize_grok_responses_effort(
    std::string_view raw_effort,
    std::string_view model) {
    std::string effort = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(raw_effort));
    std::erase_if(effort, [](unsigned char ch) {
        return std::isspace(ch);
    });
    if (effort == "low" || effort == "medium" || effort == "high") {
        return effort;
    }
    if (effort == "max" || effort == "xhigh" || effort == "ultra") {
        return grok_responses_supports_xhigh_effort(model)
            ? "xhigh"
            : "high";
    }
    return {};
}

} // namespace

GrokResponsesProtocol::GrokResponsesProtocol(
    std::string service_tier,
    bool enable_hosted_tools,
    std::string default_effort,
    std::shared_ptr<IGrokBillingUsageSource> billing_usage_source)
    : OpenAIResponsesProtocol(/*include_reasoning_encrypted=*/true,
                              std::move(service_tier))
    , enable_hosted_tools_(enable_hosted_tools)
    , default_effort_(std::move(default_effort))
    , billing_usage_source_(billing_usage_source
          ? std::move(billing_usage_source)
          : make_grok_billing_usage_source()) {}

void GrokProtocol::append_extra_fields(std::string&       payload,
                                        const ChatRequest& req) const {
    if (effort_ == GrokReasoningEffort::None) return;
    if (!grok_supports_reasoning_effort(req.model)) return;

    const char* effort_str = (effort_ == GrokReasoningEffort::Low) ? "low" : "high";
    payload += R"(,"reasoning_effort":")";
    payload += effort_str;
    payload += '"';
}

void GrokProtocol::on_response(const HttpResponse& response) {
    last_rate_limit_ = parse_rate_limit_headers(response.headers);
}

void GrokProtocol::prepare_headers(cpr::Header& headers,
                                   const ChatRequest& request,
                                   std::string_view base_url) {
    prepare_grok_session_headers(headers, request, base_url);
}

void GrokResponsesProtocol::prepare_headers(cpr::Header& headers,
                                            const ChatRequest& request,
                                            std::string_view base_url) {
    prepare_grok_session_headers(headers, request, base_url);
}

void GrokResponsesProtocol::on_response(const HttpResponse& response) {
    OpenAIResponsesProtocol::on_response(response);
    grok_rate_limit_ = OpenAIResponsesProtocol::last_rate_limit();
}

void GrokResponsesProtocol::enrich_rate_limit(
    std::string_view base_url,
    const cpr::Header& request_headers,
    const HttpResponse& response) {
    if (response.status_code != 200 || !billing_usage_source_) return;
    if (auto windows = billing_usage_source_->fetch(base_url, request_headers);
        !windows.empty()) {
        grok_rate_limit_.usage_windows = std::move(windows);
    }
}

std::string GrokResponsesProtocol::serialize(const ChatRequest& request) const {
    SerializationOptions options;
    ChatRequest effective = request;
    if (enable_hosted_tools_) {
        // xAI hosted server-side tools. The Grok Build session proxy resolves
        // these internally, giving the model access to real-time web search
        // and X (Twitter) data that filo's own client-side tools cannot reach.
        // `code_execution` is intentionally omitted to avoid shadowing filo's
        // local exec tool.
        static constexpr std::array<std::string_view, 2> kHostedTools{
            "web_search", "x_search",
        };
        options.hosted_tool_types = kHostedTools;

        // xAI requires tool names to be unique. Its hosted implementation is
        // authoritative for colliding names, so remove only those local tools
        // here instead of changing shared OpenAI/DashScope serialization.
        std::erase_if(effective.tools, [](const Tool& tool) {
            return tool.function.name == "web_search"
                || tool.function.name == "x_search";
        });
    }
    // A session override wins over the provider default. Normalize it here so
    // Filo's provider-neutral "max" setting maps to Grok 4.6's wire-level
    // "xhigh", while older effort-capable Grok models safely fall back high.
    std::string normalized_effort;
    if (grok_responses_supports_effort(request.model)) {
        normalized_effort = normalize_grok_responses_effort(
            request.effort.empty() ? default_effort_ : request.effort,
            request.model);
        if (!normalized_effort.empty()) {
            options.reasoning_effort_override = normalized_effort;
        }
    }
    return serialize_with_options(effective, options);
}

ParseResult GrokResponsesProtocol::parse_event(std::string_view raw_event) {
    if (auto error = parse_grok_responses_stream_error(raw_event)) {
        ParseResult result;
        result.stream_error = true;
        result.retryable_stream_error = true;
        result.stream_error_type = std::move(error->type);
        result.stream_error_message = std::move(error->message);
        return result;
    }
    return OpenAIResponsesProtocol::parse_event(raw_event);
}

std::string GrokProtocol::format_error_message(const HttpResponse& response) const {
    const int code = response.status_code;
    
    // Try to extract xAI's error message from the JSON body
    std::string xai_message;
    if (!response.body.empty()) {
        thread_local simdjson::dom::parser parser;
        simdjson::padded_string ps(response.body);
        simdjson::dom::element doc;
        if (parser.parse(ps).get(doc) == simdjson::SUCCESS) {
            // xAI error format: {"error": {"message": "...", "type": "...", "code": "..."}}
            simdjson::dom::object error_obj;
            if (doc["error"].get(error_obj) == simdjson::SUCCESS) {
                std::string_view msg;
                if (error_obj["message"].get(msg) == simdjson::SUCCESS) {
                    xai_message = std::string(msg);
                }
            }
        }
    }
    
    switch (code) {
        case 400: {
            std::string msg = "[xAI API Error 400: Bad request. ";
            if (!xai_message.empty()) {
                msg += xai_message;
            } else {
                msg += "The request body is malformed or contains invalid parameters.";
            }
            // Add helpful hint for common mistakes
            msg += " Hint: Grok 4 models do NOT support 'reasoning_effort'. "
                   "Use GrokProtocol without reasoning effort for these models.]";
            return msg;
        }
        
        case 401:
            return "[xAI API Error 401: Authentication failed. For a Grok account session, run "
                   "'filo --auth grok' again. For public API access, check XAI_API_KEY and "
                   "visit https://console.x.ai.]";
        
        case 403:
            return "[xAI API Error 403: Permission denied. Your account may not have access to "
                   "this model or feature. Some models require specific access tiers. "
                   "Visit https://console.x.ai to check your subscription.]";
        
        case 404:
            return "[xAI API Error 404: Model or endpoint not found. Verify the model name "
                   "is correct. Available models: grok-code-fast-1, grok-4, grok-3-mini, etc.]";
        
        case 422:
            return "[xAI API Error 422: Unprocessable entity. The request was well-formed but "
                   "contains semantic errors. Check that 'max_tokens' is within model limits "
                   "and 'temperature' is between 0.0 and 2.0.]";
        
        case 429: {
            std::string msg = "[xAI API Error 429: Rate limit exceeded. ";
            if (last_rate_limit_.retry_after > 0) {
                msg += "Retry after " + std::to_string(last_rate_limit_.retry_after) + " seconds. ";
            }
            msg += "Consider reducing request frequency or using a model with higher rate limits. "
                   "Check your usage at https://console.x.ai]";
            return msg;
        }
        
        case 500:
            return "[xAI API Error 500: Internal server error. This is a temporary issue on "
                   "xAI's side. Please retry with exponential backoff.]";
        
        case 502:
            return "[xAI API Error 502: Bad gateway. xAI is experiencing connectivity issues. "
                   "Please retry with exponential backoff.]";
        
        case 503:
            return "[xAI API Error 503: Service unavailable. xAI may be undergoing maintenance. "
                   "Please retry with exponential backoff.]";
        
        case 504:
            return "[xAI API Error 504: Gateway timeout. The request timed out at xAI's edge. "
                   "Consider reducing 'max_tokens' or context size, then retry.]";
        
        case 529:
            return "[xAI API Error 529: Server overloaded. xAI is experiencing high load. "
                   "This is NOT a rate limit error - retrying with exponential backoff is "
                   "recommended. Consider using 'grok-code-fast-1' for lower latency.]";
        
        default:
            if (code >= 500) {
                return "[xAI API Error " + std::to_string(code) + ": Server error. "
                       "This is a temporary issue on xAI's side. Please retry with "
                       "exponential backoff.]";
            } else if (code >= 400) {
                return "[xAI API Error " + std::to_string(code) + ": Client error. "
                       "Please check your request parameters. " +
                       (xai_message.empty() ? "" : "Details: " + xai_message + " ") + 
                       "See https://docs.x.ai for API documentation.]";
            }
            return "[xAI API Error " + std::to_string(code) + "]";
    }
}

bool GrokProtocol::is_retryable(const HttpResponse& response) const noexcept {
    return is_grok_retryable_response(response);
}

bool GrokResponsesProtocol::is_retryable(
    const HttpResponse& response) const noexcept {
    return is_grok_retryable_response(response);
}

// ─────────────────────────────────────────────────────────────────────────────
// Private Helpers
// ─────────────────────────────────────────────────────────────────────────────

int32_t GrokProtocol::parse_int_header(const cpr::Header& headers, 
                                        std::string_view key) noexcept {
    auto it = headers.find(std::string(key));
    if (it == headers.end()) return 0;
    
    // Safe parsing - handle empty strings and non-numeric values
    const std::string& value = it->second;
    if (value.empty()) return 0;
    
    try {
        // Handle potential surrounding whitespace
        size_t start = 0;
        while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
            ++start;
        }
        size_t end = value.size();
        while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
            --end;
        }
        
        return static_cast<int32_t>(std::stoi(value.substr(start, end - start)));
    } catch (...) {
        return 0;
    }
}

/*static*/ RateLimitInfo GrokProtocol::parse_rate_limit_headers(const cpr::Header& headers) noexcept {
    RateLimitInfo info;
    
    // xAI uses standard rate limit headers similar to OpenAI
    // Request-based limits
    info.requests_limit = parse_int_header(headers, "x-ratelimit-limit-requests");
    info.requests_remaining = parse_int_header(headers, "x-ratelimit-remaining-requests");
    
    // Token-based limits  
    info.tokens_limit = parse_int_header(headers, "x-ratelimit-limit-tokens");
    info.tokens_remaining = parse_int_header(headers, "x-ratelimit-remaining-tokens");
    
    // Retry-after (present on 429 responses)
    info.retry_after = parse_int_header(headers, "retry-after");
    if (info.retry_after > 0) {
        info.is_rate_limited = true;
    }
    
    // xAI may also return standard OpenAI-style headers as fallbacks
    if (info.requests_limit == 0) {
        info.requests_limit = parse_int_header(headers, "x-ratelimit-limit");
    }
    if (info.requests_remaining == 0) {
        info.requests_remaining = parse_int_header(headers, "x-ratelimit-remaining");
    }
    
    return info;
}

} // namespace core::llm::protocols
