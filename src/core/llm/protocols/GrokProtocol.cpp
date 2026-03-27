#include "GrokProtocol.hpp"
#include "../Models.hpp"
#include <simdjson.h>
#include <cctype>
#include <string_view>

namespace core::llm::protocols {

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

bool grok_supports_reasoning_effort(std::string_view model) noexcept {
    // Only the grok-3-mini family accepts reasoning_effort.
    // Grok 4 and grok-code models return HTTP 400 if the field is present.
    return model.starts_with("grok-3-mini");
}

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
            return "[xAI API Error 401: Authentication failed. Please check your XAI_API_KEY "
                   "environment variable or configuration. Visit https://console.x.ai to verify "
                   "your API key is valid and active.]";
        
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
    // xAI retryable status codes (aligned with OpenAI + xAI-specific 529)
    return response.status_code == 429 ||  // Rate limit
           response.status_code == 500 ||  // Internal server error
           response.status_code == 502 ||  // Bad gateway
           response.status_code == 503 ||  // Service unavailable
           response.status_code == 504 ||  // Gateway timeout
           response.status_code == 529;    // xAI-specific: overloaded
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
