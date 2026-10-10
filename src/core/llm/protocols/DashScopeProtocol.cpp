#include "DashScopeProtocol.hpp"
#include "SseUtils.hpp"
#include "OpenAIUsage.hpp"
#include "../Models.hpp"
#include "../ToolCallAssembly.hpp"
#include "../QwenModelTraits.hpp"
#include "../../utils/StringUtils.hpp"
#include "../../utils/AsciiUtils.hpp"
#include "core/utils/TimeUtils.hpp"
#include "../../utils/JsonUtils.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <simdjson.h>

namespace core::llm::protocols {

namespace {

// Keep provisional tool fragments inside Qwen's wire adapter until the
// request terminates successfully. The shared agent never sees partial calls.
void collect_tool_calls(ParseResult& result, std::vector<ToolCall>& pending) {
    std::erase_if(result.chunks, [&](StreamChunk& chunk) {
        if (chunk.tools.empty()) return false;
        for (const auto& call : chunk.tools) merge_tool_call_fragment(pending, call);
        chunk.tools.clear();
        return chunk.content.empty() && chunk.reasoning_content.empty()
            && chunk.continuation_items.empty() && !chunk.is_final;
    });
}

void complete_tool_calls(ParseResult& result, std::vector<ToolCall>& pending) {
    simdjson::dom::parser parser;
    for (const auto& call : pending) {
        if (call.function.arguments.empty()) continue;
        simdjson::padded_string json(call.function.arguments);
        simdjson::dom::element arguments;
        if (parser.parse(json).get(arguments) != simdjson::SUCCESS) {
            result.incomplete_tool_call = true;
            break;
        }
    }
    if (!result.incomplete_tool_call && !pending.empty()) {
        result.chunks.push_back(StreamChunk::make_tools(std::move(pending)));
    }
    pending.clear();
}

enum class ThinkingMode { None, Toggle, QwenEffort, VendorEffort };
struct ThinkingPolicy {
    ThinkingMode mode = ThinkingMode::None;
    bool required = false;
    bool low_effort = false;
    bool xhigh_maps_high = false;
};

[[nodiscard]] std::string wire_model(std::string_view model) {
    std::string id = core::utils::str::to_lower_ascii_copy(core::utils::str::trim_ascii_view(model));
    if (const auto slash = id.rfind('/'); slash != std::string::npos) id.erase(0, slash + 1);
    return id;
}

[[nodiscard]] ThinkingPolicy thinking_policy(std::string_view model) {
    const std::string id = wire_model(model);
    if (qwen_model_supports_tiered_effort(id)) {
        return {ThinkingMode::QwenEffort, id.starts_with("qwen3.8-max"), true};
    }
    if (id.starts_with("qwen3-coder")) return {};
    if (id.starts_with("glm-5.3")) return {ThinkingMode::VendorEffort, true, true};
    if (id.starts_with("glm-5")) return {ThinkingMode::VendorEffort};
    if (id.starts_with("deepseek-v4")) {
        const bool native_low = id.starts_with("deepseek-v4.1")
            || id.starts_with("deepseek-v4-flash-0731")
            || id.starts_with("deepseek-v4-pro-0813");
        return {ThinkingMode::VendorEffort, false, native_low, native_low};
    }
    if (id.starts_with("kimi-k3")) return {ThinkingMode::VendorEffort, false, true};
    if (id.starts_with("kimi-k2.7-code") || id.starts_with("kimi-k2-thinking")) {
        return {ThinkingMode::Toggle, true};
    }
    if (id.starts_with("kimi-k2.") || id.starts_with("glm-4.")
        || id == "coder-model" || id.starts_with("qwen3")
        || id.starts_with("qwen-plus") || id.starts_with("qwen-flash")
        || id.starts_with("qwen-turbo")) {
        return {ThinkingMode::Toggle};
    }
    if (id.starts_with("qwq") || id.starts_with("qvq")) return {ThinkingMode::Toggle, true};
    return {};
}

[[nodiscard]] ReasoningCapabilities qwen_reasoning_capabilities(std::string_view model) {
    const auto policy = thinking_policy(model);
    if (policy.mode == ThinkingMode::None) return {};
    ReasoningCapabilities caps{policy.required ? ReasoningCapability::Required
                                               : ReasoningCapability::Disable};
    if (policy.mode == ThinkingMode::Toggle) return caps | ReasoningCapability::ToggleOnly;
    caps = caps | ReasoningCapability::Effort | ReasoningCapability::MapsMinimalToLow;
    if (policy.mode == ThinkingMode::QwenEffort) {
        return caps | ReasoningCapability::XHighEffort | ReasoningCapability::MapsHighToXHigh;
    }
    caps = caps | ReasoningCapability::MaxEffort | ReasoningCapability::MapsMediumToHigh;
    if (!policy.low_effort) caps = caps | ReasoningCapability::MapsLowToHigh;
    return caps | (policy.xhigh_maps_high ? ReasoningCapability::MapsXHighToHigh
                                         : ReasoningCapability::MapsXHighToMax);
}

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

[[nodiscard]] std::optional<std::string_view> find_header_case_insensitive(
    const cpr::Header& headers,
    std::string_view key) noexcept {
    for (const auto& [header_name, header_val] : headers) {
        if (core::utils::ascii::iequals(header_name, key)) {
            return std::string_view{header_val};
        }
    }
    return std::nullopt;
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
    if (auto r_reset = find_header_case_insensitive(headers, "x-ratelimit-requests-reset")) {
        info.requests_reset = core::utils::time::parse_timestamp_or_duration(*r_reset);
    } else if (auto r_reset2 = find_header_case_insensitive(headers, "x-ratelimit-reset-requests")) {
        info.requests_reset = core::utils::time::parse_timestamp_or_duration(*r_reset2);
    }

    // Token-based limits.
    info.tokens_limit = parse_first_int_header(
        headers, "x-ratelimit-tokens-limit", "x-ratelimit-limit-tokens");
    info.tokens_remaining = parse_first_int_header(
        headers, "x-ratelimit-tokens-remaining", "x-ratelimit-remaining-tokens");
    if (auto t_reset = find_header_case_insensitive(headers, "x-ratelimit-tokens-reset")) {
        info.tokens_reset = core::utils::time::parse_timestamp_or_duration(*t_reset);
    } else if (auto t_reset2 = find_header_case_insensitive(headers, "x-ratelimit-reset-tokens")) {
        info.tokens_reset = core::utils::time::parse_timestamp_or_duration(*t_reset2);
    }

    info.retry_after     = parse_int_header(headers, "retry-after");
    info.is_rate_limited = (status_code == 429 || info.retry_after > 0);

    // Avoid false "0 remaining" alarms when the header is simply absent.
    if (info.requests_limit > 0 && info.requests_remaining == 0 && !info.is_rate_limited)
        info.requests_remaining = info.requests_limit;
    if (info.tokens_limit > 0 && info.tokens_remaining == 0 && !info.is_rate_limited)
        info.tokens_remaining = info.tokens_limit;

    return info;
}

[[nodiscard]] std::string normalize_dashscope_effort(
    std::string_view configured, const ThinkingPolicy& policy) {
    std::string effort = core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(configured));
    if (effort == "off" || effort == "disabled") effort = "none";
    if (effort == "none") return policy.required ? "low" : "none";
    if (effort == "ultra") effort = "max";
    if (effort != "minimal" && effort != "low" && effort != "medium"
        && effort != "high" && effort != "xhigh" && effort != "max") return {};
    if (policy.mode == ThinkingMode::QwenEffort) {
        // Use QwenCloud's documented low/medium/xhigh ladder. In particular,
        // forwarding Filo's max/minimal verbatim can produce a persistent 400.
        if (effort == "minimal") return "low";
        if (effort == "high" || effort == "max") return "xhigh";
    }
    if (policy.mode == ThinkingMode::VendorEffort) {
        if (effort == "minimal") effort = policy.low_effort ? "low" : "high";
        if (effort == "low" && !policy.low_effort) effort = "high";
        if (effort == "medium") effort = "high";
        if (effort == "xhigh") effort = policy.xhigh_maps_high ? "high" : "max";
    }
    return effort;
}

[[nodiscard]] bool supports_dashscope_structured_cache_content(
    const ChatRequest& request) {
    const std::string model = wire_model(request.model);
    if (!model.starts_with("glm-")) return true;
    if (!request.tools.empty()) return true;
    return std::ranges::any_of(request.messages, [](const Message& message) {
        return message.role == "tool" || !message.tool_calls.empty();
    });
}

enum class TokenPlanLimitKind {
    Unknown,
    RequestRate,
    PlanQuota,
};

struct DashScopeErrorDetails {
    std::string code;
    std::string message;
    TokenPlanLimitKind token_plan_limit = TokenPlanLimitKind::Unknown;
};

[[nodiscard]] TokenPlanLimitKind classify_token_plan_limit(
    std::string_view code,
    std::string_view message) {
    const std::string normalized_code =
        core::utils::str::to_lower_ascii_copy(code);
    const std::string normalized_message =
        core::utils::str::to_lower_ascii_copy(message);

    // Token Plan Personal Edition documents these allocation errors as an
    // exhausted 5-hour or 7-day Credits window.
    if (normalized_code == "throttling.allocationquota"
        || normalized_code == "insufficient_quota"
        || normalized_message.find("allocated quota exceeded")
            != std::string::npos
        || normalized_message.find("exceeded your current quota")
            != std::string::npos
        // Qwen Code also recognizes the weekly-quota message carrying a
        // reset time, even when its error code is only "429".
        || (normalized_message.find("quota") != std::string::npos
            && (normalized_message.find("exhausted") != std::string::npos
                || normalized_message.find("exceeded") != std::string::npos)
            && (normalized_message.find("will reset") != std::string::npos
                || normalized_message.find("reset at") != std::string::npos))) {
        return TokenPlanLimitKind::PlanQuota;
    }

    // Do not classify the broad `Throttling.*` namespace as plan exhaustion:
    // RateQuota and BurstRate are short-lived request throttles.
    if (normalized_code == "throttling.ratequota"
        || normalized_code == "limitrequests"
        || normalized_code == "limit_requests"
        || normalized_code == "throttling.burstrate"
        || normalized_code == "limit_burst_rate"
        || normalized_message.find("requests rate limit exceeded")
            != std::string::npos
        || normalized_message.find("request rate increased too quickly")
            != std::string::npos) {
        return TokenPlanLimitKind::RequestRate;
    }
    return TokenPlanLimitKind::Unknown;
}

[[nodiscard]] DashScopeErrorDetails parse_dashscope_error(
    std::string_view body) {
    DashScopeErrorDetails details;
    if (body.empty()) return details;

    thread_local simdjson::dom::parser parser;
    const core::utils::json::ParserRetentionGuard parser_guard{parser};
    simdjson::padded_string ps(body);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) {
        details.message = body;
        details.token_plan_limit = classify_token_plan_limit({}, body);
        return details;
    }

    std::string_view value;
    if (doc["code"].get(value) == simdjson::SUCCESS) {
        details.code = value;
    } else if (doc["error"]["code"].get(value) == simdjson::SUCCESS) {
        details.code = value;
    } else if (doc["response"]["error"]["code"].get(value) == simdjson::SUCCESS) {
        details.code = value;
    }
    if (doc["message"].get(value) == simdjson::SUCCESS) {
        details.message = value;
    } else if (doc["error"]["message"].get(value) == simdjson::SUCCESS) {
        details.message = value;
    } else if (doc["response"]["error"]["message"].get(value) == simdjson::SUCCESS) {
        details.message = value;
    }

    details.token_plan_limit =
        classify_token_plan_limit(details.code, details.message);
    return details;
}

void mark_rate_limit(RateLimitInfo& info, const DashScopeErrorDetails& error) {
    info.is_rate_limited = true;
    info.unified_status = "rate_limited";
    if (error.token_plan_limit == TokenPlanLimitKind::PlanQuota) info.unified_overage_status = "rejected";
}

ParseResult stream_failure(const DashScopeErrorDetails& error, std::string type,
                           RateLimitInfo& rate_limit, bool& saw_limit,
                           ParseResult result = {}) {
    const auto message = core::utils::str::to_lower_ascii_copy(error.message);
    const bool throttle = error.token_plan_limit == TokenPlanLimitKind::RequestRate
        || message.find("rate limit") != std::string::npos
        || message.find("throttl") != std::string::npos || message.find("tpm") != std::string::npos;
    if (error.token_plan_limit != TokenPlanLimitKind::Unknown || throttle) {
        rate_limit = {};
        mark_rate_limit(rate_limit, error);
        saw_limit = true;
    }
    result.stream_error = true;
    result.retryable_stream_error = throttle && error.token_plan_limit != TokenPlanLimitKind::PlanQuota;
    result.stream_error_type = std::move(type);
    result.stream_error_message = error.message.empty() ? "Qwen generation failed." : error.message;
    return result;
}

bool retryable_response(const HttpResponse& response) noexcept {
    if (response.status_code == 429) {
        try {
            return parse_dashscope_error(response.body).token_plan_limit != TokenPlanLimitKind::PlanQuota;
        } catch (...) { return true; }
    }
    return response.status_code == 500 || response.status_code == 502
        || response.status_code == 503 || response.status_code == 504;
}

[[nodiscard]] std::string format_dashscope_error(
    const HttpResponse& response,
    DashScopeDeployment deployment,
    int retry_after_seconds) {
    const int code = response.status_code;
    const bool token_plan = deployment == DashScopeDeployment::TokenPlan;

    const DashScopeErrorDetails error =
        parse_dashscope_error(response.body);

    const auto detail = [&]() -> std::string {
        if (!error.code.empty() && !error.message.empty()) {
            return " (" + error.code + ": " + error.message + ")";
        }
        return error.message.empty()
            ? std::string{}
            : " " + error.message;
    };

    switch (code) {
        case 400:
            return std::string(token_plan ? "[Qwen Token Plan Error 400: Bad request."
                                          : "[DashScope Error 400: Bad request.") + detail()
                + " Check model-specific parameters, reasoning controls, and tool types.]";
        case 401:
            if (token_plan) {
                return "[Qwen Token Plan Error 401: Authentication failed." + detail()
                    + " Use the dedicated Token Plan key with the token-plan base URL; "
                      "pay-as-you-go and Coding Plan keys are not interchangeable. "
                      "Manage it at https://home.qwencloud.com/token-plan]";
            }
            return "[DashScope Error 401: Authentication failed." + detail()
                + " Verify QWEN_API_KEY (public DashScope) or "
                  "QWEN_CODING_PLAN_API_KEY (Coding Plan) matches this endpoint.]";
        case 403:
            return std::string(token_plan ? "[Qwen Token Plan Error 403: Access denied."
                                          : "[DashScope Error 403: Access denied.") + detail()
                + (token_plan
                       ? " Confirm the subscription is active and the base URL contains token-plan.]"
                       : " Your account may not have access to this model or tier.]");
        case 404:
            return std::string(token_plan
                    ? "[Qwen Token Plan Error 404: Model or endpoint not found."
                    : "[DashScope Error 404: Model not found.") + detail()
                + (token_plan
                       ? " Check the exact model ID and use /compatible-mode/v1.]"
                       : " Check the model ID in the Qwen Cloud model catalog.]");
        case 429: {
            // Distinguish a hard plan-quota block (5h/7d window exhausted) from
            // a transient per-minute request rate limit. The two call for very
            // different user action, and the Token Plan surfaces both as 429.
            const bool quota = error.token_plan_limit
                == TokenPlanLimitKind::PlanQuota;
            const bool request_rate = error.token_plan_limit
                == TokenPlanLimitKind::RequestRate;
            std::string message;
            if (!token_plan) {
                message = "[DashScope Error 429: Rate limit or quota exceeded.";
            } else if (quota) {
                message = "[Qwen Token Plan Error 429: Credits quota exceeded.";
            } else if (request_rate) {
                message = "[Qwen Token Plan Error 429: Request rate limit exceeded.";
            } else {
                message = "[Qwen Token Plan Error 429: Rate limit or quota exceeded.";
            }
            message += detail();
            if (retry_after_seconds > 0) {
                message += " Retry after " + std::to_string(retry_after_seconds) + "s.";
            }
            if (token_plan) {
                if (quota) {
                    message +=
                        " The 5-hour or 7-day window is exhausted — the service "
                        "resumes when that window resets. Manage usage at "
                        "https://home.qwencloud.com/token-plan]";
                } else if (request_rate) {
                    message += " Reduce request frequency and retry.]";
                } else {
                    message +=
                        " Retry with backoff; if the error persists, check "
                        "Credits at https://home.qwencloud.com/token-plan]";
                }
            } else {
                message += " Check Qwen Cloud usage in the console.]";
            }
            return message;
        }
        case 500:
            return "[DashScope Error 500: Internal server error." + detail()
                + " Temporary Alibaba Cloud issue — retry with backoff.]";
        case 502:
            return "[DashScope Error 502: Bad gateway." + detail()
                + " DashScope connectivity issue — retry with backoff.]";
        case 503:
            return "[DashScope Error 503: Service unavailable." + detail()
                + " DashScope may be under maintenance — retry with backoff.]";
        case 504:
            return "[DashScope Error 504: Gateway timeout." + detail()
                + " Request timed out. Reduce max_tokens or context size and retry.]";
        default:
            if (code >= 500) {
                return "[DashScope Error " + std::to_string(code) + ": Server error."
                    + detail() + " Retry with exponential backoff.]";
            }
            if (code >= 400) {
                return "[DashScope Error " + std::to_string(code) + ": Client error."
                    + detail() + " See https://help.aliyun.com/product/610990.html]";
            }
            return "[DashScope Error " + std::to_string(code) + detail() + "]";
    }
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
    headers["X-DashScope-AuthType"]     = "openai";
    return headers;
}

void DashScopeProtocol::prepare_request(ChatRequest& request) {
    if (lifecycle_.prepare) lifecycle_.prepare(request);
}

ReasoningCapabilities DashScopeProtocol::reasoning_capabilities(std::string_view model) const noexcept {
    return qwen_reasoning_capabilities(model);
}

std::string DashScopeProtocol::serialize(const ChatRequest& req) const {
    Serializer::Options options;
    // Qwen requires this field on the assistant message when a thinking tool
    // call is followed by tool results.
    options.reasoning_content_policy =
        Serializer::ReasoningContentPolicy::NonEmptyOwned;
    options.reasoning_protocol = std::string(name());
    // Qwen requires content to be an empty string, rather than null, for
    // tool-call-only assistant messages. This matters when history originated
    // from another provider (for example Claude), whose tool turn has no
    // visible text. Third-party models on DashScope retain the shared default.
    options.empty_assistant_content_policy =
        qwen_model_requires_nonnull_assistant_content(req.model)
        ? Serializer::EmptyAssistantContentPolicy::EmptyString
        : Serializer::EmptyAssistantContentPolicy::EmptyStringWhenReasoning;
    if (supports_dashscope_structured_cache_content(req)) {
        options.prompt_cache = {
            .first_system_message = true,
            .latest_message = req.stream,
            .last_tool = req.stream,
        };
    }

    // Qwen's OpenAI-compatible endpoint rejects a request at 256 images.
    // Match Qwen Code's 250-image margin while leaving the caller's durable
    // history intact: only an outbound copy is compacted, and only when needed.
    constexpr std::size_t kMaxRequestImages = 250;
    std::string payload;
    if (const auto images = count_image_inputs(req); images > kMaxRequestImages) {
        ChatRequest bounded = req;
        evict_oldest_image_inputs(bounded, images - kMaxRequestImages);
        payload = Serializer::serialize(bounded, options);
    } else {
        payload = Serializer::serialize(req, options);
    }
    if (payload.ends_with('}')) {
        payload.pop_back();
        append_extra_fields(payload, req);
        if (qwen_model_requires_nonnull_assistant_content(req.model)
            && (!req.session_id.empty() || !req.transport_turn_id.empty())) {
            payload += R"(,"metadata":{)";
            bool has_metadata_field = false;
            if (!req.session_id.empty()) {
                payload += R"("sessionId":")"
                    + core::utils::escape_json_string(req.session_id) + '"';
                has_metadata_field = true;
            }
            if (!req.transport_turn_id.empty()) {
                if (has_metadata_field) payload += ',';
                payload += R"("promptId":")"
                    + core::utils::escape_json_string(req.transport_turn_id) + '"';
            }
            payload += '}';
        }
        if (req.stream) {
            payload += R"(,"stream_options":{"include_usage":true})";
        }
        payload += '}';
    }
    return payload;
}

void DashScopeProtocol::append_extra_fields(std::string&       payload,
                                             const ChatRequest& req) const {
    // Qwen Code always sends vl_high_resolution_images for DashScope vision
    // models. Omitting it silently downgrades screenshot / image-tool quality.
    if (qwen_model_supports_vision(req.model)) {
        payload += R"(,"vl_high_resolution_images":true)";
    }

    const auto policy = thinking_policy(req.model);
    const bool mandatory = policy.required;
    const std::string effort = normalize_dashscope_effort(
        req.effort.empty() ? std::string_view(default_effort_) : std::string_view(req.effort), policy);
    if (policy.mode != ThinkingMode::None) {
        if (effort == "none") {
            payload += R"(,"enable_thinking":false)";
        } else if (policy.mode == ThinkingMode::QwenEffort) {
            if (req.effort.empty() && thinking_budget_ > 0) {
                payload += R"(,"enable_thinking":true,"thinking_budget":)" + std::to_string(thinking_budget_);
            } else if (!effort.empty()) {
                payload += R"(,"reasoning_effort":")" + effort + '"';
            }
        } else {
            if (!effort.empty() || thinking_budget_ > 0 || mandatory) {
                payload += R"(,"enable_thinking":true)";
            }
            if (policy.mode == ThinkingMode::VendorEffort && !effort.empty()) {
                payload += R"(,"reasoning_effort":")" + effort + '"';
            } else if (thinking_budget_ > 0) {
                payload += R"(,"thinking_budget":)" + std::to_string(thinking_budget_);
            }
        }
    }
    const std::string id = wire_model(req.model);
    if (id.starts_with("glm-")) payload += R"(,"clear_thinking":false)";
    if (id.starts_with("kimi-k2.") || id.starts_with("kimi-k3")) {
        payload += R"(,"preserve_thinking":true)";
    }

    // Qwen Code stamps preserve_thinking on every DashScope chat request for
    // the Qwen family so multi-turn tool use keeps the thinking trace.
    if (qwen_model_supports_preserve_thinking(req.model)) {
        payload += R"(,"preserve_thinking":true)";
    }
}

DashScopeResponsesProtocol::DashScopeResponsesProtocol()
    : DashScopeResponsesProtocol(Options{}) {}

DashScopeResponsesProtocol::DashScopeResponsesProtocol(Options options)
    : OpenAIResponsesProtocol(/*include_reasoning_encrypted=*/false)
    , options_(std::move(options)) {}

void DashScopeResponsesProtocol::prepare_request(ChatRequest& request) {
    if (options_.lifecycle.prepare) options_.lifecycle.prepare(request);
    session_id_ = request.session_id;
    OpenAIResponsesProtocol::prepare_request(request);
}

ReasoningCapabilities DashScopeResponsesProtocol::reasoning_capabilities(
    std::string_view model) const noexcept {
    return qwen_reasoning_capabilities(model);
}

std::string DashScopeResponsesProtocol::serialize(const ChatRequest& req) const {
    ChatRequest effective = req;
    if (!effective.previous_response_id.empty()) {
        // Qwen appends `input` to the context identified by previous_response_id.
        // Send only the messages produced since the previous model response to
        // avoid duplicating the complete Filo transcript on every turn. System
        // instructions are retained because Qwen does not inherit them.
        auto last_assistant = effective.messages.end();
        for (auto it = effective.messages.begin(); it != effective.messages.end(); ++it) {
            if (it->role == "assistant") last_assistant = it;
        }
        if (last_assistant != effective.messages.end()) {
            std::vector<Message> incremental;
            for (const auto& message : effective.messages) {
                if (message.role == "system") incremental.push_back(message);
            }
            incremental.insert(
                incremental.end(), std::next(last_assistant), effective.messages.end());
            effective.messages = std::move(incremental);
        }
    }

    const auto policy = thinking_policy(req.model);
    const std::string effort = normalize_dashscope_effort(
        req.effort.empty() ? std::string_view(options_.default_effort) : std::string_view(req.effort),
        policy);
    const std::string effective_effort = effort.empty() ? "xhigh" : effort;
    const auto model_reasoning =
        qwen_reasoning_capabilities(req.model);
    const std::optional<std::string_view> reasoning_effort =
        model_reasoning.supports_effort()
        ? std::optional<std::string_view>{effective_effort}
        : std::nullopt;
    static constexpr std::array<std::string_view, 5> kHostedTools{
        "web_search",
        "code_interpreter",
        "web_extractor",
        "web_search_image",
        "image_search",
    };
    const bool enable_hosted_tools =
        options_.enable_hosted_tools
        && qwen_model_supports_token_plan_hosted_tools(req.model);
    const std::span<const std::string_view> hosted_tools = enable_hosted_tools
        ? std::span<const std::string_view>{kHostedTools}
        : std::span<const std::string_view>{};
    std::string payload = serialize_with_options(
        effective,
        SerializationOptions{
            .include_store = false,
            .include_prompt_cache_key = false,
            .include_response_include = false,
            .reasoning_effort_override = reasoning_effort,
            .hosted_tool_types = hosted_tools,
        });
    if (policy.mode == ThinkingMode::Toggle && (!effort.empty() || policy.required)) {
        payload.pop_back();
        payload += effort == "none" ? R"(,"enable_thinking":false})" : R"(,"enable_thinking":true})";
    }
    return payload;
}

cpr::Header DashScopeResponsesProtocol::build_headers(
    const core::auth::AuthInfo& auth) const {
    auto headers = OpenAIResponsesProtocol::build_headers(auth);
    headers["X-DashScope-Session-Cache"] = "enable";
    headers["X-DashScope-UserAgent"] = "filo/0.1 (responses; token-plan)";
    return headers;
}

std::string DashScopeResponsesProtocol::format_error_message(
    const HttpResponse& response) const {
    return format_dashscope_error(
        response, options_.deployment, last_rate_limit().retry_after);
}

void DashScopeResponsesProtocol::on_response(const HttpResponse& response) {
    if (response.status_code >= 400 && options_.lifecycle.failure) {
        options_.lifecycle.failure(response.status_code, parse_dashscope_error(response.body).message);
    }
    OpenAIResponsesProtocol::on_response(response);
    if (stream_rate_limit_seen_ && response.status_code == 200) {
        stream_rate_limit_seen_ = false;
        return;
    }
    stream_rate_limit_seen_ = false;
    response_rate_limit_ = OpenAIResponsesProtocol::last_rate_limit();
    if (response.status_code == 429) mark_rate_limit(response_rate_limit_, parse_dashscope_error(response.body));
}

ParseResult DashScopeResponsesProtocol::parse_event(std::string_view raw) {
    sse::ParsedEventView envelope;
    if (!sse::parse_event_payload(raw, envelope) || envelope.is_done) {
        auto result = OpenAIResponsesProtocol::parse_event(raw);
        if (result.done) complete_tool_calls(result, pending_tool_calls_);
        return result;
    }
    simdjson::dom::parser parser;
    simdjson::padded_string json(envelope.data);
    simdjson::dom::element doc;
    if (parser.parse(json).get(doc) != simdjson::SUCCESS) return {};
    std::string_view type = envelope.event;
    if (type.empty()) (void)doc["type"].get(type);
    simdjson::dom::object bare_error;
    if (type.empty() && doc["error"].get(bare_error) == simdjson::SUCCESS) type = "error";
    if (type == "error" || type == "response.failed" || type == "response.incomplete") {
        ParseResult result;
        simdjson::dom::object usage;
        if (doc["response"]["usage"].get(usage) == simdjson::SUCCESS) {
            (void)parse_openai_usage(usage, result);
        }
        if (result.prompt_tokens > 0 && options_.lifecycle.prompt_usage) options_.lifecycle.prompt_usage(result.prompt_tokens);
        // Never release client calls from an interrupted response.
        result.incomplete_tool_call = !pending_tool_calls_.empty();
        pending_tool_calls_.clear();
        // An interrupted response cannot extend the preceding successful
        // response's chain. Clear only this session so recovery replays input.
        {
            std::lock_guard lock(shared_state_->mutex);
            shared_state_->sessions[session_id_].previous_response_id.clear();
        }
        if (type == "response.incomplete") {
            std::string_view reason;
            (void)doc["response"]["incomplete_details"]["reason"].get(reason);
            if (reason == "max_output_tokens" || reason == "length") {
                result.done = true;
                result.stop_reason = "max_tokens";
                return result;
            }
        }
        const auto error = parse_dashscope_error(envelope.data);
        return stream_failure(error, error.code.empty() ? std::string(type) : error.code,
                              response_rate_limit_, stream_rate_limit_seen_, std::move(result));
    }
    auto result = OpenAIResponsesProtocol::parse_event(raw);
    collect_tool_calls(result, pending_tool_calls_);
    if (result.prompt_tokens > 0 && options_.lifecycle.prompt_usage) options_.lifecycle.prompt_usage(result.prompt_tokens);
    result.stream_started = type.starts_with("response.");
    if (result.done) {
        result.stop_reason = "stop";
        complete_tool_calls(result, pending_tool_calls_);
    }
    return result;
}

void DashScopeResponsesProtocol::reset_state() {
    OpenAIResponsesProtocol::reset_state();
    response_rate_limit_ = {};
    stream_rate_limit_seen_ = false;
    pending_tool_calls_.clear();
}

bool DashScopeResponsesProtocol::is_retryable(const HttpResponse& response) const noexcept {
    return retryable_response(response);
}

DashScopeTokenPlanProtocol::DashScopeTokenPlanProtocol()
    : DashScopeTokenPlanProtocol(Options{}) {}

DashScopeTokenPlanProtocol::DashScopeTokenPlanProtocol(Options options)
    : DashScopeTokenPlanProtocol(options, std::make_unique<DashScopeProtocol>(
          options.thinking_budget, options.default_effort,
          DashScopeDeployment::TokenPlan, options.lifecycle)) {}

DashScopeTokenPlanProtocol::DashScopeTokenPlanProtocol(
    Options options,
    std::unique_ptr<ApiProtocolBase> delegate)
    : options_(std::move(options))
    , delegate_(std::move(delegate)) {}

void DashScopeTokenPlanProtocol::prepare_request(ChatRequest& request) {
    delegate_->prepare_request(request);
}

std::string DashScopeTokenPlanProtocol::serialize(
    const ChatRequest& req) const {
    return delegate_->serialize(req);
}

cpr::Header DashScopeTokenPlanProtocol::build_headers(
    const core::auth::AuthInfo& auth) const {
    return delegate_->build_headers(auth);
}

std::string DashScopeTokenPlanProtocol::build_url(
    std::string_view base_url,
    std::string_view model) const {
    return delegate_->build_url(base_url, model);
}

std::string_view DashScopeTokenPlanProtocol::event_delimiter() const noexcept {
    return delegate_->event_delimiter();
}

ParseResult DashScopeProtocol::parse_event(
    std::string_view raw_event) {
    sse::ParsedEventView parsed;
    if (sse::parse_event_payload(raw_event, parsed) && !parsed.is_done) {
        DashScopeErrorDetails error;
        std::string error_type;
        bool is_error = parsed.event == "error";

        if (is_error) {
            error = parse_dashscope_error(parsed.data);
            error_type = error.code.empty() ? "error" : error.code;
        } else {
            thread_local simdjson::dom::parser parser;
            const core::utils::json::ParserRetentionGuard parser_guard{parser};
            simdjson::padded_string json(parsed.data);
            simdjson::dom::element doc;
            if (parser.parse(json).get(doc) == simdjson::SUCCESS) {
                simdjson::dom::object error_object;
                if (doc["error"].get(error_object) == simdjson::SUCCESS) {
                    // OpenAI SDK streams throw on data:{"error":{...}} even
                    // without an `event: error` envelope.
                    is_error = true;
                    error = parse_dashscope_error(parsed.data);
                    error_type = error.code.empty() ? "error" : error.code;
                }
                std::string_view code;
                simdjson::dom::array normal_choices;
                if (!is_error && doc["code"].get(code) == simdjson::SUCCESS
                    && !code.empty() && code != "0"
                    && doc["choices"].get(normal_choices) != simdjson::SUCCESS) {
                    is_error = true;
                    error = parse_dashscope_error(parsed.data);
                    error_type = error.code;
                }
                simdjson::dom::array choices;
                if (doc["choices"].get(choices) == simdjson::SUCCESS) {
                    for (const auto choice : choices) {
                        std::string_view finish_reason;
                        if (choice["finish_reason"].get(finish_reason)
                                != simdjson::SUCCESS
                            || finish_reason != "error_finish") {
                            continue;
                        }
                        is_error = true;
                        error_type = "error_finish";
                        std::string_view content;
                        if (choice["delta"]["content"].get(content)
                            == simdjson::SUCCESS) {
                            error.message = content;
                        }
                        error.token_plan_limit = classify_token_plan_limit(
                            error.code, error.message);
                        break;
                    }
                }
            }
        }

        if (is_error) return stream_failure(error, std::move(error_type), last_rate_limit_, stream_rate_limit_seen_);
    }
    auto result = OpenAIProtocol::parse_event(raw_event);
    collect_tool_calls(result, pending_tool_calls_);
    if (result.prompt_tokens > 0 && lifecycle_.prompt_usage) lifecycle_.prompt_usage(result.prompt_tokens);
    if (parsed.is_done) {
        result.stop_reason = stop_reason_;
        complete_tool_calls(result, pending_tool_calls_);
        return result;
    }
    thread_local simdjson::dom::parser finish_parser;
    const core::utils::json::ParserRetentionGuard guard{finish_parser};
    simdjson::padded_string json(parsed.data);
    simdjson::dom::element doc;
    if (finish_parser.parse(json).get(doc) != simdjson::SUCCESS) return result;
    simdjson::dom::array choices;
    if (doc["choices"].get(choices) != simdjson::SUCCESS) return result;
    for (const auto choice : choices) {
        result.stream_started = true;
        std::string_view reason;
        if (choice["finish_reason"].get(reason) != simdjson::SUCCESS || reason.empty()) continue;
        stop_reason_ = reason == "length" ? "max_tokens"
            : reason == "tool_calls" ? "tool_use"
            : reason == "content_filter" ? "refusal" : std::string(reason);
    }
    return result;
}

ParseResult DashScopeTokenPlanProtocol::parse_event(std::string_view raw_event) {
    return delegate_->parse_event(raw_event);
}

ReasoningCapabilities DashScopeTokenPlanProtocol::reasoning_capabilities(
    std::string_view model) const noexcept {
    return delegate_->reasoning_capabilities(model);
}

std::unique_ptr<ApiProtocolBase> DashScopeTokenPlanProtocol::clone() const {
    return std::unique_ptr<ApiProtocolBase>{
        new DashScopeTokenPlanProtocol(
            options_, delegate_->clone())};
}

void DashScopeTokenPlanProtocol::on_response(const HttpResponse& response) {
    delegate_->on_response(response);
}

std::string DashScopeTokenPlanProtocol::format_error_message(
    const HttpResponse& response) const {
    return delegate_->format_error_message(response);
}

bool DashScopeTokenPlanProtocol::is_retryable(const HttpResponse& response) const noexcept {
    return delegate_->is_retryable(response);
}

RateLimitInfo DashScopeTokenPlanProtocol::last_rate_limit() const noexcept {
    return delegate_->last_rate_limit();
}

void DashScopeTokenPlanProtocol::reset_state() { delegate_->reset_state(); }

void DashScopeProtocol::reset_state() {
    stop_reason_.clear();
    pending_tool_calls_.clear();
    stream_rate_limit_seen_ = false;
    last_rate_limit_ = {};
}

// ─────────────────────────────────────────────────────────────────────────────
// Response lifecycle
// ─────────────────────────────────────────────────────────────────────────────

void DashScopeProtocol::on_response(const HttpResponse& response) {
    if (response.status_code >= 400 && lifecycle_.failure) {
        lifecycle_.failure(response.status_code, parse_dashscope_error(response.body).message);
    }
    // An HTTP-200 stream may fail later; preserve its authoritative quota signal.
    const bool preserve_limit = stream_rate_limit_seen_ && response.status_code == 200;
    stream_rate_limit_seen_ = false;
    if (preserve_limit) return;
    last_rate_limit_ = build_rate_limit_info(response.headers, response.status_code);
    if (response.status_code == 429) mark_rate_limit(last_rate_limit_, parse_dashscope_error(response.body));
}

// ─────────────────────────────────────────────────────────────────────────────
// Error handling
// ─────────────────────────────────────────────────────────────────────────────

std::string DashScopeProtocol::format_error_message(const HttpResponse& response) const {
    return format_dashscope_error(response, deployment_, last_rate_limit_.retry_after);
}

bool DashScopeProtocol::is_retryable(const HttpResponse& response) const noexcept {
    return retryable_response(response);
}

} // namespace core::llm::protocols
