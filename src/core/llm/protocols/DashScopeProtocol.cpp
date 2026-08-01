#include "DashScopeProtocol.hpp"
#include "SseUtils.hpp"
#include "../Models.hpp"
#include "../QwenModelTraits.hpp"
#include "../../utils/StringUtils.hpp"
#include "../../utils/AsciiUtils.hpp"
#include <array>
#include <simdjson.h>

namespace core::llm::protocols {

namespace {

[[nodiscard]] ReasoningCapabilities qwen_reasoning_capabilities(
    std::string_view model) noexcept {
    using core::utils::ascii::istarts_with;
    const auto generation = qwen_model_generation(model);
    const bool supported = (!generation.empty()
                            && generation >= std::vector<int>{3})
        || istarts_with(model, "qwen-plus")
        || istarts_with(model, "qwen-flash")
        || istarts_with(model, "qwen-turbo")
        || istarts_with(model, "qwq")
        || istarts_with(model, "qvq");
    if (!supported) return {};
    return ReasoningCapability::Effort
        | ReasoningCapability::MaxEffort
        | ReasoningCapability::XHighEffort
        | ReasoningCapability::Disable;
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

[[nodiscard]] std::string normalize_qwen_effort(std::string_view configured) {
    const std::string lowered = core::utils::str::to_lower_ascii_copy(configured);
    if (lowered == "off" || lowered == "none" || lowered == "disabled") return "none";
    if (lowered == "minimal" || lowered == "low" || lowered == "medium"
        || lowered == "high" || lowered == "xhigh" || lowered == "max") {
        return lowered;
    }
    return {};
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
            != std::string::npos) {
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
    simdjson::padded_string ps(body);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) return details;

    std::string_view value;
    if (doc["code"].get(value) == simdjson::SUCCESS) {
        details.code = value;
    } else if (doc["error"]["code"].get(value) == simdjson::SUCCESS) {
        details.code = value;
    }
    if (doc["message"].get(value) == simdjson::SUCCESS) {
        details.message = value;
    } else if (doc["error"]["message"].get(value) == simdjson::SUCCESS) {
        details.message = value;
    }

    details.token_plan_limit =
        classify_token_plan_limit(details.code, details.message);
    return details;
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
                + " Verify DASHSCOPE_API_KEY is set correctly.]";
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
    return headers;
}

ReasoningCapabilities DashScopeProtocol::reasoning_capabilities(
    std::string_view model) const noexcept {
    return qwen_reasoning_capabilities(model);
}

std::string DashScopeProtocol::serialize(const ChatRequest& req) const {
    Serializer::Options options;
    // Qwen requires this field on the assistant message when a thinking tool
    // call is followed by tool results.
    options.reasoning_content_policy =
        Serializer::ReasoningContentPolicy::NonEmptyOwned;
    options.reasoning_protocol = std::string(name());

    std::string payload = Serializer::serialize(req, options);
    if (payload.ends_with('}')) {
        payload.pop_back();
        append_extra_fields(payload, req);
        if (req.stream) {
            payload += R"(,"stream_options":{"include_usage":true})";
        }
        payload += '}';
    }
    return payload;
}

void DashScopeProtocol::append_extra_fields(std::string&       payload,
                                             const ChatRequest& req) const {
    if (!qwen_reasoning_capabilities(req.model).supports_effort()) {
        return;
    }

    const std::string effort = normalize_qwen_effort(
        req.effort.empty() ? std::string_view(default_effort_) : std::string_view(req.effort));

    if (effort == "none") {
        payload += R"(,"enable_thinking":false)";
        return;
    }
    if (effort.empty() && thinking_budget_ <= 0) return;

    payload += R"(,"enable_thinking":true)";
    if (thinking_budget_ > 0) {
        payload += R"(,"thinking_budget":)";
        payload += std::to_string(thinking_budget_);
    }
    if (qwen_model_supports_preserve_thinking(req.model)) {
        payload += R"(,"preserve_thinking":true)";
    }
}

DashScopeResponsesProtocol::DashScopeResponsesProtocol()
    : DashScopeResponsesProtocol(Options{}) {}

DashScopeResponsesProtocol::DashScopeResponsesProtocol(Options options)
    : OpenAIResponsesProtocol(/*include_reasoning_encrypted=*/false)
    , options_(std::move(options)) {}

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

    const std::string effort = normalize_qwen_effort(
        req.effort.empty()
            ? std::string_view(options_.default_effort)
            : std::string_view(req.effort));
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
    return serialize_with_options(
        effective,
        SerializationOptions{
            .include_store = false,
            .include_prompt_cache_key = false,
            .include_response_include = false,
            .reasoning_effort_override = reasoning_effort,
            .hosted_tool_types = hosted_tools,
        });
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

bool DashScopeResponsesProtocol::is_retryable(
    const HttpResponse& response) const noexcept {
    return response.status_code == 429 || response.status_code == 500
        || response.status_code == 502 || response.status_code == 503
        || response.status_code == 504;
}

DashScopeTokenPlanProtocol::DashScopeTokenPlanProtocol()
    : DashScopeTokenPlanProtocol(Options{}) {}

DashScopeTokenPlanProtocol::DashScopeTokenPlanProtocol(Options options)
    : DashScopeTokenPlanProtocol(
          options,
          std::make_unique<DashScopeProtocol>(
              options.thinking_budget,
              options.default_effort,
              DashScopeDeployment::TokenPlan),
          std::make_unique<DashScopeResponsesProtocol>(
              DashScopeResponsesProtocol::Options{
                  .default_effort = options.default_effort,
                  .enable_hosted_tools = options.enable_hosted_tools,
                  .deployment = DashScopeDeployment::TokenPlan,
              })) {}

DashScopeTokenPlanProtocol::DashScopeTokenPlanProtocol(
    Options options,
    std::unique_ptr<ApiProtocolBase> chat,
    std::unique_ptr<ApiProtocolBase> responses)
    : options_(std::move(options))
    , chat_(std::move(chat))
    , responses_(std::move(responses)) {}

void DashScopeTokenPlanProtocol::select_for_model(
    std::string_view model) noexcept {
    active_wire_api_ = qwen_token_plan_wire_api(model);
}

ApiProtocolBase& DashScopeTokenPlanProtocol::active() noexcept {
    return active_wire_api_ == QwenTokenPlanWireApi::Responses
        ? *responses_
        : *chat_;
}

const ApiProtocolBase& DashScopeTokenPlanProtocol::active() const noexcept {
    return active_wire_api_ == QwenTokenPlanWireApi::Responses
        ? *responses_
        : *chat_;
}

const ApiProtocolBase& DashScopeTokenPlanProtocol::protocol_for(
    std::string_view model) const noexcept {
    return qwen_token_plan_wire_api(model) == QwenTokenPlanWireApi::Responses
        ? *responses_
        : *chat_;
}

void DashScopeTokenPlanProtocol::prepare_request(ChatRequest& request) {
    select_for_model(request.model);
    active().prepare_request(request);
}

std::string DashScopeTokenPlanProtocol::serialize(
    const ChatRequest& req) const {
    return protocol_for(req.model).serialize(req);
}

cpr::Header DashScopeTokenPlanProtocol::build_headers(
    const core::auth::AuthInfo& auth) const {
    return active().build_headers(auth);
}

std::string DashScopeTokenPlanProtocol::build_url(
    std::string_view base_url,
    std::string_view model) const {
    return protocol_for(model).build_url(base_url, model);
}

std::string_view DashScopeTokenPlanProtocol::event_delimiter() const noexcept {
    return active().event_delimiter();
}

ParseResult DashScopeTokenPlanProtocol::parse_event(
    std::string_view raw_event) {
    return active().parse_event(raw_event);
}

ReasoningCapabilities DashScopeTokenPlanProtocol::reasoning_capabilities(
    std::string_view model) const noexcept {
    return qwen_reasoning_capabilities(model);
}

std::unique_ptr<ApiProtocolBase> DashScopeTokenPlanProtocol::clone() const {
    return std::unique_ptr<ApiProtocolBase>{
        new DashScopeTokenPlanProtocol(
            options_, chat_->clone(), responses_->clone())};
}

void DashScopeTokenPlanProtocol::on_response(
    const HttpResponse& response) {
    // Keep the inner protocol's behaviour (usage parsing, header-based limits).
    active().on_response(response);

    // The Token Plan inference endpoints expose no rate-limit/usage headers, so
    // on a successful response there is nothing additional to capture. The only
    // provider-backed quota signal is a 429 body; translate it into a
    // rate-limit snapshot so the status bar and quota notifications can react.
    has_rate_limit_override_ = false;
    rate_limit_override_ = {};

    if (response.status_code != 429) return;

    RateLimitInfo info = active().last_rate_limit();
    info.is_rate_limited = true;
    info.unified_status = "rate_limited";
    if (const int32_t retry_after = parse_int_header(response.headers, "retry-after");
        retry_after > 0) {
        info.retry_after = retry_after;
    }
    // A plan-quota block is a hard rejection (not a transient throttle).
    if (parse_dashscope_error(response.body).token_plan_limit
        == TokenPlanLimitKind::PlanQuota) {
        info.unified_overage_status = "rejected";
    }

    rate_limit_override_ = info;
    has_rate_limit_override_ = true;
}

std::string DashScopeTokenPlanProtocol::format_error_message(
    const HttpResponse& response) const {
    return active().format_error_message(response);
}

bool DashScopeTokenPlanProtocol::is_retryable(
    const HttpResponse& response) const noexcept {
    return active().is_retryable(response);
}

RateLimitInfo DashScopeTokenPlanProtocol::last_rate_limit() const noexcept {
    if (has_rate_limit_override_) return rate_limit_override_;
    return active().last_rate_limit();
}

void DashScopeTokenPlanProtocol::reset_state() {
    chat_->reset_state();
    responses_->reset_state();
    active_wire_api_ = QwenTokenPlanWireApi::Responses;
    has_rate_limit_override_ = false;
    rate_limit_override_ = {};
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE parsing — reasoning_content support for Qwen3 thinking models
// ─────────────────────────────────────────────────────────────────────────────

ParseResult DashScopeProtocol::parse_event(std::string_view raw_event) {
    // Base parser handles: content, tool_calls, usage, [DONE].
    ParseResult result = OpenAIProtocol::parse_event(raw_event);
    if (result.done) return result;

    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed)) return result;
    if (parsed.is_done) return result;
    const std::string_view json_sv = parsed.data;

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
        thinking.reasoning_protocol = std::string(name());
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
    return format_dashscope_error(response, deployment_, last_rate_limit_.retry_after);
}

bool DashScopeProtocol::is_retryable(const HttpResponse& response) const noexcept {
    return response.status_code == 429 ||
           response.status_code == 500 ||
           response.status_code == 502 ||
           response.status_code == 503 ||
           response.status_code == 504;
}

} // namespace core::llm::protocols
