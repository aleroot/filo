#include "ZaiProtocol.hpp"

#include "../Models.hpp"
#include "OpenAIUsage.hpp"
#include "SseUtils.hpp"
#include "../ZaiModelTraits.hpp"
#include "../transport/HttpHeaderUtils.hpp"
#include "../../utils/AsciiUtils.hpp"
#include "../../utils/JsonUtils.hpp"
#include "../../utils/StringUtils.hpp"
#include "../../utils/TimeUtils.hpp"
#include "../../utils/Uuid.hpp"

#include <simdjson.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace core::llm::protocols {

namespace {

constexpr auto kUsageSnapshotTtl = std::chrono::seconds{20};

struct UsageSnapshot {
    std::vector<UsageWindow> windows;
};

struct CachedUsageSnapshot {
    UsageSnapshot value;
    std::chrono::steady_clock::time_point expires_at;
};

struct ApiError {
    int code = 0;
    std::string message;
};

[[nodiscard]] std::optional<float> parse_float_header(
    const cpr::Header& headers,
    std::string_view name) noexcept {
    const auto value = transport::find_header(headers, name);
    if (!value.has_value() || value->empty()) return std::nullopt;
    try {
        float parsed = std::stof(*value);
        if (parsed > 1.5f && parsed <= 100.0f) parsed /= 100.0f;
        return std::clamp(parsed, 0.0f, 1.5f);
    } catch (...) {
        return std::nullopt;
    }
}

[[nodiscard]] int32_t parse_int_header(
    const cpr::Header& headers,
    std::string_view name) noexcept {
    const auto value = transport::find_header(headers, name);
    if (!value.has_value() || value->empty()) return 0;
    try {
        return static_cast<int32_t>(std::stoi(*value));
    } catch (...) {
        return 0;
    }
}

void merge_header_quota(RateLimitInfo& info, const cpr::Header& headers) {
    struct WindowHeader {
        std::string_view label;
        std::string_view name;
    };
    static constexpr std::array<WindowHeader, 2> kWindows{{
        {"5h", "x-ratelimit-unified-5h-utilization"},
        {"7d", "x-ratelimit-unified-7d-utilization"},
    }};

    for (const auto& window : kWindows) {
        if (const auto value = parse_float_header(headers, window.name);
            value.has_value()) {
            int64_t w_reset = 0;
            const std::string reset_key = "x-ratelimit-unified-" + std::string(window.label) + "-reset";
            const std::string resets_at_key = "x-ratelimit-unified-" + std::string(window.label) + "-resets-at";
            if (const auto rval = transport::find_header(headers, reset_key); rval.has_value()) {
                w_reset = core::utils::time::parse_timestamp_or_duration(*rval);
            } else if (const auto rval2 = transport::find_header(headers, resets_at_key); rval2.has_value()) {
                w_reset = core::utils::time::parse_timestamp_or_duration(*rval2);
            }
            info.usage_windows.push_back({std::string(window.label), *value, w_reset});
        }
    }

    if (const auto status = transport::find_header(
            headers, "x-ratelimit-unified-status"); status.has_value()) {
        info.unified_status = *status;
        if (*status == "rate_limited" || *status == "rejected") {
            info.is_rate_limited = true;
        }
    }
    if (const auto claim = transport::find_header(
            headers, "x-ratelimit-unified-representative-claim");
        claim.has_value()) {
        info.unified_representative_claim = *claim;
    }
}

void merge_usage_window(RateLimitInfo& info, UsageWindow incoming) {
    auto existing = std::find_if(
        info.usage_windows.begin(), info.usage_windows.end(),
        [&](const UsageWindow& current) {
            return current.label == incoming.label;
        });
    if (existing != info.usage_windows.end()) {
        existing->utilization = incoming.utilization;
        if (incoming.resets_at > 0) {
            existing->resets_at = incoming.resets_at;
        }
    } else {
        info.usage_windows.push_back(std::move(incoming));
    }
}

[[nodiscard]] std::optional<float> read_json_number(
    simdjson::dom::object object,
    std::string_view key) {
    double as_double = 0.0;
    if (object[key].get(as_double) == simdjson::SUCCESS) {
        return static_cast<float>(as_double);
    }
    int64_t as_int = 0;
    if (object[key].get(as_int) == simdjson::SUCCESS) {
        return static_cast<float>(as_int);
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string_view> read_json_string(
    simdjson::dom::object object,
    std::string_view key) {
    std::string_view value;
    if (object[key].get(value) == simdjson::SUCCESS) return value;
    return std::nullopt;
}

[[nodiscard]] std::optional<int64_t> read_json_int(
    simdjson::dom::object object,
    std::string_view key) {
    int64_t value = 0;
    if (object[key].get(value) == simdjson::SUCCESS) return value;
    double as_double = 0.0;
    if (object[key].get(as_double) == simdjson::SUCCESS) {
        return static_cast<int64_t>(as_double);
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<ApiError> parse_api_error(
    std::string_view payload) noexcept {
    try {
        simdjson::dom::parser parser;
        simdjson::dom::element document;
        if (parser.parse(payload).get(document) != simdjson::SUCCESS) {
            return std::nullopt;
        }
        simdjson::dom::object root;
        if (document.get(root) != simdjson::SUCCESS) return std::nullopt;

        simdjson::dom::object error = root;
        simdjson::dom::object nested;
        if (root["error"].get(nested) == simdjson::SUCCESS) error = nested;

        ApiError parsed;
        if (const auto numeric = read_json_int(error, "code"); numeric.has_value()) {
            parsed.code = static_cast<int>(*numeric);
        } else if (const auto text = read_json_string(error, "code");
                   text.has_value()) {
            try {
                parsed.code = std::stoi(std::string(*text));
            } catch (...) {
                return std::nullopt;
            }
        } else {
            return std::nullopt;
        }

        if (const auto message = read_json_string(error, "message");
            message.has_value()) {
            parsed.message = std::string(*message);
        } else if (const auto message = read_json_string(root, "msg");
                   message.has_value()) {
            parsed.message = std::string(*message);
        }
        return parsed;
    } catch (...) {
        return std::nullopt;
    }
}

[[nodiscard]] bool is_transient_error(int code) noexcept {
    return zai_is_transient_business_code(code);
}

[[nodiscard]] bool is_resettable_limit(int code) noexcept {
    return zai_is_resettable_limit(code);
}

[[nodiscard]] std::optional<std::string> quota_label(
    simdjson::dom::object limit) {
    const auto type = read_json_string(limit, "type");
    const auto unit = read_json_int(limit, "unit");
    if (!type.has_value() || !unit.has_value()) return std::nullopt;

    // Z.ai renamed Coding Plan buckets from TOKENS_LIMIT to CREDIT_LIMIT.
    // Their unit identifiers and percentage semantics stayed unchanged.
    const bool is_coding_consumption =
        *type == "TOKENS_LIMIT" || *type == "CREDIT_LIMIT";
    if (is_coding_consumption && *unit == 3) return std::string{"5h"};
    if (is_coding_consumption && *unit == 6) return std::string{"7d"};
    if (*type == "TIME_LIMIT" && *unit == 5) return std::string{"web"};
    return std::nullopt;
}

[[nodiscard]] std::optional<UsageSnapshot> parse_usage_payload(
    std::string_view payload) {
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    if (parser.parse(payload).get(document) != simdjson::SUCCESS) {
        return std::nullopt;
    }

    simdjson::dom::object data;
    if (document["data"].get(data) != simdjson::SUCCESS) return std::nullopt;
    simdjson::dom::array limits;
    if (data["limits"].get(limits) != simdjson::SUCCESS) return std::nullopt;

    UsageSnapshot snapshot;
    for (simdjson::dom::element element : limits) {
        simdjson::dom::object limit;
        if (element.get(limit) != simdjson::SUCCESS) continue;

        const auto label = quota_label(limit);
        if (!label.has_value()) continue;

        std::optional<float> percentage = read_json_number(limit, "percentage");
        if (!percentage.has_value()) {
            const auto current = read_json_number(limit, "currentValue");
            const auto remaining = read_json_number(limit, "remaining");
            if (current.has_value() && remaining.has_value()
                && (*current + *remaining) > 0.0f) {
                percentage = (*current / (*current + *remaining)) * 100.0f;
            }
        }
        if (!percentage.has_value()) continue;
        int64_t resets_at = 0;
        static constexpr std::array<std::string_view, 5> kResetKeys{
            "resetTime", "nextResetTime", "expiresAt", "expireTime", "resetsAt"
        };
        for (const auto& key : kResetKeys) {
            std::string_view str_val;
            if (limit[key].get(str_val) == simdjson::SUCCESS) {
                if (const int64_t parsed = core::utils::time::parse_timestamp_or_duration(str_val); parsed > 0) {
                    resets_at = parsed;
                    break;
                }
            }
            int64_t int_val = 0;
            if (limit[key].get(int_val) == simdjson::SUCCESS && int_val > 0) {
                if (int_val > 100'000'000'000LL) int_val /= 1000LL;
                resets_at = int_val;
                break;
            }
        }
        snapshot.windows.push_back({
            *label,
            std::clamp(*percentage / 100.0f, 0.0f, 1.5f),
            resets_at,
        });
    }

    if (snapshot.windows.empty()) return std::nullopt;
    const auto rank = [](const std::string& label) {
        if (label == "5h") return 0;
        if (label == "7d") return 1;
        if (label == "web") return 2;
        return 3;
    };
    std::stable_sort(
        snapshot.windows.begin(), snapshot.windows.end(),
        [&](const UsageWindow& lhs, const UsageWindow& rhs) {
            return rank(lhs.label) < rank(rhs.label);
        });
    return snapshot;
}

[[nodiscard]] std::string management_api_base_url(std::string_view base_url) {
    std::string url = core::utils::str::trim_trailing_slashes(base_url);
    static constexpr std::array<std::string_view, 3> kSuffixes{
        "/api/coding/paas/v4",
        "/api/paas/v4",
        "/api/anthropic",
    };
    for (const auto suffix : kSuffixes) {
        if (url.ends_with(suffix)) {
            url.resize(url.size() - suffix.size());
            return url + "/api";
        }
    }
    return url;
}

[[nodiscard]] std::string usage_cache_key(
    std::string_view base_url,
    const cpr::Header& request_headers) {
    std::string key = management_api_base_url(base_url);
    auto auth = transport::find_header(request_headers, "Authorization");
    if (!auth.has_value()) {
        auth = transport::find_header(request_headers, "x-api-key");
    }
    const std::size_t auth_hash = std::hash<std::string_view>{}(
        auth.has_value() ? std::string_view(*auth) : std::string_view{});
    key += "|auth:" + std::to_string(auth_hash);
    return key;
}

[[nodiscard]] std::unordered_map<std::string, CachedUsageSnapshot>& usage_cache() {
    static std::unordered_map<std::string, CachedUsageSnapshot> cache;
    return cache;
}

[[nodiscard]] std::mutex& usage_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

[[nodiscard]] std::optional<UsageSnapshot> load_cached_usage(
    const std::string& key) {
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lock(usage_cache_mutex());
    auto& cache = usage_cache();
    const auto it = cache.find(key);
    if (it == cache.end()) return std::nullopt;
    if (it->second.expires_at <= now) {
        cache.erase(it);
        return std::nullopt;
    }
    return it->second.value;
}

void store_cached_usage(std::string key, const UsageSnapshot& snapshot) {
    std::scoped_lock lock(usage_cache_mutex());
    usage_cache()[std::move(key)] = {
        snapshot,
        std::chrono::steady_clock::now() + kUsageSnapshotTtl,
    };
}

[[nodiscard]] std::optional<UsageSnapshot> fetch_usage(
    std::string_view base_url,
    const cpr::Header& request_headers) {
    const std::string url = management_api_base_url(base_url)
        + "/monitor/usage/quota/limit";
    cpr::Header headers = request_headers;
    headers["Accept"] = "application/json";
    if (!transport::find_header(headers, "Accept-Language").has_value()) {
        headers["Accept-Language"] = "en";
    }

    const cpr::Response response = cpr::Get(
        cpr::Url{url}, headers, cpr::Timeout{1500});
    if (response.error.code != cpr::ErrorCode::OK
        || response.status_code != 200
        || response.text.empty()) {
        return std::nullopt;
    }
    return parse_usage_payload(response.text);
}

void merge_usage(RateLimitInfo& info, const UsageSnapshot& snapshot) {
    for (const auto& window : snapshot.windows) {
        merge_usage_window(info, window);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Subscription end date (GET /api/biz/subscription/list)
// ─────────────────────────────────────────────────────────────────────────────

// The subscription end date changes rarely, so cache it longer than the quota
// windows. Failed/absent lookups are cached too (value 0) so accounts without
// a Coding Plan subscription are not hammered on every response.
constexpr auto kSubscriptionCacheTtl = std::chrono::minutes{5};

struct CachedSubscriptionEnd {
    int64_t value = 0; ///< Unix seconds; 0 = provider reported no end date
    std::chrono::steady_clock::time_point expires_at;
};

[[nodiscard]] std::unordered_map<std::string, CachedSubscriptionEnd>&
subscription_end_cache() {
    static std::unordered_map<std::string, CachedSubscriptionEnd> cache;
    return cache;
}

[[nodiscard]] std::mutex& subscription_end_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

/// Parse a timestamp string, additionally tolerating the "YYYY-MM-DD HH:MM:SS"
/// and bare "YYYY-MM-DD" shapes used by Z.ai's business APIs (treated as UTC).
[[nodiscard]] int64_t parse_subscription_timestamp(std::string_view raw) noexcept {
    // Try the Z.ai business-API shapes first: their leading year digits would
    // otherwise be misread as an epoch by the generic numeric fallback.
    std::string normalized(raw);
    if (const auto space = normalized.find(' '); space != std::string::npos) {
        normalized[space] = 'T';
        if (const int64_t parsed =
                core::utils::time::parse_timestamp_or_duration(normalized);
            parsed > 0) {
            return parsed;
        }
    } else if (normalized.size() == 10
               && normalized[4] == '-' && normalized[7] == '-') {
        normalized += "T00:00:00";
        if (const int64_t parsed =
                core::utils::time::parse_timestamp_or_duration(normalized);
            parsed > 0) {
            return parsed;
        }
    }
    return core::utils::time::parse_timestamp_or_duration(raw);
}

[[nodiscard]] int64_t fetch_subscription_end(
    std::string_view base_url,
    const cpr::Header& request_headers) {
    const std::string url = management_api_base_url(base_url)
        + "/biz/subscription/list";
    cpr::Header headers = request_headers;
    headers["Accept"] = "application/json";
    if (!transport::find_header(headers, "Accept-Language").has_value()) {
        headers["Accept-Language"] = "en";
    }

    const cpr::Response response = cpr::Get(
        cpr::Url{url}, headers, cpr::Timeout{1500});
    if (response.error.code != cpr::ErrorCode::OK
        || response.status_code != 200
        || response.text.empty()) {
        return 0;
    }
    return parse_zai_subscription_end(response.text);
}

[[nodiscard]] int64_t cached_or_fetch_subscription_end(
    std::string_view base_url,
    const cpr::Header& request_headers) {
    const std::string key = usage_cache_key(base_url, request_headers)
        + "|subscription";
    {
        const auto now = std::chrono::steady_clock::now();
        std::scoped_lock lock(subscription_end_cache_mutex());
        auto& cache = subscription_end_cache();
        const auto it = cache.find(key);
        if (it != cache.end()) {
            if (it->second.expires_at > now) return it->second.value;
            cache.erase(it);
        }
    }

    const int64_t end = fetch_subscription_end(base_url, request_headers);
    std::scoped_lock lock(subscription_end_cache_mutex());
    subscription_end_cache()[std::move(key)] = CachedSubscriptionEnd{
        end,
        std::chrono::steady_clock::now() + kSubscriptionCacheTtl,
    };
    return end;
}

[[nodiscard]] bool has_usage_window(
    const RateLimitInfo& info,
    std::string_view label) {
    return std::ranges::any_of(info.usage_windows, [&](const UsageWindow& window) {
        return window.label == label;
    });
}

[[nodiscard]] bool supports_tool_stream(std::string_view model) {
    return is_glm_model_family(model, "glm-5")
        || is_glm_model_family(model, "glm-4.7")
        || is_glm_model_family(model, "glm-4.6");
}

[[nodiscard]] ParseResult parse_stream_chunk(std::string_view json) {
    thread_local simdjson::dom::parser parser;
    const core::utils::json::ParserRetentionGuard parser_guard{parser};
    simdjson::padded_string padded(json);
    simdjson::dom::element document;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS) return {};

    ParseResult result;
    simdjson::dom::object usage;
    bool has_usage = document["usage"].get(usage) == simdjson::SUCCESS
        && parse_openai_usage(usage, result);

    simdjson::dom::array choices;
    if (document["choices"].get(choices) != simdjson::SUCCESS) return result;
    StreamChunk chunk;
    for (simdjson::dom::element choice : choices) {
        if (!has_usage) {
            simdjson::dom::object choice_usage;
            has_usage = choice["usage"].get(choice_usage) == simdjson::SUCCESS
                && parse_openai_usage(choice_usage, result);
        }

        simdjson::dom::object delta;
        if (choice["delta"].get(delta) != simdjson::SUCCESS) continue;
        std::string_view content;
        if (delta["content"].get(content) == simdjson::SUCCESS) {
            chunk.content = content;
        }
        std::string_view reasoning;
        if (delta["reasoning_content"].get(reasoning) == simdjson::SUCCESS
            && !reasoning.empty()) {
            chunk.reasoning_content.append(reasoning);
        }

        simdjson::dom::array tool_calls;
        if (delta["tool_calls"].get(tool_calls) != simdjson::SUCCESS) continue;
        for (simdjson::dom::element element : tool_calls) {
            ToolCall call;
            int64_t index = -1;
            if (element["index"].get(index) == simdjson::SUCCESS) {
                call.index = static_cast<int>(index);
            }
            std::string_view value;
            if (element["id"].get(value) == simdjson::SUCCESS) call.id = value;
            if (element["type"].get(value) == simdjson::SUCCESS) call.type = value;
            simdjson::dom::object function;
            if (element["function"].get(function) == simdjson::SUCCESS) {
                if (function["name"].get(value) == simdjson::SUCCESS) {
                    call.function.name = value;
                }
                if (function["arguments"].get(value) == simdjson::SUCCESS) {
                    call.function.arguments = value;
                }
            }
            chunk.tools.push_back(std::move(call));
        }
    }
    if (!chunk.content.empty() || !chunk.reasoning_content.empty()
        || !chunk.tools.empty()) {
        result.chunks.push_back(std::move(chunk));
    }
    return result;
}

/// ZCode runner parity: chat-completions error frames carry
/// {"error":{"code":1302,…}} with no choices array. Surface them as
/// classified stream errors instead of dropping the frame.
[[nodiscard]] std::optional<ParseResult> parse_zai_stream_error_frame(
    std::string_view data) {
    thread_local simdjson::dom::parser parser;
    const core::utils::json::ParserRetentionGuard parser_guard{parser};
    simdjson::padded_string padded(data);
    simdjson::dom::element document;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS) {
        return std::nullopt;
    }
    simdjson::dom::array choices;
    if (document["choices"].get(choices) == simdjson::SUCCESS) {
        return std::nullopt;  // ordinary completion chunk
    }

    const auto error = parse_api_error(data);
    if (!error.has_value()) return std::nullopt;

    ParseResult failure;
    failure.stream_error = true;
    failure.stream_error_type = std::to_string(error->code);
    failure.stream_error_message = error->message;
    failure.retryable_stream_error =
        zai_is_transient_business_code(error->code);
    return failure;
}

/// ZCode runner parity: ten attempts with 2s → 60s exponential backoff and
/// jitter; Retry-After still wins via the server-delay path. Retries stop at
/// the first non-empty output token, matching ZCode's retry boundary —
/// committed output cannot be retracted.
[[nodiscard]] transport::RetryPolicy zai_stream_retry_policy() noexcept {
    transport::RetryPolicy policy;
    policy.max_retries = 10;
    policy.initial_backoff = std::chrono::seconds(2);
    policy.maximum_backoff = std::chrono::seconds(60);
    policy.retry_only_before_output = true;
    return policy;
}

/// GLM streams fail with business codes instead of the Claude error-type
/// vocabulary ("[1312][upstream][request_id]" messages, bare numeric types).
/// Classify those so transient waves are retried and terminal codes are not.
void classify_zai_stream_error(ParseResult& result) {
    if (!result.stream_error || result.retryable_stream_error) return;
    const int code = zai_stream_business_code(
        result.stream_error_type, result.stream_error_message);
    if (code == 0) return;
    result.retryable_stream_error = zai_is_transient_business_code(code);
    if (!result.stream_error_message.empty()) {
        const std::string code_text = std::to_string(code);
        if (result.stream_error_message.find(code_text)
            == std::string::npos) {
            result.stream_error_message += " [Z.ai business code "
                + code_text + "]";
        }
    } else {
        result.stream_error_message = "business code "
            + std::to_string(code);
    }
}

} // namespace

std::string ZaiProtocol::serialize(const ChatRequest& req) const {
    Serializer::Options options;
    options.reasoning_content_policy =
        Serializer::ReasoningContentPolicy::NonEmptyOwned;
    options.reasoning_protocol = std::string(name());

    std::string payload = Serializer::serialize(req, options);
    if (payload.ends_with('}')) {
        payload.pop_back();
        append_extra_fields(payload, req);
        if (req.stream && !req.tools.empty() && supports_tool_stream(req.model)) {
            payload += R"(,"tool_stream":true)";
        }
        if ((stream_usage_ || req.stream_include_usage) && req.stream) {
            payload += R"(,"stream_options":{"include_usage":true})";
        }
        payload += '}';
    }
    return payload;
}

transport::RetryPolicy ZaiProtocol::stream_retry_policy() const noexcept {
    return zai_stream_retry_policy();
}

ParseResult ZaiProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed)) return {};
    if (parsed.is_done) return {.done = true};

    if (auto error_frame = parse_zai_stream_error_frame(parsed.data)) {
        return std::move(*error_frame);
    }

    ParseResult result = parse_stream_chunk(parsed.data);
    for (auto& chunk : result.chunks) {
        if (!chunk.reasoning_content.empty()) {
            chunk.reasoning_protocol = std::string(name());
        }
    }
    return result;
}

void ZaiProtocol::on_response(const HttpResponse& response) {
    OpenAIProtocol::on_response(response);
    last_rate_limit_ = OpenAIProtocol::last_rate_limit();
    merge_header_quota(last_rate_limit_, response.headers);

    if (response.status_code != 429) return;
    if (const auto error = parse_api_error(response.body); error.has_value()) {
        last_rate_limit_.is_rate_limited = is_resettable_limit(error->code);
        return;
    }
    const auto unified = transport::find_header(
        response.headers, "x-ratelimit-unified-status");
    last_rate_limit_.is_rate_limited =
        last_rate_limit_.retry_after > 0
        || (unified.has_value()
            && (*unified == "rate_limited" || *unified == "rejected"));
}

bool ZaiProtocol::is_retryable(const HttpResponse& response) const noexcept {
    if (response.status_code != 429) {
        return OpenAIProtocol::is_retryable(response);
    }
    if (const auto error = parse_api_error(response.body); error.has_value()) {
        return is_transient_error(error->code);
    }
    const auto unified = transport::find_header(
        response.headers, "x-ratelimit-unified-status");
    return parse_int_header(response.headers, "retry-after") > 0
        || (unified.has_value()
            && (*unified == "rate_limited" || *unified == "rejected"));
}

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Coding Plan request attribution (ZCode parity)
//
// The Coding Plan service consumes per-request attribution: header-family
// x-request-id / x-zcode-trace-id / x-session-id / x-zcode-session-type, plus
// body metadata.user_id = {"device_id","account_uuid","session_id"}.
// Filo maps its own scopes onto the family: request ids are minted per
// request, transport_turn_id carries the turn scope, and session_id carries
// the session scope. Filo has no per-request agent-kind marker, so the
// session type is always "main" — the value the server expects for ordinary
// agent steps.
// ─────────────────────────────────────────────────────────────────────────────

[[nodiscard]] std::string normalized_session_id(std::string_view session_id) {
    auto stripped = core::utils::str::trim_ascii_view(session_id);
    // Defensive: never leak internal storage prefixes over the wire.
    for (const std::string_view prefix : {std::string_view("sess_"),
                                          std::string_view("subagent_agent_")}) {
        if (stripped.starts_with(prefix) && stripped.size() > prefix.size()) {
            stripped = stripped.substr(prefix.size());
            break;
        }
    }
    return std::string(stripped);
}

/// Anonymous, process-stable stand-in for ZCode's persisted device id. No
/// durable install identity is reachable from the protocol layer; OAuth
/// flows that project a real device_id into auth_properties take precedence.
[[nodiscard]] const std::string& anonymous_device_id() {
    static const std::string device_id = core::utils::random_uuid_v4();
    return device_id;
}

[[nodiscard]] std::string build_anthropic_metadata_user_id(const ChatRequest& req) {
    const auto projected = req.auth_properties.find("device_id");
    const std::string& device_id =
        projected != req.auth_properties.end() && !projected->second.empty()
            ? projected->second
            : anonymous_device_id();

    std::string user_id = R"({"device_id":")";
    user_id += core::utils::escape_json_string(device_id);
    user_id += R"(","account_uuid":"","session_id":")";
    user_id += core::utils::escape_json_string(normalized_session_id(req.session_id));
    user_id += '"';
    user_id += '}';
    return user_id;
}

void add_coding_plan_attribution_headers(cpr::Header& headers, const ChatRequest& req) {
    const std::string request_id = core::utils::random_uuid_v4();
    headers["x-request-id"] = request_id;
    headers["x-zcode-session-type"] = "main";
    headers["x-zcode-trace-id"] = req.transport_turn_id.empty()
        ? request_id
        : req.transport_turn_id;
    if (!req.session_id.empty()) {
        headers["x-session-id"] = normalized_session_id(req.session_id);
    }
}

[[nodiscard]] std::string zai_format_error(
    std::string_view protocol_name,
    const HttpResponse& response);

} // namespace

std::string ZaiProtocol::format_error_message(const HttpResponse& response) const {
    // Shared with the Coding wire: business-code formatting, General-API hints,
    // and context-overflow guidance all live in one place.
    return zai_format_error(name(), response);
}

int64_t parse_zai_subscription_end(std::string_view payload) noexcept {
    try {
        simdjson::dom::parser parser;
        simdjson::dom::element document;
        if (parser.parse(payload).get(document) != simdjson::SUCCESS) return 0;

        simdjson::dom::array data;
        if (document["data"].get(data) != simdjson::SUCCESS) return 0;

        // Candidate keys ordered by likelihood. Z.ai's subscription objects
        // expose `valid` (current-term expiry) plus, depending on rollout,
        // explicit expiry/renewal timestamps.
        static constexpr std::array<std::string_view, 9> kExpiryKeys{
            "valid", "validUntil", "valid_until",
            "expireTime", "expiredTime", "expiresAt",
            "endTime", "renewTime", "nextRenewTime",
        };

        for (simdjson::dom::element element : data) {
            simdjson::dom::object subscription;
            if (element.get(subscription) != simdjson::SUCCESS) continue;

            for (const auto key : kExpiryKeys) {
                if (const auto text = read_json_string(subscription, key);
                    text.has_value() && !text->empty()) {
                    if (const int64_t parsed = parse_subscription_timestamp(*text);
                        parsed > 0) {
                        return parsed;
                    }
                }
                if (const auto numeric = read_json_int(subscription, key);
                    numeric.has_value() && *numeric > 0) {
                    int64_t value = *numeric;
                    if (value > 100'000'000'000LL) value /= 1000LL; // ms -> s
                    return value;
                }
            }
            // Only the first subscription entry is considered.
            return 0;
        }
        return 0;
    } catch (...) {
        return 0;
    }
}

void enrich_zai_quota(
    RateLimitInfo& info,
    std::string_view base_url,
    const cpr::Header& request_headers,
    const HttpResponse& response) {
    if (response.status_code <= 0
        || response.status_code == 401
        || response.status_code == 403) {
        return;
    }

    // Plan-level metadata lives on a separate endpoint from the quota
    // windows; fetch it independently (long-lived, negatively cached).
    if (info.subscription_ends_at <= 0) {
        if (const int64_t end = cached_or_fetch_subscription_end(
                base_url, request_headers);
            end > 0) {
            info.subscription_ends_at = end;
        }
    }

    if (has_usage_window(info, "5h") && has_usage_window(info, "7d")) {
        return;
    }

    const std::string key = usage_cache_key(base_url, request_headers);
    if (const auto cached = load_cached_usage(key); cached.has_value()) {
        merge_usage(info, *cached);
        return;
    }
    if (const auto usage = fetch_usage(base_url, request_headers);
        usage.has_value()) {
        store_cached_usage(key, *usage);
        merge_usage(info, *usage);
    }
}

void ZaiProtocol::enrich_rate_limit(
    std::string_view base_url,
    const cpr::Header& request_headers,
    const HttpResponse& response) {
    enrich_zai_quota(last_rate_limit_, base_url, request_headers, response);
}

void ZaiProtocol::append_extra_fields(
    std::string& payload,
    const ChatRequest& req) const {
    append_glm_openai_thinking_fields(payload, req.model, req.effort);
}

ReasoningCapabilities ZaiProtocol::reasoning_capabilities(
    std::string_view model) const noexcept {
    return glm_reasoning_capabilities(model);
}

void apply_zai_http_response(RateLimitInfo& info, const HttpResponse& response) {
    merge_header_quota(info, response.headers);
    if (const int32_t retry_after = parse_int_header(response.headers, "retry-after");
        retry_after > 0) {
        info.retry_after = retry_after;
    }
    if (response.status_code != 429) return;
    if (const auto error = parse_api_error(response.body); error.has_value()) {
        info.is_rate_limited = is_resettable_limit(error->code);
        return;
    }
    const auto unified = transport::find_header(
        response.headers, "x-ratelimit-unified-status");
    info.is_rate_limited =
        info.retry_after > 0
        || (unified.has_value()
            && (*unified == "rate_limited" || *unified == "rejected"));
}

[[nodiscard]] bool zai_response_is_retryable(const HttpResponse& response) noexcept {
    if (const auto error = parse_api_error(response.body); error.has_value()) {
        return is_transient_error(error->code);
    }
    if (response.status_code != 429) {
        return response.status_code == 500
            || response.status_code == 502
            || response.status_code == 503
            || response.status_code == 504
            || response.status_code == 529;
    }
    const auto unified = transport::find_header(
        response.headers, "x-ratelimit-unified-status");
    return parse_int_header(response.headers, "retry-after") > 0
        || (unified.has_value()
            && (*unified == "rate_limited" || *unified == "rejected"));
}

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// Thinking-replay hygiene (ZCode parity)
//
// GLM binds thinking-block signatures to the model that produced them and
// rejects replayed signatures with a 400. ZCode guards this in two ways that
// Filo mirrors here, entirely inside the Z.ai protocol:
//   1. parse_event stamps every replayable thinking item with the producing
//      model; serialize() refuses cross-model replay of signed/redacted items.
//   2. A 400 whose wording matches a signature rejection is remembered per
//      request fingerprint; the next serialize() of the same history strips
//      the rejected blocks and inserts a "[Thinking removed]" fallback.
// ─────────────────────────────────────────────────────────────────────────────

constexpr std::string_view kRejectedReasoningFallback = "[Thinking removed]";

/// Filo's Anthropic parser emits replay blocks as
/// {"type":"thinking","thinking":"…","signature":"…"} — an empty signature
/// is exactly the unsigned case. The needle cannot false-positive inside
/// escaped thinking text because escaping removes unescaped quotes.
[[nodiscard]] bool continuation_is_signed(const ContinuationItem& item) noexcept {
    if (item.kind == "redacted_thinking") return false;
    return item.payload.find(R"("signature":"")") == std::string::npos;
}

[[nodiscard]] bool continuation_is_redacted(const ContinuationItem& item) noexcept {
    return item.kind == "redacted_thinking";
}

using ContinuationRejectPredicate =
    bool (*)(const ContinuationItem&, std::string_view target_model);

/// ZCode removeCrossModelReasoning parity: signed/redacted thinking is bound
/// to the model that produced it. Unsigned text stays — the backend accepts
/// it as ordinary reasoning context.
[[nodiscard]] bool continuation_rejected_across_models(
    const ContinuationItem& item,
    std::string_view target_model) noexcept {
    if (item.provider.empty() || item.provider == target_model) return false;
    return continuation_is_signed(item) || continuation_is_redacted(item);
}

/// ZCode removeRejectedReasoning parity: after a signature rejection the
/// backend refuses replayed signatures, so every signed/redacted block goes.
[[nodiscard]] bool continuation_rejected_after_signature_failure(
    const ContinuationItem& item,
    std::string_view /*target_model*/) noexcept {
    return continuation_is_signed(item) || continuation_is_redacted(item);
}

/// Returns a copy of @p req whose replayable thinking items have been filtered
/// through @p rejects. Survivors are re-tagged for the generic Anthropic
/// serializer (which only replays items carrying its own wire tag); an
/// assistant message that loses all of its content receives the
/// "[Thinking removed]" fallback so it cannot become an empty turn.
[[nodiscard]] ChatRequest strip_rejected_continuations(
    const ChatRequest& req,
    ContinuationRejectPredicate rejects) {
    ChatRequest sanitized = req;
    for (auto& msg : sanitized.messages) {
        if (msg.continuation_items.empty()) continue;

        bool dropped = false;
        std::vector<ContinuationItem> kept;
        kept.reserve(msg.continuation_items.size());
        for (auto& item : msg.continuation_items) {
            if (rejects(item, sanitized.model)) {
                dropped = true;
                continue;
            }
            // Survivors must always be re-tagged: the generic Anthropic
            // serializer replays only items carrying its own wire tag, and
            // Z.ai-stamped items arrive with the producing model instead.
            item.provider = "anthropic";
            kept.push_back(std::move(item));
        }
        msg.continuation_items = std::move(kept);
        if (!dropped) continue;

        // Only a turn that lost everything becomes an empty request payload;
        // surviving blocks keep the assistant message self-sufficient.
        if (msg.continuation_items.empty() && msg.role == "assistant"
            && msg.tool_calls.empty() && msg.content.empty()) {
            msg.content = std::string(kRejectedReasoningFallback);
        }
    }
    return sanitized;
}

[[nodiscard]] std::uint64_t fingerprint_request(const ChatRequest& req) noexcept {
    // FNV-1a over the model and every replayed thinking item. Attempts of the
    // same request serialize the identical history, so equal fingerprints
    // mean "same payload" for repair purposes.
    std::uint64_t hash = 1469598103934665603ull;
    const auto mix = [&hash](std::string_view text) {
        for (const char ch : text) {
            hash ^= static_cast<unsigned char>(ch);
            hash *= 1099511628211ull;
        }
    };
    mix(req.model);
    for (const auto& msg : req.messages) {
        for (const auto& item : msg.continuation_items) {
            mix(item.kind);
            mix(item.payload);
        }
    }
    return hash;
}

struct SignatureRejectionRecord {
    std::chrono::steady_clock::time_point expires_at;
};

constexpr auto kSignatureRejectionTtl = std::chrono::minutes{15};
constexpr std::size_t kSignatureRejectionCacheCap = 64;

std::unordered_map<std::uint64_t, SignatureRejectionRecord>&
signature_rejection_registry() {
    static std::unordered_map<std::uint64_t, SignatureRejectionRecord> registry;
    return registry;
}

void record_signature_rejection(std::uint64_t fingerprint) {
    auto& registry = signature_rejection_registry();
    const auto now = std::chrono::steady_clock::now();
    std::erase_if(registry, [now](const auto& entry) {
        return entry.second.expires_at <= now;
    });
    while (registry.size() >= kSignatureRejectionCacheCap) {
        registry.erase(registry.begin());
    }
    registry[fingerprint] = {now + kSignatureRejectionTtl};
}

[[nodiscard]] bool take_signature_rejection(std::uint64_t fingerprint) {
    auto& registry = signature_rejection_registry();
    const auto it = registry.find(fingerprint);
    if (it == registry.end()) return false;
    const bool expired = it->second.expires_at <= std::chrono::steady_clock::now();
    registry.erase(it);
    return !expired;
}

} // namespace

void zai_clear_signature_rejection_cache_for_testing() {
    signature_rejection_registry().clear();
}

namespace {

// ─────────────────────────────────────────────────────────────────────────────
// GLM video input on the Anthropic wire (ZCode patch parity)
//
// The generic Anthropic serializer degrades video parts to placeholder text;
// GLM-5.3-Flash accepts video blocks shaped like image blocks ("video" +
// base64 source, exactly what ZCode's patched AI SDK emits). This pass swaps
// the placeholders back out for real video blocks; every other model keeps
// the placeholder.
// ─────────────────────────────────────────────────────────────────────────────

[[nodiscard]] std::optional<std::pair<std::string, std::string>> video_media_payload(
    const ContentPart& part) {
    constexpr std::string_view kDataPrefix = "data:";
    constexpr std::string_view kBase64Marker = ";base64,";
    if (part.url.starts_with(kDataPrefix)) {
        const std::string_view url = part.url;
        const auto semi = url.find(';', kDataPrefix.size());
        if (semi == std::string_view::npos) return std::nullopt;
        const auto mime = url.substr(kDataPrefix.size(), semi - kDataPrefix.size());
        if (!mime.starts_with("video/")) return std::nullopt;
        if (!url.substr(semi).starts_with(kBase64Marker)) return std::nullopt;
        return std::make_pair(
            std::string(mime),
            std::string(url.substr(semi + kBase64Marker.size())));
    }
    if (!part.path.empty()) {
        const auto encoded = encode_media_part(part);
        if (!encoded.has_value() || encoded->is_url_reference()) return std::nullopt;
        return std::make_pair(encoded->mime_type, encoded->base64_data);
    }
    return std::nullopt;
}

void substitute_glm_video_blocks(std::string& payload, const ChatRequest& req) {
    if (!is_glm_multimodal_model(req.model)) return;

    // Parts stream into the payload in message order, so a rolling search
    // position keeps each placeholder matched to its own part.
    std::size_t search_from = 0;
    for (const auto& msg : req.messages) {
        for (const auto& part : msg.content_parts) {
            if (part.type != ContentPartType::Video) continue;

            const std::string needle =
                std::string(R"({"type":"text","text":")")
                + core::utils::escape_json_string(
                    unavailable_media_attachment_text(part.type, media_reference(part)))
                + R"("})";
            const auto at = payload.find(needle, search_from);
            if (at == std::string::npos) continue;

            const auto media = video_media_payload(part);
            search_from = at + needle.size();
            if (!media.has_value()) continue; // keep the degraded placeholder

            const std::string block =
                std::string(R"({"type":"video","source":{"type":"base64","media_type":")")
                + core::utils::escape_json_string(media->first)
                + R"(","data":")"
                + core::utils::escape_json_string(media->second)
                + R"("}})";
            payload.replace(at, needle.size(), block);
            search_from = at + block.size();
        }
    }
}

[[nodiscard]] std::string zai_format_error(
    std::string_view protocol_name,
    const HttpResponse& response) {
    const auto error = parse_api_error(response.body);
    if (!error.has_value()) {
        std::string plain = "[HTTP Error: " + std::to_string(response.status_code) +
               " - " + std::string(response.body) + "]";
        if (zai_error_mentions_context_overflow(response.body)) {
            plain += "\n[Z.ai] Context window exceeded — compact the session (/compact) "
                     "or start a new one, then retry.";
        }
        return plain;
    }

    std::string message = "[Z.ai Error " + std::to_string(error->code);
    if (!error->message.empty()) message += ": " + error->message;
    message += ']';
    if (protocol_name == "zai"
        && (error->code == 1113 || error->code == 1311 || error->code == 1315)) {
        message += " This request used the General API. If your API key is for "
                   "the GLM Coding Plan, pick a model listed under GLM Coding Plan.";
    }
    // ZCode parity: business code 1261 (and equivalent wordings) mean the
    // conversation no longer fits the model's context window. That failure is
    // terminal — surface the compaction escape hatch instead of a bare code.
    if (zai_is_context_overflow_code(error->code)
        || zai_error_mentions_context_overflow(error->message)
        || zai_error_mentions_context_overflow(response.body)) {
        message += "\n[Z.ai] Context window exceeded — compact the session (/compact) "
                   "or start a new one, then retry.";
    }
    if (zai_error_is_signature_rejection(response.status_code, response.body)) {
        message += "\n[Z.ai] Thinking-block signatures were rejected; replaying "
                   "this request will drop the rejected thinking blocks.";
    }
    return message;
}

} // namespace

void ZaiCodingProtocol::prepare_request(ChatRequest& req) {
    if (!is_glm_model(req.model)) {
        throw std::invalid_argument(
            "Z.ai Coding Plan requires a GLM model available to your account; got '"
            + req.model + "'");
    }
    // Remember the bound model so parse_event can stamp replayable thinking
    // with its producer (see strip_rejected_continuations).
    current_model_ = req.model;
}

std::string ZaiCodingProtocol::serialize(const ChatRequest& req) const {
    const std::uint64_t fingerprint = fingerprint_request(req);
    last_request_fingerprint_ = fingerprint;
    has_request_fingerprint_ = true;

    // If the backend just rejected this history's thinking signatures, the
    // retry carries the same payload fields — strip the rejected blocks now.
    // Otherwise apply the standing cross-model replay guard.
    const ContinuationRejectPredicate rejects =
        take_signature_rejection(fingerprint)
        ? &continuation_rejected_after_signature_failure
        : &continuation_rejected_across_models;

    std::string payload = AnthropicProtocol::serialize(
        strip_rejected_continuations(req, rejects));
    substitute_glm_video_blocks(payload, req);

    // ZCode parity: anthropic-kind requests carry attribution in the body.
    if (!payload.empty() && payload.back() == '}') {
        payload.pop_back();
        payload += R"(,"metadata":{"user_id":")";
        payload += core::utils::escape_json_string(build_anthropic_metadata_user_id(req));
        payload += R"("}})";
    }
    return payload;
}

ParseResult ZaiCodingProtocol::parse_event(std::string_view raw_event) {
    ParseResult result = AnthropicProtocol::parse_event(raw_event);
    if (!current_model_.empty()) {
        for (auto& chunk : result.chunks) {
            for (auto& item : chunk.continuation_items) {
                if (item.provider == "anthropic") item.provider = current_model_;
            }
        }
    }
    classify_zai_stream_error(result);
    return result;
}

transport::RetryPolicy ZaiCodingProtocol::stream_retry_policy()
    const noexcept {
    return zai_stream_retry_policy();
}

AnthropicReasoningEmitter ZaiCodingProtocol::reasoning_emitter() const {
    // GLM Coding Plan on the Anthropic wire uses thinking.type plus
    // output_config.effort. Claude budget_tokens / adaptive thinking and the
    // temperature=1 constraint do not apply.
    return [](std::string& payload, const ChatRequest& req) {
        append_glm_anthropic_thinking_fields(payload, req.model, req.effort);
        if (req.temperature.has_value()) {
            payload += R"(,"temperature":)";
            payload += std::to_string(req.temperature.value());
        }
    };
}

ReasoningCapabilities ZaiCodingProtocol::reasoning_capabilities(
    std::string_view model) const noexcept {
    return glm_reasoning_capabilities(model);
}

cpr::Header ZaiCodingProtocol::build_headers(
    const core::auth::AuthInfo& auth) const {
    cpr::Header headers{
        {"anthropic-version", "2023-06-01"},
        {"Content-Type", "application/json"},
        {"Accept", "text/event-stream"},
    };
    for (const auto& [key, value] : auth.headers) {
        headers[key] = value;
    }

    // ZCode parity: Anthropic-compatible gateways read both x-api-key and
    // Bearer Authorization; an explicitly configured Authorization wins.
    std::string api_key;
    bool has_authorization = false;
    for (const auto& [key, value] : headers) {
        if (core::utils::ascii::iequals(key, "x-api-key")) api_key = value;
        if (core::utils::ascii::iequals(key, "authorization")) has_authorization = true;
    }
    if (!api_key.empty() && !has_authorization) {
        headers["Authorization"] = "Bearer " + api_key;
    }
    return headers;
}

void ZaiCodingProtocol::prepare_headers(
    cpr::Header& headers,
    const ChatRequest& request,
    std::string_view /*base_url*/) {
    add_coding_plan_attribution_headers(headers, request);
}

void ZaiCodingProtocol::on_response(const HttpResponse& response) {
    last_rate_limit_ = {};
    apply_zai_http_response(last_rate_limit_, response);
    if (has_request_fingerprint_
        && zai_error_is_signature_rejection(response.status_code, response.body)) {
        record_signature_rejection(last_request_fingerprint_);
    }
}

bool ZaiCodingProtocol::is_retryable(const HttpResponse& response) const noexcept {
    return zai_response_is_retryable(response);
}

std::string ZaiCodingProtocol::format_error_message(
    const HttpResponse& response) const {
    return zai_format_error(name(), response);
}

void ZaiCodingProtocol::enrich_rate_limit(std::string_view base_url,
                                          const cpr::Header& request_headers,
                                          const HttpResponse& response) {
    enrich_zai_quota(last_rate_limit_, base_url, request_headers, response);
}

void ZaiCodingProtocol::reset_state() {
    AnthropicProtocol::reset_state();
    last_rate_limit_ = {};
}

} // namespace core::llm::protocols
