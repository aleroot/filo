#include "ZaiProtocol.hpp"

#include "OpenAIUsage.hpp"
#include "SseUtils.hpp"
#include "../transport/HttpHeaderUtils.hpp"
#include "../../utils/AsciiUtils.hpp"
#include "../../utils/StringUtils.hpp"
#include "../../utils/TimeUtils.hpp"

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
    return code == 1302 || code == 1305;
}

[[nodiscard]] bool is_resettable_limit(int code) noexcept {
    return code == 1302
        || code == 1308
        || code == 1310
        || (code >= 1316 && code <= 1321);
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
    static constexpr std::array<std::string_view, 2> kSuffixes{
        "/api/coding/paas/v4",
        "/api/paas/v4",
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
    const auto auth = transport::find_header(request_headers, "Authorization");
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

[[nodiscard]] bool is_glm_model(std::string_view model) {
    return core::utils::ascii::istarts_with(
        core::utils::str::trim_ascii_view(model), "glm-");
}

[[nodiscard]] bool is_model_family(
    std::string_view model,
    std::string_view family) {
    const std::string_view normalized = core::utils::str::trim_ascii_view(model);
    if (!core::utils::ascii::istarts_with(normalized, family)) return false;
    if (normalized.size() == family.size()) return true;
    const char suffix = normalized[family.size()];
    return suffix == '-' || suffix == '.' || suffix == ':'
        || suffix == '/' || suffix == '[';
}

[[nodiscard]] bool is_glm_53_model(std::string_view model) {
    return is_model_family(model, "glm-5.3");
}

[[nodiscard]] bool is_glm_52_model(std::string_view model) {
    return is_model_family(model, "glm-5.2");
}

[[nodiscard]] bool supports_tool_stream(std::string_view model) {
    return is_model_family(model, "glm-5")
        || is_model_family(model, "glm-4.7")
        || is_model_family(model, "glm-4.6");
}

enum class ZaiReasoningEffort {
    ProviderDefault,
    Off,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
    Ultra,
};

[[nodiscard]] ZaiReasoningEffort parse_reasoning_effort(
    std::string_view raw_effort) noexcept {
    const std::string_view effort = core::utils::str::trim_ascii_view(raw_effort);
    using core::utils::ascii::iequals;
    if (effort.empty() || iequals(effort, "auto") || iequals(effort, "unset")
        || iequals(effort, "default")) {
        return ZaiReasoningEffort::ProviderDefault;
    }
    if (iequals(effort, "off") || iequals(effort, "none")
        || iequals(effort, "disabled") || iequals(effort, "disable")) {
        return ZaiReasoningEffort::Off;
    }
    if (iequals(effort, "minimal")) return ZaiReasoningEffort::Minimal;
    if (iequals(effort, "low")) return ZaiReasoningEffort::Low;
    if (iequals(effort, "medium")) return ZaiReasoningEffort::Medium;
    if (iequals(effort, "high")) return ZaiReasoningEffort::High;
    if (iequals(effort, "xhigh")) return ZaiReasoningEffort::XHigh;
    if (iequals(effort, "max")) return ZaiReasoningEffort::Max;
    if (iequals(effort, "ultra")) return ZaiReasoningEffort::Ultra;
    return ZaiReasoningEffort::ProviderDefault;
}

[[nodiscard]] ZaiReasoningEffort effort_for_model(
    ZaiReasoningEffort effort,
    std::string_view model) noexcept {
    if (is_glm_53_model(model)) {
        switch (effort) {
        case ZaiReasoningEffort::Minimal:
        case ZaiReasoningEffort::Off:
        case ZaiReasoningEffort::Low:
            return ZaiReasoningEffort::Low;
        case ZaiReasoningEffort::Medium:
        case ZaiReasoningEffort::XHigh:
        case ZaiReasoningEffort::High:
            return ZaiReasoningEffort::High;
        case ZaiReasoningEffort::Ultra:
        case ZaiReasoningEffort::Max:
            return ZaiReasoningEffort::Max;
        case ZaiReasoningEffort::ProviderDefault:
            return effort;
        }
    }
    if (is_glm_52_model(model)) {
        switch (effort) {
        case ZaiReasoningEffort::Low:
        case ZaiReasoningEffort::Medium:
            return ZaiReasoningEffort::High;
        case ZaiReasoningEffort::XHigh:
        case ZaiReasoningEffort::Ultra:
            return ZaiReasoningEffort::Max;
        default:
            return effort;
        }
    }
    return effort;
}

[[nodiscard]] std::string_view reasoning_effort_name(
    ZaiReasoningEffort effort) noexcept {
    switch (effort) {
    case ZaiReasoningEffort::Minimal: return "minimal";
    case ZaiReasoningEffort::Low: return "low";
    case ZaiReasoningEffort::Medium: return "medium";
    case ZaiReasoningEffort::High: return "high";
    case ZaiReasoningEffort::XHigh: return "xhigh";
    case ZaiReasoningEffort::Max: return "max";
    case ZaiReasoningEffort::Ultra: return "ultra";
    case ZaiReasoningEffort::ProviderDefault:
    case ZaiReasoningEffort::Off:
        return {};
    }
    return {};
}

[[nodiscard]] ParseResult parse_stream_chunk(std::string_view json) {
    thread_local simdjson::dom::parser parser;
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

ParseResult ZaiProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed)) return {};
    if (parsed.is_done) return {.done = true};

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

std::string ZaiProtocol::format_error_message(const HttpResponse& response) const {
    const auto error = parse_api_error(response.body);
    if (!error.has_value()) return ApiProtocolBase::format_error_message(response);

    std::string message = "[Z.ai Error " + std::to_string(error->code);
    if (!error->message.empty()) message += ": " + error->message;
    message += ']';
    if (name() == "zai"
        && (error->code == 1113 || error->code == 1311 || error->code == 1315)) {
        message += " This request used the General API. If your API key is for "
                   "the GLM Coding Plan, pick a model listed under GLM Coding Plan.";
    }
    return message;
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

void ZaiProtocol::enrich_rate_limit(
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
    if (last_rate_limit_.subscription_ends_at <= 0) {
        if (const int64_t end = cached_or_fetch_subscription_end(
                base_url, request_headers);
            end > 0) {
            last_rate_limit_.subscription_ends_at = end;
        }
    }

    if (has_usage_window(last_rate_limit_, "5h")
        && has_usage_window(last_rate_limit_, "7d")) {
        return;
    }

    const std::string key = usage_cache_key(base_url, request_headers);
    if (const auto cached = load_cached_usage(key); cached.has_value()) {
        merge_usage(last_rate_limit_, *cached);
        return;
    }
    if (const auto usage = fetch_usage(base_url, request_headers);
        usage.has_value()) {
        store_cached_usage(key, *usage);
        merge_usage(last_rate_limit_, *usage);
    }
}

void ZaiProtocol::append_extra_fields(
    std::string& payload,
    const ChatRequest& req) const {
    if (!is_glm_model(req.model)) return;
    const ReasoningCapabilities capabilities = reasoning_capabilities(req.model);
    ZaiReasoningEffort effort = parse_reasoning_effort(req.effort);

    const bool skip_glm_52_thinking = is_glm_52_model(req.model)
        && effort == ZaiReasoningEffort::Minimal;
    if (effort == ZaiReasoningEffort::Off || skip_glm_52_thinking) {
        if (capabilities.supports(ReasoningCapability::Disable)) {
            payload += R"(,"thinking":{"type":"disabled"})";
            return;
        }
        if (capabilities.supports(ReasoningCapability::Required)
            && capabilities.supports_effort()) {
            // GLM-5.3 and Flash require reasoning, so use the lowest supported
            // effort instead of sending a request that the API rejects.
            effort = ZaiReasoningEffort::Low;
        } else {
            effort = ZaiReasoningEffort::ProviderDefault;
        }
    }

    payload += R"(,"thinking":{"type":"enabled","clear_thinking":false})";
    if (effort != ZaiReasoningEffort::ProviderDefault
        && capabilities.supports_effort()) {
        effort = effort_for_model(effort, req.model);
        const std::string_view effort_name = reasoning_effort_name(effort);
        if (effort_name.empty()) return;
        payload += R"(,"reasoning_effort":")";
        payload += effort_name;
        payload += '"';
    }
}

ReasoningCapabilities ZaiProtocol::reasoning_capabilities(
    std::string_view model) const noexcept {
    if (!is_glm_model(model)) return {};
    if (is_glm_53_model(model)) {
        return ReasoningCapability::Effort
            | ReasoningCapability::MaxEffort
            | ReasoningCapability::Required
            | ReasoningCapability::MapsMediumToHigh
            | ReasoningCapability::MapsMinimalToLow
            | ReasoningCapability::MapsXHighToHigh;
    }
    if (is_glm_52_model(model)) {
        return ReasoningCapability::Effort
            | ReasoningCapability::MaxEffort
            | ReasoningCapability::XHighEffort
            | ReasoningCapability::Disable
            | ReasoningCapability::MapsLowToHigh
            | ReasoningCapability::MapsMediumToHigh
            | ReasoningCapability::MapsXHighToMax
            | ReasoningCapability::MapsMinimalToOff;
    }
    if (is_model_family(model, "glm-5")
        || is_model_family(model, "glm-4.7")
        || is_model_family(model, "glm-4.6")
        || is_model_family(model, "glm-4.5")) {
        return ReasoningCapabilities{ReasoningCapability::Disable};
    }
    return {};
}

void ZaiCodingProtocol::prepare_request(ChatRequest& req) {
    if (is_glm_model(req.model)) return;
    throw std::invalid_argument(
        "Z.ai Coding Plan requires a GLM model available to your account; got '"
        + req.model + "'");
}

} // namespace core::llm::protocols
