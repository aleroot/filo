#include "ZaiProtocol.hpp"

#include "SseUtils.hpp"
#include "../transport/HttpHeaderUtils.hpp"
#include "../../utils/StringUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
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

struct StreamParseResult {
    std::string content;
    std::string reasoning_content;
    std::vector<ToolCall> tools;
    int32_t prompt_tokens = 0;
    int32_t completion_tokens = 0;
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
            info.usage_windows.push_back({std::string(window.label), *value});
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
        snapshot.windows.push_back({
            *label,
            std::clamp(*percentage / 100.0f, 0.0f, 1.5f),
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

[[nodiscard]] bool has_usage_window(
    const RateLimitInfo& info,
    std::string_view label) {
    return std::ranges::any_of(info.usage_windows, [&](const UsageWindow& window) {
        return window.label == label;
    });
}

[[nodiscard]] bool is_glm_model(std::string_view model) {
    return core::utils::str::to_lower_ascii_copy(
        core::utils::str::trim_ascii_view(model)).starts_with("glm-");
}

[[nodiscard]] std::string normalize_effort(std::string_view raw_effort) {
    std::string effort = core::utils::str::to_lower_ascii_copy(raw_effort);
    std::erase_if(effort, [](unsigned char ch) { return std::isspace(ch); });
    if (effort == "auto" || effort == "unset" || effort == "default") return {};
    if (effort == "off" || effort == "disabled" || effort == "disable") return "off";
    if (effort == "low" || effort == "medium" || effort == "high" || effort == "max") {
        return effort;
    }
    return {};
}

[[nodiscard]] bool read_usage_object(
    simdjson::dom::object usage,
    int32_t& prompt_tokens,
    int32_t& completion_tokens) {
    int64_t prompt = 0;
    int64_t completion = 0;
    const bool has_prompt = usage["prompt_tokens"].get(prompt) == simdjson::SUCCESS;
    const bool has_completion =
        usage["completion_tokens"].get(completion) == simdjson::SUCCESS;
    if (!has_prompt && !has_completion) return false;
    prompt_tokens = static_cast<int32_t>(prompt);
    completion_tokens = static_cast<int32_t>(completion);
    return prompt > 0 || completion > 0;
}

[[nodiscard]] StreamParseResult parse_stream_chunk(std::string_view json) {
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string padded(json);
    simdjson::dom::element document;
    if (parser.parse(padded).get(document) != simdjson::SUCCESS) return {};

    StreamParseResult result;
    simdjson::dom::object usage;
    bool has_usage = document["usage"].get(usage) == simdjson::SUCCESS
        && read_usage_object(
            usage, result.prompt_tokens, result.completion_tokens);

    simdjson::dom::array choices;
    if (document["choices"].get(choices) != simdjson::SUCCESS) return result;
    for (simdjson::dom::element choice : choices) {
        if (!has_usage) {
            simdjson::dom::object choice_usage;
            has_usage = choice["usage"].get(choice_usage) == simdjson::SUCCESS
                && read_usage_object(
                    choice_usage, result.prompt_tokens, result.completion_tokens);
        }

        simdjson::dom::object delta;
        if (choice["delta"].get(delta) != simdjson::SUCCESS) continue;
        std::string_view content;
        if (delta["content"].get(content) == simdjson::SUCCESS) {
            result.content = std::string(content);
        }
        std::string_view reasoning;
        if (delta["reasoning_content"].get(reasoning) == simdjson::SUCCESS
            && !reasoning.empty()) {
            result.reasoning_content.append(reasoning);
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
            result.tools.push_back(std::move(call));
        }
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

    ParseResult result;
    auto parsed_chunk = parse_stream_chunk(parsed.data);
    result.prompt_tokens = parsed_chunk.prompt_tokens;
    result.completion_tokens = parsed_chunk.completion_tokens;
    if (!parsed_chunk.content.empty()
        || !parsed_chunk.reasoning_content.empty()
        || !parsed_chunk.tools.empty()) {
        StreamChunk chunk;
        chunk.content = std::move(parsed_chunk.content);
        chunk.reasoning_content = std::move(parsed_chunk.reasoning_content);
        if (!chunk.reasoning_content.empty()) {
            chunk.reasoning_protocol = std::string(name());
        }
        chunk.tools = std::move(parsed_chunk.tools);
        result.chunks.push_back(std::move(chunk));
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
                   "the GLM Coding Plan, select the model under Coding endpoint.";
    }
    return message;
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
    const std::string effort = normalize_effort(req.effort);
    if (effort == "off") {
        payload += R"(,"thinking":{"type":"disabled"})";
        return;
    }
    payload += R"(,"thinking":{"type":"enabled","clear_thinking":false})";
    if (!effort.empty()) {
        payload += R"(,"reasoning_effort":")";
        payload += core::utils::escape_json_string(effort);
        payload += '"';
    }
}

void ZaiCodingProtocol::prepare_request(ChatRequest& req) {
    if (is_glm_model(req.model)) return;
    throw std::invalid_argument(
        "Z.ai Coding Plan requires a GLM model available to your account; got '"
        + req.model + "'");
}

} // namespace core::llm::protocols
