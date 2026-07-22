#include "OpenAIProtocol.hpp"
#include "OpenAIUsage.hpp"
#include "SseUtils.hpp"
#include "core/utils/AsciiUtils.hpp"
#include "../Models.hpp"
#include "../OpenAIEndpointUtils.hpp"
#include "../../logging/Logger.hpp"
#include "../../utils/StringUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace core::llm::protocols {

ReasoningCapabilities openai_reasoning_capabilities(std::string_view model) noexcept {
    using core::utils::ascii::istarts_with;
    if (!istarts_with(model, "gpt-5")
        && !istarts_with(model, "o1")
        && !istarts_with(model, "o3")
        && !istarts_with(model, "o4")) {
        return {};
    }

    ReasoningCapabilities features{ReasoningCapability::Effort};
    if (istarts_with(model, "gpt-5.6")) {
        features = features | ReasoningCapability::MaxEffort;
    }
    return features;
}

namespace {

constexpr auto kZaiUsageSnapshotTtl = std::chrono::seconds{20};

[[nodiscard]] std::string lower_ascii(std::string_view value) {
    std::string lowered;
    lowered.reserve(value.size());
    for (const char ch : value) {
        lowered.push_back(static_cast<char>(
            std::tolower(static_cast<unsigned char>(ch))));
    }
    return lowered;
}

[[nodiscard]] std::optional<std::string_view>
find_header_case_insensitive(const cpr::Header& headers, std::string_view key) {
    if (const auto it = headers.find(std::string(key)); it != headers.end()) {
        return it->second;
    }

    const std::string key_lower = lower_ascii(key);
    for (const auto& [k, v] : headers) {
        if (k.size() != key_lower.size()) continue;
        bool equal = true;
        for (std::size_t i = 0; i < k.size(); ++i) {
            const char lhs = static_cast<char>(
                std::tolower(static_cast<unsigned char>(k[i])));
            if (lhs != key_lower[i]) {
                equal = false;
                break;
            }
        }
        if (equal) return v;
    }
    return std::nullopt;
}

[[nodiscard]] int32_t parse_int_header_case_insensitive(const cpr::Header& headers,
                                                        std::string_view key) noexcept {
    const auto value = find_header_case_insensitive(headers, key);
    if (!value.has_value() || value->empty()) return 0;
    try {
        return static_cast<int32_t>(std::stoi(std::string(*value)));
    } catch (...) {
        return 0;
    }
}

[[nodiscard]] std::optional<float>
parse_float_header_case_insensitive(const cpr::Header& headers,
                                    std::string_view key) noexcept {
    const auto value = find_header_case_insensitive(headers, key);
    if (!value.has_value() || value->empty()) return std::nullopt;
    try {
        float parsed = std::stof(std::string(*value));
        if (parsed > 1.5f && parsed <= 100.0f) {
            parsed /= 100.0f;
        }
        return std::clamp(parsed, 0.0f, 1.5f);
    } catch (...) {
        return std::nullopt;
    }
}

[[nodiscard]] RateLimitInfo parse_openai_compatible_rate_limit_headers(
    const cpr::Header& headers,
    int response_status_code) noexcept {
    RateLimitInfo info;

    info.requests_limit =
        parse_int_header_case_insensitive(headers, "x-ratelimit-limit-requests");
    info.requests_remaining =
        parse_int_header_case_insensitive(headers, "x-ratelimit-remaining-requests");
    info.tokens_limit =
        parse_int_header_case_insensitive(headers, "x-ratelimit-limit-tokens");
    info.tokens_remaining =
        parse_int_header_case_insensitive(headers, "x-ratelimit-remaining-tokens");
    info.retry_after = parse_int_header_case_insensitive(headers, "retry-after");
    info.is_rate_limited = (response_status_code == 429 || info.retry_after > 0);

    if (info.requests_limit > 0 && info.requests_remaining == 0 && !info.is_rate_limited) {
        info.requests_remaining = info.requests_limit;
    }
    if (info.tokens_limit > 0 && info.tokens_remaining == 0 && !info.is_rate_limited) {
        info.tokens_remaining = info.tokens_limit;
    }

    return info;
}

[[nodiscard]] RateLimitInfo parse_zai_rate_limit_headers(const cpr::Header& headers,
                                                         int response_status_code) noexcept {
    RateLimitInfo info =
        parse_openai_compatible_rate_limit_headers(headers, response_status_code);

    struct WindowHeader {
        std::string_view label;
        std::string_view key;
    };
    static constexpr std::array<WindowHeader, 2> kWindows{{
        {"5h", "x-ratelimit-unified-5h-utilization"},
        {"7d", "x-ratelimit-unified-7d-utilization"},
    }};

    for (const auto& window : kWindows) {
        if (auto value = parse_float_header_case_insensitive(headers, window.key);
            value.has_value()) {
            info.usage_windows.push_back({
                std::string(window.label),
                *value,
            });
        }
    }

    if (auto status = find_header_case_insensitive(headers, "x-ratelimit-unified-status");
        status.has_value()) {
        info.unified_status = std::string(*status);
        if (*status == "rate_limited" || *status == "rejected") {
            info.is_rate_limited = true;
        }
    }
    if (auto claim = find_header_case_insensitive(
            headers, "x-ratelimit-unified-representative-claim");
        claim.has_value()) {
        info.unified_representative_claim = std::string(*claim);
    }

    return info;
}

struct ZaiUsageSnapshot {
    std::vector<UsageWindow> windows;
};

struct CachedZaiUsageSnapshot {
    ZaiUsageSnapshot value;
    std::chrono::steady_clock::time_point expires_at;
};

void merge_usage_window(RateLimitInfo& info, UsageWindow incoming) {
    auto existing = std::find_if(
        info.usage_windows.begin(),
        info.usage_windows.end(),
        [&](const UsageWindow& current) {
            return current.label == incoming.label;
        });
    if (existing != info.usage_windows.end()) {
        existing->utilization = incoming.utilization;
    } else {
        info.usage_windows.push_back(std::move(incoming));
    }
}

[[nodiscard]] std::optional<float> read_json_number(simdjson::dom::object object,
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

[[nodiscard]] std::optional<std::string_view> read_json_string(simdjson::dom::object object,
                                                               std::string_view key) {
    std::string_view value;
    if (object[key].get(value) == simdjson::SUCCESS) return value;
    return std::nullopt;
}

[[nodiscard]] std::optional<int64_t> read_json_int(simdjson::dom::object object,
                                                   std::string_view key) {
    int64_t value = 0;
    if (object[key].get(value) == simdjson::SUCCESS) return value;
    double as_double = 0.0;
    if (object[key].get(as_double) == simdjson::SUCCESS) {
        return static_cast<int64_t>(as_double);
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<std::string> zai_quota_label(simdjson::dom::object limit) {
    const auto type = read_json_string(limit, "type");
    const auto unit = read_json_int(limit, "unit");
    if (!type.has_value() || !unit.has_value()) return std::nullopt;

    if (*type == "TOKENS_LIMIT" && *unit == 3) return std::string{"5h"};
    if (*type == "TOKENS_LIMIT" && *unit == 6) return std::string{"7d"};
    if (*type == "TIME_LIMIT" && *unit == 5) return std::string{"web"};
    return std::nullopt;
}

[[nodiscard]] std::optional<ZaiUsageSnapshot>
parse_zai_usage_quota_payload(std::string_view payload) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    if (parser.parse(payload).get(doc) != simdjson::SUCCESS) return std::nullopt;

    simdjson::dom::object data;
    if (doc["data"].get(data) != simdjson::SUCCESS) return std::nullopt;

    simdjson::dom::array limits;
    if (data["limits"].get(limits) != simdjson::SUCCESS) return std::nullopt;

    ZaiUsageSnapshot snapshot;
    for (simdjson::dom::element element : limits) {
        simdjson::dom::object limit;
        if (element.get(limit) != simdjson::SUCCESS) continue;

        const auto label = zai_quota_label(limit);
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

        snapshot.windows.push_back(UsageWindow{
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
        snapshot.windows.begin(),
        snapshot.windows.end(),
        [&](const UsageWindow& lhs, const UsageWindow& rhs) {
            return rank(lhs.label) < rank(rhs.label);
        });
    return snapshot;
}

[[nodiscard]] std::string trim_trailing_slash(std::string_view url) {
    while (!url.empty() && url.back() == '/') {
        url.remove_suffix(1);
    }
    return std::string(url);
}

[[nodiscard]] std::string zai_management_api_base_url(std::string_view base_url) {
    std::string url = trim_trailing_slash(base_url);
    const std::array<std::string_view, 2> suffixes{
        "/api/coding/paas/v4",
        "/api/paas/v4",
    };
    for (const auto suffix : suffixes) {
        if (url.size() >= suffix.size()
            && url.compare(url.size() - suffix.size(), suffix.size(), suffix) == 0) {
            url.resize(url.size() - suffix.size());
            url += "/api";
            return url;
        }
    }
    return url;
}

[[nodiscard]] std::string zai_usage_cache_key(std::string_view base_url,
                                              const cpr::Header& request_headers) {
    std::string key = zai_management_api_base_url(base_url);
    const auto auth_header = find_header_case_insensitive(request_headers, "Authorization");
    const std::size_t auth_hash = std::hash<std::string_view>{}(
        auth_header.has_value() ? *auth_header : std::string_view{});
    key += "|auth:";
    key += std::to_string(auth_hash);
    return key;
}

[[nodiscard]] std::unordered_map<std::string, CachedZaiUsageSnapshot>&
zai_usage_cache() {
    static std::unordered_map<std::string, CachedZaiUsageSnapshot> cache;
    return cache;
}

[[nodiscard]] std::mutex& zai_usage_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

[[nodiscard]] std::optional<ZaiUsageSnapshot>
load_cached_zai_usage_snapshot(const std::string& key) {
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lock(zai_usage_cache_mutex());
    auto& cache = zai_usage_cache();
    auto it = cache.find(key);
    if (it == cache.end()) return std::nullopt;
    if (it->second.expires_at <= now) {
        cache.erase(it);
        return std::nullopt;
    }
    return it->second.value;
}

void store_cached_zai_usage_snapshot(std::string key,
                                     const ZaiUsageSnapshot& snapshot) {
    const auto expires_at = std::chrono::steady_clock::now() + kZaiUsageSnapshotTtl;
    std::scoped_lock lock(zai_usage_cache_mutex());
    auto& cache = zai_usage_cache();
    cache[std::move(key)] = CachedZaiUsageSnapshot{snapshot, expires_at};
}

[[nodiscard]] std::optional<ZaiUsageSnapshot>
fetch_zai_usage_snapshot(std::string_view base_url,
                         const cpr::Header& request_headers) {
    std::string url = zai_management_api_base_url(base_url);
    url += "/monitor/usage/quota/limit";

    cpr::Header headers = request_headers;
    headers["Accept"] = "application/json";
    if (!find_header_case_insensitive(headers, "Accept-Language").has_value()) {
        headers["Accept-Language"] = "en";
    }

    const cpr::Response response = cpr::Get(
        cpr::Url{url},
        headers,
        cpr::Timeout{1500});

    if (response.error.code != cpr::ErrorCode::OK) return std::nullopt;
    if (response.status_code != 200 || response.text.empty()) return std::nullopt;
    return parse_zai_usage_quota_payload(response.text);
}

void merge_zai_usage_snapshot(RateLimitInfo& info, const ZaiUsageSnapshot& snapshot) {
    for (const auto& incoming : snapshot.windows) {
        merge_usage_window(info, incoming);
    }
}

[[nodiscard]] bool has_usage_window(const RateLimitInfo& info, std::string_view label) {
    return std::any_of(
        info.usage_windows.begin(),
        info.usage_windows.end(),
        [&](const UsageWindow& window) {
            return window.label == label;
        });
}

[[nodiscard]] std::string normalize_openai_effort(std::string_view raw_effort,
                                                  std::string_view model) {
    std::string effort = lower_ascii(raw_effort);
    std::erase_if(effort, [](unsigned char ch) {
        return std::isspace(ch);
    });
    if (effort == "auto" || effort == "unset" || effort == "default") {
        return {};
    }
    if (effort == "low" || effort == "medium" || effort == "high") {
        return effort;
    }
    if (effort == "max") {
        return openai_reasoning_capabilities(model).supports(
            ReasoningCapability::MaxEffort) ? "max" : "high";
    }
    return {};
}

[[nodiscard]] bool is_zai_coding_model(std::string_view model) {
    std::string normalized = lower_ascii(core::utils::str::trim_ascii_view(model));
    return normalized == "glm-5.2"
        || normalized == "glm-5-turbo"
        || normalized == "glm-4.7"
        || normalized == "glm-4.5-air";
}

[[nodiscard]] bool is_zai_glm_model(std::string_view model) {
    std::string normalized = lower_ascii(core::utils::str::trim_ascii_view(model));
    return normalized.starts_with("glm-");
}

[[nodiscard]] std::string normalize_zai_effort(std::string_view raw_effort) {
    std::string effort = lower_ascii(raw_effort);
    std::erase_if(effort, [](unsigned char ch) {
        return std::isspace(ch);
    });
    if (effort == "auto" || effort == "unset" || effort == "default") {
        return {};
    }
    if (effort == "off" || effort == "disabled" || effort == "disable") {
        return "off";
    }
    if (effort == "low" || effort == "medium" || effort == "high" || effort == "max") {
        return effort;
    }
    return {};
}

struct ZaiParseResult {
    std::string content;
    std::string reasoning_content;
    std::vector<ToolCall> tools;
    int32_t prompt_tokens = 0;
    int32_t completion_tokens = 0;
};

[[nodiscard]] bool read_usage_object(simdjson::dom::object usage,
                                     int32_t& prompt_tokens,
                                     int32_t& completion_tokens) {
    int64_t pt = 0;
    int64_t ct = 0;
    const bool has_prompt = usage["prompt_tokens"].get(pt) == simdjson::SUCCESS;
    const bool has_completion = usage["completion_tokens"].get(ct) == simdjson::SUCCESS;
    if (!has_prompt && !has_completion) return false;
    prompt_tokens = static_cast<int32_t>(pt);
    completion_tokens = static_cast<int32_t>(ct);
    return pt > 0 || ct > 0;
}

[[nodiscard]] ZaiParseResult parse_zai_sse_chunk(std::string_view json_str) {
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(json_str);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) {
        return {};
    }

    ZaiParseResult result;

    simdjson::dom::object usage_obj;
    bool has_usage = false;
    if (doc["usage"].get(usage_obj) == simdjson::SUCCESS
        && read_usage_object(usage_obj, result.prompt_tokens, result.completion_tokens)) {
        has_usage = true;
    }

    simdjson::dom::array choices;
    if (doc["choices"].get(choices) != simdjson::SUCCESS) {
        return result;
    }

    for (simdjson::dom::element choice : choices) {
        if (!has_usage) {
            simdjson::dom::object choice_usage;
            if (choice["usage"].get(choice_usage) == simdjson::SUCCESS
                && read_usage_object(choice_usage, result.prompt_tokens,
                                     result.completion_tokens)) {
                has_usage = true;
            }
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
            result.reasoning_content.append(reasoning.data(), reasoning.size());
        }

        simdjson::dom::array tool_calls_arr;
        if (delta["tool_calls"].get(tool_calls_arr) == simdjson::SUCCESS) {
            for (simdjson::dom::element tc : tool_calls_arr) {
                ToolCall call;
                int64_t index_v = -1;
                if (tc["index"].get(index_v) == simdjson::SUCCESS) {
                    call.index = static_cast<int>(index_v);
                }

                std::string_view id_v;
                if (tc["id"].get(id_v) == simdjson::SUCCESS) {
                    call.id = std::string(id_v);
                }

                std::string_view type_v;
                if (tc["type"].get(type_v) == simdjson::SUCCESS) {
                    call.type = std::string(type_v);
                }

                simdjson::dom::object func;
                if (tc["function"].get(func) == simdjson::SUCCESS) {
                    std::string_view name_v;
                    if (func["name"].get(name_v) == simdjson::SUCCESS) {
                        call.function.name = std::string(name_v);
                    }
                    std::string_view args_v;
                    if (func["arguments"].get(args_v) == simdjson::SUCCESS) {
                        call.function.arguments = std::string(args_v);
                    }
                }

                result.tools.push_back(std::move(call));
            }
        }
    }

    return result;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// parse_openai_sse_chunk — shared pure parser for any OpenAI-compatible stream
// ─────────────────────────────────────────────────────────────────────────────

OpenAIChatChunk
parse_openai_sse_chunk(std::string_view json_str) {
    if (json_str.empty()) {
        return {};
    }

    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(json_str);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) {
        return {};
    }

    simdjson::dom::array choices;
    if (doc["choices"].get(choices) != simdjson::SUCCESS) {
        return {};
    }
    std::string              content_str;
    std::vector<ToolCall>    tools;
    std::string              reasoning;

    for (simdjson::dom::element choice : choices) {
        simdjson::dom::object delta;
        if (choice["delta"].get(delta) != simdjson::SUCCESS) {
            continue;
        }

        std::string_view content;
        if (delta["content"].get(content) == simdjson::SUCCESS) {
            content_str = std::string(content);
        }

        // OpenAI-compatible thinking models (DeepSeek-R1, Qwen-thinking via
        // OpenRouter, etc.) stream chain-of-thought as delta.reasoning_content.
        // Capture it so the TUI reasoning disclosure lights up for these models.
        std::string_view reasoning_delta;
        if (delta["reasoning_content"].get(reasoning_delta) == simdjson::SUCCESS
            && !reasoning_delta.empty()) {
            reasoning.append(reasoning_delta.data(), reasoning_delta.size());
        }

        simdjson::dom::array tool_calls_arr;
        if (delta["tool_calls"].get(tool_calls_arr) == simdjson::SUCCESS) {
            for (simdjson::dom::element tc : tool_calls_arr) {
                ToolCall call;
                int64_t index_v;
                if (tc["index"].get(index_v) == simdjson::SUCCESS)
                    call.index = static_cast<int>(index_v);
                std::string_view id_v;
                if (tc["id"].get(id_v) == simdjson::SUCCESS)
                    call.id = std::string(id_v);
                std::string_view type_v;
                if (tc["type"].get(type_v) == simdjson::SUCCESS)
                    call.type = std::string(type_v);
                simdjson::dom::object func;
                if (tc["function"].get(func) == simdjson::SUCCESS) {
                    std::string_view name_v;
                    if (func["name"].get(name_v) == simdjson::SUCCESS)
                        call.function.name = std::string(name_v);
                    std::string_view args_v;
                    if (func["arguments"].get(args_v) == simdjson::SUCCESS)
                        call.function.arguments = std::string(args_v);
                }
                tools.push_back(std::move(call));
            }
        }
    }

    return {std::move(content_str), std::move(tools), std::move(reasoning)};
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenAIProtocol
// ─────────────────────────────────────────────────────────────────────────────

std::string OpenAIProtocol::serialize(const ChatRequest& req) const {
    std::string payload = Serializer::serialize(req);

    if (payload.ends_with('}')) {
        payload.pop_back();
        if (openai_reasoning_capabilities(req.model).supports_effort()) {
            const std::string effort = normalize_openai_effort(req.effort, req.model);
            if (!effort.empty()) {
                payload += R"(,"reasoning_effort":")";
                payload += core::utils::escape_json_string(effort);
                payload += '"';
            }
        }
        append_extra_fields(payload, req);
        if ((stream_usage_ || req.stream_include_usage) && req.stream) {
            payload += R"(,"stream_options":{"include_usage":true})";
        }
        payload += '}';
    }

    return payload;
}

cpr::Header OpenAIProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Accept",       "text/event-stream"},
    };
    for (const auto& [k, v] : auth.headers) {
        headers[k] = v;
    }

    // OpenAI Codex backend account-scoped tokens require this header.
    if (auto it = auth.properties.find("account_id");
        it != auth.properties.end() && !it->second.empty()
        && headers.count("chatgpt-account-id") == 0) {
        headers["chatgpt-account-id"] = it->second;
    }

    return headers;
}

std::string OpenAIProtocol::build_url(std::string_view base_url,
                                      std::string_view model) const {
    if (openai_endpoint::is_azure_openai_base_url(base_url)) {
        const char* api_version_env = std::getenv("AZURE_OPENAI_API_VERSION");
        const std::string api_version = (api_version_env && api_version_env[0] != '\0')
            ? std::string(api_version_env)
            : "2024-12-01-preview";
        return openai_endpoint::build_azure_chat_completions_url(
            base_url,
            model,
            api_version);
    }
    std::string normalized(base_url);
    while (!normalized.empty() && normalized.back() == '/') {
        normalized.pop_back();
    }
    return normalized + "/chat/completions";
}

ParseResult OpenAIProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed)) return {};
    const std::string_view json_sv = parsed.data;

    if (parsed.is_done) {
        ParseResult r;
        r.done = true;
        return r;
    }

    ParseResult result;

    // Extract usage from stream_options chunk (choices array will be empty).
    {
        thread_local simdjson::dom::parser usage_parser;
        simdjson::padded_string ps(json_sv);
        simdjson::dom::element doc;
        if (usage_parser.parse(ps).get(doc) == simdjson::SUCCESS) {
            simdjson::dom::object usage_obj;
            if (doc["usage"].get(usage_obj) == simdjson::SUCCESS) {
                (void)parse_openai_usage(usage_obj, result);
            }
        }
    }

    auto [content, tools, reasoning] = parse_openai_sse_chunk(json_sv);
    if (!content.empty() || !tools.empty() || !reasoning.empty()) {
        // Combine content, tool calls, and reasoning into a single chunk. The
        // Agent routes each field independently, and merging here (mirroring the
        // Z.ai parser) keeps reasoning on the same chunk as its content so that
        // subclasses which call this base and then augment (e.g. Mistral) don't
        // end up with a duplicate reasoning-only chunk.
        StreamChunk chunk;
        chunk.content           = std::move(content);
        chunk.tools             = std::move(tools);
        chunk.reasoning_content = std::move(reasoning);
        result.chunks.push_back(std::move(chunk));
    }

    return result;
}

void OpenAIProtocol::on_response(const HttpResponse& response) {
    last_rate_limit_ =
        parse_openai_compatible_rate_limit_headers(response.headers, response.status_code);
}

std::string ZaiProtocol::serialize(const ChatRequest& req) const {
    Serializer::Options options;
    options.include_reasoning_content = true;

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

    if (parsed.is_done) {
        ParseResult result;
        result.done = true;
        return result;
    }

    ParseResult result;
    auto zai_result = parse_zai_sse_chunk(parsed.data);
    result.prompt_tokens = zai_result.prompt_tokens;
    result.completion_tokens = zai_result.completion_tokens;

    if (!zai_result.content.empty()
        || !zai_result.reasoning_content.empty()
        || !zai_result.tools.empty()) {
        StreamChunk chunk;
        chunk.content = std::move(zai_result.content);
        chunk.reasoning_content = std::move(zai_result.reasoning_content);
        chunk.tools = std::move(zai_result.tools);
        result.chunks.push_back(std::move(chunk));
    }

    return result;
}

void ZaiProtocol::on_response(const HttpResponse& response) {
    last_rate_limit_ = parse_zai_rate_limit_headers(response.headers, response.status_code);
}

void ZaiProtocol::enrich_rate_limit(std::string_view base_url,
                                    const cpr::Header& request_headers,
                                    const HttpResponse& response) {
    if (response.status_code <= 0 || response.status_code == 401 || response.status_code == 403) {
        return;
    }

    const bool needs_enrichment =
        !has_usage_window(last_rate_limit_, "5h")
        || !has_usage_window(last_rate_limit_, "7d");
    if (!needs_enrichment) return;

    const std::string cache_key = zai_usage_cache_key(base_url, request_headers);
    if (const auto cached = load_cached_zai_usage_snapshot(cache_key);
        cached.has_value()) {
        merge_zai_usage_snapshot(last_rate_limit_, *cached);
        return;
    }

    if (const auto usage = fetch_zai_usage_snapshot(base_url, request_headers);
        usage.has_value()) {
        store_cached_zai_usage_snapshot(cache_key, *usage);
        merge_zai_usage_snapshot(last_rate_limit_, *usage);
    }
}

void ZaiProtocol::append_extra_fields(std::string& payload, const ChatRequest& req) const {
    if (!is_zai_glm_model(req.model)) {
        return;
    }

    const std::string effort = normalize_zai_effort(req.effort);
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
    if (is_zai_coding_model(req.model)) {
        return;
    }

    throw std::invalid_argument(
        "Z.ai Coding Plan supports only glm-5.2, glm-5-turbo, "
        "glm-4.7, and glm-4.5-air; got '" + req.model + "'");
}

} // namespace core::llm::protocols
