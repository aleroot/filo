#include "AnthropicProtocol.hpp"
#include "SseUtils.hpp"
#include "../Models.hpp"
#include "../ModelRegistry.hpp"
#include "../AnthropicCompatibility.hpp"
#include "../../logging/Logger.hpp"
#include "../../utils/JsonUtils.hpp"
#include "../../utils/StringUtils.hpp"
#include "../../utils/AsciiUtils.hpp"
#include "../../utils/TimeUtils.hpp"
#include "../../tools/StrictToolSchema.hpp"
#include "../../tools/ToolSchema.hpp"
#include "../StrictToolPolicy.hpp"
#include <simdjson.h>
#include <curl/curl.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cctype>
#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace core::llm::protocols {

using core::llm::ToolSchemaWire;
using core::llm::configured_strict_tool_dialect;

namespace {
    constexpr std::string_view ANTHROPIC_VERSION       = "2023-06-01";
    constexpr std::string_view ANTHROPIC_BETA_CLAUDE_CODE = "claude-code-20250219";
    constexpr std::string_view ANTHROPIC_BETA_THINKING = "interleaved-thinking-2025-05-14";
    constexpr std::string_view ANTHROPIC_BETA_CONTEXT_1M = "context-1m-2025-08-07";
    constexpr std::string_view ANTHROPIC_BETA_OAUTH    = "oauth-2025-04-20";
    constexpr std::string_view ANTHROPIC_BILLING_HEADER = anthropic::kBillingHeader;

    struct HeaderWindowDef {
        std::string_view suffix;
        std::string_view label;
    };

    struct ClaudeUsageSnapshot {
        std::vector<UsageWindow> windows;
    };

    struct CachedClaudeUsageSnapshot {
        ClaudeUsageSnapshot value;
        std::chrono::steady_clock::time_point expires_at;
    };

    constexpr auto kClaudeUsageSnapshotTtl = std::chrono::seconds(20);

    [[nodiscard]] std::string lower_header_key(std::string_view value) {
        std::string out;
        out.reserve(value.size());
        for (char ch : value) {
            out.push_back(static_cast<char>(
                std::tolower(static_cast<unsigned char>(ch))));
        }
        return out;
    }

    [[nodiscard]] std::optional<std::string_view>
    find_header_case_insensitive(const cpr::Header& headers, std::string_view key) {
        if (auto it = headers.find(std::string(key)); it != headers.end()) {
            return it->second;
        }

        const std::string key_lower = lower_header_key(key);
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

    // Helper to safely convert string header to int32_t
    int32_t safe_stoi32(std::string_view sv, int32_t default_val = 0) {
        if (sv.empty()) return default_val;
        try {
            return static_cast<int32_t>(std::stoi(std::string(sv)));
        } catch (...) {
            return default_val;
        }
    }
    
    // Helper to safely convert string header to int64_t
    int64_t safe_stoi64(std::string_view sv, int64_t default_val = 0) {
        if (sv.empty()) return default_val;
        try {
            return std::stoll(std::string(sv));
        } catch (...) {
            return default_val;
        }
    }
    
    bool try_parse_float(std::string_view sv, float& out) {
        if (sv.empty()) return false;
        try {
            std::size_t parsed = 0;
            const std::string raw(sv);
            const float value = std::stof(raw, &parsed);
            while (parsed < raw.size()
                   && std::isspace(static_cast<unsigned char>(raw[parsed]))) {
                ++parsed;
            }
            if (parsed != raw.size()) return false;
            out = value;
            return true;
        } catch (...) {
            return false;
        }
    }

    [[nodiscard]] float normalize_utilization(float value, bool endpoint_percent) {
        if (endpoint_percent || (value > 1.5f && value <= 100.0f)) {
            value /= 100.0f;
        }
        return std::clamp(value, 0.0f, 1.5f);
    }

    [[nodiscard]] std::optional<float>
    parse_float_element(simdjson::dom::element element) noexcept {
        double as_double = 0.0;
        if (element.get(as_double) == simdjson::SUCCESS) {
            return static_cast<float>(as_double);
        }

        int64_t as_i64 = 0;
        if (element.get(as_i64) == simdjson::SUCCESS) {
            return static_cast<float>(as_i64);
        }

        std::string_view as_string;
        if (element.get(as_string) == simdjson::SUCCESS) {
            float value = 0.0f;
            if (try_parse_float(as_string, value)) return value;
        }
        return std::nullopt;
    }

    bool safe_parse_bool(std::string_view sv, bool default_val = false) {
        if (sv.empty()) return default_val;
        std::string lowered;
        lowered.reserve(sv.size());
        for (char ch : sv) {
            lowered.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
        if (lowered == "1" || lowered == "true" || lowered == "yes") return true;
        if (lowered == "0" || lowered == "false" || lowered == "no") return false;
        return default_val;
    }

    [[nodiscard]] std::string infer_anthropic_event_type(std::string_view payload) {
        thread_local simdjson::dom::parser parser;
        const core::utils::json::ParserRetentionGuard parser_guard{parser};
        simdjson::padded_string padded(payload);
        simdjson::dom::element doc;
        if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return {};

        std::string_view type;
        if (doc["type"].get(type) != simdjson::SUCCESS) return {};
        return std::string(type);
    }

    [[nodiscard]] bool is_retryable_anthropic_stream_error(std::string_view type) {
        return type == "overloaded_error"
            || type == "rate_limit_error"
            || type == "api_error"
            || type == "timeout_error";
    }

    [[nodiscard]] int model_default_max_tokens(std::string_view model,
                                                int fallback) {
        const auto info = ModelRegistry::instance().lookup(model);
        if (!info) return fallback;
        const int32_t max_tokens = info->effective_max_tokens();
        return max_tokens > 0 ? max_tokens : fallback;
    }

    std::string normalized_effort_or_empty(std::string_view raw) {
        std::string normalized = core::utils::str::to_lower_ascii_copy(core::utils::str::trim_ascii_copy(raw));
        std::erase_if(normalized, [](unsigned char ch) {
            return std::isspace(ch);
        });
        if (normalized == "auto" || normalized == "unset" || normalized == "default") {
            return {};
        }
        if (normalized == "x-high" || normalized == "x_high"
            || normalized == "extra-high" || normalized == "extra_high") {
            normalized = "xhigh";
        }
        if (normalized == "low" || normalized == "medium"
            || normalized == "high" || normalized == "xhigh"
            || normalized == "max") {
            return normalized;
        }
        return {};
    }

    // True when the user explicitly turned reasoning off via `/effort off`
    // (mapped to "none"/"off"/"disabled"). Distinct from "auto"/empty, which
    // both normalize to an empty effort but mean "provider default".
    bool effort_explicitly_disabled(std::string_view raw) {
        std::string normalized =
            core::utils::str::to_lower_ascii_copy(core::utils::str::trim_ascii_copy(raw));
        std::erase_if(normalized, [](unsigned char ch) { return std::isspace(ch); });
        return normalized == "none" || normalized == "off" || normalized == "disabled";
    }

    bool ends_with_1m_suffix(std::string_view model) {
        if (model.size() < 4) return false;
        const std::size_t tail = model.size() - 4;
        return model[tail] == '['
            && model[tail + 1] == '1'
            && (model[tail + 2] == 'm' || model[tail + 2] == 'M')
            && model[tail + 3] == ']';
    }

    struct ModelNormalization {
        std::string model;
        bool use_context_1m = false;
    };

    ModelNormalization normalize_requested_claude_model(std::string_view raw_model) {
        ModelNormalization out{
            .model = core::utils::str::trim_ascii_copy(raw_model),
            .use_context_1m = false,
        };
        if (out.model.empty()) return out;

        if (ends_with_1m_suffix(out.model)) {
            out.model = core::utils::str::trim_ascii_copy(std::string_view(out.model).substr(0, out.model.size() - 4));
            out.use_context_1m = true;
        }

        const std::string lowered = core::utils::str::to_lower_ascii_copy(out.model);
        if (lowered == "sonnet") {
            out.model = std::string(anthropic::kDefaultSonnet);
        } else if (anthropic::is_fable_alias(lowered)) {
            out.model = std::string(anthropic::kDefaultFable);
        } else if (lowered == "opus") {
            out.model = std::string(anthropic::kDefaultOpus);
        } else if (anthropic::is_opus_55_alias(lowered)) {
            out.model = std::string(anthropic::kDefaultOpus55);
        } else if (lowered == "haiku") {
            out.model = std::string(anthropic::kDefaultHaiku);
        } else if (lowered == "opusplan") {
            out.model = std::string(anthropic::kDefaultSonnet);
        } else if (const auto card = ModelRegistry::instance().lookup(lowered);
                   card && card->provider == "anthropic") {
            out.model = card->canonical_id;
        }

        return out;
    }

    std::string compose_anthropic_system_prompt(const ChatRequest& req) {
        std::string base_system =
            "x-anthropic-billing-header: " + std::string(ANTHROPIC_BILLING_HEADER);

        for (const auto& msg : req.messages) {
            if (msg.role != "system" || msg.content.empty()) continue;

            // Avoid duplicating the billing marker if callers already provide it.
            if (msg.content.find("x-anthropic-billing-header:") != std::string::npos) {
                return msg.content;
            }

            base_system += "\n\n";
            base_system += msg.content;
            return base_system;
        }

        return base_system;
    }

    void append_anthropic_system_field(std::string& payload, const ChatRequest& req) {
        payload += R"(,"system":)";
        if (req.prompt_plan.empty()) {
            payload += '"';
            payload += core::utils::escape_json_string(compose_anthropic_system_prompt(req));
            payload += '"';
            return;
        }

        const auto& layers = req.prompt_plan.layers();
        std::optional<std::size_t> workspace_cache_boundary;
        for (std::size_t i = 0; i < layers.size(); ++i) {
            if (layers[i].stability <= core::context::PromptStability::Workspace) {
                workspace_cache_boundary = i;
            }
        }

        payload += R"([{"type":"text","text":")";
        payload += core::utils::escape_json_string(
            "x-anthropic-billing-header: " + std::string(ANTHROPIC_BILLING_HEADER));
        payload += R"("})";
        for (std::size_t i = 0; i < layers.size(); ++i) {
            payload += R"(,{"type":"text","text":")";
            payload += core::utils::escape_json_string(layers[i].content);
            payload += '"';
            if (workspace_cache_boundary == i) {
                payload += R"(,"cache_control":{"type":"ephemeral"})";
            }
            payload += '}';
        }
        payload += ']';
    }
    
    // Parse ISO 8601 timestamp or duration to unix epoch seconds.
    int64_t parse_iso8601_timestamp(std::string_view sv) {
        return core::utils::time::parse_timestamp_or_duration(sv);
    }

    // ── Response lifecycle hook implementations ──────────────────────────────
    // Named with an `impl_` prefix to avoid name-hiding issues when called
    // from member functions that share the same base name (e.g. the virtual
    // override `format_error_message` would shadow a plain `format_error_message`
    // inside the class's own method body due to C++ name-lookup rules).
    // ─────────────────────────────────────────────────────────────────────────

    [[nodiscard]] std::unordered_map<std::string, CachedClaudeUsageSnapshot>&
    claude_usage_cache() {
        static std::unordered_map<std::string, CachedClaudeUsageSnapshot> cache;
        return cache;
    }

    [[nodiscard]] std::mutex& claude_usage_cache_mutex() {
        static std::mutex mutex;
        return mutex;
    }

    [[nodiscard]] std::string trim_trailing_slash(std::string_view url) {
        while (!url.empty() && url.back() == '/') {
            url.remove_suffix(1);
        }
        return std::string(url);
    }

    [[nodiscard]] std::string usage_cache_key(std::string_view base_url,
                                              const cpr::Header& request_headers) {
        std::string key = trim_trailing_slash(base_url);
        const auto auth_header = find_header_case_insensitive(request_headers, "Authorization");
        const std::size_t auth_hash = std::hash<std::string_view>{}(
            auth_header.has_value() ? *auth_header : std::string_view{});
        key += "|auth:";
        key += std::to_string(auth_hash);
        return key;
    }

    [[nodiscard]] std::optional<ClaudeUsageSnapshot>
    load_cached_claude_usage_snapshot(const std::string& key) {
        const auto now = std::chrono::steady_clock::now();
        std::scoped_lock lock(claude_usage_cache_mutex());
        auto& cache = claude_usage_cache();
        auto it = cache.find(key);
        if (it == cache.end()) return std::nullopt;
        if (it->second.expires_at <= now) {
            cache.erase(it);
            return std::nullopt;
        }
        return it->second.value;
    }

    void store_cached_claude_usage_snapshot(std::string key,
                                            const ClaudeUsageSnapshot& snapshot) {
        const auto expires_at = std::chrono::steady_clock::now() + kClaudeUsageSnapshotTtl;
        std::scoped_lock lock(claude_usage_cache_mutex());
        auto& cache = claude_usage_cache();
        cache[std::move(key)] = CachedClaudeUsageSnapshot{snapshot, expires_at};
    }

    void merge_usage_window(RateLimitInfo& info, UsageWindow incoming) {
        auto existing = std::find_if(
            info.usage_windows.begin(),
            info.usage_windows.end(),
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

    void merge_claude_usage_snapshot(RateLimitInfo& info,
                                     const ClaudeUsageSnapshot& snapshot) {
        for (const auto& window : snapshot.windows) {
            merge_usage_window(info, window);
        }
    }

    [[nodiscard]] std::optional<ClaudeUsageSnapshot>
    parse_claude_usage_payload(std::string_view payload) {
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        simdjson::padded_string padded(payload);
        if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
            return std::nullopt;
        }

        ClaudeUsageSnapshot snapshot;
        static constexpr std::array kWindows{
            HeaderWindowDef{"five_hour", "5h"},
            HeaderWindowDef{"seven_day", "7d"},
            HeaderWindowDef{"seven_day_oauth_apps", "oauth-apps"},
            HeaderWindowDef{"seven_day_opus", "opus"},
            HeaderWindowDef{"seven_day_sonnet", "sonnet"},
        };

        for (const auto& w : kWindows) {
            simdjson::dom::object obj;
            if (doc[w.suffix].get(obj) != simdjson::SUCCESS) continue;

            simdjson::dom::element utilization_el;
            if (obj["utilization"].get(utilization_el) != simdjson::SUCCESS) continue;
            const auto utilization = parse_float_element(utilization_el);
            if (!utilization.has_value()) continue;

            int64_t resets_at = 0;
            std::string_view resets_at_str;
            if (obj["resets_at"].get(resets_at_str) == simdjson::SUCCESS) {
                resets_at = parse_iso8601_timestamp(resets_at_str);
            }

            snapshot.windows.push_back(UsageWindow{
                .label = std::string(w.label),
                .utilization = normalize_utilization(*utilization, true),
                .resets_at = resets_at,
            });
        }

        simdjson::dom::object extra_usage;
        if (doc["extra_usage"].get(extra_usage) == simdjson::SUCCESS) {
            simdjson::dom::element utilization_el;
            if (extra_usage["utilization"].get(utilization_el) == simdjson::SUCCESS) {
                if (const auto utilization = parse_float_element(utilization_el);
                    utilization.has_value()) {
                    int64_t resets_at = 0;
                    std::string_view resets_at_str;
                    if (extra_usage["resets_at"].get(resets_at_str) == simdjson::SUCCESS) {
                        resets_at = parse_iso8601_timestamp(resets_at_str);
                    }
                    snapshot.windows.push_back(UsageWindow{
                        .label = "overage",
                        .utilization = normalize_utilization(*utilization, true),
                        .resets_at = resets_at,
                    });
                }
            }
        }

        if (snapshot.windows.empty()) return std::nullopt;
        return snapshot;
    }

    [[nodiscard]] bool should_query_claude_usage_endpoint(const cpr::Header& request_headers,
                                                          const HttpResponse& response) {
        if (response.status_code != 200) return false;
        if (!find_header_case_insensitive(request_headers, "Authorization").has_value()) {
            return false;
        }
        return true;
    }

    [[nodiscard]] std::optional<ClaudeUsageSnapshot>
    fetch_claude_usage_snapshot(std::string_view base_url,
                                const cpr::Header& request_headers) {
        std::string url = trim_trailing_slash(base_url);
        url += "/api/oauth/usage";

        cpr::Header headers = request_headers;
        headers["Accept"] = "application/json";
        headers["Content-Type"] = "application/json";

        const cpr::Response response = cpr::Get(
            cpr::Url{url},
            headers,
            cpr::Timeout{1500});

        if (response.error.code != cpr::ErrorCode::OK) {
            return std::nullopt;
        }
        if (response.status_code != 200 || response.text.empty()) {
            return std::nullopt;
        }
        return parse_claude_usage_payload(response.text);
    }

    RateLimitInfo impl_parse_rate_limit_headers(const cpr::Header& headers) {
        RateLimitInfo info;

        // Standard rate limit headers (requests per minute)
        if (auto value = find_header_case_insensitive(headers, "anthropic-ratelimit-requests-limit")) {
            info.requests_limit = safe_stoi32(*value);
        }
        if (auto value = find_header_case_insensitive(headers, "anthropic-ratelimit-requests-remaining")) {
            info.requests_remaining = safe_stoi32(*value);
        }
        if (auto value = find_header_case_insensitive(headers, "anthropic-ratelimit-requests-reset")) {
            info.requests_reset = parse_iso8601_timestamp(*value);
        }

        // Token-based rate limits (tokens per minute)
        if (auto value = find_header_case_insensitive(headers, "anthropic-ratelimit-tokens-limit")) {
            info.tokens_limit = safe_stoi32(*value);
        }
        if (auto value = find_header_case_insensitive(headers, "anthropic-ratelimit-tokens-remaining")) {
            info.tokens_remaining = safe_stoi32(*value);
        }
        if (auto value = find_header_case_insensitive(headers, "anthropic-ratelimit-tokens-reset")) {
            info.tokens_reset = parse_iso8601_timestamp(*value);
        }

        // Retry-after header (present on 429 responses)
        if (auto value = find_header_case_insensitive(headers, "retry-after")) {
            float seconds = 0;
            if (try_parse_float(*value, seconds) && std::isfinite(seconds) && seconds >= 0) {
                info.retry_after = static_cast<int32_t>(std::ceil(std::min<double>(
                    seconds, std::numeric_limits<int32_t>::max())));
            } else {
                const std::string date(*value);
                const auto deadline = curl_getdate(date.c_str(), nullptr);
                if (deadline >= 0) {
                    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
                    info.retry_after = static_cast<int32_t>(std::clamp<int64_t>(
                        deadline - now, 0, std::numeric_limits<int32_t>::max()));
                }
            }
            if (info.retry_after > 0) {
                info.is_rate_limited = true;
            }
        }

        // Subscription usage windows (for OAuth/claude.ai users).
        // Anthropic returns two windows: a 5-hour rolling window and a 7-day window.
        // The representative-claim header indicates which is currently authoritative.
        // Add new labels here if Anthropic introduces additional windows.
        for (const auto& [window, label] : std::array{
                 HeaderWindowDef{"5h", "5h"},
                 HeaderWindowDef{"7d", "7d"},
                 HeaderWindowDef{"seven_day_oauth_apps", "oauth-apps"},
                 HeaderWindowDef{"seven_day_opus", "opus"},
                 HeaderWindowDef{"seven_day_sonnet", "sonnet"},
                 HeaderWindowDef{"overage", "overage"},
             }) {
            const std::string header_name =
                "anthropic-ratelimit-unified-" + std::string(window) + "-utilization";
            if (auto value = find_header_case_insensitive(headers, header_name)) {
                float val = 0.0f;
                if (try_parse_float(*value, val) && val >= 0.0f) {
                    int64_t window_reset = 0;
                    const std::string reset_header_name =
                        "anthropic-ratelimit-unified-" + std::string(window) + "-reset";
                    if (auto rval = find_header_case_insensitive(headers, reset_header_name)) {
                        window_reset = parse_iso8601_timestamp(*rval);
                    } else if (auto rval2 = find_header_case_insensitive(headers, "anthropic-ratelimit-unified-" + std::string(window) + "-resets-at")) {
                        window_reset = parse_iso8601_timestamp(*rval2);
                    } else if (auto rval3 = find_header_case_insensitive(headers, "anthropic-ratelimit-unified-" + std::string(window) + "-reset-time")) {
                        window_reset = parse_iso8601_timestamp(*rval3);
                    }
                    info.usage_windows.push_back({
                        std::string(label),
                        normalize_utilization(val, false),
                        window_reset,
                    });
                }
            }
        }
        if (auto value = find_header_case_insensitive(headers, "anthropic-ratelimit-unified-status")) {
            info.unified_status = std::string(*value);
            if (*value == "rate_limited" || *value == "rejected") {
                info.is_rate_limited = true;
            }
        }
        if (auto value = find_header_case_insensitive(headers, "anthropic-ratelimit-unified-representative-claim")) {
            info.unified_representative_claim = std::string(*value);
        }
        if (auto value = find_header_case_insensitive(headers, "anthropic-ratelimit-unified-overage-status")) {
            info.unified_overage_status = std::string(*value);
        }
        if (auto value = find_header_case_insensitive(headers, "anthropic-ratelimit-unified-overage-reset")) {
            info.unified_overage_reset = safe_stoi64(*value);
        }
        if (auto value = find_header_case_insensitive(headers, "anthropic-ratelimit-unified-overage-disabled-reason")) {
            info.unified_overage_disabled_reason = std::string(*value);
        }
        if (auto value = find_header_case_insensitive(headers, "anthropic-ratelimit-unified-fallback")) {
            info.unified_fallback_available = (*value == "available");
        }
        if (auto value = find_header_case_insensitive(headers, "anthropic-ratelimit-unified-fallback-available")) {
            info.unified_fallback_available = safe_parse_bool(*value);
        }

        return info;
    }

    struct AnthropicErrorDetails {
        std::string type;
        std::string message;
        std::string request_id;
    };

    [[nodiscard]] AnthropicErrorDetails error_details(const HttpResponse& response) {
        AnthropicErrorDetails details;
        simdjson::dom::parser parser;
        simdjson::dom::element doc;
        simdjson::padded_string body(response.body);
        if (parser.parse(body).get(doc) == simdjson::SUCCESS) {
            std::string_view value;
            if (doc["error"]["type"].get(value) == simdjson::SUCCESS) details.type = value;
            if (doc["error"]["message"].get(value) == simdjson::SUCCESS) details.message = value.substr(0, 2000);
            if (doc["request_id"].get(value) == simdjson::SUCCESS) details.request_id = value.substr(0, 200);
        }
        if (const auto id = find_header_case_insensitive(response.headers, "request-id")) {
            details.request_id = id->substr(0, 200);
        }
        return details;
    }

    [[nodiscard]] bool permanent_rate_limit(const AnthropicErrorDetails& error) {
        const auto message = core::utils::str::to_lower_ascii_copy(error.message);
        return error.type == "billing_error"
            || message.find("credit balance is too low") != std::string::npos
            || message.find("monthly spend") != std::string::npos
            || message.find("spend limit") != std::string::npos
            || message.find("usage tier's spend cap") != std::string::npos;
    }

    std::string impl_format_error_message(int status_code,
                                          const AnthropicErrorDetails& error) {
        switch (status_code) {
            case 400:
                return "[Anthropic API Error 400: Invalid request. "
                    + (error.message.empty()
                        ? std::string("The request body is malformed or contains invalid parameters.")
                        : error.message) + "]";
            case 401:
                return "[Anthropic API Error 401: Authentication failed. Please check your API key or session token.]";
            case 403:
                return "[Anthropic API Error 403: Permission denied. Your account may not have access to this model or feature.]";
            case 404:
                return "[Anthropic API Error 404: Not found. The requested model or endpoint does not exist.]";
            case 429: {
                std::string msg = "[Anthropic API Error 429: Rate limit exceeded. ";
                if (!error.message.empty()) {
                    msg += "Please wait before retrying. Consider reducing request frequency or context size.";
                } else {
                    msg += "Please wait before retrying.";
                }
                msg += ']';
                return msg;
            }
            case 500:
                return "[Anthropic API Error 500: Internal server error. This is a temporary issue on Anthropic's side.]";
            case 529:
                return "[Anthropic API Error 529: Server overloaded. Anthropic is experiencing high load. "
                       "This is NOT a rate limit - retrying with backoff is recommended.]";
            default:
                if (status_code >= 500) {
                    return "[Anthropic API Error " + std::to_string(status_code) +
                           ": Server error. Please retry with exponential backoff.]";
                } else if (status_code >= 400) {
                    return "[Anthropic API Error " + std::to_string(status_code) +
                           ": Client error. Please check your request parameters.]";
                }
                return "[Anthropic API Error " + std::to_string(status_code) + "]";
        }
    }

    bool impl_is_retryable_status(int status_code) noexcept {
        return status_code == 429 ||  // Rate limit — retry after delay
               status_code == 500 ||  // Internal server error
               status_code == 502 ||  // Bad gateway
               status_code == 503 ||  // Service unavailable
               status_code == 504 ||  // Gateway timeout
               status_code == 529;    // Overloaded
    }

} // namespace

AnthropicWirePolicy anthropic_wire_policy(
    std::string_view model,
    const ModelReasoningProfile* live) {
    AnthropicWirePolicy policy;

    const std::string normalized = anthropic::normalized_claude_id(model);
    auto card = ModelRegistry::instance().lookup(model);
    if (!card) card = ModelRegistry::instance().lookup(normalized);
    const auto* generation = anthropic::model_policy(normalized);

    // Reasoning controls: the serving endpoint's own catalog is authoritative,
    // the curated card covers an offline session, and the generation table
    // covers ids that match neither (pinned snapshots, new deployments, models
    // released after this build).
    const ModelReasoningProfile* reasoning = nullptr;
    if (live && live->complete) {
        reasoning = live;
    } else if (card && card->reasoning.complete) {
        reasoning = &card->reasoning;
    }
    if (reasoning) {
        policy.known = true;
        policy.effort = reasoning->effort;
        policy.adaptive_thinking = reasoning->adaptive_thinking;
        policy.manual_thinking = reasoning->manual_thinking;
    } else if (generation) {
        policy.known = true;
        policy.effort = generation->effort;
        policy.adaptive_thinking = generation->adaptive_thinking;
        policy.manual_thinking = generation->manual_thinking;
    } else if (card) {
        policy.known = true;
    }

    // No model catalog advertises rejected request fields, so a matched card or
    // generation is the only curated source. A matched generation counts as
    // curated even when it carries no constraints.
    const ModelWireConstraints* wire = nullptr;
    if (card && !card->wire.empty()) {
        wire = &card->wire;
    } else if (generation) {
        wire = &generation->wire;
    }

    if (wire) {
        policy.thinking_always_on = wire->thinking_always_on;
        policy.reasoning_text_hidden = wire->reasoning_text_hidden;
        policy.reasoning_bound_to_prefix = wire->reasoning_bound_to_prefix;
        policy.fixed_sampling = wire->fixed_sampling;
        policy.forced_tool_choice_rejected = wire->forced_tool_choice_rejected;
        policy.between_tools_thinking = wire->between_tools_thinking;
        policy.disabled_thinking = wire->disabled_thinking;
        return policy;
    }

    if (policy.known && policy.adaptive_thinking && !policy.manual_thinking) {
        // An unseen model that offers only adaptive thinking belongs to the
        // generations that hide reasoning text by default and reject
        // non-default sampling: every adaptive-only Claude documented so far
        // does both, and the catalog advertises neither. Prefix binding and
        // forced-tool-choice rejection are NOT implied by adaptive thinking
        // (Claude Opus 5 is adaptive-only and accepts both), so they stay off.
        policy.reasoning_text_hidden = true;
        policy.fixed_sampling = true;
    }
    return policy;
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicSerializer
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// Messages API image sources accept image/jpeg, image/png, image/gif, and
// image/webp. `source.data` is raw base64, never a data: URL.
[[nodiscard]] std::optional<std::string> anthropic_image_media_type(
    std::string_view mime) {
    std::string normalized;
    normalized.reserve(mime.size());
    for (const unsigned char ch : mime) {
        if (ch == ' ' || ch == '\t') continue;
        normalized.push_back(static_cast<char>(std::tolower(ch)));
    }
    if (normalized == "image/jpg" || normalized == "image/pjpeg") {
        return std::string("image/jpeg");
    }
    if (normalized == "image/x-png") {
        return std::string("image/png");
    }
    if (normalized == "image/jpeg" || normalized == "image/png"
        || normalized == "image/gif" || normalized == "image/webp") {
        return normalized;
    }
    return std::nullopt;
}

// Grammar limits apply to the combined output schema and strict tools.
// Tools outside the budget stay available with normal local argument validation.
class AnthropicSchemaBudget {
public:
    void reserve_output(std::string_view schema) { counts_ = count(schema); }
    bool admit_tool(std::string_view schema) {
        const auto added = count(schema);
        if (tools_ >= 20 || counts_.optional + added.optional > 24
            || counts_.unions + added.unions > 16) return false;
        ++tools_;
        counts_.optional += added.optional;
        counts_.unions += added.unions;
        return true;
    }
private:
    struct Counts { int optional = 0; int unions = 0; };
    static Counts count(std::string_view schema) {
        Counts counts;
        simdjson::dom::parser parser;
        simdjson::padded_string input(schema);
        simdjson::dom::element root;
        if (parser.parse(input).get(root) != simdjson::SUCCESS) return {25, 17};
        const auto visit = [&](auto&& self, simdjson::dom::element node) -> void {
            simdjson::dom::object object;
            simdjson::dom::array array;
            if (node.get(object) == simdjson::SUCCESS) {
                simdjson::dom::object properties;
                if (object["properties"].get(properties) == simdjson::SUCCESS) {
                    for (const auto property : properties) {
                        bool required = false;
                        simdjson::dom::array names;
                        if (object["required"].get(names) == simdjson::SUCCESS) {
                            for (const auto name : names) {
                                std::string_view value;
                                if (name.get(value) == simdjson::SUCCESS && value == property.key) required = true;
                            }
                        }
                        counts.optional += !required;
                    }
                }
                if (object["anyOf"].get(array) == simdjson::SUCCESS) ++counts.unions;
                if (object["type"].get(array) == simdjson::SUCCESS && array.size() > 1) ++counts.unions;
                for (const auto field : object) self(self, field.value);
            } else if (node.get(array) == simdjson::SUCCESS) {
                for (const auto element : array) self(self, element);
            }
        };
        visit(visit, root);
        return counts;
    }
    Counts counts_;
    int tools_ = 0;
};

// Replay native block order only while the visible transcript still matches it.
// Session edits fall back to the normalized representation and binding controls.
[[nodiscard]] std::optional<std::string> native_assistant_content(
    const Message& message, bool omit_thinking) {
    for (const auto& item : message.continuation_items) {
        if (item.provider != "anthropic" || item.kind != "assistant_content") continue;
        simdjson::dom::parser parser;
        simdjson::padded_string payload(item.payload);
        simdjson::dom::element doc;
        simdjson::dom::array blocks;
        if (parser.parse(payload).get(doc) != simdjson::SUCCESS
            || doc["content"].get(blocks) != simdjson::SUCCESS) continue;
        std::string text;
        std::string content = "[";
        std::size_t tool_index = 0;
        bool matches = true;
        for (const auto block : blocks) {
            std::string_view type;
            if (block["type"].get(type) != simdjson::SUCCESS) { matches = false; break; }
            if (type == "text") {
                std::string_view value;
                if (block["text"].get(value) != simdjson::SUCCESS) { matches = false; break; }
                text += value;
            } else if (type == "tool_use") {
                if (tool_index >= message.tool_calls.size()) { matches = false; break; }
                const auto& tool = message.tool_calls[tool_index++];
                std::string_view id, name;
                simdjson::dom::object input;
                simdjson::dom::parser arguments_parser;
                simdjson::padded_string arguments(tool.function.arguments.empty() ? "{}" : tool.function.arguments);
                simdjson::dom::object expected;
                if (block["id"].get(id) != simdjson::SUCCESS || id != tool.id
                    || block["name"].get(name) != simdjson::SUCCESS || name != tool.function.name
                    || block["input"].get(input) != simdjson::SUCCESS
                    || arguments_parser.parse(arguments).get_object().get(expected) != simdjson::SUCCESS
                    || simdjson::to_string(input) != simdjson::to_string(expected)) { matches = false; break; }
            } else if (type != "thinking" && type != "redacted_thinking") {
                matches = false; break;
            }
            if (omit_thinking && (type == "thinking" || type == "redacted_thinking")) continue;
            if (content.back() != '[') content += ',';
            content += simdjson::to_string(block);
        }
        if (matches && text == message.content && tool_index == message.tool_calls.size()
            && content.size() > 1) return content + "]";
    }
    return std::nullopt;
}

} // namespace

std::string AnthropicSerializer::serialize(
    const ChatRequest& req,
    int default_max_tokens,
    const AnthropicThinkingConfig& thinking,
    const AnthropicReasoningEmitter& reasoning_emitter) {
    std::string payload;
    payload.reserve(8192);

    payload += R"({"model":")";
    payload += core::utils::escape_json_string(req.model);
    payload += R"(","stream":)";
    payload += req.stream ? "true" : "false";

    // max_tokens is mandatory in the Anthropic API.
    payload += R"(,"max_tokens":)";
    payload += std::to_string(req.max_tokens.value_or(
        model_default_max_tokens(req.model, default_max_tokens)));

    // Automatic prompt caching materially reduces repeated-prefix cost for
    // tool-heavy multi-step agent loops while remaining transparent to callers.
    const bool should_enable_auto_cache =
        !req.tools.empty() || req.messages.size() > 2;
    if (should_enable_auto_cache) {
        payload += R"(,"cache_control":{"type":"ephemeral"})";
    }

    const auto wire_policy = anthropic_wire_policy(
        req.model, req.catalog_reasoning ? &*req.catalog_reasoning : nullptr);
    if (reasoning_emitter) {
        reasoning_emitter(payload, req);
    } else {
    // Effort can reduce output/token spend for tool-heavy sessions.
    // Anthropic docs (Apr 2026): generally available, no beta header needed.
    ReasoningCapabilities reasoning_capabilities = wire_policy.effort;
    if (wire_policy.thinking_always_on) {
        reasoning_capabilities =
            reasoning_capabilities | ReasoningCapability::Required;
    }
    std::string effective_effort;
    if (reasoning_capabilities.supports_effort()) {
        effective_effort = normalized_effort_or_empty(req.effort);
        if (effective_effort == "max"
            && !reasoning_capabilities.supports(ReasoningCapability::MaxEffort)) {
            effective_effort = "high";
        }
        if (effective_effort == "xhigh"
            && !reasoning_capabilities.supports(ReasoningCapability::XHighEffort)) {
            effective_effort = "high";
        }
        // A model that always thinks cannot honor `off`. Omitting effort would
        // run it at the API default, which thinks more than the user asked for;
        // `low` is the supported substitute.
        if (effective_effort.empty()
            && effort_explicitly_disabled(req.effort)
            && wire_policy.thinking_always_on) {
            effective_effort = "low";
        }
    }
    if (!effective_effort.empty() || req.response_format.type == ResponseFormat::Type::JsonSchema) {
        payload += R"(,"output_config":{)";
        if (!effective_effort.empty()) {
            payload += R"("effort":")" + core::utils::escape_json_string(effective_effort) + '"';
        }
        if (req.response_format.type == ResponseFormat::Type::JsonSchema) {
            simdjson::dom::parser schema_parser;
            simdjson::padded_string schema(req.response_format.schema);
            simdjson::dom::object object;
            if (schema_parser.parse(schema).get_object().get(object) != simdjson::SUCCESS) {
                throw std::invalid_argument("Anthropic structured output requires a valid JSON schema object");
            }
            if (!effective_effort.empty()) payload += ',';
            payload += R"("format":{"type":"json_schema","schema":)";
            payload += req.response_format.schema;
            payload += '}';
        }
        payload += '}';
    }

    // Thinking is wanted when either the config enabled it (thinking_budget > 0)
    // OR the model supports effort-based reasoning and the user has not turned
    // effort off. This makes `/effort` the single user-facing control for Claude
    // reasoning: `auto`/low/medium/high/max stream chain-of-thought, `off`
    // suppresses it. `thinking_budget` remains an explicit override for manual
    // budgets and never forces thinking off.
    const bool effort_wants_thinking =
        reasoning_capabilities.supports_effort()
        && !effort_explicitly_disabled(req.effort);
    const bool thinking_wanted = thinking.enabled || effort_wants_thinking;

    const bool use_adaptive_thinking =
        thinking_wanted
        && wire_policy.adaptive_thinking
        && (wire_policy.rejects_manual_thinking() || !thinking.enabled);
    const bool use_manual_thinking =
        thinking_wanted
        && (!wire_policy.known
            || (!wire_policy.rejects_manual_thinking()
                && (!wire_policy.adaptive_thinking || thinking.enabled)));

    // Anthropic requires temperature=1 when extended thinking is enabled, and
    // the newest generations reject every non-default sampling value outright.
    // For those the field is omitted so the API never sees a value it refuses.
    if (!wire_policy.fixed_sampling) {
        if (use_manual_thinking) {
            payload += R"(,"temperature":1)";
        } else if (req.temperature.has_value()) {
            payload += R"(,"temperature":)";
            payload += std::to_string(req.temperature.value());
        }
    }

    // Models that hide reasoning text return thinking blocks with an empty
    // `thinking` field, which would leave Filo's reasoning channel silent.
    // `summarized` needs no beta header and carries no extra token cost.
    const std::string_view thinking_display =
        wire_policy.reasoning_text_hidden ? R"(,"display":"summarized")" : "";

    // Extended thinking block (must come before messages).
    if (wire_policy.between_tools_thinking && effort_explicitly_disabled(req.effort)) {
        payload += R"(,"thinking":{"type":"between_tools"})";
    } else if (wire_policy.disabled_thinking && effort_explicitly_disabled(req.effort)) {
        payload += R"(,"thinking":{"type":"disabled"})";
    } else if (wire_policy.reasoning_bound_to_prefix) {
        // Filo intentionally changes tools/context during a session. Let the
        // API discard only invalidated reasoning instead of rejecting a turn.
        // Always-on models keep adaptive thinking at their lowest supported effort.
        payload += R"(,"thinking":{"type":"adaptive")";
        payload += thinking_display;
        payload += R"(,"block_binding":{"prefix_mismatch_behavior":"drop_block"}})";
    } else if (use_manual_thinking) {
        payload += R"(,"thinking":{"type":"enabled")";
        payload += thinking_display;
        payload += R"(,"budget_tokens":)";
        const int max_tokens = req.max_tokens.value_or(
            model_default_max_tokens(req.model, default_max_tokens));
        if (max_tokens <= 1024) {
            throw std::invalid_argument("Anthropic manual thinking requires max_tokens greater than 1024");
        }
        payload += std::to_string(std::clamp(thinking.budget_tokens, 1024, max_tokens - 1));
        payload += '}';
    } else if (use_adaptive_thinking) {
        payload += R"(,"thinking":{"type":"adaptive")";
        payload += thinking_display;
        payload += '}';
    }
    }

    // Claude Code-style attribution is carried in the top-level system field.
    // Structured plans retain layer boundaries so the stable workspace prefix
    // remains reusable when session or dynamic context changes.
    append_anthropic_system_field(payload, req);

    const bool omit_historical_thinking = effort_explicitly_disabled(req.effort)
        && (wire_policy.between_tools_thinking || wire_policy.disabled_thinking);

    // Tools (Anthropic uses "input_schema" instead of "parameters").
    if (!req.tools.empty()) {
        // Strict tool use constrains the sampler to the declared shape, so a
        // mis-typed argument cannot be generated in the first place. A tool
        // whose contract does not survive the projection is sent unflagged.
        const auto strict_dialect = configured_strict_tool_dialect(
            ToolSchemaWire::Anthropic, req.model);
        AnthropicSchemaBudget schema_budget;
        if (req.response_format.type == ResponseFormat::Type::JsonSchema) {
            schema_budget.reserve_output(req.response_format.schema);
        }
        payload += R"(,"tools":[)";
        for (size_t i = 0; i < req.tools.size(); ++i) {
            const auto& def = req.tools[i].function;
            payload += R"({"name":")";
            payload += core::utils::escape_json_string(def.name);
            payload += R"(","description":")";
            payload += core::utils::escape_json_string(def.description);
            payload += R"(","input_schema":)";
            const std::string canonical_schema =
                core::tools::schema::canonical_input_schema(def);
            const auto strict_schema = strict_dialect.has_value()
                ? core::tools::schema::strict_input_schema(
                      canonical_schema, *strict_dialect)
                : std::nullopt;
            const bool use_strict = strict_schema.has_value() && schema_budget.admit_tool(*strict_schema);
            payload += use_strict ? *strict_schema : canonical_schema;
            if (use_strict) {
                payload += R"(,"strict":true)";
            }
            if (i + 1 == req.tools.size()) {
                payload += R"(,"cache_control":{"type":"ephemeral"})";
            }
            payload += '}';
            if (i + 1 < req.tools.size()) payload += ',';
        }
        payload += ']';
    }

    // Messages — system messages are skipped (handled above as top-level field).
    payload += R"(,"messages":[)";
    bool first_msg = true;
    for (std::size_t message_index = 0;
         message_index < req.messages.size();
         ++message_index) {
        const auto& msg = req.messages[message_index];
        if (msg.role == "system") continue;
        if (omit_historical_thinking && msg.role == "assistant"
            && msg.content.empty() && msg.tool_calls.empty() && msg.content_parts.empty()) continue;

        if (!first_msg) payload += ',';
        first_msg = false;

        if (msg.role == "tool") {
            // Every result for a parallel assistant tool-use step belongs in
            // the immediately following Claude user message. Filo stores one
            // provider-neutral `tool` message per result, so coalesce the
            // consecutive run into one Anthropic content array.
            payload += R"({"role":"user","content":[)";
            bool first_result = true;
            while (message_index < req.messages.size()
                   && req.messages[message_index].role == "tool") {
                const auto& result = req.messages[message_index];
                if (!first_result) payload += ',';
                first_result = false;
                payload += R"({"type":"tool_result","tool_use_id":")";
                payload += core::utils::escape_json_string(result.tool_call_id);
                payload += R"(","content":")";
                payload += core::utils::escape_json_string(result.content);
                payload += "\"}";
                ++message_index;
            }
            --message_index;
            payload += "]}";

        } else if (msg.role == "assistant"
                   && (!msg.tool_calls.empty() || !msg.continuation_items.empty())) {
            if (const auto content = native_assistant_content(msg, omit_historical_thinking)) {
                payload += R"({"role":"assistant","content":)";
                payload += *content;
                payload += '}';
                continue;
            }
            // OpenAI assistant tool_calls → Claude content blocks.
            payload += R"({"role":"assistant","content":[)";
            bool first_block = true;

            for (const auto& continuation : msg.continuation_items) {
                if (continuation.kind == "assistant_content") continue;
                // These low-thinking modes reject block_binding. Remove historical
                // reasoning as documented so edited prefixes remain resumable.
                if (omit_historical_thinking
                    && (continuation.kind == "thinking" || continuation.kind == "redacted_thinking")) continue;
                if ((!continuation.provider.empty()
                     && continuation.provider != "anthropic")
                    || !has_valid_continuation_payload(continuation)) {
                    continue;
                }
                if (!first_block) payload += ',';
                payload += continuation.payload;
                first_block = false;
            }

            if (!msg.content.empty()) {
                if (!first_block) payload += ',';
                payload += R"({"type":"text","text":")";
                payload += core::utils::escape_json_string(msg.content);
                payload += "\"}";
                first_block = false;
            }

            for (const auto& tc : msg.tool_calls) {
                if (!first_block) payload += ',';
                first_block = false;
                payload += R"({"type":"tool_use","id":")";
                payload += core::utils::escape_json_string(tc.id);
                payload += R"(","name":")";
                payload += core::utils::escape_json_string(tc.function.name);
                // Only validated JSON objects may be embedded as native JSON.
                payload += R"(","input":)";
                payload += core::utils::json::object_or_empty(
                    tc.function.arguments);
                payload += '}';
            }
            payload += "]}";

        } else {
            payload += R"({"role":")";
            payload += core::utils::escape_json_string(msg.role);
            if (!msg.content_parts.empty()) {
                payload += R"(","content":[)";
                bool first_block = true;
                for (const auto& part : msg.content_parts) {
                    if (!first_block) payload += ',';
                    first_block = false;

                    if (part.type == ContentPartType::Text) {
                        payload += R"({"type":"text","text":")";
                        payload += core::utils::escape_json_string(part.text);
                        payload += R"("})";
                        continue;
                    }

                    const auto encoded = encode_image_part(part);
                    const auto media_type = encoded.has_value() && !encoded->is_url_reference()
                        ? anthropic_image_media_type(encoded->mime_type)
                        : std::nullopt;
                    if (media_type.has_value()) {
                        payload += R"({"type":"image","source":{"type":"base64","media_type":")";
                        payload += core::utils::escape_json_string(*media_type);
                        payload += R"(","data":")";
                        payload += core::utils::escape_json_string(encoded->base64_data);
                        payload += R"("}})";
                    } else {
                        payload += R"({"type":"text","text":")";
                        payload += core::utils::escape_json_string(
                            unavailable_media_attachment_text(part.type, media_reference(part)));
                        payload += R"("})";
                    }
                }
                payload += "]}";
            } else {
                payload += R"(","content":")";
                payload += core::utils::escape_json_string(msg.content);
                payload += "\"}";
            }
        }
    }
    payload += "]}";
    return payload;
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicSSEParser
// ─────────────────────────────────────────────────────────────────────────────

AnthropicSSEParser::Result AnthropicSSEParser::process_event(std::string_view event_type,
                                                              std::string_view json_str) {
    Result result;
    if (json_str.empty()) return result;
    if (event_type == "ping") return result;

    thread_local simdjson::dom::parser parser;
    const core::utils::json::ParserRetentionGuard parser_guard{parser};
    simdjson::padded_string input(json_str);
    simdjson::dom::element doc;
    if (parser.parse(input).get(doc) != simdjson::SUCCESS) {
        if (event_type == "message_start" || event_type == "message_delta"
            || event_type == "message_stop" || event_type == "content_block_start"
            || event_type == "content_block_delta" || event_type == "content_block_stop"
            || event_type == "error") {
            result.stream_error = true;
            result.retryable_stream_error = true;
            result.error_type = "invalid_stream_event";
            result.error_message = "Claude returned malformed streaming JSON";
        }
        return result;
    }

    const auto read_usage = [&](simdjson::dom::object usage) {
        const auto read_input = [&](std::string_view key, int64_t& value) {
            int64_t reported = 0;
            if (usage[key].get(reported) == simdjson::SUCCESS) {
                value = std::clamp<int64_t>(reported, 0, std::numeric_limits<int32_t>::max());
                result.input_usage_reported = true;
            }
        };
        read_input("input_tokens", uncached_input_);
        read_input("cache_read_input_tokens", cached_input_);
        read_input("cache_creation_input_tokens", cache_creation_input_);
        if (result.input_usage_reported) {
            result.input_tokens = static_cast<int32_t>(std::min<int64_t>(
                uncached_input_ + cached_input_ + cache_creation_input_, std::numeric_limits<int32_t>::max()));
            result.cached_input_tokens = static_cast<int32_t>(cached_input_);
            result.cache_creation_input_tokens = static_cast<int32_t>(cache_creation_input_);
        }
        int64_t output = 0;
        if (usage["output_tokens"].get(output) == simdjson::SUCCESS) {
            result.output_tokens = static_cast<int32_t>(std::clamp<int64_t>(output, 0, std::numeric_limits<int32_t>::max()));
        }
    };

    if (event_type == "error") {
        result.stream_error = true;
        simdjson::dom::object error_obj;
        if (doc["error"].get(error_obj) == simdjson::SUCCESS) {
            std::string_view type;
            if (error_obj["type"].get(type) == simdjson::SUCCESS) {
                result.error_type = std::string(type);
                result.retryable_stream_error =
                    is_retryable_anthropic_stream_error(type);
            }
            std::string_view message;
            if (error_obj["message"].get(message) == simdjson::SUCCESS) {
                result.error_message = std::string(message);
            }
        }
        return result;
    }

    if (event_type == "message_start") {
        result.stream_started = true;
        simdjson::dom::array transformations;
        if (doc["input_transformations"].get(transformations) == simdjson::SUCCESS) {
            for (const auto item : transformations) {
                std::string_view type;
                std::string_view reason;
                if (item["type"].get(type) != simdjson::SUCCESS
                    || type != "thinking_dropped"
                    || item["reason"].get(reason) != simdjson::SUCCESS) continue;
                if (reason == "prefix_binding_mismatch") ++result.prefix_binding_mismatches;
                if (reason == "model_binding_mismatch") ++result.model_binding_mismatches;
            }
        }
        simdjson::dom::object msg_obj;
        if (doc["message"].get(msg_obj) == simdjson::SUCCESS) {
            simdjson::dom::object usage_obj;
            if (msg_obj["usage"].get(usage_obj) == simdjson::SUCCESS) {
                read_usage(usage_obj);
            }
        }
        return result;
    }

    if (event_type == "message_delta") {
        simdjson::dom::object delta_obj;
        if (doc["delta"].get(delta_obj) == simdjson::SUCCESS) {
            std::string_view stop_reason;
            if (delta_obj["stop_reason"].get(stop_reason) == simdjson::SUCCESS) {
                result.stop_reason = std::string(stop_reason);
                // Classifier refusals are HTTP 200 with an empty content array.
                // Surface the API's explanation so the turn is not a blank stop.
                if (stop_reason == "refusal") {
                    simdjson::dom::object details;
                    if (delta_obj["stop_details"].get(details) == simdjson::SUCCESS) {
                        std::string_view explanation;
                        std::string_view category;
                        if (details["explanation"].get(explanation) == simdjson::SUCCESS
                            && !explanation.empty()) {
                            result.text = std::string(explanation);
                        } else if (details["category"].get(category) == simdjson::SUCCESS
                                   && !category.empty()) {
                            result.text = "Request declined (";
                            result.text.append(category);
                            result.text += ").";
                        }
                    }
                }
            }
        }

        simdjson::dom::object usage_obj;
        if (doc["usage"].get(usage_obj) == simdjson::SUCCESS) {
            read_usage(usage_obj);
        }
        return result;
    }

    if (event_type == "content_block_start") {
        simdjson::dom::object content_block;
        if (doc["content_block"].get(content_block) != simdjson::SUCCESS) return result;

        std::string_view type_v;
        if (content_block["type"].get(type_v) != simdjson::SUCCESS) return result;

        if (type_v == "tool_use") {
            AnthropicToolBlockState state;
            int64_t index_v;
            if (doc["index"].get(index_v) == simdjson::SUCCESS)
                state.index = static_cast<int>(index_v);
            std::string_view id_v;
            if (content_block["id"].get(id_v) == simdjson::SUCCESS)
                state.id = std::string(id_v);
            std::string_view name_v;
            if (content_block["name"].get(name_v) == simdjson::SUCCESS)
                state.name = std::string(name_v);
            simdjson::dom::object initial_input;
            if (content_block["input"].get(initial_input) == simdjson::SUCCESS) {
                state.initial_args = simdjson::to_string(initial_input);
            }
            current_tool_ = std::move(state);
        } else if (type_v == "thinking" || type_v == "redacted_thinking") {
            AnthropicContinuationBlockState state;
            int64_t index = -1;
            if (doc["index"].get(index) == simdjson::SUCCESS) state.index = static_cast<int>(index);
            state.type = std::string(type_v);
            state.initial_payload = simdjson::to_string(content_block);
            std::string_view value;
            if (content_block["thinking"].get(value) == simdjson::SUCCESS) {
                state.thinking = std::string(value);
            }
            if (content_block["signature"].get(value) == simdjson::SUCCESS) {
                state.signature = std::string(value);
            }
            current_continuation_ = std::move(state);
        } else if (type_v == "text") {
            int64_t index = -1;
            if (doc["index"].get(index) == simdjson::SUCCESS) text_index_ = static_cast<int>(index);
            std::string_view text;
            current_text_ = content_block["text"].get(text) == simdjson::SUCCESS ? std::string(text) : "";
            result.text = *current_text_;
        } else {
            replay_complete_ = false;
        }
        return result;
    }

    if (event_type == "content_block_delta") {
        simdjson::dom::object delta;
        if (doc["delta"].get(delta) != simdjson::SUCCESS) return result;

        std::string_view delta_type;
        if (delta["type"].get(delta_type) != simdjson::SUCCESS) return result;

        int64_t index = -1;
        const bool has_index = doc["index"].get(index) == simdjson::SUCCESS;
        if (delta_type == "text_delta" && (!current_text_ || !has_index || index == text_index_)) {
            std::string_view text;
            if (delta["text"].get(text) == simdjson::SUCCESS) {
                result.text = std::string(text);
                if (current_text_) *current_text_ += text;
            }
        } else if (delta_type == "input_json_delta" && current_tool_.has_value()
                   && (!has_index || index == current_tool_->index)) {
            std::string_view partial_json;
            if (delta["partial_json"].get(partial_json) == simdjson::SUCCESS)
                current_tool_->accumulated_args += partial_json;
        } else if (delta_type == "thinking_delta" && current_continuation_.has_value()
                   && (!has_index || index == current_continuation_->index)) {
            std::string_view thinking;
            if (delta["thinking"].get(thinking) == simdjson::SUCCESS) {
                // Authoritative, signed thinking accumulated for upstream replay.
                current_continuation_->thinking += thinking;
                // Display-only mirror surfaced to the UI reasoning channel. This
                // is a copy: the signed block above remains the source of truth
                // for API continuation and is never derived from this field.
                result.reasoning_delta.append(thinking.data(), thinking.size());
            }
        } else if (delta_type == "signature_delta" && current_continuation_.has_value()
                   && (!has_index || index == current_continuation_->index)) {
            std::string_view signature;
            if (delta["signature"].get(signature) == simdjson::SUCCESS) {
                current_continuation_->signature += signature;
            }
        }
        return result;
    }

    if (event_type == "content_block_stop") {
        int64_t index = -1;
        const bool has_index = doc["index"].get(index) == simdjson::SUCCESS;
        if (current_text_ && (!has_index || index == text_index_)) {
            replay_blocks_.push_back(R"({"type":"text","text":")"
                + core::utils::escape_json_string(*current_text_) + R"("})");
            current_text_.reset();
        }
        if (current_tool_.has_value() && (!has_index || index == current_tool_->index)) {
            const std::string args = current_tool_->accumulated_args.empty()
                ? (current_tool_->initial_args.empty() ? "{}" : current_tool_->initial_args)
                : current_tool_->accumulated_args;
            simdjson::dom::parser arguments_parser;
            simdjson::dom::object arguments;
            simdjson::padded_string padded_args(args);
            if (arguments_parser.parse(padded_args).get_object().get(arguments) != simdjson::SUCCESS) {
                result.stream_error = true;
                result.retryable_stream_error = true;
                result.error_type = "invalid_tool_input";
                result.error_message = "Claude returned incomplete or malformed tool arguments";
                current_tool_.reset();
                return result;
            }
            replay_blocks_.push_back(R"({"type":"tool_use","id":")"
                + core::utils::escape_json_string(current_tool_->id) + R"(","name":")"
                + core::utils::escape_json_string(current_tool_->name) + R"(","input":)" + args + "}");
            ToolCall tc;
            tc.index              = current_tool_->index;
            tc.id                 = std::move(current_tool_->id);
            tc.type               = "function";
            tc.function.name      = std::move(current_tool_->name);
            tc.function.arguments = args;
            result.completed_tools.push_back(std::move(tc));
            current_tool_.reset();
        }
        if (current_continuation_.has_value() && (!has_index || index == current_continuation_->index)) {
            std::string block;
            if (current_continuation_->type == "redacted_thinking") {
                block = std::move(current_continuation_->initial_payload);
            } else {
                block = R"({"type":"thinking","thinking":")";
                block += core::utils::escape_json_string(current_continuation_->thinking);
                block += R"(","signature":")";
                block += core::utils::escape_json_string(current_continuation_->signature);
                block += R"("})";
            }
            replay_blocks_.push_back(block);
            result.continuation_items.push_back(ContinuationItem{
                .provider = "anthropic",
                .kind = current_continuation_->type,
                .payload = std::move(block),
            });
            current_continuation_.reset();
        }
        return result;
    }

    if (event_type == "message_stop") {
        result.done = true;
        result.incomplete_tool_call = current_tool_.has_value();
        // A terminal max_tokens stop can legitimately interrupt a block. The
        // agent already recovers such turns and never executes unfinished tools.
        if (!current_tool_ && !current_continuation_ && !current_text_
            && replay_complete_ && !replay_blocks_.empty()) {
            std::string content = R"({"content":[)";
            for (const auto& block : replay_blocks_) {
                if (content.back() != '[') content += ',';
                content += block;
            }
            content += "]}";
            result.continuation_items.push_back({.provider = "anthropic", .kind = "assistant_content", .payload = std::move(content)});
        }
        current_tool_.reset();
        current_continuation_.reset();
        current_text_.reset();
        replay_blocks_.clear();
    }

    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicProtocol
// ─────────────────────────────────────────────────────────────────────────────

void AnthropicProtocol::prepare_request(ChatRequest& request) {
    request_uses_context_1m_ = false;
    last_requested_model_.clear();
    if (request.model.empty()) return;

    ModelNormalization normalized = normalize_requested_claude_model(request.model);
    if (!normalized.model.empty()) {
        request.model = std::move(normalized.model);
    }
    request_uses_context_1m_ = normalized.use_context_1m;
    last_requested_model_ = request.model;
}

std::string AnthropicProtocol::serialize(const ChatRequest& req) const {
    return AnthropicSerializer::serialize(
        req, default_max_tokens_, thinking_, reasoning_emitter());
}

AnthropicReasoningEmitter AnthropicProtocol::reasoning_emitter() const {
    return {};
}

ReasoningCapabilities AnthropicProtocol::reasoning_capabilities(
    std::string_view model) const noexcept {
    try {
        const auto policy = anthropic_wire_policy(model);
        if (policy.thinking_always_on) {
            return policy.effort | ReasoningCapability::Required;
        }
        return policy.effort;
    } catch (...) {
        return {};
    }
}

cpr::Header AnthropicProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{
        {"anthropic-version",          std::string(ANTHROPIC_VERSION)},
        {"Content-Type",               "application/json"},
        {"Accept",                     "text/event-stream"},
        {"x-anthropic-billing-header", std::string(ANTHROPIC_BILLING_HEADER)},
    };

    std::string anthropic_beta;
    const auto append_beta_unique = [&](std::string_view beta) {
        if (beta.empty()) return;

        const std::string candidate = core::utils::str::trim_ascii_copy(beta);
        if (candidate.empty()) return;

        std::size_t start = 0;
        while (start < anthropic_beta.size()) {
            std::size_t comma = anthropic_beta.find(',', start);
            if (comma == std::string::npos) comma = anthropic_beta.size();
            const std::string existing = core::utils::str::trim_ascii_copy(
                std::string_view(anthropic_beta).substr(start, comma - start));
            if (existing == candidate) return;
            if (comma == anthropic_beta.size()) break;
            start = comma + 1;
        }

        if (!anthropic_beta.empty()) anthropic_beta += ",";
        anthropic_beta += candidate;
    };

    append_beta_unique(ANTHROPIC_BETA_CLAUDE_CODE);
    const auto wire_policy = anthropic_wire_policy(last_requested_model_);
    if (wire_policy.reasoning_bound_to_prefix) {
        append_beta_unique(anthropic::kThinkingBindingBeta);
    }
    if (request_uses_context_1m_) append_beta_unique(ANTHROPIC_BETA_CONTEXT_1M);
    if (thinking_.enabled && !wire_policy.rejects_manual_thinking()) {
        append_beta_unique(ANTHROPIC_BETA_THINKING);
    }
    if (auto it = auth.properties.find("oauth");
        it != auth.properties.end() && it->second == "1") {
        append_beta_unique(ANTHROPIC_BETA_OAUTH);
    }

    for (const auto& [k, raw_v] : auth.headers) {
        if (core::utils::ascii::iequals(k, "anthropic-beta") && !raw_v.empty()) {
            std::size_t start = 0;
            while (start < raw_v.size()) {
                std::size_t comma = raw_v.find(',', start);
                if (comma == std::string::npos) comma = raw_v.size();
                append_beta_unique(
                    std::string_view(raw_v).substr(start, comma - start));
                if (comma == raw_v.size()) break;
                start = comma + 1;
            }
        } else {
            headers[k] = raw_v;
        }
    }

    if (!anthropic_beta.empty()) {
        headers["anthropic-beta"] = anthropic_beta;
    }

    return headers;
}

std::string AnthropicProtocol::build_url(std::string_view base_url,
                                          [[maybe_unused]] std::string_view model) const {
    return std::string(base_url) + "/v1/messages";
}

ParseResult AnthropicProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed)) return {};
    if (parsed.is_done || parsed.data.empty()) return {};

    const std::string event_data = std::string(parsed.data);
    std::string event_type = std::string(parsed.event);
    if (event_type.empty()) {
        event_type = infer_anthropic_event_type(event_data);
    }
    if (event_type.empty()) return {};

    auto r = sse_parser_.process_event(event_type, event_data);

    ParseResult result;
    result.stream_started = r.stream_started;
    result.stream_error = r.stream_error;
    result.retryable_stream_error = r.retryable_stream_error;
    result.stream_error_type = std::move(r.error_type);
    result.stream_error_message = std::move(r.error_message);
    if (r.input_usage_reported) {
        accumulated_input_ = r.input_tokens;
        accumulated_cached_input_ = r.cached_input_tokens;
        accumulated_cache_creation_ = r.cache_creation_input_tokens;
    }
    if (r.output_tokens > 0) accumulated_output_ = r.output_tokens;
    if (!r.stop_reason.empty()) last_stop_reason_ = r.stop_reason;
    if (r.prefix_binding_mismatches > 0 || r.model_binding_mismatches > 0) {
        core::logging::debug(
            "[Anthropic] Dropped thinking blocks: {} after context edits, {} after model switches",
            r.prefix_binding_mismatches, r.model_binding_mismatches);
    }

    if (!r.text.empty())           result.chunks.push_back(StreamChunk::make_content(r.text));
    if (!r.reasoning_delta.empty()) {
        auto chunk = StreamChunk::make_reasoning(std::move(r.reasoning_delta));
        chunk.reasoning_protocol = std::string(name());
        result.chunks.push_back(std::move(chunk));
    }
    if (!r.completed_tools.empty()) result.chunks.push_back(StreamChunk::make_tools(r.completed_tools));
    if (!preserves_native_assistant_content()) {
        std::erase_if(r.continuation_items, [](const auto& item) { return item.kind == "assistant_content"; });
    }
    if (!r.continuation_items.empty()) {
        StreamChunk chunk;
        chunk.continuation_items = std::move(r.continuation_items);
        result.chunks.push_back(std::move(chunk));
    }

    if (r.done) {
        result.done              = true;
        result.prompt_tokens     = accumulated_input_;
        result.completion_tokens = accumulated_output_;
        result.cached_prompt_tokens = accumulated_cached_input_;
        result.cache_creation_prompt_tokens = accumulated_cache_creation_;
        result.stop_reason       = last_stop_reason_;
        result.incomplete_tool_call = r.incomplete_tool_call;
    }

    return result;
}

std::unique_ptr<ApiProtocolBase> AnthropicProtocol::clone() const {
    return std::make_unique<AnthropicProtocol>(thinking_, default_max_tokens_);
}

void AnthropicProtocol::reset_state() {
    sse_parser_ = AnthropicSSEParser{};
    accumulated_input_ = 0;
    accumulated_output_ = 0;
    accumulated_cached_input_ = 0;
    accumulated_cache_creation_ = 0;
    last_stop_reason_.clear();
    last_rate_limit_ = RateLimitInfo{};
}

// ── Response lifecycle hook overrides ────────────────────────────────────────

void AnthropicProtocol::on_response(const HttpResponse& response) {
    last_rate_limit_ = impl_parse_rate_limit_headers(response.headers);
    last_rate_limit_.is_rate_limited = last_rate_limit_.is_rate_limited || response.status_code == 429;
}

void AnthropicProtocol::enrich_rate_limit(std::string_view base_url,
                                          const cpr::Header& request_headers,
                                          const HttpResponse& response) {
    if (!should_query_claude_usage_endpoint(request_headers, response)) return;
    if (!last_rate_limit_.usage_windows.empty()) return;

    const std::string cache_key = usage_cache_key(base_url, request_headers);
    if (const auto cached = load_cached_claude_usage_snapshot(cache_key);
        cached.has_value()) {
        merge_claude_usage_snapshot(last_rate_limit_, *cached);
        return;
    }

    if (const auto usage = fetch_claude_usage_snapshot(base_url, request_headers);
        usage.has_value()) {
        store_cached_claude_usage_snapshot(cache_key, *usage);
        merge_claude_usage_snapshot(last_rate_limit_, *usage);
    }
}

std::string AnthropicProtocol::format_error_message(const HttpResponse& response) const {
    const auto error = error_details(response);
    std::string message;
    if (response.status_code == 429 && permanent_rate_limit(error)) {
        message = "[Anthropic API Error 429: Account spend limit reached. Check billing or increase the spend limit before retrying.]";
    } else {
        message = impl_format_error_message(response.status_code, error);
    }
    if (response.status_code != 400 && !error.message.empty()) {
        message += " " + error.message;
    }
    if (!error.request_id.empty()) message += " (request-id: " + error.request_id + ")";
    return message;
}

bool AnthropicProtocol::is_retryable(const HttpResponse& response) const noexcept {
    if (!impl_is_retryable_status(response.status_code)) return false;
    if (const auto should_retry = find_header_case_insensitive(response.headers, "x-should-retry");
        should_retry && core::utils::ascii::iequals(*should_retry, "false")) return false;
    try {
        if (response.status_code == 429 && permanent_rate_limit(error_details(response))) return false;
    } catch (...) {
        // Diagnostic parsing must not break transport error classification.
    }
    return true;
}

} // namespace core::llm::protocols
