#include "KimiProtocol.hpp"
#include "SseUtils.hpp"
#include "../../auth/KimiOAuthFlow.hpp"
#include "../../logging/Logger.hpp"
#include "../../utils/JsonUtils.hpp"
#include "../../utils/StringUtils.hpp"
#include <simdjson.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cctype>
#include <format>
#include <functional>
#include <initializer_list>
#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>

namespace core::llm::protocols {

namespace {

[[nodiscard]] std::optional<std::string_view>
find_header_case_insensitive(const cpr::Header& headers, std::string_view key) {
    if (const auto it = headers.find(std::string(key)); it != headers.end()) {
        return it->second;
    }

    std::string key_lower;
    key_lower.reserve(key.size());
    for (char c : key) {
        key_lower.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }

    for (const auto& [k, v] : headers) {
        if (k.size() != key_lower.size()) continue;
        bool equal = true;
        for (std::size_t i = 0; i < k.size(); ++i) {
            const char lhs = static_cast<char>(std::tolower(static_cast<unsigned char>(k[i])));
            if (lhs != key_lower[i]) {
                equal = false;
                break;
            }
        }
        if (equal) {
            return v;
        }
    }
    return std::nullopt;
}

[[nodiscard]] int32_t parse_int_header(const cpr::Header& headers, std::string_view key) noexcept {
    const auto value = find_header_case_insensitive(headers, key);
    if (!value.has_value() || value->empty()) return 0;

    try {
        return static_cast<int32_t>(std::stoi(std::string(*value)));
    } catch (...) {
        return 0;
    }
}

[[nodiscard]] std::optional<int32_t>
parse_optional_int_header(const cpr::Header& headers, std::string_view key) noexcept {
    const auto value = find_header_case_insensitive(headers, key);
    if (!value.has_value() || value->empty()) return std::nullopt;

    try {
        return static_cast<int32_t>(std::stoi(std::string(*value)));
    } catch (...) {
        return std::nullopt;
    }
}

[[nodiscard]] std::optional<int32_t>
parse_first_present_int_header(const cpr::Header& headers,
                               std::initializer_list<std::string_view> keys) noexcept {
    for (std::string_view key : keys) {
        if (auto parsed = parse_optional_int_header(headers, key); parsed.has_value()) {
            return parsed;
        }
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<float>
parse_optional_float_header(const cpr::Header& headers, std::string_view key) noexcept {
    const auto value = find_header_case_insensitive(headers, key);
    if (!value.has_value() || value->empty()) return std::nullopt;

    try {
        float parsed = std::stof(std::string(*value));
        // Some APIs report percentages as 0-100 instead of 0-1.
        if (parsed > 1.5f && parsed <= 100.0f) {
            parsed /= 100.0f;
        }
        return parsed;
    } catch (...) {
        return std::nullopt;
    }
}

[[nodiscard]] std::string parse_string_header(const cpr::Header& headers, std::string_view key) {
    const auto value = find_header_case_insensitive(headers, key);
    return value.has_value() ? std::string(*value) : std::string{};
}

[[nodiscard]] RateLimitInfo parse_rate_limit_headers(const cpr::Header& headers,
                                                     int response_status_code) noexcept {
    RateLimitInfo info;

    // Request-based limits.
    if (auto requests_limit = parse_first_present_int_header(
            headers,
            {"x-ratelimit-limit-requests", "x-msh-ratelimit-limit-requests", "x-ratelimit-limit"});
        requests_limit.has_value()) {
        info.requests_limit = *requests_limit;
    }

    if (auto requests_remaining = parse_first_present_int_header(
            headers,
            {"x-ratelimit-remaining-requests", "x-msh-ratelimit-remaining-requests",
             "x-ratelimit-remaining"});
        requests_remaining.has_value()) {
        info.requests_remaining = *requests_remaining;
    } else if (info.requests_limit > 0) {
        // Remaining was not reported: avoid showing false "0 remaining" alarms.
        info.requests_remaining = info.requests_limit;
    }

    // Token-based limits.
    if (auto tokens_limit = parse_first_present_int_header(
            headers, {"x-ratelimit-limit-tokens", "x-msh-ratelimit-limit-tokens"});
        tokens_limit.has_value()) {
        info.tokens_limit = *tokens_limit;
    }

    if (auto tokens_remaining = parse_first_present_int_header(
            headers, {"x-ratelimit-remaining-tokens", "x-msh-ratelimit-remaining-tokens"});
        tokens_remaining.has_value()) {
        info.tokens_remaining = *tokens_remaining;
    } else if (info.tokens_limit > 0) {
        // Remaining was not reported: avoid showing false "0 remaining" alarms.
        info.tokens_remaining = info.tokens_limit;
    }

    // Retry / rate-limited signals.
    info.retry_after = parse_int_header(headers, "retry-after");
    info.is_rate_limited = (response_status_code == 429 || info.retry_after > 0);

    // Unified utilization headers (OAuth/subscription style).
    // For each known window, try the standard header first then the Kimi-prefixed variant.
    // Table-driven: add a new row here if the server introduces a new window.
    struct WindowDef { std::string_view label, standard_key, msh_key; };
    static constexpr std::array<WindowDef, 2> kWindows{{
        {"5h", "x-ratelimit-unified-5h-utilization", "x-msh-ratelimit-unified-5h-utilization"},
        {"7d", "x-ratelimit-unified-7d-utilization", "x-msh-ratelimit-unified-7d-utilization"},
    }};
    for (const auto& w : kWindows) {
        auto val = parse_optional_float_header(headers, w.standard_key);
        if (!val.has_value()) {
            val = parse_optional_float_header(headers, w.msh_key);
        }
        if (val.has_value()) {
            info.usage_windows.push_back({std::string(w.label), *val});
        }
    }

    info.unified_status = parse_string_header(headers, "x-ratelimit-unified-status");
    if (info.unified_status.empty()) {
        info.unified_status = parse_string_header(headers, "x-msh-ratelimit-unified-status");
    }

    info.unified_representative_claim =
        parse_string_header(headers, "x-ratelimit-unified-representative-claim");
    if (info.unified_representative_claim.empty()) {
        info.unified_representative_claim =
            parse_string_header(headers, "x-msh-ratelimit-unified-representative-claim");
    }

    if (info.unified_status == "rate_limited") {
        info.is_rate_limited = true;
    }

    return info;
}

struct KimiUsageSnapshot {
    int32_t tokens_limit = 0;
    int32_t tokens_remaining = 0;
    bool has_tokens_summary = false;
    std::vector<UsageWindow> windows;
};

struct CachedUsageSnapshot {
    KimiUsageSnapshot value;
    std::chrono::steady_clock::time_point expires_at;
};

constexpr auto kUsageSnapshotTtl = std::chrono::seconds(20);

[[nodiscard]] std::optional<int32_t>
parse_int_string(std::string_view value) noexcept {
    value = core::utils::str::trim_ascii_view(value);
    if (value.empty()) return std::nullopt;
    int32_t parsed = 0;
    const auto [ptr, ec] =
        std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (ec != std::errc{} || ptr != value.data() + value.size()) {
        return std::nullopt;
    }
    return parsed;
}

[[nodiscard]] std::optional<int32_t>
parse_int_element(simdjson::dom::element element) noexcept {
    int64_t as_i64 = 0;
    if (element.get(as_i64) == simdjson::SUCCESS) {
        if (as_i64 < std::numeric_limits<int32_t>::min()
            || as_i64 > std::numeric_limits<int32_t>::max()) {
            return std::nullopt;
        }
        return static_cast<int32_t>(as_i64);
    }

    double as_double = 0.0;
    if (element.get(as_double) == simdjson::SUCCESS) {
        if (as_double < static_cast<double>(std::numeric_limits<int32_t>::min())
            || as_double > static_cast<double>(std::numeric_limits<int32_t>::max())) {
            return std::nullopt;
        }
        return static_cast<int32_t>(as_double);
    }

    std::string_view as_str;
    if (element.get(as_str) == simdjson::SUCCESS) {
        return parse_int_string(as_str);
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<int32_t>
parse_int_field(const simdjson::dom::object* obj, std::string_view key) noexcept {
    if (obj == nullptr) return std::nullopt;
    simdjson::dom::element element;
    if ((*obj)[key].get(element) != simdjson::SUCCESS) return std::nullopt;
    return parse_int_element(element);
}

[[nodiscard]] std::optional<std::string>
parse_string_field(const simdjson::dom::object* obj, std::string_view key) {
    if (obj == nullptr) return std::nullopt;
    std::string_view value;
    if ((*obj)[key].get(value) != simdjson::SUCCESS) return std::nullopt;
    value = core::utils::str::trim_ascii_view(value);
    if (value.empty()) return std::nullopt;
    return std::string(value);
}

[[nodiscard]] std::string lower_ascii(std::string_view value) {
    std::string out;
    out.reserve(value.size());
    for (const unsigned char ch : value) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

[[nodiscard]] std::string normalize_kimi_effort(std::string_view raw_effort) {
    std::string effort = lower_ascii(raw_effort);
    std::erase_if(effort, [](unsigned char ch) {
        return std::isspace(ch) || ch == '-' || ch == '_';
    });

    if (effort.empty() || effort == "auto" || effort == "unset" || effort == "default") {
        return {};
    }
    if (effort == "off" || effort == "none" || effort == "disable" || effort == "disabled") {
        return "off";
    }
    if (effort == "low" || effort == "medium" || effort == "high") {
        return effort;
    }
    if (effort == "max" || effort == "xhigh") {
        // Kimi's public control surface caps effort at "high".
        return "high";
    }
    return {};
}

[[nodiscard]] bool model_supports_kimi_thinking(std::string_view model) {
    const std::string lowered = lower_ascii(model);
    return lowered == "kimi-for-coding"
        || lowered.starts_with("kimi-k2")
        || lowered.find("thinking") != std::string::npos;
}

[[nodiscard]] std::string sanitize_os_version(std::string version) {
    std::string sanitized;
    sanitized.reserve(version.size());
    for (char c : version) {
        // Keep only printable ASCII that's safe for HTTP headers.
        // Remove: # < > [ ] { } \ | ` and control characters.
        if (c >= 32 && c < 127
            && c != '#' && c != '<' && c != '>'
            && c != '[' && c != ']' && c != '{' && c != '}'
            && c != '\\' && c != '|' && c != '`') {
            sanitized.push_back(c);
        }
    }

    std::string result;
    result.reserve(sanitized.size());
    bool last_was_space = false;
    for (char c : sanitized) {
        if (c == ' ') {
            if (!last_was_space) {
                result.push_back(c);
                last_was_space = true;
            }
        } else {
            result.push_back(c);
            last_was_space = false;
        }
    }

    const size_t start = result.find_first_not_of(" \t");
    const size_t end = result.find_last_not_of(" \t");
    if (start == std::string::npos) return "unknown";
    return result.substr(start, end - start + 1);
}

[[nodiscard]] const cpr::Header& cached_kimi_metadata_headers() {
    static const cpr::Header kCachedHeaders = [] {
        auto raw = core::auth::KimiOAuthFlow::getCommonHeaders();
        if (raw.count("X-Msh-Os-Version")) {
            raw["X-Msh-Os-Version"] = sanitize_os_version(raw["X-Msh-Os-Version"]);
        }

        cpr::Header headers;
        for (const auto& [key, value] : raw) {
            headers[key] = value;
        }
        return headers;
    }();
    return kCachedHeaders;
}

[[nodiscard]] std::string label_from_duration(int32_t duration, std::string_view time_unit) {
    if (duration <= 0) return {};
    const std::string lower = lower_ascii(time_unit);
    if (lower.find("minute") != std::string::npos) {
        if (duration % 60 == 0) {
            return std::format("{}h", duration / 60);
        }
        return std::format("{}m", duration);
    }
    if (lower.find("hour") != std::string::npos) {
        return std::format("{}h", duration);
    }
    if (lower.find("day") != std::string::npos) {
        return std::format("{}d", duration);
    }
    return std::format("{}s", duration);
}

[[nodiscard]] std::string infer_kimi_window_label(
    const simdjson::dom::object* item,
    const simdjson::dom::object* detail,
    const simdjson::dom::object* window,
    int index) {
    static constexpr std::array<std::string_view, 3> kLabelKeys{
        "name", "title", "scope"
    };
    for (const auto key : kLabelKeys) {
        if (auto label = parse_string_field(item, key); label.has_value()) {
            return *label;
        }
        if (auto label = parse_string_field(detail, key); label.has_value()) {
            return *label;
        }
    }

    const std::optional<int32_t> duration =
        parse_int_field(window, "duration").has_value()
            ? parse_int_field(window, "duration")
            : (parse_int_field(item, "duration").has_value()
                ? parse_int_field(item, "duration")
                : parse_int_field(detail, "duration"));
    std::optional<std::string> time_unit =
        parse_string_field(window, "timeUnit").has_value()
            ? parse_string_field(window, "timeUnit")
            : (parse_string_field(item, "timeUnit").has_value()
                ? parse_string_field(item, "timeUnit")
                : parse_string_field(detail, "timeUnit"));

    if (duration.has_value()) {
        const std::string label = label_from_duration(
            *duration, time_unit.value_or(""));
        if (!label.empty()) return label;
    }

    return std::format("window{}", index + 1);
}

[[nodiscard]] std::optional<float>
utilization_from_limit_detail(const simdjson::dom::object* detail) noexcept {
    const auto limit_opt = parse_int_field(detail, "limit");
    if (!limit_opt.has_value() || *limit_opt <= 0) return std::nullopt;
    const int32_t limit = *limit_opt;

    std::optional<int32_t> used_opt = parse_int_field(detail, "used");
    if (!used_opt.has_value()) {
        if (const auto remaining = parse_int_field(detail, "remaining");
            remaining.has_value()) {
            used_opt = std::max<int32_t>(0, limit - *remaining);
        }
    }
    const int32_t used = std::max<int32_t>(0, used_opt.value_or(0));
    const float utilization = static_cast<float>(used)
                            / static_cast<float>(limit);
    return std::clamp(utilization, 0.0f, 1.5f);
}

[[nodiscard]] std::optional<KimiUsageSnapshot>
parse_kimi_usage_payload(std::string_view payload) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    simdjson::padded_string padded(payload);
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) {
        return std::nullopt;
    }

    KimiUsageSnapshot snapshot;

    simdjson::dom::object usage_obj;
    if (doc["usage"].get(usage_obj) == simdjson::SUCCESS) {
        if (const auto limit = parse_int_field(&usage_obj, "limit");
            limit.has_value() && *limit > 0) {
            snapshot.has_tokens_summary = true;
            snapshot.tokens_limit = *limit;

            if (const auto remaining = parse_int_field(&usage_obj, "remaining");
                remaining.has_value()) {
                snapshot.tokens_remaining = std::max<int32_t>(0, *remaining);
            } else if (const auto used = parse_int_field(&usage_obj, "used");
                       used.has_value()) {
                snapshot.tokens_remaining = std::max<int32_t>(0, *limit - *used);
            } else {
                snapshot.tokens_remaining = *limit;
            }
        }
    }

    simdjson::dom::array limits;
    if (doc["limits"].get(limits) == simdjson::SUCCESS) {
        int index = 0;
        for (simdjson::dom::element limit_element : limits) {
            simdjson::dom::object item_obj;
            if (limit_element.get(item_obj) != simdjson::SUCCESS) {
                ++index;
                continue;
            }

            simdjson::dom::object detail_obj = item_obj;
            simdjson::dom::object nested_detail;
            if (item_obj["detail"].get(nested_detail) == simdjson::SUCCESS) {
                detail_obj = nested_detail;
            }

            simdjson::dom::object window_obj;
            const simdjson::dom::object* window_ptr = nullptr;
            if (item_obj["window"].get(window_obj) == simdjson::SUCCESS) {
                window_ptr = &window_obj;
            }

            const auto utilization = utilization_from_limit_detail(&detail_obj);
            if (utilization.has_value()) {
                snapshot.windows.push_back(UsageWindow{
                    .label = infer_kimi_window_label(
                        &item_obj, &detail_obj, window_ptr, index),
                    .utilization = *utilization,
                });
            }
            ++index;
        }
    }

    if (!snapshot.has_tokens_summary && snapshot.windows.empty()) {
        return std::nullopt;
    }
    return snapshot;
}

[[nodiscard]] bool should_query_kimi_usages_endpoint(std::string_view base_url) {
    // Match kimi-cli behavior: usage endpoint is available on Kimi Code
    // managed platform base URLs (*/coding/*), regardless of OAuth/API key.
    return base_url.find("/coding/") != std::string_view::npos;
}

[[nodiscard]] std::string trim_trailing_slash(std::string_view url) {
    while (!url.empty() && url.back() == '/') {
        url.remove_suffix(1);
    }
    return std::string(url);
}

[[nodiscard]] std::string usage_cache_key(std::string_view base_url,
                                          const cpr::Header& request_headers) {
    // Scope cache to endpoint + auth identity so we don't leak usage state
    // between different accounts in the same process.
    std::string key = trim_trailing_slash(base_url);
    const auto auth_header = find_header_case_insensitive(request_headers, "Authorization");
    const std::size_t auth_hash = std::hash<std::string_view>{}(
        auth_header.has_value() ? *auth_header : std::string_view{});
    key += "|auth:";
    key += std::to_string(auth_hash);
    return key;
}

[[nodiscard]] std::unordered_map<std::string, CachedUsageSnapshot>& kimi_usage_cache() {
    static std::unordered_map<std::string, CachedUsageSnapshot> cache;
    return cache;
}

[[nodiscard]] std::mutex& kimi_usage_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

[[nodiscard]] std::optional<KimiUsageSnapshot>
load_cached_kimi_usage_snapshot(const std::string& key) {
    const auto now = std::chrono::steady_clock::now();
    std::scoped_lock lock(kimi_usage_cache_mutex());
    auto& cache = kimi_usage_cache();
    auto it = cache.find(key);
    if (it == cache.end()) return std::nullopt;
    if (it->second.expires_at <= now) {
        cache.erase(it);
        return std::nullopt;
    }
    return it->second.value;
}

void store_cached_kimi_usage_snapshot(std::string key,
                                      const KimiUsageSnapshot& snapshot) {
    const auto expires_at = std::chrono::steady_clock::now() + kUsageSnapshotTtl;
    std::scoped_lock lock(kimi_usage_cache_mutex());
    auto& cache = kimi_usage_cache();
    cache[std::move(key)] = CachedUsageSnapshot{snapshot, expires_at};
}

[[nodiscard]] std::optional<KimiUsageSnapshot>
fetch_kimi_usage_snapshot(std::string_view base_url,
                          const cpr::Header& request_headers) {
    std::string url = trim_trailing_slash(base_url);
    url += "/usages";

    cpr::Header headers = request_headers;
    headers["Accept"] = "application/json";

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
    return parse_kimi_usage_payload(response.text);
}

void merge_kimi_usage_snapshot(RateLimitInfo& info, const KimiUsageSnapshot& snapshot) {
    if (snapshot.has_tokens_summary && info.tokens_limit <= 0) {
        info.tokens_limit = snapshot.tokens_limit;
        info.tokens_remaining = snapshot.tokens_remaining;
    }

    for (const auto& incoming : snapshot.windows) {
        auto existing = std::find_if(
            info.usage_windows.begin(),
            info.usage_windows.end(),
            [&](const UsageWindow& current) {
                return current.label == incoming.label;
            });
        if (existing != info.usage_windows.end()) {
            existing->utilization = incoming.utilization;
        } else {
            info.usage_windows.push_back(incoming);
        }
    }
}

[[nodiscard]] bool read_usage_object(simdjson::dom::object usage_obj,
                                     int32_t& prompt_tokens,
                                     int32_t& completion_tokens) {
    int64_t pt = 0;
    int64_t ct = 0;
    [[maybe_unused]] const auto e1 = usage_obj["prompt_tokens"].get(pt);
    [[maybe_unused]] const auto e2 = usage_obj["completion_tokens"].get(ct);

    if (pt > 0 || ct > 0) {
        prompt_tokens = static_cast<int32_t>(pt);
        completion_tokens = static_cast<int32_t>(ct);
        return true;
    }
    return false;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Kimi-specific SSE parsing — handles reasoning_content for K2.5 models
// ─────────────────────────────────────────────────────────────────────────────

/**
 * @brief Parse Kimi SSE chunk, handling both content and reasoning_content.
 * 
 * Kimi K2.5 models return reasoning_content for thinking tokens and content
 * for the actual response. We collect both as they represent the model's output.
 */
struct KimiParseResult {
    std::string content;
    std::string reasoning_content;
    std::vector<ToolCall> tools;
    int32_t prompt_tokens = 0;
    int32_t completion_tokens = 0;
};

[[nodiscard]] KimiParseResult parse_kimi_sse_chunk(std::string_view json_str) {
    thread_local simdjson::dom::parser parser;
    simdjson::padded_string ps(json_str);
    simdjson::dom::element doc;
    if (parser.parse(ps).get(doc) != simdjson::SUCCESS) {
        return {};
    }

    KimiParseResult result;

    // Kimi can emit usage in top-level "usage" (OpenAI-style) or in
    // "choices[i].usage" (vendor-specific streaming chunks).
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

        // delta.content
        std::string_view c;
        if (delta["content"].get(c) == simdjson::SUCCESS) {
            result.content = std::string(c);
        }

        // delta.reasoning_content (Kimi thinking stream)
        std::string_view rc;
        if (delta["reasoning_content"].get(rc) == simdjson::SUCCESS && rc.size() > 0) {
            result.reasoning_content.append(rc.data(), rc.size());
        }

        // delta.tool_calls
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

// ─────────────────────────────────────────────────────────────────────────────
// KimiProtocol
// ─────────────────────────────────────────────────────────────────────────────

std::string KimiProtocol::serialize(const ChatRequest& req) const {
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

void KimiProtocol::append_extra_fields(std::string& payload, const ChatRequest& req) const {
    if (!model_supports_kimi_thinking(req.model)) {
        return;
    }

    const std::string effort = normalize_kimi_effort(req.effort);
    if (effort.empty()) {
        return;
    }

    if (effort == "off") {
        payload += R"(,"thinking":{"type":"disabled"})";
        return;
    }

    payload += R"(,"reasoning_effort":")";
    payload += core::utils::escape_json_string(effort);
    payload += R"(","thinking":{"type":"enabled"})";
}

cpr::Header KimiProtocol::build_headers(const core::auth::AuthInfo& auth) const {
    cpr::Header headers{
        {"Content-Type", "application/json"},
        {"Accept",       "text/event-stream"},
        {"User-Agent",   "KimiCLI/1.41.0"},
    };
    
    // Add Kimi-specific X-Msh-* headers for device identification.
    cpr::Header kimi_headers = cached_kimi_metadata_headers();
    
    // CRITICAL: Use device_id from OAuth token if available.
    // The X-Msh-Device-Id header MUST match the device_id claim in the JWT token,
    // otherwise the server returns 401 Authentication failed.
    // The device_id is passed via auth.properties from OAuthCredentialSource.
    auto it = auth.properties.find("device_id");
    if (it != auth.properties.end() && !it->second.empty()) {
        kimi_headers["X-Msh-Device-Id"] = it->second;
    }
    
    for (const auto& [key, value] : kimi_headers) {
        headers[key] = value;
    }
    
    // Add auth headers (e.g., Authorization: Bearer <token>)
    for (const auto& [key, value] : auth.headers) {
        headers[key] = value;
    }
    
    return headers;
}

std::string KimiProtocol::format_error_message(const HttpResponse& response) const {
    const int code = response.status_code;
    
    // Try to extract Kimi's error message from the JSON body
    std::string kimi_message;
    if (!response.body.empty()) {
        thread_local simdjson::dom::parser parser;
        simdjson::padded_string ps(response.body);
        simdjson::dom::element doc;
        if (parser.parse(ps).get(doc) == simdjson::SUCCESS) {
            // Kimi error format varies, try common patterns
            std::string_view msg;
            if (doc["error"]["message"].get(msg) == simdjson::SUCCESS) {
                kimi_message = std::string(msg);
            } else if (doc["message"].get(msg) == simdjson::SUCCESS) {
                kimi_message = std::string(msg);
            }
        }
    }
    
    switch (code) {
        case 400:
            return "[Kimi API Error 400: Bad request. " +
                   (kimi_message.empty() 
                       ? std::string("The request body is malformed or contains invalid parameters.")
                       : kimi_message) + "]";
        
        case 401:
            return "[Kimi API Error 401: Authentication failed. "
                   "If using OAuth, ensure the X-Msh-* headers are present. "
                   "If using an API key, check your KIMI_API_KEY is valid. "
                   "Visit https://platform.moonshot.cn to verify credentials.]";

        case 402:
            return "[Kimi API Error 402: Membership expired or payment required. "
                   "Please renew your Kimi plan to continue using this model.]";
        
        case 403:
            return "[Kimi API Error 403: Permission denied. Your account may not have access to "
                   "this model or feature, or your quota may be exhausted. "
                   "Visit https://platform.moonshot.cn to check remaining credits and access.]";
        
        case 404:
            return "[Kimi API Error 404: Model or endpoint not found. Verify the model name "
                   "is correct. Available models: kimi-k2.6, kimi-k2.5, moonshot-v1-8k, etc.]";
        
        case 429:
            return "[Kimi API Error 429: Rate limit exceeded. Please reduce request frequency. "
                   "Check your usage at https://platform.moonshot.cn]";
        
        case 500:
            return "[Kimi API Error 500: Internal server error. This is a temporary issue on "
                   "Moonshot AI's side. Please retry with exponential backoff.]";
        
        case 502:
            return "[Kimi API Error 502: Bad gateway. Kimi is experiencing connectivity issues. "
                   "Please retry with exponential backoff.]";
        
        case 503:
            return "[Kimi API Error 503: Service unavailable. Kimi may be undergoing maintenance. "
                   "Please retry with exponential backoff.]";
        
        case 504:
            return "[Kimi API Error 504: Gateway timeout. The request timed out at Kimi's edge. "
                   "Consider reducing context size, then retry.]";
        
        default:
            if (code >= 500) {
                return "[Kimi API Error " + std::to_string(code) + ": Server error. "
                       "This is a temporary issue on Moonshot AI's side. Please retry.]";
            } else if (code >= 400) {
                return "[Kimi API Error " + std::to_string(code) + ": Client error. " +
                       (kimi_message.empty() ? "" : "Details: " + kimi_message + " ") + 
                       "See https://platform.moonshot.cn for API documentation.]";
            }
            return "[Kimi API Error " + std::to_string(code) + "]";
    }
}

bool KimiProtocol::is_retryable(const HttpResponse& response) const noexcept {
    // Kimi retryable status codes (standard OpenAI-compatible)
    return response.status_code == 429 ||  // Rate limit
           response.status_code == 500 ||  // Internal server error
           response.status_code == 502 ||  // Bad gateway
           response.status_code == 503 ||  // Service unavailable
           response.status_code == 504;    // Gateway timeout
}

void KimiProtocol::on_response(const HttpResponse& response) {
    last_rate_limit_ = parse_rate_limit_headers(response.headers, response.status_code);
}

void KimiProtocol::enrich_rate_limit(std::string_view base_url,
                                     const cpr::Header& request_headers,
                                     const HttpResponse& response) {
    if (response.status_code != 200) return;
    if (!should_query_kimi_usages_endpoint(base_url)) return;

    const bool needs_enrichment =
        last_rate_limit_.usage_windows.empty()
        || (last_rate_limit_.tokens_limit <= 0
            && last_rate_limit_.tokens_remaining <= 0);
    if (!needs_enrichment) return;

    const std::string cache_key = usage_cache_key(base_url, request_headers);
    if (const auto cached = load_cached_kimi_usage_snapshot(cache_key);
        cached.has_value()) {
        merge_kimi_usage_snapshot(last_rate_limit_, *cached);
        return;
    }

    if (const auto usage = fetch_kimi_usage_snapshot(base_url, request_headers);
        usage.has_value()) {
        store_cached_kimi_usage_snapshot(cache_key, *usage);
        merge_kimi_usage_snapshot(last_rate_limit_, *usage);
    }
}

ParseResult KimiProtocol::parse_event(std::string_view raw_event) {
    sse::ParsedEventView parsed;
    if (!sse::parse_event_payload(raw_event, parsed)) return {};
    const std::string_view json_sv = parsed.data;

    if (parsed.is_done) {
        ParseResult r;
        r.done = true;
        return r;
    }

    ParseResult result;
    auto kimi_result = parse_kimi_sse_chunk(json_sv);
    result.prompt_tokens = kimi_result.prompt_tokens;
    result.completion_tokens = kimi_result.completion_tokens;
    
    // Create chunk with content, reasoning_content, and tools
    // Kimi can send both reasoning_content and content/tool_calls in the same or different deltas
    if (!kimi_result.content.empty() || 
        !kimi_result.reasoning_content.empty() || 
        !kimi_result.tools.empty()) {
        StreamChunk chunk;
        chunk.content = std::move(kimi_result.content);
        chunk.reasoning_content = std::move(kimi_result.reasoning_content);
        chunk.tools   = std::move(kimi_result.tools);
        result.chunks.push_back(std::move(chunk));
    }

    return result;
}

} // namespace core::llm::protocols
