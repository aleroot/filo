#include "GrokBillingUsage.hpp"

#include "GrokBuildEndpoint.hpp"
#include "core/llm/transport/HttpHeaderUtils.hpp"
#include "core/utils/StringUtils.hpp"

#include <simdjson.h>

#include <algorithm>
#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace core::llm::protocols {

namespace {

constexpr auto kBillingCacheTtl = std::chrono::seconds(60);

[[nodiscard]] std::optional<float> numeric_field(
    simdjson::dom::object object,
    std::string_view key) {
    double double_value = 0.0;
    if (object[key].get(double_value) == simdjson::SUCCESS) {
        return static_cast<float>(double_value);
    }
    int64_t int_value = 0;
    if (object[key].get(int_value) == simdjson::SUCCESS) {
        return static_cast<float>(int_value);
    }
    return std::nullopt;
}

[[nodiscard]] std::optional<float> legacy_credit_value(
    simdjson::dom::object config,
    std::string_view key) {
    simdjson::dom::object cents;
    if (config[key].get(cents) != simdjson::SUCCESS) return std::nullopt;
    return numeric_field(cents, "val").value_or(0.0f);
}

struct CachedUsage {
    std::vector<UsageWindow> value;
    std::chrono::steady_clock::time_point expires_at;
};

class GrokBillingUsageSource final : public IGrokBillingUsageSource {
public:
    [[nodiscard]] std::vector<UsageWindow> fetch(
        std::string_view base_url,
        const cpr::Header& request_headers) override {
        // Never forward an OAuth credential to a custom/OpenAI-compatible URL.
        if (!grok_build::is_official_proxy(base_url)) return {};

        const auto authorization = transport::find_header(
            request_headers, "Authorization");
        const auto user_id = transport::find_header(
            request_headers, "x-grok-user-id");
        const auto token_auth = transport::find_header(
            request_headers, "X-XAI-Token-Auth");
        if (!authorization.has_value() || !user_id.has_value()
            || !token_auth.has_value() || *token_auth != "xai-grok-cli") {
            return {};
        }

        std::string cache_key = core::utils::str::trim_trailing_slashes(base_url);
        cache_key += "|auth:";
        cache_key += std::to_string(std::hash<std::string>{}(*authorization));
        cache_key += "|user:";
        cache_key += std::to_string(std::hash<std::string>{}(*user_id));
        if (const auto cached = load_cached(cache_key); cached.has_value()) {
            return *cached;
        }

        cpr::Header headers;
        headers["Authorization"] = *authorization;
        headers["X-XAI-Token-Auth"] = *token_auth;
        headers["x-userid"] = *user_id;
        copy_header(request_headers, headers, "x-grok-client-version");
        copy_header(request_headers, headers, "x-grok-client-mode");
        copy_header(request_headers, headers, "User-Agent");
        headers["Accept"] = "application/json";

        const std::string url =
            core::utils::str::trim_trailing_slashes(base_url)
            + "/billing?format=credits";
        const cpr::Response response = cpr::Get(
            cpr::Url{url},
            headers,
            cpr::Timeout{1500});
        if (response.error.code != cpr::ErrorCode::OK
            || response.status_code != 200
            || response.text.empty()) {
            return {};
        }

        auto windows = parse_grok_billing_usage(response.text);
        if (!windows.empty()) store_cached(std::move(cache_key), windows);
        return windows;
    }

private:
    static void copy_header(const cpr::Header& source,
                            cpr::Header& destination,
                            std::string_view key) {
        if (const auto value = transport::find_header(source, key);
            value.has_value()) {
            destination[std::string(key)] = *value;
        }
    }

    [[nodiscard]] std::optional<std::vector<UsageWindow>> load_cached(
        const std::string& key) {
        const auto now = std::chrono::steady_clock::now();
        std::scoped_lock lock(cache_mutex_);
        const auto it = cache_.find(key);
        if (it == cache_.end()) return std::nullopt;
        if (it->second.expires_at <= now) {
            cache_.erase(it);
            return std::nullopt;
        }
        return it->second.value;
    }

    void store_cached(std::string key, std::vector<UsageWindow> windows) {
        std::scoped_lock lock(cache_mutex_);
        cache_[std::move(key)] = CachedUsage{
            std::move(windows),
            std::chrono::steady_clock::now() + kBillingCacheTtl,
        };
    }

    std::mutex cache_mutex_;
    std::unordered_map<std::string, CachedUsage> cache_;
};

} // namespace

std::vector<UsageWindow> parse_grok_billing_usage(std::string_view payload) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(payload);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return {};

    simdjson::dom::object config;
    if (doc["config"].get(config) != simdjson::SUCCESS) return {};

    std::string label = "usage";
    bool has_period = false;
    simdjson::dom::object period;
    if (config["currentPeriod"].get(period) == simdjson::SUCCESS) {
        has_period = true;
        std::string_view type;
        if (period["type"].get(type) == simdjson::SUCCESS) {
            if (type.find("WEEKLY") != std::string_view::npos) label = "7d";
            else if (type.find("MONTHLY") != std::string_view::npos) label = "30d";
        }
    }

    const auto percentage = numeric_field(config, "creditUsagePercent");
    const float monthly_limit =
        legacy_credit_value(config, "monthlyLimit").value_or(0.0f);
    const float monthly_used =
        legacy_credit_value(config, "used").value_or(0.0f);
    const std::optional<float> monthly_percentage = monthly_limit > 0.0f
        ? std::optional<float>{monthly_used / monthly_limit * 100.0f}
        : std::nullopt;

    std::vector<UsageWindow> windows;
    if (has_period) {
        // Proto3 JSON omits zero-valued scalars. Grok Build treats a missing
        // percentage for a typed current period as zero usage.
        windows.push_back(UsageWindow{
            label,
            std::clamp(percentage.value_or(0.0f) / 100.0f, 0.0f, 1.5f),
        });

        // New responses normally expose exactly one typed period. If a
        // transitional response also contains an independent legacy monthly
        // allowance, retain both real windows instead of discarding the 30d
        // data. Never synthesize 30d from the weekly percentage.
        if (label != "30d" && monthly_percentage.has_value()) {
            windows.push_back(UsageWindow{
                "30d",
                std::clamp(*monthly_percentage / 100.0f, 0.0f, 1.5f),
            });
        }
        return windows;
    }

    if (percentage.has_value()) {
        windows.push_back(UsageWindow{
            monthly_percentage.has_value() ? "30d" : "usage",
            std::clamp(*percentage / 100.0f, 0.0f, 1.5f),
        });
    } else if (monthly_percentage.has_value()) {
        windows.push_back(UsageWindow{
            "30d",
            std::clamp(*monthly_percentage / 100.0f, 0.0f, 1.5f),
        });
    }
    return windows;
}

std::shared_ptr<IGrokBillingUsageSource> make_grok_billing_usage_source() {
    return std::make_shared<GrokBillingUsageSource>();
}

} // namespace core::llm::protocols
