#include "MiniMaxUsageClient.hpp"
#include "MiniMaxModelTraits.hpp"
#include "transport/HttpHeaderUtils.hpp"
#include "core/auth/MiniMaxClientIdentity.hpp"
#include "core/utils/TimeUtils.hpp"
#include "core/utils/Crypto.hpp"
#include <simdjson.h>
#include <cpr/cpr.h>
#include <algorithm>
#include <format>
#include <chrono>
#include <cmath>
#include <mutex>
#include <limits>
#include <unordered_map>

namespace core::llm {
using protocols::RateLimitInfo;
namespace {
constexpr std::size_t kMaxQuotaBodyBytes = 256 * 1024;

std::optional<double> number(simdjson::dom::element node, std::string_view key) {
    double value;
    if (node[key].get(value) == simdjson::SUCCESS && std::isfinite(value)) return value;
    return std::nullopt;
}
int64_t reset_time(simdjson::dom::element node, std::string_view key) {
    if (const auto value = number(node, key); value && *value > 0) {
        const double seconds = *value > 100000000000.0 ? *value / 1000.0 : *value;
        return seconds < static_cast<double>(std::numeric_limits<int64_t>::max())
            ? static_cast<int64_t>(seconds) : 0;
    }
    std::string_view text;
    if (node[key].get(text) == simdjson::SUCCESS) return core::utils::time::parse_timestamp_or_duration(text);
    return 0;
}
void append_window(RateLimitInfo& info, simdjson::dom::element node,
                   std::string_view prefix, std::string_view label, std::string_view reset) {
    const auto percent_key = std::string(prefix) + "_remaining_percent";
    auto remaining = number(node, percent_key);
    const auto status = number(node, std::string(prefix) + "_status");
    if (status && *status == 3) return; // An unlimited bucket is not an exhausted zero limit.
    // Status 2 is the server's own verdict that this window is exhausted,
    // whatever percentage accompanies it.
    if (status && *status == 2) remaining = 0.0;
    simdjson::dom::element explicit_percent;
    if (!remaining && node[percent_key].get(explicit_percent) != simdjson::SUCCESS) {
        const auto total = number(node, std::string(prefix) + "_total_count");
        const auto count = number(node, std::string(prefix) + "_usage_count");
        // The official CLI treats pre-percentage responses as remaining counts.
        // Never apply that historical interpretation when explicit % is present.
        if (total && count && *total > 0 && *count >= 0 && *count <= *total) remaining = *count / *total * 100.0;
    }
    if (!remaining || *remaining < 0 || *remaining > 100) return;
    const float utilization = static_cast<float>((100.0 - *remaining) / 100.0);
    info.usage_windows.push_back({std::string(label), utilization, reset_time(node, reset)});
    info.is_rate_limited |= *remaining == 0;
}

std::optional<RateLimitInfo> fetch_quota(std::string_view key) {
    std::string body;
    cpr::Header headers{{"Authorization", "Bearer " + std::string(key)}};
    core::auth::minimax_cli::apply_identity_headers(headers);
    const auto response = cpr::Get(
        cpr::Url{std::string(kMiniMaxQuotaUrl)}, headers,
        cpr::Timeout{2500}, cpr::ConnectTimeout{1500}, cpr::Redirect{false},
        cpr::WriteCallback{[&](std::string_view bytes, intptr_t) {
            if (bytes.size() > kMaxQuotaBodyBytes - body.size()) return false;
            body.append(bytes);
            return true;
        }});
    if (response.status_code != 200 || response.error.code != cpr::ErrorCode::OK) {
        return std::nullopt;
    }
    return parse_minimax_quota(body);
}

class MiniMaxUsageClient final : public IMiniMaxUsageSource {
    struct Entry {
        std::mutex mutex;
        std::optional<RateLimitInfo> value;
        std::chrono::steady_clock::time_point expires{};
    };
    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<Entry>> cache_;
    MiniMaxQuotaFetcher fetcher_;

    std::shared_ptr<Entry> entry_for(const std::string& identity) {
        std::lock_guard lock(mutex_);
        if (const auto it = cache_.find(identity); it != cache_.end()) return it->second;
        if (cache_.size() >= 64) {
            // Never evict an in-flight entry: its waiters must share its read.
            const auto idle = std::find_if(cache_.begin(), cache_.end(), [](const auto& item) {
                return item.second.use_count() == 1;
            });
            if (idle == cache_.end()) return {};
            cache_.erase(idle);
        }
        auto entry = std::make_shared<Entry>();
        cache_.emplace(identity, entry);
        return entry;
    }

public:
    explicit MiniMaxUsageClient(MiniMaxQuotaFetcher fetcher) : fetcher_(std::move(fetcher)) {}

    std::optional<RateLimitInfo> read(std::string_view base_url, const cpr::Header& headers) override {
        // A custom proxy must never cause its credential to be forwarded to MiniMax.
        if (!is_official_minimax_base_url(base_url)) return std::nullopt;
        auto key = transport::find_header(headers, "x-api-key");
        if (!key || key->empty()) return std::nullopt;
        const auto identity = core::utils::crypto::sha256_hex(*key);
        const auto entry = entry_for(identity);
        if (!entry) return std::nullopt;
        // Network I/O holds only this account's lock, never the cache lock.
        std::lock_guard lock(entry->mutex);
        if (entry->expires > std::chrono::steady_clock::now()) return entry->value;
        entry->value.reset();
        try {
            entry->value = fetcher_(*key);
        } catch (...) { /* Usage is best-effort and must not fail generation. */ }
        // Quota windows span hours, so a minute of staleness is invisible;
        // the read sits on the response path and should rarely be paid.
        entry->expires = std::chrono::steady_clock::now() + std::chrono::seconds(60);
        return entry->value;
    }
};
} // namespace

std::optional<RateLimitInfo> parse_minimax_quota(std::string_view body) {
    if (body.size() > kMaxQuotaBodyBytes) return std::nullopt;
    simdjson::dom::parser parser;
    simdjson::padded_string padded(body);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return std::nullopt;
    simdjson::dom::element base;
    if (doc["base_resp"].get(base) == simdjson::SUCCESS) {
        if (const auto status = number(base, "status_code"); status && *status != 0) return std::nullopt;
    }
    RateLimitInfo info;
    info.subscription.tier_label = "M Plan";
    const auto append = [&](simdjson::dom::element node) {
        append_window(info, node, "current_interval", "5h", "end_time");
        append_window(info, node, "current_weekly", "7d", "weekly_end_time");
        // A boost enlarges the weekly allowance; remaining percentages stay
        // relative to it, so it is reported rather than folded into usage.
        if (const auto boost = number(node, "weekly_boost_permille"); boost && *boost > 1000.0 && *boost < 100000.0) {
            info.subscription.notice = std::format("Weekly quota boosted to {:g}%", std::round(*boost / 10.0));
        }
    };
    simdjson::dom::array models;
    if (doc["model_remains"].get(models) == simdjson::SUCCESS) {
        for (const auto model : models) {
            std::string_view name;
            if (model["model_name"].get(name) == simdjson::SUCCESS
                && (name == "general" || name == "MiniMax-M*" || name == kMiniMaxDefaultModel)) {
                append(model);
                if (!info.usage_windows.empty()) break;
            }
        }
    }
    if (info.usage_windows.empty()) append(doc);
    if (info.usage_windows.empty()) return std::nullopt;
    return info;
}

std::shared_ptr<IMiniMaxUsageSource> make_minimax_usage_source(MiniMaxQuotaFetcher fetcher) {
    if (fetcher) return std::make_shared<MiniMaxUsageClient>(std::move(fetcher));
    static const auto source = std::make_shared<MiniMaxUsageClient>(fetch_quota);
    return source;
}
} // namespace core::llm
