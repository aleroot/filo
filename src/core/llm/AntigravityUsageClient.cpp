#include "AntigravityUsageClient.hpp"

#include "AntigravityModelTraits.hpp"
#include "UsageSnapshotCache.hpp"
#include "transport/HttpHeaderUtils.hpp"
#include "core/auth/GoogleAntigravityIdentity.hpp"
#include "core/auth/GoogleCodeAssist.hpp"
#include "core/utils/Crypto.hpp"
#include "core/utils/JsonUtils.hpp"
#include "core/utils/TimeUtils.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <future>
#include <utility>

namespace core::llm::antigravity {
using protocols::RateLimitInfo;
using protocols::UsageWindow;

namespace {

constexpr std::size_t kMaxBodyBytes = 256 * 1024;

/// The production transport, bounded like the other providers' usage reads.
std::optional<std::string> post(std::string_view url, std::string_view authorization,
                                std::string_view body) {
    std::string response_body;
    const auto response = cpr::Post(
        cpr::Url{std::string(url)},
        cpr::Header{{"Authorization", std::string(authorization)},
                    {"Content-Type", "application/json"},
                    {"User-Agent", core::auth::antigravity::user_agent()}},
        cpr::Body{std::string(body)},
        cpr::Timeout{2500}, cpr::ConnectTimeout{1500}, cpr::Redirect{false},
        cpr::WriteCallback{[&](std::string_view bytes, intptr_t) {
            if (bytes.size() > kMaxBodyBytes - response_body.size()) return false;
            response_body.append(bytes);
            return true;
        }});
    if (response.status_code != 200 || response.error.code != cpr::ErrorCode::OK) return std::nullopt;
    return response_body;
}

std::string endpoint(std::string_view base_url, std::string_view method) {
    while (!base_url.empty() && base_url.back() == '/') base_url.remove_suffix(1);
    return std::string(base_url) + "/v1internal:" + std::string(method);
}

std::string project_body(std::string_view project) {
    if (project.empty()) return "{}";
    return R"({"project":")" + core::utils::escape_json_string(project) + "\"}";
}

/// A share in [0, 1]; a limit counted in units (`remainingAmount`) has none.
std::optional<double> remaining_fraction(simdjson::dom::element entry) {
    double remaining = 0.0;
    if (entry["remainingFraction"].get(remaining) != simdjson::SUCCESS
        || !std::isfinite(remaining) || remaining < 0.0 || remaining > 1.0) {
        return std::nullopt;
    }
    return remaining;
}

int64_t reset_time(simdjson::dom::element entry) {
    std::string_view reset;
    return entry["resetTime"].get(reset) == simdjson::SUCCESS
        ? core::utils::time::parse_timestamp_or_duration(reset) : 0;
}

const QuotaBucket* bucket_for(std::span<const QuotaBucket> buckets, std::string_view wire_model) {
    const auto exact = std::ranges::find(buckets, wire_model, &QuotaBucket::model);
    if (exact != buckets.end()) return &*exact;
    const auto variant = split_thinking_variant(wire_model);
    const std::string prefix = std::string(variant ? variant->family : wire_model) + "-";
    // A sibling differs by one trailing segment (`-tiered`, `-low`), so a
    // shorter id never borrows another family's bucket. Siblings share one
    // group, so any of them names it.
    const auto sibling = std::ranges::find_if(buckets, [&](const QuotaBucket& bucket) {
        return bucket.model.starts_with(prefix)
            && bucket.model.find('-', prefix.size()) == std::string::npos;
    });
    return sibling != buckets.end() ? &*sibling : nullptr;
}

/// The group `bucket` is the tightest window of. A touched bucket carries that
/// window's reset time; an untouched one can only come from a group no request
/// has touched. Two candidates mean the reading cannot tell, so none is chosen.
const QuotaGroup* group_of(std::span<const QuotaGroup> groups, const QuotaBucket& bucket) {
    const QuotaGroup* found = nullptr;
    for (const auto& group : groups) {
        const bool member = !group.windows.empty()
            && (bucket.resets_at != 0
                    ? std::ranges::any_of(group.windows, [&](const QuotaWindow& window) {
                          return window.resets_at == bucket.resets_at;
                      })
                    : std::ranges::all_of(group.windows, [](const QuotaWindow& window) {
                          return window.resets_at == 0;
                      }));
        if (!member) continue;
        if (found) return nullptr;
        found = &group;
    }
    return found;
}

/// The status-bar label of a summary window; the TUI knows "5h" and "7d".
std::string window_label(std::string_view window) {
    return window == "weekly" ? "7d" : std::string(window);
}

float utilization(double remaining_fraction) {
    return static_cast<float>(1.0 - remaining_fraction);
}

class UsageClient final : public IUsageSource {
public:
    explicit UsageClient(UsageTransport transport) : transport_(std::move(transport)) {}

    std::optional<RateLimitInfo> read(std::string_view base_url, const cpr::Header& headers,
                                      std::string_view project,
                                      std::string_view wire_model) override {
        const auto authorization = transport::find_header(headers, "Authorization");
        if (!authorization || authorization->empty()) return std::nullopt;
        const Account account{base_url, *authorization, project};
        const std::string identity = core::utils::crypto::sha256_hex(
            *authorization + '\n' + std::string(base_url) + '\n' + std::string(project));
        // The plan changes only on an upgrade, the quota with every request.
        const auto plan = [&] {
            return plans_.get(identity, [&] { return read_plan(account); });
        };
        const auto reading = readings_.get(identity, [&]() -> std::optional<Reading> {
            // Everything due is read side by side: a refresh on the response
            // path costs the slowest read, not the sum. Leaving this scope
            // joins both futures, even when the bucket read throws.
            auto plan_read = std::async(std::launch::async, plan);
            auto summary = std::async(std::launch::async, [&] { return read_summary(account); });
            auto buckets = read_buckets(account);
            std::vector<QuotaGroup> groups;
            try {
                groups = summary.get();
            } catch (...) { /* The windows stay unknown; the buckets still count. */ }
            if (!buckets) return std::nullopt;
            return Reading{std::move(*buckets), std::move(groups)};
        });
        if (!reading) return std::nullopt;
        auto info = quota_for_model(reading->buckets, reading->groups, wire_model);
        if (info) {
            if (auto name = plan()) info->subscription.tier_label = std::move(*name);
        }
        return info;
    }

private:
    /// Whose quota is read, and from which host.
    struct Account {
        std::string_view base_url;
        std::string_view authorization;
        std::string_view project;
    };
    struct Reading {
        std::vector<QuotaBucket> buckets;
        std::vector<QuotaGroup> groups; ///< Empty when the summary could not be read.
    };

    std::optional<std::string> call(const Account& account, std::string_view method,
                                    std::string_view body) const {
        return transport_(endpoint(account.base_url, method), account.authorization, body);
    }

    std::optional<std::vector<QuotaBucket>> read_buckets(const Account& account) const {
        const auto body = call(account, "retrieveUserQuota", project_body(account.project));
        if (!body) return std::nullopt;
        return parse_quota_buckets(*body);
    }

    std::vector<QuotaGroup> read_summary(const Account& account) const {
        const auto body = call(account, "retrieveUserQuotaSummary", project_body(account.project));
        return body ? parse_quota_summary(*body) : std::vector<QuotaGroup>{};
    }

    std::optional<std::string> read_plan(const Account& account) const {
        const auto body = call(account, "loadCodeAssist",
            core::auth::google_code_assist::load_code_assist_payload(account.project, "ANTIGRAVITY"));
        if (!body) return std::nullopt;
        return parse_plan_name(*body);
    }

    UsageTransport transport_;
    UsageSnapshotCache<Reading> readings_{std::chrono::minutes(1)};
    // A failed plan read only costs the tier label, so it is retried within
    // minutes instead of hiding the plan for the hour.
    UsageSnapshotCache<std::string> plans_{std::chrono::hours(1), std::chrono::minutes(5)};
};

} // namespace

std::vector<QuotaBucket> parse_quota_buckets(std::string_view body) {
    std::vector<QuotaBucket> buckets;
    if (body.size() > kMaxBodyBytes) return buckets;
    simdjson::dom::parser parser;
    const simdjson::padded_string padded(body);
    simdjson::dom::array entries;
    if (parser.parse(padded)["buckets"].get(entries) != simdjson::SUCCESS) return buckets;
    for (const auto entry : entries) {
        std::string_view model;
        const auto remaining = remaining_fraction(entry);
        if (entry["modelId"].get(model) != simdjson::SUCCESS || model.empty() || !remaining) continue;
        buckets.push_back({std::string(model), *remaining, reset_time(entry)});
    }
    return buckets;
}

std::vector<QuotaGroup> parse_quota_summary(std::string_view body) {
    std::vector<QuotaGroup> groups;
    if (body.size() > kMaxBodyBytes) return groups;
    simdjson::dom::parser parser;
    const simdjson::padded_string padded(body);
    simdjson::dom::array entries;
    if (parser.parse(padded)["groups"].get(entries) != simdjson::SUCCESS) return groups;
    for (const auto entry : entries) {
        QuotaGroup group;
        std::string_view name;
        if (entry["displayName"].get(name) == simdjson::SUCCESS) group.name = std::string(name);
        simdjson::dom::array windows;
        if (entry["buckets"].get(windows) != simdjson::SUCCESS) continue;
        for (const auto window : windows) {
            std::string_view kind;
            bool disabled = false;
            const auto remaining = remaining_fraction(window);
            // A disabled limit does not apply.
            if (window["window"].get(kind) != simdjson::SUCCESS || kind.empty() || !remaining
                || (window["disabled"].get(disabled) == simdjson::SUCCESS && disabled)) {
                continue;
            }
            group.windows.push_back({std::string(kind), *remaining, reset_time(window)});
        }
        if (!group.windows.empty()) groups.push_back(std::move(group));
    }
    return groups;
}

std::optional<std::string> parse_plan_name(std::string_view body) {
    try {
        const auto response = core::auth::google_code_assist::parse_load_code_assist_response(body);
        const auto& tier = response.paid_tier && !response.paid_tier->id.empty()
            ? response.paid_tier : response.current_tier;
        if (tier && !tier->name.empty()) return tier->name;
    } catch (...) { /* Malformed: the plan stays unknown. */ }
    return std::nullopt;
}

std::optional<RateLimitInfo> quota_for_model(std::span<const QuotaBucket> buckets,
                                             std::span<const QuotaGroup> groups,
                                             std::string_view wire_model) {
    const QuotaBucket* bucket = bucket_for(buckets, wire_model);
    if (!bucket) return std::nullopt;
    RateLimitInfo info;
    info.is_rate_limited = bucket->remaining_fraction <= 0.0;
    const QuotaGroup* group = group_of(groups, *bucket);
    if (!group) {
        info.usage_windows.push_back({"quota", utilization(bucket->remaining_fraction), bucket->resets_at});
        return info;
    }
    for (const auto& window : group->windows) {
        info.usage_windows.push_back(
            {window_label(window.window), utilization(window.remaining_fraction), window.resets_at});
        info.is_rate_limited |= window.remaining_fraction <= 0.0;
        if (bucket->resets_at != 0 && window.resets_at == bucket->resets_at
            && info.unified_representative_claim.empty()) {
            info.unified_representative_claim = info.usage_windows.back().label;
        }
    }
    // Shortest window first, as Claude's are shown.
    std::ranges::stable_sort(info.usage_windows, {}, [](const UsageWindow& window) {
        return window.label == "5h" ? 0 : window.label == "7d" ? 1 : 2;
    });
    return info;
}

std::shared_ptr<IUsageSource> make_usage_source(UsageTransport transport) {
    if (transport) return std::make_shared<UsageClient>(std::move(transport));
    static const auto source = std::make_shared<UsageClient>(post);
    return source;
}

} // namespace core::llm::antigravity
