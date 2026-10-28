#include "CodexUsageClient.hpp"

#include "UsageSnapshotCache.hpp"
#include "transport/HttpHeaderUtils.hpp"
#include "core/utils/Crypto.hpp"
#include "core/utils/StringUtils.hpp"

#include <cpr/cpr.h>
#include <simdjson.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <format>
#include <utility>

namespace core::llm::codex {
using protocols::RateLimitInfo;
using protocols::SupplementalCredits;
using protocols::UsageWindow;

namespace {

constexpr std::size_t kMaxBodyBytes = 256 * 1024;

/// The request headers the usage read needs, as Codex's backend client sends them.
constexpr std::array kForwardedHeaders{
    std::string_view{"Authorization"},
    std::string_view{"chatgpt-account-id"},
    std::string_view{"User-Agent"},
    std::string_view{"originator"},
    std::string_view{"X-OpenAI-Fedramp"},
};

/// The production transport, bounded like the other providers' usage reads.
std::optional<std::string> get(std::string_view url, const cpr::Header& headers) {
    std::string body;
    const auto response = cpr::Get(
        cpr::Url{std::string(url)}, headers,
        cpr::Timeout{2500}, cpr::ConnectTimeout{1500}, cpr::Redirect{false},
        cpr::WriteCallback{[&](std::string_view bytes, intptr_t) {
            if (bytes.size() > kMaxBodyBytes - body.size()) return false;
            body.append(bytes);
            return true;
        }});
    if (response.status_code != 200 || response.error.code != cpr::ErrorCode::OK) return std::nullopt;
    return body;
}

std::optional<double> number(simdjson::dom::element node, std::string_view key) {
    double value = 0.0;
    if (node[key].get(value) == simdjson::SUCCESS && std::isfinite(value)) return value;
    return std::nullopt;
}

std::string_view text(simdjson::dom::element node, std::string_view key) {
    std::string_view value;
    return node[key].get(value) == simdjson::SUCCESS ? value : std::string_view{};
}

bool flag(simdjson::dom::element node, std::string_view key) {
    bool value = false;
    return node[key].get(value) == simdjson::SUCCESS && value;
}

/// The reset instant: `reset_at`, else now plus `reset_after_seconds`.
int64_t reset_time(simdjson::dom::element node) {
    if (const auto at = number(node, "reset_at"); at && *at > 0) return static_cast<int64_t>(*at);
    if (const auto after = number(node, "reset_after_seconds"); after && *after > 0) {
        const auto now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        return now + static_cast<int64_t>(*after);
    }
    return 0;
}

/// "5h", "7d", "90m", prefixed by the limit id unless it is the default
/// "codex" one, as the `x-<limit>-*` header windows are labelled.
std::string window_label(int64_t seconds, std::string_view limit_id, std::string_view position) {
    std::string duration;
    const int64_t minutes = seconds / 60;
    if (minutes <= 0) duration = std::string(position);
    else if (minutes % (24 * 60) == 0) duration = std::format("{}d", minutes / (24 * 60));
    else if (minutes % 60 == 0) duration = std::format("{}h", minutes / 60);
    else duration = std::format("{}m", minutes);
    if (limit_id.empty() || limit_id == "codex") return duration;
    return std::string(limit_id) + " " + duration;
}

std::string normalize_limit_id(std::string_view value) {
    std::string normalized = core::utils::str::to_lower_ascii_copy(value);
    std::ranges::replace(normalized, '-', '_');
    return normalized;
}

void append_windows(RateLimitInfo& info, simdjson::dom::element rate_limit,
                    std::string_view limit_id) {
    if (!rate_limit.is_object()) return;
    if (flag(rate_limit, "limit_reached")) info.is_rate_limited = true;
    for (const auto& [key, position] : {std::pair{"primary_window", "primary"},
                                        std::pair{"secondary_window", "secondary"}}) {
        simdjson::dom::element window;
        if (rate_limit[key].get(window) != simdjson::SUCCESS || !window.is_object()) continue;
        const auto used = number(window, "used_percent");
        if (!used) continue;
        const auto seconds = number(window, "limit_window_seconds").value_or(0.0);
        info.usage_windows.push_back(UsageWindow{
            .label = window_label(static_cast<int64_t>(seconds), limit_id, position),
            .utilization = static_cast<float>(std::max(0.0, *used) / 100.0),
            .resets_at = reset_time(window),
        });
    }
}

class UsageClient final : public IUsageSource {
public:
    explicit UsageClient(UsageTransport transport) : transport_(std::move(transport)) {}

    std::optional<RateLimitInfo> read(std::string_view base_url,
                                      const cpr::Header& request_headers) override {
        const auto url = usage_url(base_url);
        if (!url) return std::nullopt;
        const auto authorization = transport::find_header(request_headers, "Authorization");
        if (!authorization || authorization->empty()) return std::nullopt;
        cpr::Header headers;
        for (const auto name : kForwardedHeaders) {
            if (auto value = transport::find_header(request_headers, name); value && !value->empty()) {
                headers[std::string(name)] = std::move(*value);
            }
        }
        const std::string identity = core::utils::crypto::sha256_hex(
            *authorization + '\n'
            + transport::find_header(request_headers, "chatgpt-account-id").value_or("")
            + '\n' + *url);
        return cache_.get(identity, [&]() -> std::optional<RateLimitInfo> {
            const auto body = transport_(*url, headers);
            return body ? parse_usage(*body) : std::nullopt;
        });
    }

private:
    UsageTransport transport_;
    UsageSnapshotCache<RateLimitInfo> cache_{std::chrono::minutes(1)};
};

} // namespace

std::optional<std::string> usage_url(std::string_view base_url) {
    std::string base = core::utils::str::trim_trailing_slashes(base_url);
    const std::string lower = core::utils::str::to_lower_ascii_copy(base);
    if (!lower.starts_with("https://chatgpt.com/") && !lower.starts_with("https://chat.openai.com/")) {
        return std::nullopt;
    }
    constexpr std::string_view kBackendApi = "/backend-api";
    const auto at = lower.find(kBackendApi);
    if (at == std::string::npos) return std::nullopt;
    return base.substr(0, at + kBackendApi.size()) + "/wham/usage";
}

std::optional<RateLimitInfo> parse_usage(std::string_view body) {
    if (body.size() > kMaxBodyBytes) return std::nullopt;
    simdjson::dom::parser parser;
    const simdjson::padded_string padded(body);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS || !doc.is_object()) return std::nullopt;

    RateLimitInfo info;
    if (const auto plan = text(doc, "plan_type"); !plan.empty()) {
        info.subscription.tier_label = "ChatGPT " + std::string(plan);
    }

    simdjson::dom::element element;
    if (doc["rate_limit"].get(element) == simdjson::SUCCESS) append_windows(info, element, "codex");

    simdjson::dom::array additional;
    if (doc["additional_rate_limits"].get(additional) == simdjson::SUCCESS) {
        for (const auto limit : additional) {
            std::string_view id = text(limit, "metered_feature");
            if (id.empty()) id = text(limit, "limit_name");
            if (id.empty() || limit["rate_limit"].get(element) != simdjson::SUCCESS) continue;
            append_windows(info, element, normalize_limit_id(id));
        }
    }

    simdjson::dom::element spend;
    if (doc["spend_control"].get(spend) == simdjson::SUCCESS && spend.is_object()) {
        if (flag(spend, "reached")) info.is_rate_limited = true;
        simdjson::dom::element limit;
        if (spend["individual_limit"].get(limit) == simdjson::SUCCESS && limit.is_object()) {
            // Codex's "Monthly credit limit" row.
            if (const auto remaining = number(limit, "remaining_percent")) {
                info.usage_windows.push_back(UsageWindow{
                    .label = "30d",
                    .utilization = static_cast<float>((100.0 - std::clamp(*remaining, 0.0, 100.0)) / 100.0),
                    .resets_at = reset_time(limit),
                });
            }
        }
    }

    if (doc["credits"].get(element) == simdjson::SUCCESS && element.is_object()) {
        bool has_credits = false;
        bool unlimited = false;
        if (element["has_credits"].get(has_credits) == simdjson::SUCCESS
            && element["unlimited"].get(unlimited) == simdjson::SUCCESS) {
            info.subscription.supplemental_credits = SupplementalCredits{
                .available = has_credits,
                .unlimited = unlimited,
                .balance = std::string(text(element, "balance")),
            };
        }
    }

    if (doc["rate_limit_reached_type"].get(element) == simdjson::SUCCESS && element.is_object()) {
        info.subscription.limit_reached_reason = std::string(text(element, "type"));
    }
    return info;
}

std::shared_ptr<IUsageSource> make_usage_source(UsageTransport transport) {
    if (transport) return std::make_shared<UsageClient>(std::move(transport));
    static const auto source = std::make_shared<UsageClient>(get);
    return source;
}

} // namespace core::llm::codex
