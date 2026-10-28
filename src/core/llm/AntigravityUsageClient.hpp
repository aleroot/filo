#pragma once

#include "protocols/ApiProtocol.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::llm::antigravity {

/// One `retrieveUserQuota` bucket: what is left of one SKU's allowance. The
/// backend reports the tightest window of the SKU's quota group here, without
/// saying which window that is.
struct QuotaBucket {
    std::string model;
    double remaining_fraction = 1.0; ///< 1 = untouched, 0 = exhausted.
    int64_t resets_at = 0;           ///< Unix seconds; 0 while the window is untouched.
};

/// One limit of a `retrieveUserQuotaSummary` group.
struct QuotaWindow {
    std::string window;              ///< As served: "5h", "weekly".
    double remaining_fraction = 1.0;
    int64_t resets_at = 0;
};

/// Models that share their limits, e.g. "Gemini Models" (Flash and Pro) or
/// "Claude and GPT models": every window applies to each model of the group.
struct QuotaGroup {
    std::string name;
    std::vector<QuotaWindow> windows;
};

/// Pure decoders; malformed input yields nothing rather than zero usage.
[[nodiscard]] std::vector<QuotaBucket> parse_quota_buckets(std::string_view body);
[[nodiscard]] std::vector<QuotaGroup> parse_quota_summary(std::string_view body);
/// The plan `loadCodeAssist` names: the Google plan the account pays for
/// ("Google AI Pro"), else its Antigravity tier.
[[nodiscard]] std::optional<std::string> parse_plan_name(std::string_view body);

/// The status-bar view of `wire_model`'s quota: every window of its group,
/// the 5-hour one first and the weekly one as "7d", with the tightest marked
/// as governing.
///
/// Antigravity meters every SKU separately, so the bucket of the id actually
/// sent wins; a SKU without its own bucket uses one of its family
/// (`gemini-3.8-flash-high` -> `gemini-3.8-flash-tiered`). The summary does
/// not list a group's models, but a SKU's bucket repeats its group's tightest
/// window, reset time included, and that names the group. When no single
/// group matches, the bucket is reported alone as "quota", since its window
/// is unknown.
[[nodiscard]] std::optional<protocols::RateLimitInfo> quota_for_model(
    std::span<const QuotaBucket> buckets,
    std::span<const QuotaGroup> groups,
    std::string_view wire_model);

class IUsageSource {
public:
    virtual ~IUsageSource() = default;
    /// The quota of `wire_model` for the account authorised by
    /// `request_headers`, read from `base_url`: the Code Assist host the
    /// provider is configured with, which already receives this credential.
    [[nodiscard]] virtual std::optional<protocols::RateLimitInfo> read(
        std::string_view base_url, const cpr::Header& request_headers,
        std::string_view project, std::string_view wire_model) = 0;
};

/// One Code Assist call: POST `body` to `url` with the `Authorization` value,
/// giving the body of a 200 and nothing on any failure. Implementations must
/// bound their I/O and allow concurrent calls.
using UsageTransport = std::function<std::optional<std::string>(
    std::string_view url, std::string_view authorization, std::string_view body)>;

/// Defaults to one shared source over the network so protocol clones coalesce
/// reads per account. An injected transport gets an isolated cache for
/// deterministic verification.
[[nodiscard]] std::shared_ptr<IUsageSource> make_usage_source(UsageTransport transport = {});

} // namespace core::llm::antigravity
