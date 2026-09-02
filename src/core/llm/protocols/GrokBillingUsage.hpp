#pragma once

#include "ApiProtocol.hpp"

#include <memory>
#include <string_view>
#include <vector>

namespace core::llm::protocols {

/**
 * Parsed result of Grok Build's billing API.
 *
 * Besides the utilization windows, the API reports the current billing
 * period's endTime — effectively the subscription renewal boundary — which
 * is surfaced separately so the UI can show a subscription end date.
 */
struct GrokBillingUsage {
    std::vector<UsageWindow> windows;
    int64_t period_ends_at = 0; ///< Unix seconds when the current billing period ends (0 = unknown)

    // Vector-like accessors so existing call sites keep working unchanged.
    [[nodiscard]] bool empty() const noexcept { return windows.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return windows.size(); }
    [[nodiscard]] const UsageWindow& operator[](std::size_t index) const {
        return windows[index];
    }
};

/**
 * Parse the coding-credit windows returned by Grok Build's billing API.
 *
 * The current API normally reports one period (weekly or monthly), unlike
 * Claude's independent 5-hour and 7-day limits. Transitional responses may
 * additionally carry a legacy monthly allowance. An omitted percentage
 * represents zero usage in the protobuf JSON response.
 */
[[nodiscard]] GrokBillingUsage parse_grok_billing_usage(
    std::string_view payload);

/**
 * Provider-owned source for Grok Build billing utilization.
 *
 * The Responses protocol depends on this capability rather than CPR or a
 * concrete endpoint. Tests can inject a deterministic implementation.
 */
class IGrokBillingUsageSource {
public:
    virtual ~IGrokBillingUsageSource() = default;

    [[nodiscard]] virtual GrokBillingUsage fetch(
        std::string_view base_url,
        const cpr::Header& request_headers) = 0;
};

/** Create the production, host-scoped Grok Build billing source. */
[[nodiscard]] std::shared_ptr<IGrokBillingUsageSource>
make_grok_billing_usage_source();

} // namespace core::llm::protocols
