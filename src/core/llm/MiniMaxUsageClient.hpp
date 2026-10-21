#pragma once
#include "protocols/ApiProtocol.hpp"
#include <functional>
#include <memory>
#include <optional>

namespace core::llm {
/// Pure decoder. Missing or inconsistent quota is unknown, never zero usage.
[[nodiscard]] std::optional<protocols::RateLimitInfo> parse_minimax_quota(std::string_view body);

class IMiniMaxUsageSource {
public:
    virtual ~IMiniMaxUsageSource() = default;
    [[nodiscard]] virtual std::optional<protocols::RateLimitInfo> read(
        std::string_view base_url, const cpr::Header& request_headers) = 0;
};
/// Fetch one account's quota; implementations must bound their network work
/// and permit concurrent calls for different accounts.
using MiniMaxQuotaFetcher = std::function<std::optional<protocols::RateLimitInfo>(std::string_view)>;

/// Defaults to a shared source so protocol clones coalesce reads per account.
/// An injected fetcher creates an isolated cache for deterministic verification.
[[nodiscard]] std::shared_ptr<IMiniMaxUsageSource> make_minimax_usage_source(
    MiniMaxQuotaFetcher fetcher = {});
} // namespace core::llm
