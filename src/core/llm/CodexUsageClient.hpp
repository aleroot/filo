#pragma once

#include "protocols/ApiProtocol.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace core::llm::codex {

/// Pure decoder of the ChatGPT backend's `GET /wham/usage` body, the reading
/// Codex shows in `/status`: the 5-hour and weekly windows (`rate_limit`),
/// the per-feature ones (`additional_rate_limits`, labelled like their
/// `x-<feature>-*` headers), the workspace's monthly credit limit
/// (`spend_control.individual_limit`, as "30d"), plan and credits. Business
/// workspaces metered by credits report `rate_limit: null`, so the monthly
/// limit is all they have. Malformed input yields nothing.
[[nodiscard]] std::optional<protocols::RateLimitInfo> parse_usage(std::string_view body);

/// `<...>/backend-api/wham/usage` for a ChatGPT backend base URL (the Codex
/// one is `https://chatgpt.com/backend-api/codex`); nothing for any other
/// host, so the credential never goes to an endpoint that did not ask for it.
[[nodiscard]] std::optional<std::string> usage_url(std::string_view base_url);

class IUsageSource {
public:
    virtual ~IUsageSource() = default;
    /// The usage of the account authorised by `request_headers`, read next to
    /// `base_url`.
    [[nodiscard]] virtual std::optional<protocols::RateLimitInfo> read(
        std::string_view base_url, const cpr::Header& request_headers) = 0;
};

/// One GET of `url` with `headers`, giving the body of a 200 and nothing on
/// any failure. Implementations must bound their I/O and allow concurrent calls.
using UsageTransport = std::function<std::optional<std::string>(
    std::string_view url, const cpr::Header& headers)>;

/// Defaults to one shared source over the network so protocol clones coalesce
/// reads per account. An injected transport gets an isolated cache.
[[nodiscard]] std::shared_ptr<IUsageSource> make_usage_source(UsageTransport transport = {});

} // namespace core::llm::codex
