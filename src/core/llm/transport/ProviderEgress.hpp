#pragma once

#include "../../net/EgressScope.hpp"
#include "../../policy/ApprovalChannel.hpp"
#include "../protocols/ProtocolRequest.hpp"

#include <atomic>
#include <optional>
#include <string>
#include <string_view>

namespace cpr {
class Session;
}

namespace core::llm::transport {

/**
 * The network policy, applied to one request of a model provider.
 *
 * Opened when a request starts and closed when it ends. The provider's
 * configured endpoint is decided first, so a refused provider is sent nothing:
 * no upload, no token count, no conversation. Every request sent while it is
 * open, the stream and a protocol's own preparation requests alike, goes out
 * under that decision. A question the policy asks is shown in the chat that
 * sent the request and withdrawn when the request is cancelled.
 *
 * Without a network policy in the process it decides and binds nothing.
 */
class ProviderEgress {
public:
    ProviderEgress(std::string_view provider_name,
                   std::string_view base_url,
                   const std::string& session_id,
                   const std::atomic_bool& cancelled);

    /// Why the provider's endpoint was refused, or nullopt when it may be reached.
    [[nodiscard]] const std::optional<std::string>& refusal() const noexcept { return refusal_; }

    /// Binds a session the provider is about to perform.
    /// @return Why it must not be performed, or nullopt when it may be.
    [[nodiscard]] std::optional<std::string> admit(cpr::Session& session) {
        return scope_.admit(session);
    }

    /**
     * Performs a request a protocol asked for.
     *
     * The provider owns the transport, so a protocol describes the request it
     * needs and the provider performs it here, under the same decision as the
     * chat request itself. A protocol therefore cannot send anything the
     * policy has not seen.
     */
    [[nodiscard]] protocols::ProtocolResponse post(
        const protocols::ProtocolRequest& request);

private:
    core::net::EgressScope scope_;
    std::optional<core::policy::ApprovalOriginScope> origin_;
    std::optional<std::string> refusal_;
    const std::atomic_bool& cancelled_;
};

} // namespace core::llm::transport
