#pragma once

#include "../policy/PolicyTypes.hpp"

#include <cstdint>
#include <expected>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace cpr {
class Session;
}

namespace core::policy {
class PolicySnapshot;
}

namespace core::net {

// The destination of an HTTP(S) URL, as the policy sees it.
struct UrlDestination {
    std::string host;
    std::uint16_t port{};
    bool valid{false};
};

/**
 * Parses the authority of an HTTP(S) URL.
 *
 * The URL is parsed by the library the transport uses, so the destination the
 * policy judges is the one the client connects to. A URL with credentials, or
 * with a scheme other than HTTP(S), is refused: both hide a destination from a
 * reader.
 */
[[nodiscard]] UrlDestination parse_http_destination(std::string_view url);

// Who chose where a request goes.
enum class Destination {
    // The request names it. The scope recognises its own configured endpoint;
    // any other destination is the policy's to decide.
    requested,
    // The caller vouches that the user configured it, as a sign-in configures
    // its token endpoint. An unmatched ask may allow it implicitly.
    configured,
};

/**
 * The network policy, applied to the requests of one operation.
 *
 * Whatever reaches the network on behalf of a principal (a model provider, an
 * MCP server, a web tool) opens a scope for each operation and passes every
 * HTTP session of it to admit(). admit() decides the session's destination and
 * binds the decision to the socket libcurl opens: no proxy, no redirect, and
 * only an address the decision covers. A destination is decided once per
 * scope, so a retry is never asked about twice.
 *
 * Until it closes, the scope is also the one in effect on the thread that
 * opened it. Code that cannot be handed the scope, such as a protocol's
 * preparation request or an OAuth token refresh, uses admit_in_scope() and is
 * decided as part of the same operation.
 *
 * Without a policy the scope is inert: nothing is decided, no session is
 * changed and nothing is allocated.
 */
class EgressScope {
public:
    /**
     * @param configured_url The endpoint the user configured for this
     *        principal, if it has one. An unmatched ask allows it implicitly;
     *        a deny default or an explicit rule still applies.
     */
    EgressScope(std::shared_ptr<const core::policy::PolicySnapshot> policy,
                core::policy::Subject subject,
                core::policy::Action action,
                std::string_view configured_url = {});
    ~EgressScope();

    EgressScope(const EgressScope&) = delete;
    EgressScope& operator=(const EgressScope&) = delete;
    EgressScope(EgressScope&&) = delete;
    EgressScope& operator=(EgressScope&&) = delete;

    [[nodiscard]] bool enforcing() const noexcept { return state_ != nullptr; }

    /// Decides where @p url goes without sending anything.
    /// @return Why it is refused, or nullopt when it may be reached.
    [[nodiscard]] std::optional<std::string> authorize(
        std::string_view url, Destination destination = Destination::requested);

    /// Decides where @p session goes and binds it there. Call it once the
    /// session is configured, and keep the scope open until the transfer
    /// ends: a later SetUrl, SetRedirect or SetProxies would undo the binding.
    /// @return Why the session must not be performed, or nullopt when it may be.
    [[nodiscard]] std::optional<std::string> admit(
        cpr::Session& session, Destination destination = Destination::requested);

private:
    struct State;

    [[nodiscard]] std::expected<const core::policy::Resource*, std::string> decide(
        std::string_view url, Destination destination);

    std::unique_ptr<State> state_;
    // The scope that was in effect on this thread, restored on destruction.
    EgressScope* previous_{nullptr};
};

/**
 * Admits @p session through the scope in effect on this thread.
 *
 * Outside any scope the session is left as the caller built it.
 */
[[nodiscard]] std::optional<std::string> admit_in_scope(
    cpr::Session& session, Destination destination = Destination::requested);

/**
 * Whether a connection opened on this thread must be bound to a decision.
 *
 * A transport that cannot bind its connection, such as a WebSocket, must not
 * be used while it must.
 */
[[nodiscard]] bool binding_required() noexcept;

} // namespace core::net
