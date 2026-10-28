#include "EgressScope.hpp"

#include "NetworkAddress.hpp"
#include "../policy/PolicyRules.hpp"
#include "../policy/PolicySnapshot.hpp"

#include <cpr/session.h>
#include <curl/curl.h>

#include <charconv>
#include <format>
#include <list>
#include <netinet/in.h>
#include <sys/socket.h>

namespace core::net {
namespace {

thread_local EgressScope* current_scope = nullptr;

// libcurl asks this for every socket it opens. Only a socket to an address the
// decision covers, on the port it covers, is opened.
curl_socket_t open_bound_socket(void* client, curlsocktype purpose,
                                curl_sockaddr* address) noexcept {
    const auto& allowed = *static_cast<const core::policy::Resource*>(client);
    if (purpose != CURLSOCKTYPE_IPCXN || !address
        || !permits_resolved_address(&address->addr, address->addrlen, allowed.host))
        return CURL_SOCKET_BAD;
    unsigned port = 0;
    if (address->family == AF_INET && address->addrlen >= sizeof(sockaddr_in))
        port = ntohs(reinterpret_cast<const sockaddr_in*>(&address->addr)->sin_port);
    else if (address->family == AF_INET6 && address->addrlen >= sizeof(sockaddr_in6))
        port = ntohs(reinterpret_cast<const sockaddr_in6*>(&address->addr)->sin6_port);
    if (port != allowed.port) return CURL_SOCKET_BAD;
    return ::socket(address->family, address->socktype, address->protocol);
}

// Binds the session's connection to @p allowed, which must outlive the transfer.
[[nodiscard]] bool bind(cpr::Session& session, const core::policy::Resource& allowed) {
    const auto holder = session.GetCurlHolder();
    if (!holder || !holder->handle) return false;
    CURL* const handle = holder->handle;
    // An ambient proxy would carry the request to a host the decision never
    // named, and so would a redirect, which libcurl follows with the body.
    // Both are switched off, and the socket check admits only the approved
    // address and port.
    return curl_easy_setopt(handle, CURLOPT_PROXY, "") == CURLE_OK
        && curl_easy_setopt(handle, CURLOPT_FOLLOWLOCATION, 0L) == CURLE_OK
        && curl_easy_setopt(handle, CURLOPT_MAXREDIRS, 0L) == CURLE_OK
        && curl_easy_setopt(handle, CURLOPT_POSTREDIR, 0L) == CURLE_OK
        && curl_easy_setopt(handle, CURLOPT_OPENSOCKETFUNCTION, &open_bound_socket) == CURLE_OK
        && curl_easy_setopt(handle, CURLOPT_OPENSOCKETDATA, &allowed) == CURLE_OK;
}

[[nodiscard]] std::string describe(const core::policy::Resource& resource) {
    return resource.host.find(':') == std::string::npos
        ? std::format("{}:{}", resource.host, resource.port)
        : std::format("[{}]:{}", resource.host, resource.port);
}

} // namespace

UrlDestination parse_http_destination(std::string_view url) {
    using UrlPtr = std::unique_ptr<CURLU, decltype(&curl_url_cleanup)>;
    using PartPtr = std::unique_ptr<char, decltype(&curl_free)>;
    UrlPtr parsed(curl_url(), &curl_url_cleanup);
    const std::string text(url);
    if (!parsed || curl_url_set(parsed.get(), CURLUPART_URL, text.c_str(), 0) != CURLUE_OK)
        return {};
    auto part = [&](CURLUPart name, unsigned flags = 0) -> PartPtr {
        char* value = nullptr;
        if (curl_url_get(parsed.get(), name, &value, flags) != CURLUE_OK) value = nullptr;
        return PartPtr(value, &curl_free);
    };
    const auto scheme = part(CURLUPART_SCHEME);
    const auto host = part(CURLUPART_HOST);
    const auto port = part(CURLUPART_PORT, CURLU_DEFAULT_PORT);
    const auto user = part(CURLUPART_USER);
    const auto password = part(CURLUPART_PASSWORD);
    if (!scheme || !host || !port || user || password) return {};
    if (std::string_view(scheme.get()) != "http" && std::string_view(scheme.get()) != "https")
        return {};
    unsigned port_value = 0;
    const std::string_view port_text(port.get());
    const auto [end, error] = std::from_chars(port_text.data(),
                                              port_text.data() + port_text.size(),
                                              port_value);
    if (error != std::errc{} || end != port_text.data() + port_text.size()
        || port_value == 0 || port_value > 65535)
        return {};
    const auto normalized = core::policy::normalize_host(host.get());
    if (normalized.empty()) return {};
    return UrlDestination{.host = normalized,
                          .port = static_cast<std::uint16_t>(port_value),
                          .valid = true};
}

struct EgressScope::State {
    std::shared_ptr<const core::policy::PolicySnapshot> policy;
    core::policy::Subject subject;
    core::policy::Action action;
    std::optional<core::policy::Resource> configured;
    // The destinations allowed so far. A list, so that an element keeps its
    // address: libcurl reads the one a connection is bound to.
    std::list<core::policy::Resource> allowed;
};

EgressScope::EgressScope(std::shared_ptr<const core::policy::PolicySnapshot> policy,
                         core::policy::Subject subject,
                         core::policy::Action action,
                         std::string_view configured_url)
    : previous_(current_scope) {
    if (policy) {
        state_ = std::make_unique<State>();
        state_->policy = std::move(policy);
        state_->subject = std::move(subject);
        state_->action = action;
        if (!configured_url.empty()) {
            if (const auto endpoint = parse_http_destination(configured_url); endpoint.valid)
                state_->configured = core::policy::Resource{.host = endpoint.host, .port = endpoint.port};
        }
    }
    current_scope = this;
}

EgressScope::~EgressScope() {
    current_scope = previous_;
}

std::expected<const core::policy::Resource*, std::string> EgressScope::decide(
    std::string_view url, Destination destination) {
    const auto target = parse_http_destination(url);
    if (!target.valid)
        return std::unexpected("Network policy: the destination of this request cannot be determined");
    core::policy::Resource resource{.host = target.host, .port = target.port};
    for (const auto& allowed : state_->allowed)
        if (allowed == resource) return &allowed;

    const core::policy::NetworkIntent intent{
        .subject = state_->subject, .action = state_->action, .resource = resource};
    std::optional<core::policy::ConfiguredDestination> configured;
    if (destination == Destination::configured || state_->configured == resource) {
        configured = core::policy::ConfiguredDestination{.kind = intent.subject.kind,
                                                         .id = intent.subject.id,
                                                         .operation = intent.action.operation,
                                                         .flow = intent.action.flow,
                                                         .resource = resource};
    }
    const auto decision = state_->policy->authorize(intent, configured ? &*configured : nullptr);
    if (decision.effect != core::policy::Effect::allow)
        return std::unexpected(
            std::format("Network policy refused {}: {}", describe(resource), decision.reason));
    return &state_->allowed.emplace_back(std::move(resource));
}

std::optional<std::string> EgressScope::authorize(std::string_view url, Destination destination) {
    if (!state_) return std::nullopt;
    auto allowed = decide(url, destination);
    if (!allowed) return std::move(allowed.error());
    return std::nullopt;
}

std::optional<std::string> EgressScope::admit(cpr::Session& session, Destination destination) {
    if (!state_) return std::nullopt;
    auto allowed = decide(session.GetFullRequestUrl(), destination);
    if (!allowed) return std::move(allowed.error());
    if (!bind(session, **allowed))
        return "Network policy: the connection cannot be bound to its approved destination";
    return std::nullopt;
}

std::optional<std::string> admit_in_scope(cpr::Session& session, Destination destination) {
    if (current_scope == nullptr) return std::nullopt;
    return current_scope->admit(session, destination);
}

bool binding_required() noexcept {
    return current_scope != nullptr && current_scope->enforcing();
}

} // namespace core::net
