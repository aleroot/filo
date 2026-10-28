#pragma once

#include <sys/socket.h>

#include <string_view>

namespace core::net {

/**
 * Checks the address a connection actually reaches.
 *
 * Public addresses are accepted. Non-public addresses require an exact
 * IP-literal destination, except the names `localhost` and `*.localhost`,
 * which may connect only to a loopback address. Known cloud metadata
 * addresses are always denied. A name is never trusted to mean whatever it
 * resolved to.
 *
 * This belongs to the transport, not to the policy: the policy decides on a
 * host, and this is what binds that decision to the socket libcurl opens.
 */
[[nodiscard]] bool permits_resolved_address(const sockaddr* address,
                                            socklen_t length,
                                            std::string_view approved_host) noexcept;

} // namespace core::net
