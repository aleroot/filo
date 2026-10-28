#include "NetworkAddress.hpp"

#include <arpa/inet.h>
#include <cstring>
#include <netinet/in.h>
#include <string>

namespace core::net {
namespace {

bool public_v4(const unsigned char* b) noexcept {
    return b[0] != 0 && b[0] != 10 && b[0] != 127 && b[0] < 224
        && !(b[0] == 100 && (b[1] & 0xc0) == 64)
        && !(b[0] == 169 && b[1] == 254)
        && !(b[0] == 172 && (b[1] & 0xf0) == 16)
        && !(b[0] == 192 && b[1] == 168)
        && !(b[0] == 192 && b[1] == 0 && (b[2] == 0 || b[2] == 2))
        && !(b[0] == 192 && b[1] == 88 && b[2] == 99)
        && !(b[0] == 198 && (b[1] == 18 || b[1] == 19))
        && !(b[0] == 198 && b[1] == 51 && b[2] == 100)
        && !(b[0] == 203 && b[1] == 0 && b[2] == 113);
}

bool public_v6(const unsigned char* b) noexcept {
    // Limit to global unicast and conservatively exclude special-use space,
    // documentation, 6to4 and IPv4-mapped addresses.
    if ((b[0] & 0xe0) != 0x20) return false;
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] < 0x02) return false;
    if (b[0] == 0x20 && b[1] == 0x01 && b[2] == 0x0d && b[3] == 0xb8) return false;
    if (b[0] == 0x20 && b[1] == 0x02) return false;
    return true;
}

// RFC 6761 reserves `localhost` and every name under it for loopback. The
// host is canonical (lower case, no trailing dot), as every resource is.
bool localhost_name(std::string_view host) noexcept {
    return host == "localhost" || host.ends_with(".localhost");
}

} // namespace

bool permits_resolved_address(const sockaddr* address, socklen_t length,
                              std::string_view approved_host) noexcept {
    if (!address || approved_host.empty() || approved_host.size() > 253) return false;
    // inet_pton requires a terminated string; reject allocation failure.
    try {
        const std::string host(approved_host);
        if (address->sa_family == AF_INET && length >= sizeof(sockaddr_in)) {
            const auto* in = reinterpret_cast<const sockaddr_in*>(address);
            const auto* bytes = reinterpret_cast<const unsigned char*>(&in->sin_addr);
            if ((bytes[0] == 169 && bytes[1] == 254 && bytes[2] == 169 && bytes[3] == 254)
                || (bytes[0] == 169 && bytes[1] == 254 && bytes[2] == 170 && bytes[3] == 2)
                || (bytes[0] == 100 && bytes[1] == 100 && bytes[2] == 100 && bytes[3] == 200)
                || (bytes[0] == 168 && bytes[1] == 63 && bytes[2] == 129 && bytes[3] == 16))
                return false;
            if (localhost_name(host)) return bytes[0] == 127;
            if (public_v4(bytes)) return true;
            in_addr literal{};
            return ::inet_pton(AF_INET, host.c_str(), &literal) == 1
                && std::memcmp(&literal, &in->sin_addr, sizeof(literal)) == 0;
        }
        if (address->sa_family == AF_INET6 && length >= sizeof(sockaddr_in6)) {
            const auto* in = reinterpret_cast<const sockaddr_in6*>(address);
            const auto* bytes = reinterpret_cast<const unsigned char*>(&in->sin6_addr);
            if (IN6_IS_ADDR_V4MAPPED(&in->sin6_addr)) return false;
            in6_addr metadata{};
            (void)::inet_pton(AF_INET6, "fd00:ec2::254", &metadata);
            if (std::memcmp(&metadata, &in->sin6_addr, sizeof(metadata)) == 0) return false;
            if (localhost_name(host)) return IN6_IS_ADDR_LOOPBACK(&in->sin6_addr) != 0;
            if (public_v6(bytes)) return true;
            in6_addr literal{};
            return in->sin6_scope_id == 0
                && ::inet_pton(AF_INET6, host.c_str(), &literal) == 1
                && std::memcmp(&literal, &in->sin6_addr, sizeof(literal)) == 0;
        }
    } catch (...) {
        return false;
    }
    return false;
}

} // namespace core::net
