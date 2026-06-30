#pragma once

#include <atomic>
#include <cstdint>
#include <string>

namespace core::net {

struct NetworkTraffic {
    uint64_t bytes_sent = 0;
    uint64_t bytes_received = 0;

    [[nodiscard]] constexpr uint64_t total_bytes() const noexcept {
        return bytes_sent + bytes_received;
    }

    [[nodiscard]] constexpr bool empty() const noexcept {
        return bytes_sent == 0 && bytes_received == 0;
    }
};

class NetworkTrafficStats {
public:
    static NetworkTrafficStats& get_instance() noexcept;

    void record_sent(uint64_t bytes) noexcept;
    void record_received(uint64_t bytes) noexcept;
    void record(NetworkTraffic traffic) noexcept;
    void reset() noexcept;

    [[nodiscard]] NetworkTraffic snapshot() const noexcept;

private:
    NetworkTrafficStats() = default;

    uint64_t bytes_sent() const noexcept;
    uint64_t bytes_received() const noexcept;

    std::atomic<uint64_t> bytes_sent_{0};
    std::atomic<uint64_t> bytes_received_{0};
};

[[nodiscard]] std::string format_bytes(uint64_t bytes);
[[nodiscard]] std::string format_network_traffic(NetworkTraffic traffic);
[[nodiscard]] uint64_t estimated_http_header_bytes(auto&& headers) {
    uint64_t total = 0;
    for (const auto& [name, value] : headers) {
        total += static_cast<uint64_t>(name.size() + value.size() + 4); // ": " + CRLF
    }
    return total == 0 ? 0 : total + 2; // terminal CRLF
}

} // namespace core::net
