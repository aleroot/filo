#include "NetworkTraffic.hpp"

#include <algorithm>
#include <array>
#include <format>

namespace core::net {

namespace {

constexpr std::array kUnits{"B", "KB", "MB", "GB", "TB"};

void add_saturating(std::atomic<uint64_t>& target, uint64_t value) noexcept {
    if (value == 0) return;

    uint64_t current = target.load(std::memory_order_relaxed);
    while (true) {
        const uint64_t available = UINT64_MAX - current;
        const uint64_t next = current + std::min(value, available);
        if (target.compare_exchange_weak(
                current, next, std::memory_order_relaxed, std::memory_order_relaxed)) {
            return;
        }
    }
}

} // namespace

NetworkTrafficStats& NetworkTrafficStats::get_instance() noexcept {
    static NetworkTrafficStats instance;
    return instance;
}

void NetworkTrafficStats::record_sent(uint64_t bytes) noexcept {
    add_saturating(bytes_sent_, bytes);
}

void NetworkTrafficStats::record_received(uint64_t bytes) noexcept {
    add_saturating(bytes_received_, bytes);
}

void NetworkTrafficStats::record(NetworkTraffic traffic) noexcept {
    record_sent(traffic.bytes_sent);
    record_received(traffic.bytes_received);
}

void NetworkTrafficStats::reset() noexcept {
    bytes_sent_.store(0, std::memory_order_release);
    bytes_received_.store(0, std::memory_order_release);
}

NetworkTraffic NetworkTrafficStats::snapshot() const noexcept {
    return {
        .bytes_sent = bytes_sent(),
        .bytes_received = bytes_received(),
    };
}

uint64_t NetworkTrafficStats::bytes_sent() const noexcept {
    return bytes_sent_.load(std::memory_order_acquire);
}

uint64_t NetworkTrafficStats::bytes_received() const noexcept {
    return bytes_received_.load(std::memory_order_acquire);
}

std::string format_bytes(uint64_t bytes) {
    double value = static_cast<double>(bytes);
    std::size_t unit = 0;
    while (value >= 1024.0 && unit + 1 < kUnits.size()) {
        value /= 1024.0;
        ++unit;
    }

    if (unit == 0 || value >= 100.0) {
        return std::format("{:.0f} {}", value, kUnits[unit]);
    }
    return std::format("{:.1f} {}", value, kUnits[unit]);
}

std::string format_network_traffic(NetworkTraffic traffic) {
    return std::format(
        "\xe2\x86\x91{}  \xe2\x86\x93{}  total {}",
        format_bytes(traffic.bytes_sent),
        format_bytes(traffic.bytes_received),
        format_bytes(traffic.total_bytes()));
}

} // namespace core::net
