#include <catch2/catch_test_macros.hpp>

#include "core/net/NetworkTraffic.hpp"

TEST_CASE("NetworkTrafficStats accumulates and resets byte counters", "[network][traffic]") {
    auto& stats = core::net::NetworkTrafficStats::get_instance();
    stats.reset();

    stats.record_sent(512);
    stats.record_received(2048);
    stats.record({.bytes_sent = 1536, .bytes_received = 1024});

    const auto snapshot = stats.snapshot();
    CHECK(snapshot.bytes_sent == 2048);
    CHECK(snapshot.bytes_received == 3072);
    CHECK(snapshot.total_bytes() == 5120);

    stats.reset();
    CHECK(stats.snapshot().empty());
}

TEST_CASE("Network traffic byte formatter uses compact binary units", "[network][traffic]") {
    CHECK(core::net::format_bytes(842ULL * 1024ULL) == "842 KB");
    CHECK(core::net::format_bytes(12ULL * 1024ULL * 1024ULL + 410ULL * 1024ULL) == "12.4 MB");
    CHECK(core::net::format_bytes(999) == "999 B");
}

TEST_CASE("Network traffic formatter matches report row value shape", "[network][traffic]") {
    const auto formatted = core::net::format_network_traffic({
        .bytes_sent = 842ULL * 1024ULL,
        .bytes_received = 12ULL * 1024ULL * 1024ULL + 410ULL * 1024ULL,
    });

    CHECK(formatted == "\xe2\x86\x91" "842 KB  \xe2\x86\x93" "12.4 MB  total 13.2 MB");
}
