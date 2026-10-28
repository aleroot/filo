#include <catch2/catch_test_macros.hpp>

#include "core/llm/UsageSnapshotCache.hpp"

#include <atomic>
#include <chrono>
#include <latch>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

using core::llm::UsageSnapshotCache;
using namespace std::chrono_literals;

TEST_CASE("UsageSnapshotCache keeps a reading and retries a failure sooner", "[usage][cache]") {
    UsageSnapshotCache<int> cache(1h, 0s);
    int reads = 0;
    std::optional<int> served;
    const auto fetch = [&] {
        ++reads;
        return served;
    };

    CHECK_FALSE(cache.get("a", fetch));
    served = 7;
    CHECK(cache.get("a", fetch) == 7);
    served = 8;
    CHECK(cache.get("a", fetch) == 7);
    CHECK(reads == 2);

    // Each account has its own reading.
    CHECK(cache.get("b", fetch) == 8);
    CHECK(reads == 3);
}

TEST_CASE("UsageSnapshotCache holds a failure as long as a reading by default", "[usage][cache]") {
    UsageSnapshotCache<int> cache(1h);
    int reads = 0;
    CHECK_FALSE(cache.get("a", [&]() -> std::optional<int> {
        ++reads;
        throw std::runtime_error("offline");
    }));
    CHECK_FALSE(cache.get("a", [&] {
        ++reads;
        return std::optional<int>{1};
    }));
    CHECK(reads == 1);
}

TEST_CASE("UsageSnapshotCache shares one read among concurrent callers", "[usage][cache]") {
    UsageSnapshotCache<int> cache(1h);
    constexpr int kCallers = 8;
    std::atomic<int> reads = 0;
    std::latch start(kCallers);
    std::vector<std::optional<int>> results(kCallers);
    {
        std::vector<std::jthread> callers;
        for (int i = 0; i < kCallers; ++i) {
            callers.emplace_back([&, i] {
                start.arrive_and_wait();
                results[i] = cache.get("a", [&] {
                    ++reads;
                    std::this_thread::sleep_for(20ms);
                    return std::optional<int>{42};
                });
            });
        }
    }
    CHECK(reads == 1);
    for (const auto& result : results) CHECK(result == 42);
}
