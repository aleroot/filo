#include <catch2/catch_test_macros.hpp>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "core/utils/ConcurrentQueue.hpp"

using namespace core::utils;

TEST_CASE("ConcurrentQueue basic operations", "[concurrent_queue]") {
    ConcurrentQueue<int> queue;
    
    SECTION("push and try_pop work") {
        queue.push(42);
        auto val = queue.try_pop();
        REQUIRE(val.has_value());
        REQUIRE(*val == 42);
    }
    
    SECTION("try_pop on empty returns nullopt") {
        auto val = queue.try_pop();
        REQUIRE_FALSE(val.has_value());
    }
    
    SECTION("FIFO order is maintained") {
        for (int i = 0; i < 5; ++i) {
            queue.push(i);
        }
        for (int i = 0; i < 5; ++i) {
            auto val = queue.try_pop();
            REQUIRE(val.has_value());
            REQUIRE(*val == i);
        }
    }
    
    SECTION("clear removes all items") {
        for (int i = 0; i < 5; ++i) {
            queue.push(i);
        }
        queue.clear();
        REQUIRE(queue.empty());
        auto val = queue.try_pop();
        REQUIRE_FALSE(val.has_value());
    }
}

TEST_CASE("ConcurrentQueue thread safety", "[concurrent_queue][threading]") {
    ConcurrentQueue<int> queue;
    std::atomic<int> pushed{0};
    std::atomic<int> popped{0};
    
    SECTION("single producer single consumer") {
        std::thread producer([&]() {
            for (int i = 0; i < 1000; ++i) {
                queue.push(i);
                ++pushed;
            }
        });
        
        std::thread consumer([&]() {
            while (popped < 1000) {
                if (auto val = queue.try_pop()) {
                    ++popped;
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            }
        });
        
        producer.join();
        consumer.join();
        
        REQUIRE(pushed == 1000);
        REQUIRE(popped == 1000);
    }
    
    SECTION("multiple producers single consumer") {
        std::vector<std::thread> producers;
        for (int t = 0; t < 4; ++t) {
            producers.emplace_back([&]() {
                for (int i = 0; i < 250; ++i) {
                    queue.push(i);
                    ++pushed;
                }
            });
        }
        
        std::thread consumer([&]() {
            while (popped < 1000) {
                if (auto val = queue.try_pop()) {
                    ++popped;
                } else {
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                }
            }
        });
        
        for (auto& t : producers) {
            t.join();
        }
        consumer.join();
        
        REQUIRE(pushed == 1000);
        REQUIRE(popped == 1000);
    }
}

TEST_CASE("ConcurrentQueue pop_blocking", "[concurrent_queue]") {
    ConcurrentQueue<int> queue;
    
    SECTION("pop_blocking waits for item") {
        std::atomic<bool> item_received{false};
        
        std::thread consumer([&]() {
            int val = queue.pop_blocking();
            item_received = true;
            REQUIRE(val == 42);
        });
        
        // Give consumer time to start waiting
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        REQUIRE_FALSE(item_received.load());
        
        queue.push(42);
        
        consumer.join();
        REQUIRE(item_received.load());
    }
}
