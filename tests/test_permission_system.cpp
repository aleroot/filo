#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/permissions/PermissionSystem.hpp"
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace core::permissions;

TEST_CASE("SessionAllowList basic operations", "[permissions]") {
    SessionAllowList list;
    
    SECTION("empty list contains nothing") {
        REQUIRE_FALSE(list.contains("test_key"));
        REQUIRE_FALSE(list.contains(""));
    }
    
    SECTION("insert and contains work") {
        list.insert("test_key");
        REQUIRE(list.contains("test_key"));
        REQUIRE_FALSE(list.contains("other_key"));
    }
    
    SECTION("clear removes all entries") {
        list.insert("key1");
        list.insert("key2");
        list.clear();
        REQUIRE_FALSE(list.contains("key1"));
        REQUIRE_FALSE(list.contains("key2"));
    }
    
    SECTION("snapshot returns all keys") {
        list.insert("key1");
        list.insert("key2");
        auto snap = list.snapshot();
        REQUIRE(snap.size() == 2);
        // Order may vary due to hash set
        REQUIRE(std::find(snap.begin(), snap.end(), "key1") != snap.end());
        REQUIRE(std::find(snap.begin(), snap.end(), "key2") != snap.end());
    }
    
    SECTION("duplicate inserts are idempotent") {
        list.insert("key1");
        list.insert("key1");
        list.insert("key1");
        REQUIRE(list.contains("key1"));
        auto snap = list.snapshot();
        REQUIRE(snap.size() == 1);
    }
}

TEST_CASE("SessionAllowList thread safety", "[permissions][threading]") {
    SessionAllowList list;
    
    std::vector<std::thread> threads;
    
    // Writers
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&list, i]() {
            for (int j = 0; j < 100; ++j) {
                list.insert(std::format("thread_{}_key_{}", i, j));
            }
        });
    }
    
    // Readers
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&list]() {
            for (int j = 0; j < 1000; ++j) {
                auto snap = list.snapshot();
                // Just verify no crash
                (void)snap.size();
            }
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    // Should have 400 unique keys
    REQUIRE(list.snapshot().size() == 400);
}

TEST_CASE("PermissionQueue basic operations", "[permissions]") {
    PermissionQueue queue;
    
    SECTION("empty queue returns nullopt") {
        REQUIRE_FALSE(queue.dequeue());
        REQUIRE(queue.empty());
    }
    
    SECTION("enqueue and dequeue work") {
        PermissionRequest req;
        req.tool_name = "test_tool";
        req.tool_args = "{}";
        
        auto seq = queue.enqueue(std::move(req));
        REQUIRE(seq > 0);
        REQUIRE_FALSE(queue.empty());
        
        auto entry = queue.dequeue();
        REQUIRE(entry);
        REQUIRE(entry->request.tool_name == "test_tool");
        REQUIRE(entry->sequence_number == seq);
        REQUIRE(queue.empty());
    }
    
    SECTION("FIFO order is maintained") {
        for (int i = 0; i < 5; ++i) {
            PermissionRequest req;
            req.tool_name = std::format("tool_{}", i);
            queue.enqueue(std::move(req));
        }
        
        for (int i = 0; i < 5; ++i) {
            auto entry = queue.dequeue();
            REQUIRE(entry);
            REQUIRE(entry->request.tool_name == std::format("tool_{}", i));
        }
    }
    
    SECTION("clear removes all entries") {
        for (int i = 0; i < 5; ++i) {
            PermissionRequest req;
            queue.enqueue(std::move(req));
        }
        
        queue.clear();
        REQUIRE(queue.empty());
        REQUIRE_FALSE(queue.dequeue());
    }
    
    SECTION("sequence numbers are monotonic") {
        uint64_t prev = 0;
        for (int i = 0; i < 10; ++i) {
            PermissionRequest req;
            auto seq = queue.enqueue(std::move(req));
            REQUIRE(seq > prev);
            prev = seq;
        }
    }
}

TEST_CASE("PermissionQueue thread safety", "[permissions][threading]") {
    PermissionQueue queue;
    std::atomic<int> enqueued{0};
    std::atomic<int> dequeued{0};
    
    std::vector<std::thread> threads;
    
    // Producers
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&queue, &enqueued]() {
            for (int j = 0; j < 100; ++j) {
                PermissionRequest req;
                req.tool_name = "test";
                queue.enqueue(std::move(req));
                ++enqueued;
            }
        });
    }
    
    // Consumer
    threads.emplace_back([&queue, &dequeued]() {
        while (dequeued < 400) {
            if (queue.dequeue()) {
                ++dequeued;
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    });
    
    for (auto& t : threads) {
        t.join();
    }
    
    REQUIRE(enqueued == 400);
    REQUIRE(dequeued == 400);
}

TEST_CASE("PermissionSystem basic flow", "[permissions]") {
    PermissionSystem& sys = PermissionSystem::instance();
    sys.reset();
    
    SECTION("YOLO mode auto-approves everything") {
        sys.set_yolo_mode(true);
        
        bool called = false;
        bool result = sys.check_permission("run_terminal_command", 
                                           R"({"command":"rm -rf /"})",
                                           true,  // yolo
                                           [&called](bool) { called = true; });
        
        // Should return true (immediate approval)
        REQUIRE(result);
        REQUIRE(called);
    }
    
    SECTION("non-destructive tools are auto-approved") {
        sys.set_yolo_mode(false);
        
        bool called = false;
        bool result = sys.check_permission("read_file",
                                           R"({"file_path":"test.txt"})",
                                           false,
                                           [&called](bool r) { called = r; });
        
        // read_file doesn't need permission
        REQUIRE(result);
        REQUIRE(called);
    }
    
    SECTION("destructive tools require async approval") {
        sys.set_yolo_mode(false);
        
        std::atomic<bool> callback_called{false};
        std::atomic<bool> callback_result{false};
        
        bool immediate = sys.check_permission("run_terminal_command",
                                              R"({"command":"rm -rf /"})",
                                              false,
                                              [&callback_called, &callback_result](bool r) {
                                                  callback_result = r;
                                                  callback_called = true;
                                              });
        
        // Should NOT be immediate
        REQUIRE_FALSE(immediate);
        REQUIRE_FALSE(callback_called);
        
        // Process the pending request
        bool new_prompt = sys.process_pending_requests();
        REQUIRE(new_prompt);
        REQUIRE(sys.prompt_active());
        
        // Handle approval
        sys.handle_decision(0);  // 0 = Yes
        
        REQUIRE(callback_called);
        REQUIRE(callback_result);
    }
    
    SECTION("remember choice adds to allow list") {
        sys.set_yolo_mode(false);
        
        // First request - approve and remember
        bool called = false;
        sys.check_permission("run_terminal_command",
                            R"({"command":"ls"})",
                            false,
                            [&called](bool) { called = true; });
        
        sys.process_pending_requests();
        sys.handle_decision(1);  // 1 = Yes+Remember
        
        // Second request with same command should be auto-approved
        bool immediate = sys.check_permission("run_terminal_command",
                                              R"({"command":"ls"})",
                                              false,
                                              [](bool) {});
        
        REQUIRE(immediate);
    }
    
    SECTION("denial invokes callback with false") {
        sys.set_yolo_mode(false);
        
        std::atomic<bool> callback_called{false};
        std::atomic<bool> callback_result{true};  // Start with true to verify it gets set to false
        
        sys.check_permission("run_terminal_command",
                            R"({"command":"rm"})",
                            false,
                            [&callback_called, &callback_result](bool r) {
                                callback_result = r;
                                callback_called = true;
                            });
        
        sys.process_pending_requests();
        sys.handle_decision(3);  // 3 = No
        
        REQUIRE(callback_called);
        REQUIRE_FALSE(callback_result);
    }
    
    SECTION("YOLO option enables YOLO mode") {
        sys.reset();  // Explicit reset at section start
        REQUIRE_FALSE(sys.yolo_mode());  // Verify initial state
        REQUIRE_FALSE(sys.prompt_active());  // Verify no active prompt
        
        bool immediate = sys.check_permission("run_terminal_command",
                            R"({"command":"ls_yolo_test"})",
                            false,
                            [](bool) {});
        REQUIRE_FALSE(immediate);  // Should not be immediate (needs permission)
        
        bool new_prompt = sys.process_pending_requests();
        REQUIRE(new_prompt);  // Verify we got a new prompt
        REQUIRE(sys.prompt_active());  // Verify prompt is active
        
        sys.handle_decision(2);  // 2 = YOLO
        
        // After handle_decision, prompt should be inactive
        REQUIRE_FALSE(sys.prompt_active());
        REQUIRE(sys.yolo_mode());
    }
}

TEST_CASE("PermissionSystem process_pending_requests behavior", "[permissions]") {
    PermissionSystem& sys = PermissionSystem::instance();
    sys.reset();
    
    SECTION("returns false when no pending requests") {
        sys.reset();
        REQUIRE_FALSE(sys.process_pending_requests());
    }
    
    SECTION("returns false when prompt already active") {
        const bool first_immediate = sys.check_permission("run_terminal_command",
                                                          R"({"command":"rm -rf /tmp/filo_perm_probe_1"})",
                                                          false,
                                                          [](bool) {});
        REQUIRE_FALSE(first_immediate);
        
        // First call should succeed
        REQUIRE(sys.process_pending_requests());
        
        // Second call should fail (prompt still active)
        const bool second_immediate = sys.check_permission("run_terminal_command",
                                                           R"({"command":"rm -rf /tmp/filo_perm_probe_2"})",
                                                           false,
                                                           [](bool) {});
        REQUIRE_FALSE(second_immediate);
        
        REQUIRE_FALSE(sys.process_pending_requests());

        // Keep singleton state isolated for subsequent tests.
        sys.reset();
    }
    
    SECTION("multiple requests are processed sequentially") {
        std::vector<bool> results;
        
        for (int i = 0; i < 3; ++i) {
            sys.check_permission("run_terminal_command",
                                std::format(R"({{"command":"cmd{}"}})", i),
                                false,
                                [&results](bool r) { results.push_back(r); });
        }
        
        // Process first request
        REQUIRE(sys.process_pending_requests());
        REQUIRE(sys.current_tool_name() == "run_terminal_command");
        sys.handle_decision(0);  // Approve
        
        REQUIRE(results.size() == 1);
        
        // Process second request
        REQUIRE(sys.process_pending_requests());
        sys.handle_decision(0);
        
        REQUIRE(results.size() == 2);
        
        // Process third request
        REQUIRE(sys.process_pending_requests());
        sys.handle_decision(0);
        
        REQUIRE(results.size() == 3);
        
        // No more requests
        REQUIRE_FALSE(sys.process_pending_requests());
    }
}

TEST_CASE("PermissionSystem cancel and reset", "[permissions]") {
    PermissionSystem& sys = PermissionSystem::instance();
    sys.reset();
    
    SECTION("cancel_current invokes callback with false") {
        std::atomic<bool> called{false};
        std::atomic<bool> result{true};
        
        sys.check_permission("run_terminal_command",
                            R"({"command":"rm -rf /tmp/filo_perm_probe_cancel"})",
                            false,
                            [&called, &result](bool r) {
                                result = r;
                                called = true;
                            });
        
        sys.process_pending_requests();
        sys.cancel_current();
        
        REQUIRE(called);
        REQUIRE_FALSE(result);
        REQUIRE_FALSE(sys.prompt_active());
    }
    
    SECTION("reset clears everything") {
        // Add some state
        sys.set_yolo_mode(true);
        sys.allow_list().insert("test_key");
        
        sys.check_permission("run_terminal_command",
                            R"({"command":"rm -rf /tmp/filo_perm_probe_reset"})",
                            false,
                            [](bool) {});
        sys.process_pending_requests();
        
        // Reset
        sys.reset();
        
        REQUIRE_FALSE(sys.yolo_mode());
        REQUIRE_FALSE(sys.allow_list().contains("test_key"));
        REQUIRE_FALSE(sys.prompt_active());
        REQUIRE_FALSE(sys.process_pending_requests());
    }
}

TEST_CASE("make_allow_key generates correct keys", "[permissions]") {
    SECTION("run_terminal_command includes first word") {
        auto key = make_allow_key("run_terminal_command", 
                                   R"({"command":"ls -la /home"})");
        REQUIRE(key == "run_terminal_command:ls");
    }
    
    SECTION("run_terminal_command without args falls back to tool name") {
        auto key = make_allow_key("run_terminal_command", R"({"working_dir":"/tmp"})");
        REQUIRE(key == "run_terminal_command");
    }
    
    SECTION("other tools use just the name") {
        auto key = make_allow_key("write_file", R"({"file_path":"test.txt"})");
        REQUIRE(key == "write_file");
    }
}

TEST_CASE("make_allow_label generates correct labels", "[permissions]") {
    SECTION("run_terminal_command includes command name") {
        auto label = make_allow_label("run_terminal_command",
                                       R"({"command":"ls -la"})");
        REQUIRE(label == "'ls' commands");
    }
    
    SECTION("write_file returns file modifications") {
        auto label = make_allow_label("write_file", "{}");
        REQUIRE(label == "file modifications");
    }
    
    SECTION("delete_file returns file deletions") {
        auto label = make_allow_label("delete_file", "{}");
        REQUIRE(label == "file deletions");
    }
}

TEST_CASE("PermissionSystem concurrent access", "[permissions][threading]") {
    PermissionSystem& sys = PermissionSystem::instance();
    sys.reset();
    
    constexpr int kProducerThreads = 8;
    constexpr int kRequestsPerThread = 25;
    constexpr int kTotalRequests = kProducerThreads * kRequestsPerThread;

    std::atomic<int> approved{0};
    std::atomic<int> queued{0};
    std::atomic<int> processed{0};
    std::atomic<bool> producers_done{false};
    std::vector<std::thread> producer_threads;
    
    // Multiple threads requesting permission
    for (int i = 0; i < kProducerThreads; ++i) {
        producer_threads.emplace_back([&sys, &approved, &queued]() {
            for (int j = 0; j < kRequestsPerThread; ++j) {
                bool immediate = sys.check_permission("run_terminal_command",
                                                       R"({"command":"rm -rf /tmp/filo_perm_probe_concurrent"})",
                                                       false,
                                                       [&approved](bool) { ++approved; });
                if (!immediate) {
                    ++queued;
                }
            }
        });
    }
    
    // UI thread processes requests until producers are done and queue is drained.
    std::thread ui_thread([&sys, &queued, &processed, &producers_done]() {
        while (true) {
            if (sys.process_pending_requests()) {
                // Simulate user pressing Enter
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                sys.handle_decision(0);  // Approve
                ++processed;
            } else {
                if (producers_done.load(std::memory_order_acquire) &&
                    processed.load(std::memory_order_relaxed) >= queued.load(std::memory_order_relaxed)) {
                    break;
                }
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        }
    });
    
    for (auto& t : producer_threads) {
        t.join();
    }
    producers_done.store(true, std::memory_order_release);
    ui_thread.join();
    
    // All callbacks should have been invoked
    REQUIRE(approved == kTotalRequests);
    REQUIRE(processed == queued);
}
