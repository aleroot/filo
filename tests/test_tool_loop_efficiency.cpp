#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/ToolAccess.hpp"
#include "core/agent/ToolCallDeduplicator.hpp"
#include "core/agent/ToolCallPlanner.hpp"
#include "core/agent/ToolCallScheduler.hpp"
#include "core/llm/Models.hpp"
#include "TestSessionContext.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace {

[[nodiscard]] core::llm::ToolCall make_call(std::string id,
                                            std::string name,
                                            std::string arguments) {
    core::llm::ToolCall call;
    call.id = std::move(id);
    call.function.name = std::move(name);
    call.function.arguments = std::move(arguments);
    return call;
}

[[nodiscard]] core::agent::ScheduledToolTask<int> counted_task(
    core::agent::ToolAccessSet accesses,
    std::atomic<int>& active,
    std::atomic<int>& max_active,
    int value) {
    return {
        .accesses = std::move(accesses),
        .run = [&active, &max_active, value] {
            const int now = active.fetch_add(1, std::memory_order_acq_rel) + 1;
            int observed = max_active.load(std::memory_order_acquire);
            while (observed < now
                   && !max_active.compare_exchange_weak(
                       observed,
                       now,
                       std::memory_order_acq_rel,
                       std::memory_order_acquire)) {
            }
            std::this_thread::sleep_for(30ms);
            active.fetch_sub(1, std::memory_order_acq_rel);
            return value;
        },
    };
}

} // namespace

TEST_CASE("delete_file planner scopes recursive access to file_path", "[agent][tools]") {
    const auto context = test_support::make_session_context(
        {.primary = "/tmp/filo-planner", .enforce = true});
    const auto call = make_call("delete-1", "delete_file",
        R"({"file_path":"tree","recursive":true})");
    const auto planned = core::agent::plan_tool_call(call, context);

    REQUIRE(planned.accesses.size() == 1);
    REQUIRE(planned.accesses.front().kind == core::agent::ToolAccess::Kind::File);
    const auto expected = context.resolve_path("tree").generic_string();
    REQUIRE(planned.accesses.front().file.path == expected);
    REQUIRE(planned.accesses.front().file.recursive);
    REQUIRE(core::agent::tool_accesses_conflict(
        planned.accesses.front(),
        core::agent::ToolAccess::file_access(
            core::agent::ToolFileOperation::Read, expected + "/nested.txt")));
}

TEST_CASE("tool scheduler runs non-conflicting reads concurrently", "[agent][tools]") {
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};

    std::vector<core::agent::ScheduledToolTask<int>> tasks;
    tasks.push_back(counted_task(
        {core::agent::ToolAccess::file_access(
            core::agent::ToolFileOperation::Read,
            "/tmp/filo/a.txt")},
        active,
        max_active,
        1));
    tasks.push_back(counted_task(
        {core::agent::ToolAccess::file_access(
            core::agent::ToolFileOperation::Read,
            "/tmp/filo/b.txt")},
        active,
        max_active,
        2));

    core::agent::ToolCallScheduler<int> scheduler;
    const auto results = scheduler.run(std::move(tasks));

    REQUIRE(results == std::vector<int>{1, 2});
    REQUIRE(max_active.load(std::memory_order_acquire) >= 2);
}

TEST_CASE("tool scheduler reports a finished task before its batch peers", "[agent][tools]") {
    std::promise<void> slow_started;
    const auto slow_started_signal = slow_started.get_future().share();
    std::promise<void> allow_slow_finish;
    const auto allow_slow_finish_signal = allow_slow_finish.get_future().share();
    std::atomic<bool> slow_running{false};
    std::atomic<bool> fast_reported_while_slow_running{false};

    std::vector<core::agent::ScheduledToolTask<int>> tasks;
    tasks.push_back({
        .accesses = {core::agent::ToolAccess::file_access(
            core::agent::ToolFileOperation::Read, "/tmp/filo/slow.txt")},
        .run = [&] {
            slow_running.store(true, std::memory_order_release);
            slow_started.set_value();
            // The callback releases this early. A timeout keeps this test
            // bounded if a future change accidentally reintroduces a batch
            // completion barrier.
            allow_slow_finish_signal.wait_for(250ms);
            slow_running.store(false, std::memory_order_release);
            return 1;
        },
    });
    tasks.push_back({
        .accesses = {core::agent::ToolAccess::file_access(
            core::agent::ToolFileOperation::Read, "/tmp/filo/fast.txt")},
        .run = [&] {
            slow_started_signal.wait();
            return 2;
        },
    });

    core::agent::ToolCallScheduler<int> scheduler;
    const auto results = scheduler.run(
        std::move(tasks),
        [&](std::size_t index, const int&) {
            if (index == 1) {
                fast_reported_while_slow_running.store(
                    slow_running.load(std::memory_order_acquire),
                    std::memory_order_release);
                allow_slow_finish.set_value();
            }
        });

    REQUIRE(results == std::vector<int>{1, 2});
    REQUIRE(fast_reported_while_slow_running.load(std::memory_order_acquire));
}

TEST_CASE("tool scheduler serializes overlapping write conflicts", "[agent][tools]") {
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};

    std::vector<core::agent::ScheduledToolTask<int>> tasks;
    tasks.push_back(counted_task(
        {core::agent::ToolAccess::file_access(
            core::agent::ToolFileOperation::ReadWrite,
            "/tmp/filo/a.txt")},
        active,
        max_active,
        1));
    tasks.push_back(counted_task(
        {core::agent::ToolAccess::file_access(
            core::agent::ToolFileOperation::Read,
            "/tmp/filo/a.txt")},
        active,
        max_active,
        2));

    core::agent::ToolCallScheduler<int> scheduler;
    const auto results = scheduler.run(std::move(tasks));

    REQUIRE(results == std::vector<int>{1, 2});
    REQUIRE(max_active.load(std::memory_order_acquire) == 1);
}

TEST_CASE("tool scheduler overlaps workspace-wide readers", "[agent][tools][parallel]") {
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};

    std::vector<core::agent::ScheduledToolTask<int>> tasks;
    tasks.push_back(counted_task(core::agent::read_all_tool_access(), active, max_active, 1));
    tasks.push_back(counted_task(core::agent::read_all_tool_access(), active, max_active, 2));

    core::agent::ToolCallScheduler<int> scheduler;
    REQUIRE(scheduler.run(std::move(tasks)) == std::vector<int>{1, 2});
    REQUIRE(max_active.load(std::memory_order_acquire) >= 2);
}

TEST_CASE("workspace-wide readers wait behind writes", "[agent][tools][parallel]") {
    std::atomic<int> active{0};
    std::atomic<int> max_active{0};

    std::vector<core::agent::ScheduledToolTask<int>> tasks;
    tasks.push_back(counted_task(
        {core::agent::ToolAccess::file_access(
            core::agent::ToolFileOperation::Write,
            "/tmp/filo/output.txt")},
        active,
        max_active,
        1));
    tasks.push_back(counted_task(core::agent::read_all_tool_access(), active, max_active, 2));

    core::agent::ToolCallScheduler<int> scheduler;
    REQUIRE(scheduler.run(std::move(tasks)) == std::vector<int>{1, 2});
    REQUIRE(max_active.load(std::memory_order_acquire) == 1);
}

TEST_CASE("tool deduplicator reuses same-step identical reads", "[agent][tools][read]") {
    core::agent::ToolCallDeduplicator dedup;
    dedup.begin_step();

    const auto first = dedup.register_call(
        make_call("call-1", "read", R"({"path":"a.cpp"})"));
    const auto second = dedup.register_call(
        make_call("call-2", "read", "{  \"path\" : \"a.cpp\"  }"));

    REQUIRE_FALSE(first.duplicate_in_step);
    REQUIRE(second.duplicate_in_step);
    REQUIRE(second.original_index == first.original_index);

    dedup.complete_original(first.original_index, R"({"content":"hello"})");
    REQUIRE(dedup.duplicate_result(second.original_index) == R"({"content":"hello"})");
}

TEST_CASE("tool deduplicator does not collapse question reads", "[agent][tools][read]") {
    core::agent::ToolCallDeduplicator dedup;
    dedup.begin_step();

    const auto first = dedup.register_call(
        make_call("call-1", "read", R"({"path":"a.cpp","question":"Where is main?"})"));
    const auto second = dedup.register_call(
        make_call("call-2", "read", R"({"path":"a.cpp","question":"Where is main?"})"));

    REQUIRE_FALSE(first.duplicate_in_step);
    REQUIRE_FALSE(second.duplicate_in_step);
}

TEST_CASE("tool deduplicator does not skip side-effecting tools", "[agent][tools]") {
    core::agent::ToolCallDeduplicator dedup;
    dedup.begin_step();

    const auto first = dedup.register_call(
        make_call("call-1", "write_file", R"({"file_path":"a.cpp","content":"hello"})"));
    const auto second = dedup.register_call(
        make_call("call-2", "write_file", R"({"file_path":"a.cpp","content":"hello"})"));

    REQUIRE_FALSE(first.duplicate_in_step);
    REQUIRE_FALSE(second.duplicate_in_step);
    REQUIRE(second.original_index != first.original_index);
}

TEST_CASE("tool deduplicator escalates repeated cross-step calls", "[agent][tools]") {
    core::agent::ToolCallDeduplicator dedup;
    const auto call = make_call("call", "grep_search", R"({"pattern":"needle","path":"src"})");

    for (int i = 0; i < 2; ++i) {
        dedup.begin_step();
        const auto decision = dedup.register_call(call);
        auto finalized = dedup.finalize_for_model(decision.original_index, R"({"matches":[]})");
        REQUIRE_FALSE(finalized.stop_turn);
        REQUIRE_THAT(finalized.result, !Catch::Matchers::ContainsSubstring("system-reminder"));
        dedup.end_step();
    }

    dedup.begin_step();
    const auto decision = dedup.register_call(call);
    auto finalized = dedup.finalize_for_model(decision.original_index, R"({"matches":[]})");
    REQUIRE_FALSE(finalized.stop_turn);
    REQUIRE_THAT(finalized.result, Catch::Matchers::ContainsSubstring("system-reminder"));
}

TEST_CASE("tool deduplicator force-stops only after repeated read-only calls", "[agent][tools]") {
    core::agent::ToolCallDeduplicator dedup;
    const auto call = make_call("call", "read", R"({"path":"src/main.cpp"})");

    for (int i = 0; i < 11; ++i) {
        dedup.begin_step();
        const auto decision = dedup.register_call(call);
        auto finalized = dedup.finalize_for_model(decision.original_index, R"({"content":"same"})");
        REQUIRE_FALSE(finalized.stop_turn);
        dedup.end_step();
    }

    dedup.begin_step();
    const auto decision = dedup.register_call(call);
    auto finalized = dedup.finalize_for_model(decision.original_index, R"({"content":"same"})");
    REQUIRE(finalized.stop_turn);
    REQUIRE_THAT(finalized.result, Catch::Matchers::ContainsSubstring("Stop calling tools"));
}
