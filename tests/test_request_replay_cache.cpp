#include <catch2/catch_test_macros.hpp>

#include "exec/RequestReplayCache.hpp"

#include <chrono>
#include <future>
#include <string>
#include <string_view>
#include <thread>

using exec::daemon::detail::McpDispatchResult;
using exec::daemon::detail::RequestReplayCache;

namespace {

[[nodiscard]] std::string replay_key(std::string_view session_id, std::string_view suffix) {
    return RequestReplayCache::session_token(session_id) + "|" + std::string(suffix);
}

} // namespace

TEST_CASE("RequestReplayCache first access executes then replays", "[daemon][cache]") {
    RequestReplayCache cache;

    auto first = cache.begin("key-a");
    REQUIRE(first.kind == RequestReplayCache::DecisionKind::execute);
    REQUIRE(first.inflight != nullptr);

    const McpDispatchResult completed{
        .status = 200,
        .body = R"({"ok":true})",
    };
    cache.finish("key-a", first.inflight, completed, true);

    auto second = cache.begin("key-a");
    REQUIRE(second.kind == RequestReplayCache::DecisionKind::replay_completed);
    REQUIRE(second.replay_result.status == 200);
    REQUIRE(second.replay_result.body == R"({"ok":true})");
}

TEST_CASE("RequestReplayCache waiters unblock with producer result", "[daemon][cache]") {
    RequestReplayCache cache;
    auto first = cache.begin("key-b");
    REQUIRE(first.kind == RequestReplayCache::DecisionKind::execute);

    auto duplicate = cache.begin("key-b");
    REQUIRE(duplicate.kind == RequestReplayCache::DecisionKind::wait_inflight);
    REQUIRE(duplicate.inflight == first.inflight);

    auto waiter = std::async(std::launch::async, [&]() {
        return cache.wait_for(duplicate.inflight);
    });

    const McpDispatchResult completed{
        .status = 202,
        .body = "",
    };
    cache.finish("key-b", first.inflight, completed, true);

    auto replayed = waiter.get();
    REQUIRE(replayed.status == 202);
    REQUIRE(replayed.body.empty());
}

TEST_CASE("RequestReplayCache respects non-cacheable completions", "[daemon][cache]") {
    RequestReplayCache cache;
    auto first = cache.begin("key-c");
    REQUIRE(first.kind == RequestReplayCache::DecisionKind::execute);

    const McpDispatchResult failed{
        .status = 500,
        .body = R"({"error":"boom"})",
    };
    cache.finish("key-c", first.inflight, failed, false);

    auto second = cache.begin("key-c");
    REQUIRE(second.kind == RequestReplayCache::DecisionKind::execute);
}

TEST_CASE("RequestReplayCache expires old entries by TTL", "[daemon][cache]") {
    RequestReplayCache cache(std::chrono::milliseconds{5}, 16);
    auto first = cache.begin("key-ttl");
    REQUIRE(first.kind == RequestReplayCache::DecisionKind::execute);

    cache.finish("key-ttl", first.inflight, McpDispatchResult{.status = 200, .body = "{}"}, true);
    std::this_thread::sleep_for(std::chrono::milliseconds{15});

    auto second = cache.begin("key-ttl");
    REQUIRE(second.kind == RequestReplayCache::DecisionKind::execute);
}

TEST_CASE("RequestReplayCache bounds memory by max entries", "[daemon][cache]") {
    RequestReplayCache cache(std::chrono::minutes{5}, 2);

    auto a = cache.begin("a");
    cache.finish("a", a.inflight, McpDispatchResult{.status = 200, .body = "a"}, true);
    auto b = cache.begin("b");
    cache.finish("b", b.inflight, McpDispatchResult{.status = 200, .body = "b"}, true);
    auto c = cache.begin("c");
    cache.finish("c", c.inflight, McpDispatchResult{.status = 200, .body = "c"}, true);

    // Oldest entry should be evicted, newer entries remain replayable.
    auto again_a = cache.begin("a");
    REQUIRE(again_a.kind == RequestReplayCache::DecisionKind::execute);

    auto again_b = cache.begin("b");
    REQUIRE(again_b.kind == RequestReplayCache::DecisionKind::replay_completed);
    REQUIRE(again_b.replay_result.body == "b");

    auto again_c = cache.begin("c");
    REQUIRE(again_c.kind == RequestReplayCache::DecisionKind::replay_completed);
    REQUIRE(again_c.replay_result.body == "c");
}

TEST_CASE("RequestReplayCache disables completed storage when max entries is zero",
          "[daemon][cache]") {
    RequestReplayCache cache(std::chrono::minutes{5}, 0);
    auto first = cache.begin("no-store");
    REQUIRE(first.kind == RequestReplayCache::DecisionKind::execute);

    cache.finish("no-store",
                 first.inflight,
                 McpDispatchResult{.status = 200, .body = "ok"},
                 true);

    auto second = cache.begin("no-store");
    REQUIRE(second.kind == RequestReplayCache::DecisionKind::execute);
}

TEST_CASE("RequestReplayCache clear_session removes cached entries and unblocks waiters",
          "[daemon][cache]") {
    RequestReplayCache cache(std::chrono::minutes{5}, 16);
    const std::string session_a_done = replay_key("session-a", "id-1|h1");
    const std::string session_b_done = replay_key("session-b", "id-1|h1");
    const std::string session_a_inflight = replay_key("session-a", "id-2|h2");

    auto done_a = cache.begin(session_a_done);
    cache.finish(session_a_done,
                 done_a.inflight,
                 McpDispatchResult{.status = 200, .body = "a"},
                 true);

    auto done_b = cache.begin(session_b_done);
    cache.finish(session_b_done,
                 done_b.inflight,
                 McpDispatchResult{.status = 200, .body = "b"},
                 true);

    auto inflight_a = cache.begin(session_a_inflight);
    REQUIRE(inflight_a.kind == RequestReplayCache::DecisionKind::execute);
    auto duplicate_a = cache.begin(session_a_inflight);
    REQUIRE(duplicate_a.kind == RequestReplayCache::DecisionKind::wait_inflight);

    auto waiter = std::async(std::launch::async, [&]() {
        return cache.wait_for(duplicate_a.inflight);
    });

    cache.clear_session("session-a");

    const auto cancelled = waiter.get();
    REQUIRE(cancelled.status == 499);
    REQUIRE(cancelled.body.find("session was closed") != std::string::npos);

    auto replay_a_after_clear = cache.begin(session_a_done);
    REQUIRE(replay_a_after_clear.kind == RequestReplayCache::DecisionKind::execute);

    auto replay_b_still_present = cache.begin(session_b_done);
    REQUIRE(replay_b_still_present.kind
            == RequestReplayCache::DecisionKind::replay_completed);
    REQUIRE(replay_b_still_present.replay_result.body == "b");
}

TEST_CASE("RequestReplayCache does not repopulate cache from stale in-flight completion",
          "[daemon][cache]") {
    RequestReplayCache cache(std::chrono::minutes{5}, 16);
    const std::string key = replay_key("session-stale", "id-1|h1");

    auto first = cache.begin(key);
    REQUIRE(first.kind == RequestReplayCache::DecisionKind::execute);
    cache.clear_session("session-stale");

    cache.finish(key,
                 first.inflight,
                 McpDispatchResult{.status = 200, .body = "stale-result"},
                 true);

    auto after_stale_finish = cache.begin(key);
    REQUIRE(after_stale_finish.kind == RequestReplayCache::DecisionKind::execute);

    cache.finish(key,
                 after_stale_finish.inflight,
                 McpDispatchResult{.status = 200, .body = "fresh-result"},
                 true);
    auto replay = cache.begin(key);
    REQUIRE(replay.kind == RequestReplayCache::DecisionKind::replay_completed);
    REQUIRE(replay.replay_result.body == "fresh-result");
}

TEST_CASE("RequestReplayCache clears sessions whose ids contain replay delimiters",
          "[daemon][cache]") {
    RequestReplayCache cache(std::chrono::minutes{5}, 16);
    const std::string session_id = "session|with|pipe";
    const std::string key =
        RequestReplayCache::session_token(session_id) + "|id-1|h1";

    auto stale = cache.begin(key);
    REQUIRE(stale.kind == RequestReplayCache::DecisionKind::execute);

    cache.clear_session(session_id);

    cache.finish(key,
                 stale.inflight,
                 McpDispatchResult{.status = 200, .body = "stale-result"},
                 true);

    auto after_clear = cache.begin(key);
    REQUIRE(after_clear.kind == RequestReplayCache::DecisionKind::execute);

    cache.finish(key,
                 after_clear.inflight,
                 McpDispatchResult{.status = 200, .body = "fresh-result"},
                 true);

    auto replay = cache.begin(key);
    REQUIRE(replay.kind == RequestReplayCache::DecisionKind::replay_completed);
    REQUIRE(replay.replay_result.body == "fresh-result");
}

TEST_CASE("RequestReplayCache stale finish does not erase newer in-flight entry",
          "[daemon][cache]") {
    RequestReplayCache cache(std::chrono::minutes{5}, 16);
    const std::string key = replay_key("session-stale-erase", "id-1|h1");

    auto stale = cache.begin(key);
    REQUIRE(stale.kind == RequestReplayCache::DecisionKind::execute);

    cache.clear_session("session-stale-erase");

    auto fresh = cache.begin(key);
    REQUIRE(fresh.kind == RequestReplayCache::DecisionKind::execute);
    REQUIRE(fresh.inflight != stale.inflight);

    cache.finish(key,
                 stale.inflight,
                 McpDispatchResult{.status = 200, .body = "stale-result"},
                 true);

    auto duplicate = cache.begin(key);
    REQUIRE(duplicate.kind == RequestReplayCache::DecisionKind::wait_inflight);
    REQUIRE(duplicate.inflight == fresh.inflight);

    cache.finish(key,
                 fresh.inflight,
                 McpDispatchResult{.status = 200, .body = "fresh-result"},
                 true);

    auto replay = cache.begin(key);
    REQUIRE(replay.kind == RequestReplayCache::DecisionKind::replay_completed);
    REQUIRE(replay.replay_result.body == "fresh-result");
}
