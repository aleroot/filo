#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "core/budget/BudgetTracker.hpp"
#include "core/budget/TokenLedger.hpp"

#include <string>
#include <thread>
#include <vector>

using namespace core::budget;

TEST_CASE("TokenLedger records actual model usage with cost attribution",
          "[budget][token_ledger]") {
    auto& ledger = TokenLedger::get_instance();
    ledger.reset();

    const uint64_t sequence = ledger.record({
        .kind = TokenLedgerEventKind::Actual,
        .source = TokenLedgerSource::ModelCall,
        .session_id = "session-a",
        .actor = "agent",
        .model = "grok-code-fast-1",
        .usage = {.prompt_tokens = 1'000'000, .completion_tokens = 1'000'000},
    });

    REQUIRE(sequence == 1);

    const auto snapshot = ledger.snapshot();
    REQUIRE(snapshot.event_count == 1);
    CHECK(snapshot.prompt_tokens == 1'000'000);
    CHECK(snapshot.completion_tokens == 1'000'000);
    CHECK(snapshot.total_tokens == 2'000'000);
    CHECK(snapshot.cost_usd() == Catch::Approx(1.70));

    REQUIRE(snapshot.by_model.size() == 1);
    CHECK(snapshot.by_model[0].key == "grok-code-fast-1");
    CHECK(snapshot.by_model[0].cost_usd() == Catch::Approx(1.70));

    REQUIRE(snapshot.by_source.size() == 1);
    CHECK(snapshot.by_source[0].key == "model_call");
}

TEST_CASE("TokenLedger supports source and actor breakdowns",
          "[budget][token_ledger]") {
    auto& ledger = TokenLedger::get_instance();
    ledger.reset();

    ledger.record({
        .source = TokenLedgerSource::ModelCall,
        .session_id = "session-a",
        .actor = "agent",
        .model = "gpt-5.4",
        .usage = {.prompt_tokens = 100, .completion_tokens = 20},
    });
    ledger.record({
        .kind = TokenLedgerEventKind::Estimate,
        .source = TokenLedgerSource::ToolPayload,
        .session_id = "session-a",
        .actor = "agent",
        .tool_name = "read_file",
        .usage = {.prompt_tokens = 4, .completion_tokens = 80},
        .should_estimate_cost = false,
        .billable = false,
    });
    ledger.record({
        .source = TokenLedgerSource::Subagent,
        .session_id = "session-a",
        .actor = "subagent:explore",
        .model = "gemini-2.5-flash",
        .usage = {.prompt_tokens = 200, .completion_tokens = 50},
    });

    const auto all = ledger.snapshot({.session_id = "session-a"});
    CHECK(all.event_count == 3);
    CHECK(all.total_tokens == 454);
    REQUIRE(all.by_actor.size() == 2);
    CHECK(all.by_actor[0].key == "subagent:explore");
    CHECK(all.by_actor[1].key == "agent");

    const auto tools = ledger.snapshot({
        .session_id = "session-a",
        .source = TokenLedgerSource::ToolPayload,
    });
    REQUIRE(tools.event_count == 1);
    CHECK(tools.prompt_tokens == 4);
    CHECK(tools.completion_tokens == 80);
    CHECK(tools.cost_usd() == 0.0);
}

TEST_CASE("TokenLedger recent events are newest first and filterable",
          "[budget][token_ledger]") {
    auto& ledger = TokenLedger::get_instance();
    ledger.reset();

    ledger.record({
        .session_id = "a",
        .actor = "agent",
        .model = "gpt-5.4",
        .usage = {.prompt_tokens = 1},
    });
    ledger.record({
        .session_id = "b",
        .actor = "agent",
        .model = "gpt-5.4",
        .usage = {.prompt_tokens = 2},
    });
    ledger.record({
        .session_id = "a",
        .actor = "agent",
        .model = "gpt-5.4",
        .usage = {.prompt_tokens = 3},
    });

    const auto recent = ledger.recent_events(2, {.session_id = "a"});
    REQUIRE(recent.size() == 2);
    CHECK(recent[0].usage.prompt_tokens == 3);
    CHECK(recent[1].usage.prompt_tokens == 1);

    ledger.reset_session("a");
    const auto after_reset = ledger.snapshot();
    REQUIRE(after_reset.event_count == 1);
    CHECK(after_reset.prompt_tokens == 2);
}

TEST_CASE("TokenLedger is safe for concurrent recorders",
          "[budget][token_ledger]") {
    auto& ledger = TokenLedger::get_instance();
    ledger.reset();

    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&ledger, t]() {
            for (int i = 0; i < 100; ++i) {
                ledger.record({
                    .session_id = "concurrent",
                    .actor = "worker-" + std::to_string(t),
                    .model = "local",
                    .usage = {.prompt_tokens = 2, .completion_tokens = 1},
                    .should_estimate_cost = false,
                    .billable = false,
                });
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    const auto snapshot = ledger.snapshot({.session_id = "concurrent"});
    CHECK(snapshot.event_count == 400);
    CHECK(snapshot.prompt_tokens == 800);
    CHECK(snapshot.completion_tokens == 400);
    CHECK(snapshot.total_tokens == 1200);
}

TEST_CASE("BudgetTracker records through TokenLedger without duplicate counters",
          "[budget][token_ledger]") {
    auto& ledger = TokenLedger::get_instance();
    auto& tracker = BudgetTracker::get_instance();
    tracker.reset_session();

    tracker.record(
        {.prompt_tokens = 120, .completion_tokens = 30},
        BudgetRecordContext{
            .session_id = "session-budget",
            .actor = "agent",
            .model = "gpt-5.4",
            .source = TokenLedgerSource::ModelCall,
        });

    const auto total = tracker.session_total();
    CHECK(total.prompt_tokens == 120);
    CHECK(total.completion_tokens == 30);
    CHECK(total.total_tokens == 150);

    const auto snapshot = ledger.snapshot({
        .session_id = "session-budget",
        .kind = TokenLedgerEventKind::Actual,
    });
    REQUIRE(snapshot.event_count == 1);
    CHECK(snapshot.prompt_tokens == total.prompt_tokens);
    CHECK(snapshot.completion_tokens == total.completion_tokens);

    tracker.set_session_id({});
    ledger.reset();
}

TEST_CASE("BudgetTracker presentation scope excludes gateway usage from active sessions",
          "[budget][token_ledger]") {
    auto& ledger = TokenLedger::get_instance();
    auto& tracker = BudgetTracker::get_instance();
    ledger.reset();
    tracker.set_session_id("interactive-session");

    ledger.record({
        .kind = TokenLedgerEventKind::Actual,
        .source = TokenLedgerSource::Gateway,
        .actor = "api_gateway",
        .model = "gpt-5.4",
        .usage = {.prompt_tokens = 1000, .completion_tokens = 500},
    });
    ledger.record({
        .kind = TokenLedgerEventKind::Actual,
        .source = TokenLedgerSource::ModelCall,
        .session_id = "interactive-session",
        .actor = "agent",
        .model = "gpt-5.4",
        .usage = {.prompt_tokens = 120, .completion_tokens = 30},
    });

    const auto total = tracker.session_total();
    CHECK(total.prompt_tokens == 120);
    CHECK(total.completion_tokens == 30);
    CHECK(total.total_tokens == 150);

    tracker.reset_session();
    const auto gateway = ledger.snapshot({
        .source = TokenLedgerSource::Gateway,
        .kind = TokenLedgerEventKind::Actual,
    });
    CHECK(gateway.event_count == 1);
    CHECK(gateway.prompt_tokens == 1000);

    tracker.set_session_id({});
    ledger.reset();
}
