#include <catch2/catch_test_macros.hpp>
#include "core/budget/BudgetTracker.hpp"

using namespace core::budget;

// ---------------------------------------------------------------------------
// context_window_for_model
// ---------------------------------------------------------------------------
TEST_CASE("context_window_for_model returns correct sizes", "[BudgetTracker]") {
    SECTION("xAI Grok models") {
        CHECK(context_window_for_model("grok-code-fast-1")       ==   131'072);
        CHECK(context_window_for_model("grok-4.1-fast")          == 2'097'152);
        CHECK(context_window_for_model("grok-4.1-mini")          == 2'097'152);
        CHECK(context_window_for_model("grok-4-fast")            ==   256'000);
        CHECK(context_window_for_model("grok-4")                 ==   256'000);
        CHECK(context_window_for_model("grok-3-mini")            ==   131'072);
        CHECK(context_window_for_model("grok-3")                 ==   131'072);
        CHECK(context_window_for_model("grok-2")                 ==   131'072);
    }

    SECTION("Anthropic Claude models") {
        // Claude 4.6 models support 1M token context (beta)
        CHECK(context_window_for_model("claude-opus-4-6")        == 1'000'000);
        CHECK(context_window_for_model("claude-sonnet-4-6")      == 1'000'000);
        CHECK(context_window_for_model("sonnet[1m]")             == 1'000'000);
        CHECK(context_window_for_model("claude-haiku-4-5[1M]")   == 1'000'000);
        // Claude 4.5 and earlier have 200K context
        CHECK(context_window_for_model("claude-opus-4-5")        ==   200'000);
        CHECK(context_window_for_model("claude-sonnet-4-5")      ==   200'000);
        CHECK(context_window_for_model("claude-haiku-4-5")       ==   200'000);
    }

    SECTION("OpenAI models") {
        CHECK(context_window_for_model("gpt-5.4")                == 200'000);
        CHECK(context_window_for_model("gpt-4o")                 == 128'000);
        CHECK(context_window_for_model("gpt-4o-mini")            == 128'000);
        CHECK(context_window_for_model("gpt-4-turbo")            == 128'000);
        CHECK(context_window_for_model("gpt-4")                  ==   8'192);
        CHECK(context_window_for_model("gpt-3.5-turbo")          ==  16'385);
    }

    SECTION("Kimi / Moonshot models") {
        CHECK(context_window_for_model("moonshot-v1-8k")         ==   8'192);
        CHECK(context_window_for_model("moonshot-v1-32k")        ==  32'768);
        CHECK(context_window_for_model("moonshot-v1-128k")       == 128'000);
        CHECK(context_window_for_model("kimi-k2-5")              == 256'000);
        CHECK(context_window_for_model("kimi-k2.5")              == 256'000);
        CHECK(context_window_for_model("kimi-for-coding")        == 256'000);
        CHECK(context_window_for_model("kimi-unknown")           == 128'000);
    }

    SECTION("Google Gemini models") {
        CHECK(context_window_for_model("gemini-3.1-pro-preview") == 1'048'576);
        CHECK(context_window_for_model("gemini-3-flash-preview") == 1'048'576);
        CHECK(context_window_for_model("gemini-2.5-flash")       == 1'048'576);
        CHECK(context_window_for_model("gemini-2.5-flash-lite")  == 1'048'576);
        CHECK(context_window_for_model("gemini-2.0-flash")       == 1'048'576);
        CHECK(context_window_for_model("gemini-1.5-pro")         == 2'097'152);
        CHECK(context_window_for_model("gemini-1.5-flash")       == 2'097'152);
        CHECK(context_window_for_model("gemini-flash-latest")    == 1'048'576);
        CHECK(context_window_for_model("auto-gemini-3")          == 1'048'576);
    }

    SECTION("Unknown model falls back to 128K") {
        CHECK(context_window_for_model("unknown-model-xyz")      == 128'000);
        CHECK(context_window_for_model("")                       == 128'000);
    }
}

// ---------------------------------------------------------------------------
// BudgetTracker::context_remaining_pct
// ---------------------------------------------------------------------------
TEST_CASE("BudgetTracker context_remaining_pct", "[BudgetTracker]") {
    auto& tracker = BudgetTracker::get_instance();
    tracker.reset_session();

    SECTION("returns -1 when no usage recorded yet") {
        CHECK(tracker.context_remaining_pct("grok-3") == -1);
    }

    SECTION("returns correct remaining % for grok-3 (131072 ctx window)") {
        core::llm::TokenUsage usage;
        usage.prompt_tokens     = 65'536;   // exactly 50% of 131072
        usage.completion_tokens = 1'000;
        tracker.record(usage, "grok-3");

        const int32_t pct = tracker.context_remaining_pct("grok-3");
        CHECK(pct == 50);
    }

    SECTION("clamps to 0 when prompt exceeds context window") {
        core::llm::TokenUsage usage;
        usage.prompt_tokens     = 300'000;  // exceeds grok-4's 256K window
        usage.completion_tokens = 100;
        tracker.record(usage, "grok-4");

        CHECK(tracker.context_remaining_pct("grok-4") == 0);
    }

    SECTION("75% remaining when 25% used on claude-4-5 (200K)") {
        core::llm::TokenUsage usage;
        usage.prompt_tokens     = 50'000;   // 25% of 200K
        usage.completion_tokens = 500;
        tracker.record(usage, "claude-sonnet-4-5");

        const int32_t pct = tracker.context_remaining_pct("claude-sonnet-4-5");
        CHECK(pct == 75);
    }
    
    SECTION("95% remaining when 5% used on claude-4-6 (1M)") {
        core::llm::TokenUsage usage;
        usage.prompt_tokens     = 50'000;   // 5% of 1M
        usage.completion_tokens = 500;
        tracker.record(usage, "claude-sonnet-4-6");

        const int32_t pct = tracker.context_remaining_pct("claude-sonnet-4-6");
        CHECK(pct == 95);
    }

    SECTION("95% remaining when 5% used on [1m] model suffix") {
        core::llm::TokenUsage usage;
        usage.prompt_tokens     = 50'000;   // 5% of 1M
        usage.completion_tokens = 500;
        tracker.record(usage, "sonnet[1m]");

        const int32_t pct = tracker.context_remaining_pct("sonnet[1m]");
        CHECK(pct == 95);
    }

    SECTION("50% remaining for moonshot-v1-8k when half the context is used") {
        core::llm::TokenUsage usage;
        usage.prompt_tokens = 4'096;
        usage.completion_tokens = 128;
        tracker.record(usage, "moonshot-v1-8k");

        const int32_t pct = tracker.context_remaining_pct("moonshot-v1-8k");
        CHECK(pct == 50);
    }
}

TEST_CASE("BudgetTracker skips synthetic pricing for non-billable providers", "[BudgetTracker]") {
    auto& tracker = BudgetTracker::get_instance();
    tracker.reset_session();

    tracker.record({.prompt_tokens = 1'000'000, .completion_tokens = 1'000'000}, "grok-4", false);

    const auto total = tracker.session_total();
    CHECK(total.prompt_tokens == 1'000'000);
    CHECK(total.completion_tokens == 1'000'000);
    CHECK(tracker.session_cost_usd() == 0.0);
}
