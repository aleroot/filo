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
        CHECK(context_window_for_model("kimi-k2.7-code")         == 256'000);
        CHECK(context_window_for_model("kimi-k2.6")              == 256'000);
        CHECK(context_window_for_model("kimi-k2-6")              == 256'000);
        CHECK(context_window_for_model("kimi-k2-5")              == 256'000);
        CHECK(context_window_for_model("kimi-k2.5")              == 256'000);
        CHECK(context_window_for_model("kimi-for-coding")        == 256'000);
        CHECK(context_window_for_model("kimi-k2-0711-preview")   == 128'000);
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

TEST_CASE("BudgetTracker skips synthetic pricing for non-billable providers", "[BudgetTracker]") {
    auto& tracker = BudgetTracker::get_instance();
    tracker.reset_session();

    tracker.record({.prompt_tokens = 1'000'000, .completion_tokens = 1'000'000}, "grok-4", false);

    const auto total = tracker.session_total();
    CHECK(total.prompt_tokens == 1'000'000);
    CHECK(total.completion_tokens == 1'000'000);
    CHECK(tracker.session_cost_usd() == 0.0);
}
