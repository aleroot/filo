#include <catch2/catch_test_macros.hpp>
#include "core/budget/BudgetTracker.hpp"
#include "core/budget/TokenUsageFormatters.hpp"

using namespace core::budget;
using namespace core::budget::formatters;

namespace {

struct StubTokenFormatter {
    [[nodiscard]] std::string format(std::int64_t value) const {
        return "t" + std::to_string(value);
    }
};

struct StubCostFormatter {
    [[nodiscard]] std::string format(double /*usd*/) const {
        return "stub-cost";
    }

    [[nodiscard]] std::string format_fixed_4(double /*usd*/) const {
        return "stub-fixed-cost";
    }
};

static_assert(TokenCountFormatter<StubTokenFormatter>);
static_assert(UsageCostFormatting<StubCostFormatter>);

} // namespace

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
        CHECK(context_window_for_model("claude-fable-5")         == 1'000'000);
        CHECK(context_window_for_model("claude-sonnet-5")        == 1'000'000);
        CHECK(context_window_for_model("fable")                  == 1'000'000);
        CHECK(context_window_for_model("sonnet")                 == 1'000'000);
        CHECK(context_window_for_model("claude-opus-4-8")        ==   200'000);
        CHECK(context_window_for_model("claude-sonnet-4-6")      == 1'000'000);
        CHECK(context_window_for_model("sonnet[1m]")             == 1'000'000);
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

TEST_CASE("CompactTokenCountFormatter uses readable compact units", "[BudgetTracker]") {
    const CompactTokenCountFormatter formatter;

    CHECK(formatter.format(0) == "0");
    CHECK(formatter.format(999) == "999");
    CHECK(formatter.format(1'000) == "1k");
    CHECK(formatter.format(90'500) == "90.5k");
    CHECK(formatter.format(999'949) == "999.9k");
    CHECK(formatter.format(999'950) == "1M");
    CHECK(formatter.format(1'000'000) == "1M");
    CHECK(formatter.format(9'861'200) == "9.9M");
    CHECK(formatter.format(1'250'000'000) == "1.3B");
}

TEST_CASE("TokenUsageStatusFormatter renders token status lines", "[BudgetTracker]") {
    const TokenUsageStatusFormatter<> formatter;

    CHECK(formatter.format({}) == "");
    CHECK(formatter.format({
        .prompt_tokens = 9'861'200,
        .completion_tokens = 90'500,
    }) == "↑9.9M ↓90.5k");
    CHECK(formatter.format({
        .prompt_tokens = 1'000,
        .completion_tokens = 2'000,
    }, 0.123456) == "↑1k ↓2k  $0.1235");
}

TEST_CASE("TokenUsageStatusFormatter accepts concept-based formatter implementations",
          "[BudgetTracker]") {
    const TokenUsageStatusFormatter<StubTokenFormatter, StubCostFormatter> formatter;

    CHECK(formatter.format({
        .prompt_tokens = 12,
        .completion_tokens = 34,
    }, 5.0) == "↑t12 ↓t34  stub-fixed-cost");
}

TEST_CASE("BudgetTracker status string uses compact token units", "[BudgetTracker]") {
    auto& tracker = BudgetTracker::get_instance();
    tracker.set_session_id("test-compact-status");
    tracker.reset_session();

    tracker.record({.prompt_tokens = 9'861'200, .completion_tokens = 90'500},
                   "kimi-for-coding",
                   false);

    CHECK(tracker.status_string() == "↑9.9M ↓90.5k");

    tracker.reset_session();
    tracker.set_session_id("");
}
