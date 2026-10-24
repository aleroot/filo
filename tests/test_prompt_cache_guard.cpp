#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "PromptCacheTestSupport.hpp"
#include "TestSessionContext.hpp"
#include "core/agent/Agent.hpp"
#include "core/agent/PromptCacheGuard.hpp"
#include "core/agent/PromptCacheTracker.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/PromptCacheCalibration.hpp"
#include "core/llm/PromptCacheProfile.hpp"
#include "core/llm/protocols/ClaudeProtocol.hpp"
#include "core/llm/protocols/GeminiProtocol.hpp"
#include "core/llm/protocols/KimiProtocol.hpp"
#include "core/llm/protocols/OllamaProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"
#include "core/llm/protocols/OpenAIUsage.hpp"
#include "core/tools/ToolManager.hpp"

#include <simdjson.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <string>
#include <vector>

using namespace std::chrono_literals;
using core::agent::PromptCacheAssessment;
using core::agent::PromptCacheGuard;
using core::agent::PromptCacheGuardSettings;
using core::agent::PromptCacheState;
using core::agent::PromptCacheWarmth;
using core::llm::PromptCacheBasis;
using core::llm::PromptCacheProfile;

namespace {

using Clock = std::chrono::system_clock;

const Clock::time_point kT0 = Clock::time_point{} + std::chrono::hours{24 * 365 * 50};

PromptCacheState heavy_state(PromptCacheProfile profile = PromptCacheProfile::documented(5min, true)) {
    return PromptCacheState{
        .last_exchange_at = kT0,
        .context_tokens = 486'000,
        .profile = profile,
    };
}

std::size_t count_of(const std::string& haystack, std::string_view needle) {
    std::size_t count = 0;
    for (auto pos = haystack.find(needle); pos != std::string::npos;
         pos = haystack.find(needle, pos + needle.size())) {
        ++count;
    }
    return count;
}

} // namespace

// ── Pure policy ─────────────────────────────────────────────────────────────

TEST_CASE("PromptCacheGuard tracks nothing without a remote cache or an exchange",
          "[prompt-cache][guard]") {
    auto state = heavy_state(PromptCacheProfile::none());
    CHECK(PromptCacheGuard::assess(state, kT0 + 1h).warmth == PromptCacheWarmth::Untracked);

    state = heavy_state();
    state.last_exchange_at.reset();
    CHECK(PromptCacheGuard::assess(state, kT0 + 1h).warmth == PromptCacheWarmth::Untracked);
}

TEST_CASE("PromptCacheGuard ages a cache by wall-clock idle time",
          "[prompt-cache][guard]") {
    const auto state = heavy_state();

    const auto warm = PromptCacheGuard::assess(state, kT0 + 4min);
    CHECK(warm.warmth == PromptCacheWarmth::Warm);
    CHECK(warm.idle == 4min);
    CHECK(warm.remaining == 1min);

    const auto at_expiry = PromptCacheGuard::assess(state, kT0 + 5min);
    CHECK(at_expiry.expired());
    CHECK(at_expiry.remaining == 0s);

    // A clock stepped backwards reads as just used, never as expired.
    const auto rewound = PromptCacheGuard::assess(state, kT0 - 10min);
    CHECK(rewound.warmth == PromptCacheWarmth::Warm);
    CHECK(rewound.idle == 0s);
}

TEST_CASE("PromptCacheGuard asks before a send only over an expired heavy cache",
          "[prompt-cache][guard]") {
    const PromptCacheGuard guard{PromptCacheGuardSettings{.min_context_tokens = 100'000}};
    auto state = heavy_state();

    CHECK(guard.should_confirm_send(PromptCacheGuard::assess(state, kT0 + 2h)));
    CHECK_FALSE(guard.should_confirm_send(PromptCacheGuard::assess(state, kT0 + 1min)));

    state.context_tokens = 99'999;
    CHECK_FALSE(guard.should_confirm_send(PromptCacheGuard::assess(state, kT0 + 2h)));

    const PromptCacheGuard disabled{PromptCacheGuardSettings{.min_context_tokens = 0}};
    CHECK_FALSE(disabled.should_confirm_send(PromptCacheGuard::assess(heavy_state(), kT0 + 2h)));
}

TEST_CASE("PromptCacheGuard places the expiry notice inside the cache lifetime",
          "[prompt-cache][guard]") {
    const PromptCacheGuard guard{PromptCacheGuardSettings{
        .min_context_tokens = 100'000, .notice_lead = 60s}};

    const auto deadlines = guard.deadlines(heavy_state());
    REQUIRE(deadlines.has_value());
    CHECK(deadlines->expiry == kT0 + 5min);
    CHECK(deadlines->notice == kT0 + 4min);

    SECTION("a lead that does not fit the lifetime disables the notice") {
        const PromptCacheGuard long_lead{PromptCacheGuardSettings{
            .min_context_tokens = 100'000, .notice_lead = 5min}};
        CHECK_FALSE(long_lead.deadlines(heavy_state()).has_value());
    }
    SECTION("a zero lead disables the notice") {
        const PromptCacheGuard off{PromptCacheGuardSettings{
            .min_context_tokens = 100'000, .notice_lead = 0s}};
        CHECK_FALSE(off.deadlines(heavy_state()).has_value());
    }
    SECTION("a light session gets no notice") {
        auto light = heavy_state();
        light.context_tokens = 10'000;
        CHECK_FALSE(guard.deadlines(light).has_value());
    }
}

TEST_CASE("Only a replay-renewable cache can be kept warm", "[prompt-cache][guard]") {
    CHECK(PromptCacheGuard::can_keep_warm(PromptCacheProfile::documented(5min, true)));
    CHECK_FALSE(PromptCacheGuard::can_keep_warm(PromptCacheProfile::documented(5min, false)));
    CHECK_FALSE(PromptCacheGuard::can_keep_warm(PromptCacheProfile::estimated(5min)));
    CHECK_FALSE(PromptCacheGuard::can_keep_warm(PromptCacheProfile::none()));
}

TEST_CASE("Automatic keep-warm stops where a cold write would have been cheaper",
          "[prompt-cache][guard]") {
    using core::agent::PromptCacheKeepWarm;
    const auto claude = PromptCacheProfile::documented(5min, true).with_costs(10, 125);
    const auto claude_1h = PromptCacheProfile::documented(1h, true).with_costs(10, 200);
    CHECK(PromptCacheGuard::keep_warm_budget(claude) == 11);
    CHECK(PromptCacheGuard::keep_warm_budget(claude_1h) == 19);
    // No replay, or no prices to weigh it against: never automatic.
    CHECK(PromptCacheGuard::keep_warm_budget(PromptCacheProfile::documented(5min, true)) == 0);
    CHECK(PromptCacheGuard::keep_warm_budget(
              PromptCacheProfile::documented(5min, false).with_costs(10, 125)) == 0);
    CHECK(PromptCacheGuard::keep_warm_budget(
              PromptCacheProfile::estimated(core::llm::kEstimatedPromptCacheTtl)) == 0);

    const PromptCacheGuard ask{PromptCacheGuardSettings{}};
    const PromptCacheGuard automatic{PromptCacheGuardSettings{.keep_warm = PromptCacheKeepWarm::Auto}};
    CHECK_FALSE(ask.should_keep_warm(claude, 0));
    CHECK(automatic.should_keep_warm(claude, 0));
    CHECK(automatic.should_keep_warm(claude, 10));
    CHECK_FALSE(automatic.should_keep_warm(claude, 11));
}

TEST_CASE("Prompt cache guard settings come from the configuration", "[prompt-cache][config]") {
    core::config::AppConfig config;
    const auto defaults = core::agent::prompt_cache_guard_settings(config);
    CHECK(defaults.min_context_tokens == PromptCacheGuardSettings{}.min_context_tokens);
    CHECK(defaults.notice_lead == PromptCacheGuardSettings{}.notice_lead);
    CHECK(defaults.keep_warm == core::agent::PromptCacheKeepWarm::Ask);

    config.prompt_cache_guard_tokens = 0;
    config.prompt_cache_notice_seconds = 120;
    const auto tuned = core::agent::prompt_cache_guard_settings(config);
    CHECK_FALSE(tuned.enabled());
    CHECK(tuned.notice_lead == 120s);

    config.prompt_cache_keep_warm = "auto";
    CHECK(core::agent::prompt_cache_guard_settings(config).keep_warm
          == core::agent::PromptCacheKeepWarm::Auto);
    config.prompt_cache_keep_warm = "sometimes";
    CHECK(core::agent::prompt_cache_guard_settings(config).keep_warm
          == core::agent::PromptCacheKeepWarm::Ask);
}

// ── Calibration ─────────────────────────────────────────────────────────────

TEST_CASE("Calibration extends an estimated lifetime after a hit beyond it",
          "[prompt-cache][calibration]") {
    core::llm::PromptCacheCalibration calibration;
    const auto key = core::llm::PromptCacheCalibration::key("openai", "gpt-5");
    const auto estimated = PromptCacheProfile::estimated(5min);

    calibration.observe(key, 40min, {.prompt_tokens = 200'000, .cached_prompt_tokens = 199'000});
    const auto refined = calibration.refine(key, estimated);
    CHECK(refined.ttl == 40min);
    CHECK(refined.basis == PromptCacheBasis::Observed);

    SECTION("a documented lifetime is the provider's contract and never moves") {
        const auto documented = PromptCacheProfile::documented(5min, true);
        CHECK(calibration.refine(key, documented) == documented);
    }
    SECTION("other providers and models learn nothing from it") {
        const auto other = core::llm::PromptCacheCalibration::key("openai", "gpt-4.1");
        CHECK(calibration.refine(other, estimated) == estimated);
    }
}

TEST_CASE("Calibration ignores misses and tiny prompts", "[prompt-cache][calibration]") {
    core::llm::PromptCacheCalibration calibration;
    const auto key = core::llm::PromptCacheCalibration::key("grok", "grok-4");
    const auto estimated = PromptCacheProfile::estimated(5min);

    // A miss says nothing about the lifetime: providers evict best-effort.
    calibration.observe(key, 2min, {.prompt_tokens = 200'000, .cached_prompt_tokens = 0});
    // Only the system prompt survived: not a hit of the conversation.
    calibration.observe(key, 30min, {.prompt_tokens = 200'000, .cached_prompt_tokens = 4'000});
    // Too small to be cached at all.
    calibration.observe(key, 30min, {.prompt_tokens = 500, .cached_prompt_tokens = 500});

    CHECK(calibration.refine(key, estimated) == estimated);
}

// ── Protocols ───────────────────────────────────────────────────────────────

TEST_CASE("prompt_cache_ttl accepts the documented spellings", "[prompt-cache][config]") {
    using core::llm::PromptCacheTtl;
    CHECK(core::llm::parse_prompt_cache_ttl("") == PromptCacheTtl::Standard);
    CHECK(core::llm::parse_prompt_cache_ttl("5m") == PromptCacheTtl::Standard);
    CHECK(core::llm::parse_prompt_cache_ttl("1H") == PromptCacheTtl::Extended);
    CHECK(core::llm::parse_prompt_cache_ttl("extended") == PromptCacheTtl::Extended);
    CHECK_FALSE(core::llm::parse_prompt_cache_ttl("24h").has_value());
}

TEST_CASE("Each protocol reports the cache it asks for", "[prompt-cache][protocol]") {
    using namespace core::llm::protocols;
    using core::llm::PromptCacheTtl;

    const auto claude = ClaudeProtocol{}.prompt_cache_profile("claude-sonnet-4-6");
    CHECK(claude == PromptCacheProfile::documented(5min, true).with_costs(10, 125));

    const auto claude_1h = ClaudeProtocol{{}, kAnthropicDefaultMaxTokens, PromptCacheTtl::Extended}
        .prompt_cache_profile("claude-sonnet-4-6");
    CHECK(claude_1h == PromptCacheProfile::documented(1h, true).with_costs(10, 200));
    // The lifetime survives the per-request clone.
    const auto cloned = ClaudeProtocol{{}, kAnthropicDefaultMaxTokens, PromptCacheTtl::Extended}.clone();
    CHECK(cloned->prompt_cache_profile("claude-sonnet-4-6") == claude_1h);

    // OpenAI documents no fixed lifetime, and a Responses replay would chain
    // the conversation onto itself through previous_response_id: estimated,
    // and never kept warm by replay.
    const auto openai = PromptCacheProfile::estimated(core::llm::kEstimatedPromptCacheTtl);
    CHECK(OpenAIProtocol{}.prompt_cache_profile("gpt-5") == openai);
    CHECK(OpenAIResponsesProtocol{}.prompt_cache_profile("gpt-5") == openai);
    CHECK(CodexResponsesProtocol{}.prompt_cache_profile("gpt-5-codex") == openai);
    CHECK_FALSE(PromptCacheGuard::can_keep_warm(openai));
    CHECK(PromptCacheGuard::can_keep_warm(claude));

    CHECK_FALSE(OllamaProtocol{}.prompt_cache_profile("llama3").remote());
}

TEST_CASE("The one-hour Claude cache marks every breakpoint with the same lifetime",
          "[prompt-cache][protocol][claude]") {
    using namespace core::llm::protocols;
    core::llm::ChatRequest request;
    request.model = "claude-sonnet-4-6";
    request.messages = {
        {.role = "system", .content = "You are helpful."},
        {.role = "user", .content = "first"},
        {.role = "assistant", .content = "reply"},
        {.role = "user", .content = "second"},
    };
    request.tools.push_back(core::llm::Tool{.function = {.name = "read", .description = "Read"}});

    const auto standard = ClaudeSerializer::serialize(request);
    CHECK(count_of(standard, R"("ttl":"1h")") == 0);
    REQUIRE(count_of(standard, R"("cache_control":{"type":"ephemeral"})") > 0);

    const auto extended = ClaudeSerializer::serialize(
        request, kAnthropicDefaultMaxTokens, {}, core::llm::PromptCacheTtl::Extended);
    CHECK(count_of(extended, R"("cache_control":{"type":"ephemeral"})") == 0);
    CHECK(count_of(extended, R"("cache_control":{"type":"ephemeral","ttl":"1h"})")
          == count_of(standard, R"("cache_control":{"type":"ephemeral"})"));
}

TEST_CASE("Cache hits are read from every usage dialect", "[prompt-cache][usage]") {
    using namespace core::llm::protocols;

    SECTION("Gemini cachedContentTokenCount") {
        const auto usage = extract_gemini_usage_metadata(
            R"({"usageMetadata":{"promptTokenCount":9000,"candidatesTokenCount":10,"cachedContentTokenCount":8000}})");
        CHECK(usage.prompt_tokens == 9000);
        CHECK(usage.cached_prompt_tokens == 8000);
    }

    const auto parse = [](std::string_view json) {
        simdjson::dom::parser parser;
        simdjson::padded_string padded(json);
        simdjson::dom::object usage;
        REQUIRE(parser.parse(padded).get_object().get(usage) == simdjson::SUCCESS);
        ParseResult result;
        REQUIRE(parse_openai_usage(usage, result));
        return result;
    };
    SECTION("DeepSeek prompt_cache_hit_tokens") {
        const auto result = parse(
            R"({"prompt_tokens":5000,"completion_tokens":5,"prompt_cache_hit_tokens":4800,"prompt_cache_miss_tokens":200})");
        CHECK(result.cached_prompt_tokens == 4800);
    }
    SECTION("Moonshot top-level cached_tokens") {
        const auto result = parse(R"({"prompt_tokens":5000,"completion_tokens":5,"cached_tokens":4096})");
        CHECK(result.cached_prompt_tokens == 4096);
    }
    SECTION("Moonshot cache_write_tokens") {
        const auto result = parse(
            R"({"prompt_tokens":5000,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":4000,"cache_write_tokens":900}})");
        CHECK(result.cached_prompt_tokens == 4000);
        CHECK(result.cache_creation_prompt_tokens == 900);
    }
    SECTION("OpenAI details win over vendor fallbacks") {
        const auto result = parse(
            R"({"prompt_tokens":5000,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":3072},"cached_tokens":1})");
        CHECK(result.cached_prompt_tokens == 3072);
    }
}

// ── Tracker ─────────────────────────────────────────────────────────────────

TEST_CASE("PromptCacheTracker follows the cache the conversation last used",
          "[prompt-cache][tracker]") {
    using core::agent::PromptCacheTracker;
    core::llm::PromptCacheCalibration calibration;
    PromptCacheTracker tracker{calibration};
    auto claude = std::make_shared<test_support::CachingProvider>(PromptCacheProfile::documented(5min, true));
    auto request = std::make_shared<const core::llm::ChatRequest>();
    const auto exchange = [&](std::shared_ptr<core::llm::LLMProvider> provider,
                              Clock::time_point at,
                              std::shared_ptr<const core::llm::ChatRequest> sent) {
        return PromptCacheTracker::Exchange{
            .provider = std::move(provider),
            .provider_name = "caching",
            .model = "cache-model",
            .started_at = at,
            .request = std::move(sent),
            .usage = {.prompt_tokens = 120'000, .cached_prompt_tokens = 119'000},
        };
    };

    CHECK_FALSE(tracker.state(claude, "cache-model", 0).last_exchange_at.has_value());
    tracker.record(exchange(claude, kT0, request));

    const auto state = tracker.state(claude, "cache-model", 1'000);
    CHECK(state.last_exchange_at == kT0);
    CHECK(state.context_tokens == 120'000);  // the larger of reported and estimated
    CHECK(tracker.state(claude, "cache-model", 150'000).context_tokens == 150'000);
    CHECK_FALSE(tracker.state(claude, "other-model", 0).last_exchange_at.has_value());

    SECTION("a replay renews only the request it replayed") {
        const auto replay = tracker.replay();
        REQUIRE(replay.has_value());
        CHECK(replay->request == request);
        CHECK(replay->provider == claude);

        tracker.renewed(std::make_shared<const core::llm::ChatRequest>(), kT0 + 2min);
        CHECK(tracker.state(claude, "cache-model", 0).last_exchange_at == kT0);
        tracker.renewed(request, kT0 + 3min);
        CHECK(tracker.state(claude, "cache-model", 0).last_exchange_at == kT0 + 3min);
    }

    SECTION("a provider that a replay does not renew keeps no request to replay") {
        auto openai = std::make_shared<test_support::CachingProvider>(
            PromptCacheProfile::estimated(core::llm::kEstimatedPromptCacheTtl));
        tracker.record(exchange(openai, kT0 + 1min, request));
        CHECK_FALSE(tracker.replay().has_value());
        CHECK(tracker.state(openai, "cache-model", 0).last_exchange_at == kT0 + 1min);
    }

    SECTION("a provider without a billed cache clears the tracking") {
        auto local = std::make_shared<test_support::CachingProvider>(PromptCacheProfile::none());
        tracker.record(exchange(local, kT0 + 1min, request));
        CHECK_FALSE(tracker.state(claude, "cache-model", 0).last_exchange_at.has_value());
        CHECK_FALSE(tracker.replay().has_value());
    }

    SECTION("reset forgets everything") {
        tracker.reset();
        CHECK_FALSE(tracker.state(claude, "cache-model", 0).last_exchange_at.has_value());
        CHECK_FALSE(tracker.replay().has_value());
    }
}

TEST_CASE("PromptCacheTracker restores a resumed session without a request to replay",
          "[prompt-cache][tracker]") {
    using core::agent::PromptCacheTracker;
    core::llm::PromptCacheCalibration calibration;
    PromptCacheTracker tracker{calibration};
    auto claude = std::make_shared<test_support::CachingProvider>(PromptCacheProfile::documented(5min, true));

    tracker.restore(kT0, claude, "caching", "cache-model");
    CHECK(tracker.state(claude, "cache-model", 200'000).last_exchange_at == kT0);
    // The earlier process's request is gone; only a new turn can be replayed.
    CHECK_FALSE(tracker.replay().has_value());

    auto local = std::make_shared<test_support::CachingProvider>(PromptCacheProfile::none());
    tracker.restore(kT0, local, "local", "cache-model");
    CHECK_FALSE(tracker.state(local, "cache-model", 200'000).last_exchange_at.has_value());
}

TEST_CASE("Kimi renewals protect an existing five-minute prefix after requesting one hour",
          "[prompt-cache][tracker][kimi]") {
    const auto profile = core::llm::protocols::KimiProtocol{
        core::llm::PromptCacheTtl::Extended}.prompt_cache_profile("kimi-k3");
    auto provider = std::make_shared<test_support::CachingProvider>(profile);
    core::llm::PromptCacheCalibration calibration;
    core::agent::PromptCacheTracker tracker{calibration};
    tracker.restore(kT0, provider, "kimi", "kimi-k3");
    const auto state = tracker.state(provider, "kimi-k3", 200'000);
    const PromptCacheGuard guard{PromptCacheGuardSettings{
        .keep_warm = core::agent::PromptCacheKeepWarm::Auto}};

    const auto deadlines = guard.deadlines(state);
    REQUIRE(deadlines.has_value());
    CHECK(deadlines->notice == kT0 + 4min);
    CHECK(deadlines->expiry == kT0 + 5min);
    CHECK(guard.should_confirm_send(PromptCacheGuard::assess(state, kT0 + 6min)));
    // A prefix locked to the cheaper write tier cannot justify the 1h tier's
    // larger automatic replay budget.
    CHECK(PromptCacheGuard::keep_warm_budget(profile) == 9);
    CHECK_FALSE(guard.should_keep_warm(profile, 9));
}

// ── Agent bookkeeping ───────────────────────────────────────────────────────

TEST_CASE("Agent records when the conversation last used the provider cache",
          "[prompt-cache][agent]") {
    auto provider = std::make_shared<test_support::CachingProvider>(PromptCacheProfile::documented(5min, true));
    auto agent = test_support::make_prompt_cache_agent(provider);
    CHECK_FALSE(agent->prompt_cache_state().last_exchange_at.has_value());

    const auto before = Clock::now();
    test_support::send_prompt_and_wait(agent, "hello");
    const auto state = agent->prompt_cache_state();
    REQUIRE(state.last_exchange_at.has_value());
    CHECK(*state.last_exchange_at >= before);
    CHECK(*state.last_exchange_at <= Clock::now());
    CHECK(state.profile == PromptCacheProfile::documented(5min, true));
    CHECK(state.context_tokens >= 2'000);

    SECTION("a provider switch leaves the old cache's age behind") {
        agent->set_provider(std::make_shared<test_support::CachingProvider>(PromptCacheProfile::documented(5min, true)));
        CHECK_FALSE(agent->prompt_cache_state().last_exchange_at.has_value());
    }
    SECTION("clearing the conversation forgets its cache") {
        agent->clear_history();
        CHECK_FALSE(agent->prompt_cache_state().last_exchange_at.has_value());
    }
}

TEST_CASE("Agent does not track a provider without a billed cache", "[prompt-cache][agent]") {
    auto agent = test_support::make_prompt_cache_agent(std::make_shared<test_support::CachingProvider>(PromptCacheProfile::none()));
    test_support::send_prompt_and_wait(agent, "hello");
    CHECK_FALSE(agent->prompt_cache_state().last_exchange_at.has_value());
}

TEST_CASE("A resumed session seeds its cache age only for the same provider and model",
          "[prompt-cache][agent]") {
    auto agent = test_support::make_prompt_cache_agent(std::make_shared<test_support::CachingProvider>(PromptCacheProfile::documented(5min, true)));
    const auto two_hours_ago = Clock::now() - 2h;

    agent->restore_prompt_cache_activity(two_hours_ago, "other", "cache-model");
    CHECK_FALSE(agent->prompt_cache_state().last_exchange_at.has_value());
    agent->restore_prompt_cache_activity(two_hours_ago, "caching", "other-model");
    CHECK_FALSE(agent->prompt_cache_state().last_exchange_at.has_value());

    agent->restore_prompt_cache_activity(two_hours_ago, "caching", "cache-model");
    const auto state = agent->prompt_cache_state();
    REQUIRE(state.last_exchange_at.has_value());
    CHECK(PromptCacheGuard::assess(state, Clock::now()).expired());
}

TEST_CASE("Compact-then continues either way and leaves no cache to track",
          "[prompt-cache][agent]") {
    auto agent = test_support::make_prompt_cache_agent(std::make_shared<test_support::CachingProvider>(PromptCacheProfile::documented(5min, true)));
    test_support::send_prompt_and_wait(agent, "hello");
    REQUIRE(agent->prompt_cache_state().last_exchange_at.has_value());

    std::promise<bool> finished;
    auto compacted = finished.get_future();
    agent->compact_history_then({}, [&finished](bool applied) { finished.set_value(applied); });
    REQUIRE(compacted.wait_for(5s) == std::future_status::ready);
    CHECK(compacted.get());
    CHECK_FALSE(agent->prompt_cache_state().last_exchange_at.has_value());
}

TEST_CASE("Keeping the cache warm replays the last request beside the conversation",
          "[prompt-cache][agent]") {
    auto provider = std::make_shared<test_support::CachingProvider>(PromptCacheProfile::documented(5min, true));
    auto agent = test_support::make_prompt_cache_agent(provider);
    test_support::send_prompt_and_wait(agent, "hello");
    const auto first_exchange = *agent->prompt_cache_state().last_exchange_at;
    const auto history_before = agent->get_history();

    std::promise<core::agent::Agent::PromptCacheRefresh> done;
    auto refreshed = done.get_future();
    agent->refresh_prompt_cache_async(
        [&done](core::agent::Agent::PromptCacheRefresh result) { done.set_value(std::move(result)); });
    REQUIRE(refreshed.wait_for(5s) == std::future_status::ready);
    const auto result = refreshed.get();

    CHECK(result.refreshed);
    CHECK(result.usage.cached_prompt_tokens == 1'900);
    const auto requests = provider->requests();
    REQUIRE(requests.size() == 2);
    CHECK(requests[1].messages.size() == requests[0].messages.size());
    CHECK(requests[1].messages.back().content == requests[0].messages.back().content);
    // The conversation itself is untouched; only the cache age moved.
    CHECK(agent->get_history().size() == history_before.size());
    CHECK(*agent->prompt_cache_state().last_exchange_at >= first_exchange);
}

TEST_CASE("A cache that a replay does not renew is never replayed", "[prompt-cache][agent]") {
    auto agent = test_support::make_prompt_cache_agent(std::make_shared<test_support::CachingProvider>(PromptCacheProfile::documented(5min, false)));
    test_support::send_prompt_and_wait(agent, "hello");

    std::promise<core::agent::Agent::PromptCacheRefresh> done;
    auto refreshed = done.get_future();
    agent->refresh_prompt_cache_async(
        [&done](core::agent::Agent::PromptCacheRefresh result) { done.set_value(std::move(result)); });
    REQUIRE(refreshed.wait_for(5s) == std::future_status::ready);
    const auto result = refreshed.get();
    CHECK_FALSE(result.refreshed);
    CHECK_FALSE(result.error.empty());
}
