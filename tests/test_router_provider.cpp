#include <catch2/catch_test_macros.hpp>

#include "core/llm/providers/RouterProvider.hpp"
#include "core/llm/routing/PolicyLoader.hpp"
#include "core/llm/routing/RouterEngine.hpp"
#include "core/llm/ProviderManager.hpp"
#include "core/llm/Models.hpp"
#include "core/budget/BudgetTracker.hpp"

#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace core::llm;
using namespace core::llm::routing;
using namespace core::llm::providers;

// ─── Mock provider ────────────────────────────────────────────────────────────

// Succeeds after `fail_count` throws.
// retryable=true  → throws "503 Service Unavailable"
// retryable=false → throws "401 authentication failed"
class MockProvider final : public LLMProvider {
public:
    explicit MockProvider(int fail_count = 0,
                          bool retryable = true,
                          std::string success_text = "ok",
                          bool should_estimate_cost = true,
                          bool is_local = false,
                          core::llm::protocols::RateLimitInfo rate_limit_info = {})
        : fail_count_(fail_count)
        , retryable_(retryable)
        , success_text_(std::move(success_text))
        , should_estimate_cost_(should_estimate_cost)
        , is_local_(is_local)
        , rate_limit_info_(std::move(rate_limit_info)) {
        set_last_rate_limit_info(rate_limit_info_);
    }

    void stream_response(
        const ChatRequest&,
        std::function<void(const StreamChunk&)> callback) override {
        ++call_count_;
        if (call_count_ <= fail_count_) {
            if (retryable_) {
                throw std::runtime_error("503 Service Unavailable");
            } else {
                throw std::runtime_error("401 authentication failed");
            }
        }
        set_last_usage(100, 50);
        set_last_rate_limit_info(rate_limit_info_);
        core::llm::StreamChunk chunk;
        chunk.content = success_text_;
        chunk.is_final = true;
        callback(chunk);
    }

    [[nodiscard]] bool should_estimate_cost() const override { return should_estimate_cost_; }
    [[nodiscard]] ProviderCapabilities capabilities() const override {
        return ProviderCapabilities{.supports_tool_calls = true, .is_local = is_local_};
    }

    int call_count() const { return call_count_.load(); }

private:
    int fail_count_;
    bool retryable_;
    std::string success_text_;
    bool should_estimate_cost_;
    bool is_local_;
    core::llm::protocols::RateLimitInfo rate_limit_info_;
    std::atomic<int> call_count_{0};
};

// ─── Fixture ──────────────────────────────────────────────────────────────────

// Each test uses a unique namespace prefix for provider names to avoid
// cross-test contamination in the global ProviderManager singleton.
struct RouterFixture {
    std::shared_ptr<RouterEngine> engine;
    std::unordered_map<std::string, std::string> default_models;

    // providers: map from name → mock provider.  Names must be globally unique.
    RouterFixture(const std::string& router_json,
                  std::unordered_map<std::string, std::shared_ptr<MockProvider>> providers) {
        simdjson::dom::parser parser;
        simdjson::dom::element doc = parser.parse(router_json);
        simdjson::dom::object router_obj;
        if (doc.get(router_obj) != simdjson::SUCCESS)
            throw std::runtime_error("bad router json in fixture");

        RouterConfig config;
        std::string error;
        if (!parse_router_config(router_obj, config, error))
            throw std::runtime_error("parse_router_config: " + error);

        std::unordered_set<std::string> avail;
        auto& pm = ProviderManager::get_instance();
        for (auto& [name, prov] : providers) {
            pm.register_provider(name, prov);
            avail.insert(name);
            default_models[name] = name + "-default-model";
        }

        engine = std::make_shared<RouterEngine>(std::move(config), std::move(avail));
    }

    RouterProvider make_router() {
        return RouterProvider(ProviderManager::get_instance(), engine, default_models);
    }
};

// Collect all response text from a RouterProvider call.
static std::string collect(RouterProvider& rp, const std::string& prompt = "test") {
    ChatRequest req;
    Message msg;
    msg.role    = "user";
    msg.content = prompt;
    req.messages.push_back(std::move(msg));

    std::string result;
    rp.stream_response(req, [&](const StreamChunk& chunk) {
        result += chunk.content;
    });
    return result;
}

static core::llm::protocols::RateLimitInfo make_rate_limit(
    int requests_limit,
    int requests_remaining,
    int tokens_limit,
    int tokens_remaining,
    std::vector<core::llm::protocols::UsageWindow> windows = {},
    bool is_rate_limited = false,
    int retry_after = 0) {
    core::llm::protocols::RateLimitInfo info;
    info.requests_limit = requests_limit;
    info.requests_remaining = requests_remaining;
    info.tokens_limit = tokens_limit;
    info.tokens_remaining = tokens_remaining;
    info.usage_windows = std::move(windows);
    info.is_rate_limited = is_rate_limited;
    info.retry_after = retry_after;
    return info;
}

// ─── Tests ────────────────────────────────────────────────────────────────────

TEST_CASE("RouterProvider: success on first try", "[router_provider]") {
    auto prov = std::make_shared<MockProvider>(0, true, "first-ok");
    RouterFixture f(R"({
        "enabled": true, "default_policy": "p",
        "policies": { "p": { "strategy": "fallback",
            "defaults": [{ "provider": "rp_t1_prov", "model": "m", "retries": 0 }]
        }}
    })", {{"rp_t1_prov", prov}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "first-ok");
    REQUIRE(prov->call_count() == 1);
}

TEST_CASE("RouterProvider: falls back to second candidate when first always fails", "[router_provider]") {
    auto primary   = std::make_shared<MockProvider>(99, true,  "never");
    auto secondary = std::make_shared<MockProvider>(0,  true,  "secondary-ok");

    RouterFixture f(R"({
        "enabled": true, "default_policy": "p",
        "policies": { "p": { "strategy": "fallback",
            "defaults": [
                { "provider": "rp_t2_primary",   "model": "m1", "retries": 0 },
                { "provider": "rp_t2_secondary", "model": "m2", "retries": 0 }
            ]
        }}
    })", {{"rp_t2_primary", primary}, {"rp_t2_secondary", secondary}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "secondary-ok");
    REQUIRE(primary->call_count()   == 1);
    REQUIRE(secondary->call_count() == 1);
}

TEST_CASE("RouterProvider: retries candidate up to its retry count", "[router_provider]") {
    // Fails on calls 1 and 2, succeeds on call 3.
    auto prov = std::make_shared<MockProvider>(2, true, "eventual-ok");

    RouterFixture f(R"({
        "enabled": true, "default_policy": "p",
        "policies": { "p": { "strategy": "fallback",
            "defaults": [{ "provider": "rp_t3_flaky", "model": "m", "retries": 2 }]
        }}
    })", {{"rp_t3_flaky", prov}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "eventual-ok");
    REQUIRE(prov->call_count() == 3); // 2 failures + 1 success
}

TEST_CASE("RouterProvider: non-retryable auth error skips remaining retries immediately", "[router_provider]") {
    auto bad_auth  = std::make_shared<MockProvider>(99, false, "never");
    auto secondary = std::make_shared<MockProvider>(0,  true,  "backup-ok");

    RouterFixture f(R"({
        "enabled": true, "default_policy": "p",
        "policies": { "p": { "strategy": "fallback",
            "defaults": [
                { "provider": "rp_t4_badauth",   "model": "m1", "retries": 5 },
                { "provider": "rp_t4_secondary", "model": "m2", "retries": 0 }
            ]
        }}
    })", {{"rp_t4_badauth", bad_auth}, {"rp_t4_secondary", secondary}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "backup-ok");
    // bad-auth called once (non-retryable skips retry loop)
    REQUIRE(bad_auth->call_count()  == 1);
    REQUIRE(secondary->call_count() == 1);
}

TEST_CASE("RouterProvider: all candidates fail emits error message", "[router_provider]") {
    auto p1 = std::make_shared<MockProvider>(99, true, "");
    auto p2 = std::make_shared<MockProvider>(99, true, "");

    RouterFixture f(R"({
        "enabled": true, "default_policy": "p",
        "policies": { "p": { "strategy": "fallback",
            "defaults": [
                { "provider": "rp_t5_p1", "model": "m1", "retries": 0 },
                { "provider": "rp_t5_p2", "model": "m2", "retries": 0 }
            ]
        }}
    })", {{"rp_t5_p1", p1}, {"rp_t5_p2", p2}});
    auto router = f.make_router();

    const auto result = collect(router);
    REQUIRE(result.starts_with("\n[Router error:"));
    REQUIRE(result.find("all candidates failed") != std::string::npos);
}

TEST_CASE("RouterProvider: retries exhausted then falls through to next candidate", "[router_provider]") {
    // prov1 always fails; retries=1 → called twice.  prov2 succeeds.
    auto prov1 = std::make_shared<MockProvider>(99, true, "never");
    auto prov2 = std::make_shared<MockProvider>(0,  true, "prov2-ok");

    RouterFixture f(R"({
        "enabled": true, "default_policy": "p",
        "policies": { "p": { "strategy": "fallback",
            "defaults": [
                { "provider": "rp_t6_p1", "model": "m1", "retries": 1 },
                { "provider": "rp_t6_p2", "model": "m2", "retries": 0 }
            ]
        }}
    })", {{"rp_t6_p1", prov1}, {"rp_t6_p2", prov2}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "prov2-ok");
    REQUIRE(prov1->call_count() == 2); // 1 initial + 1 retry
    REQUIRE(prov2->call_count() == 1);
}

TEST_CASE("RouterProvider: last_route_summary reflects routed provider", "[router_provider]") {
    auto prov = std::make_shared<MockProvider>(0, true, "success");

    RouterFixture f(R"({
        "enabled": true, "default_policy": "p",
        "policies": { "p": { "strategy": "fallback",
            "defaults": [{ "provider": "rp_t7_myprov", "model": "mymodel", "retries": 0 }]
        }}
    })", {{"rp_t7_myprov", prov}});
    auto router = f.make_router();

    collect(router);
    const auto summary = router.last_route_summary();
    REQUIRE(summary.find("rp_t7_myprov") != std::string::npos);
}

TEST_CASE("RouterProvider: routed local providers remain non-billable", "[router_provider]") {
    auto local = std::make_shared<MockProvider>(0, true, "local-ok", false);

    RouterFixture f(R"({
        "enabled": true, "default_policy": "p",
        "policies": { "p": { "strategy": "fallback",
            "defaults": [{ "provider": "rp_t7_local", "model": "local-model", "retries": 0 }]
        }}
    })", {{"rp_t7_local", local}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "local-ok");
    REQUIRE_FALSE(router.should_estimate_cost());
}

TEST_CASE("RouterProvider: three-provider chain - cost-effective fallback", "[router_provider]") {
    // Requesty's documented use-case: cheap → mid → expensive
    auto cheap    = std::make_shared<MockProvider>(99, true, "never");
    auto mid      = std::make_shared<MockProvider>(99, true, "never");
    auto expensive = std::make_shared<MockProvider>(0,  true, "expensive-ok");

    RouterFixture f(R"({
        "enabled": true, "default_policy": "p",
        "policies": { "p": { "strategy": "fallback",
            "defaults": [
                { "provider": "rp_t8_cheap",     "model": "cheap-mini", "retries": 0 },
                { "provider": "rp_t8_mid",        "model": "mid-model",  "retries": 0 },
                { "provider": "rp_t8_expensive",  "model": "exp-model",  "retries": 0 }
            ]
        }}
    })", {{"rp_t8_cheap", cheap}, {"rp_t8_mid", mid}, {"rp_t8_expensive", expensive}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "expensive-ok");
    REQUIRE(cheap->call_count()    == 1);
    REQUIRE(mid->call_count()      == 1);
    REQUIRE(expensive->call_count() == 1);
}

TEST_CASE("RouterProvider: guardrails max_session_cost_usd skips remote and falls back to local",
          "[router_provider][guardrails]") {
    auto& budget = core::budget::BudgetTracker::get_instance();
    budget.reset_session();
    budget.record(TokenUsage{.prompt_tokens = 200000, .completion_tokens = 200000, .total_tokens = 400000},
                  "gpt-4o",
                  true);

    auto remote = std::make_shared<MockProvider>(0, true, "remote-ok", true, false);
    auto local  = std::make_shared<MockProvider>(0, true, "local-ok", false, true);

    RouterFixture f(R"({
        "enabled": true,
        "default_policy": "p",
        "guardrails": { "max_session_cost_usd": 1.0 },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "rp_guard_cost_remote", "model": "gpt-4o" },
                    { "provider": "rp_guard_cost_local",  "model": "qwen-local" }
                ]
            }
        }
    })", {{"rp_guard_cost_remote", remote}, {"rp_guard_cost_local", local}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "local-ok");
    REQUIRE(remote->call_count() == 0);
    REQUIRE(local->call_count() == 1);
    REQUIRE_FALSE(router.should_estimate_cost());

    budget.reset_session();
}

TEST_CASE("RouterProvider: guardrails request reserve triggers fallback", "[router_provider][guardrails]") {
    auto remote = std::make_shared<MockProvider>(
        0,
        true,
        "remote-ok",
        true,
        false,
        make_rate_limit(100, 5, 0, 0));
    auto local  = std::make_shared<MockProvider>(0, true, "local-ok", false, true);

    RouterFixture f(R"({
        "enabled": true,
        "default_policy": "p",
        "guardrails": { "min_requests_remaining_ratio": 0.20 },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "rp_guard_req_remote", "model": "gpt-4o" },
                    { "provider": "rp_guard_req_local",  "model": "qwen-local" }
                ]
            }
        }
    })", {{"rp_guard_req_remote", remote}, {"rp_guard_req_local", local}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "local-ok");
    REQUIRE(remote->call_count() == 0);
    REQUIRE(local->call_count() == 1);
    REQUIRE(router.last_guardrail_summary().find("rp_guard_req_remote") != std::string::npos);
}

TEST_CASE("RouterProvider: guardrails unified window reserve triggers fallback", "[router_provider][guardrails]") {
    auto remote = std::make_shared<MockProvider>(
        0,
        true,
        "remote-ok",
        true,
        false,
        make_rate_limit(0,
                        0,
                        0,
                        0,
                        {core::llm::protocols::UsageWindow{"5h", 0.85f},
                         core::llm::protocols::UsageWindow{"7d", 0.30f}}));
    auto local  = std::make_shared<MockProvider>(0, true, "local-ok", false, true);

    RouterFixture f(R"({
        "enabled": true,
        "default_policy": "p",
        "guardrails": { "min_window_remaining_ratio": 0.20 },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "rp_guard_window_remote", "model": "gpt-4o" },
                    { "provider": "rp_guard_window_local",  "model": "qwen-local" }
                ]
            }
        }
    })", {{"rp_guard_window_remote", remote}, {"rp_guard_window_local", local}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "local-ok");
    REQUIRE(remote->call_count() == 0);
    REQUIRE(local->call_count() == 1);
}

TEST_CASE("RouterProvider: guardrails token reserve triggers fallback", "[router_provider][guardrails]") {
    auto remote = std::make_shared<MockProvider>(
        0,
        true,
        "remote-ok",
        true,
        false,
        make_rate_limit(0, 0, 100000, 5000));
    auto local  = std::make_shared<MockProvider>(0, true, "local-ok", false, true);

    RouterFixture f(R"({
        "enabled": true,
        "default_policy": "p",
        "guardrails": { "min_tokens_remaining_ratio": 0.20 },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "rp_guard_tok_remote", "model": "gpt-4o" },
                    { "provider": "rp_guard_tok_local",  "model": "qwen-local" }
                ]
            }
        }
    })", {{"rp_guard_tok_remote", remote}, {"rp_guard_tok_local", local}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "local-ok");
    REQUIRE(remote->call_count() == 0);
    REQUIRE(local->call_count() == 1);
    REQUIRE(router.last_guardrail_summary().find("token reserve") != std::string::npos);
}

TEST_CASE("RouterProvider: guardrails rate-limited candidate is skipped before dispatch",
          "[router_provider][guardrails]") {
    auto remote = std::make_shared<MockProvider>(
        0,
        true,
        "remote-ok",
        true,
        false,
        make_rate_limit(100, 80, 100000, 70000, {}, true, 15));
    auto local  = std::make_shared<MockProvider>(0, true, "local-ok", false, true);

    RouterFixture f(R"({
        "enabled": true,
        "default_policy": "p",
        "guardrails": { "min_requests_remaining_ratio": 0.10 },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "rp_guard_rl_remote", "model": "gpt-4o" },
                    { "provider": "rp_guard_rl_local",  "model": "qwen-local" }
                ]
            }
        }
    })", {{"rp_guard_rl_remote", remote}, {"rp_guard_rl_local", local}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "local-ok");
    REQUIRE(remote->call_count() == 0);
    REQUIRE(local->call_count() == 1);
    REQUIRE(router.last_guardrail_summary().find("retry_after=15s") != std::string::npos);
}

TEST_CASE("RouterProvider: local providers are exempt from guardrails by default",
          "[router_provider][guardrails]") {
    auto local = std::make_shared<MockProvider>(
        0,
        true,
        "local-ok",
        false,
        true,
        make_rate_limit(100, 1, 100000, 100, {{"5h", 0.99f}}, true, 30));
    auto remote = std::make_shared<MockProvider>(0, true, "remote-ok", true, false);

    RouterFixture f(R"({
        "enabled": true,
        "default_policy": "p",
        "guardrails": {
            "min_requests_remaining_ratio": 0.50,
            "min_tokens_remaining_ratio": 0.50,
            "min_window_remaining_ratio": 0.50
        },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "rp_guard_local_exempt", "model": "qwen-local" },
                    { "provider": "rp_guard_local_remote", "model": "gpt-4o" }
                ]
            }
        }
    })", {{"rp_guard_local_exempt", local}, {"rp_guard_local_remote", remote}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "local-ok");
    REQUIRE(local->call_count() == 1);
    REQUIRE(remote->call_count() == 0);
    REQUIRE(router.last_guardrail_summary().empty());
}

TEST_CASE("RouterProvider: enforce_on_local applies guardrails to local providers",
          "[router_provider][guardrails]") {
    auto local = std::make_shared<MockProvider>(
        0,
        true,
        "local-ok",
        false,
        true,
        make_rate_limit(0, 0, 0, 0, {{"5h", 0.95f}}));
    auto remote = std::make_shared<MockProvider>(0, true, "remote-ok", true, false);

    RouterFixture f(R"({
        "enabled": true,
        "default_policy": "p",
        "guardrails": {
            "min_window_remaining_ratio": 0.10,
            "enforce_on_local": true
        },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "rp_guard_local_enforced", "model": "qwen-local" },
                    { "provider": "rp_guard_local_remote2", "model": "gpt-4o" }
                ]
            }
        }
    })", {{"rp_guard_local_enforced", local}, {"rp_guard_local_remote2", remote}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "remote-ok");
    REQUIRE(local->call_count() == 0);
    REQUIRE(remote->call_count() == 1);
    REQUIRE(router.last_guardrail_summary().find("rp_guard_local_enforced") != std::string::npos);
}

TEST_CASE("RouterProvider: guardrails block all candidates with explicit error",
          "[router_provider][guardrails]") {
    auto remote = std::make_shared<MockProvider>(
        0,
        true,
        "remote-ok",
        true,
        false,
        make_rate_limit(100, 2, 0, 0));

    RouterFixture f(R"({
        "enabled": true,
        "default_policy": "p",
        "guardrails": { "min_requests_remaining_ratio": 0.25 },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "rp_guard_all_blocked", "model": "gpt-4o" }
                ]
            }
        }
    })", {{"rp_guard_all_blocked", remote}});
    auto router = f.make_router();

    const auto result = collect(router);
    REQUIRE(result.find("Guardrails blocked") != std::string::npos);
    REQUIRE(result.find("blocked by guardrails") != std::string::npos);
    REQUIRE(remote->call_count() == 0);
}

TEST_CASE("RouterProvider: forwards routed provider rate-limit snapshot for UI", "[router_provider][rate_limit]") {
    auto remote = std::make_shared<MockProvider>(
        0,
        true,
        "remote-ok",
        true,
        false,
        make_rate_limit(120, 60, 200000, 150000, {{"5h", 0.40f}}));

    RouterFixture f(R"({
        "enabled": true,
        "default_policy": "p",
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "rp_ratelimit_remote", "model": "gpt-4o" }
                ]
            }
        }
    })", {{"rp_ratelimit_remote", remote}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "remote-ok");
    const auto rate_limit = router.get_last_rate_limit_info();
    REQUIRE(rate_limit.requests_limit == 120);
    REQUIRE(rate_limit.requests_remaining == 60);
    REQUIRE(rate_limit.tokens_limit == 200000);
    REQUIRE(rate_limit.tokens_remaining == 150000);
    REQUIRE(rate_limit.usage_windows.size() == 1);
    REQUIRE(rate_limit.usage_windows[0].label == "5h");
}

TEST_CASE("RouterProvider: route summary includes guardrail fallback note on success",
          "[router_provider][guardrails]") {
    auto remote = std::make_shared<MockProvider>(
        0,
        true,
        "remote-ok",
        true,
        false,
        make_rate_limit(100, 1, 0, 0));
    auto local = std::make_shared<MockProvider>(0, true, "local-ok", false, true);

    RouterFixture f(R"({
        "enabled": true,
        "default_policy": "p",
        "guardrails": { "min_requests_remaining_ratio": 0.20 },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "rp_guard_summary_remote", "model": "gpt-4o" },
                    { "provider": "rp_guard_summary_local", "model": "qwen-local" }
                ]
            }
        }
    })", {{"rp_guard_summary_remote", remote}, {"rp_guard_summary_local", local}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "local-ok");
    const std::string summary = router.last_route_summary();
    REQUIRE(summary.find("guardrail fallback") != std::string::npos);
    REQUIRE(summary.find("rp_guard_summary_remote") != std::string::npos);
}

TEST_CASE("RouterProvider: guardrail summary clears on later non-guardrail success",
          "[router_provider][guardrails]") {
    auto& budget = core::budget::BudgetTracker::get_instance();
    budget.reset_session();
    budget.record(TokenUsage{.prompt_tokens = 200000, .completion_tokens = 200000, .total_tokens = 400000},
                  "gpt-4o",
                  true);

    auto remote = std::make_shared<MockProvider>(0, true, "remote-ok", true, false);
    auto local  = std::make_shared<MockProvider>(0, true, "local-ok", false, true);

    RouterFixture f(R"({
        "enabled": true,
        "default_policy": "p",
        "guardrails": { "max_session_cost_usd": 1.0 },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "rp_guard_clear_remote", "model": "gpt-4o" },
                    { "provider": "rp_guard_clear_local",  "model": "qwen-local" }
                ]
            }
        }
    })", {{"rp_guard_clear_remote", remote}, {"rp_guard_clear_local", local}});
    auto router = f.make_router();

    REQUIRE(collect(router) == "local-ok");
    REQUIRE_FALSE(router.last_guardrail_summary().empty());

    budget.reset_session(); // remove spend pressure

    REQUIRE(collect(router) == "remote-ok");
    REQUIRE(router.last_guardrail_summary().empty());

    budget.reset_session();
}
