#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include "core/llm/routing/PolicyLoader.hpp"
#include "core/llm/routing/RouterEngine.hpp"

#include <simdjson.h>

#include <string_view>
#include <unordered_set>
#include <utility>

namespace {

core::llm::routing::RouterConfig parse_router_from_json(const std::string& json) {
    simdjson::dom::parser parser;
    simdjson::dom::element doc = parser.parse(json);

    simdjson::dom::object router_obj;
    REQUIRE(doc.get(router_obj) == simdjson::SUCCESS);

    core::llm::routing::RouterConfig config;
    std::string error;
    REQUIRE(core::llm::routing::parse_router_config(router_obj, config, error));
    return config;
}

core::llm::ProviderDescriptorSet providers(
    std::initializer_list<std::string_view> names) {
    core::llm::ProviderDescriptorSet descriptors;
    for (const auto name : names) {
        descriptors.insert({std::string(name), false});
    }
    return descriptors;
}

core::llm::ProviderDescriptorSet providers_with_locality(
    std::initializer_list<std::pair<std::string_view, bool>> entries) {
    core::llm::ProviderDescriptorSet descriptors;
    for (const auto& [name, is_local] : entries) {
        descriptors.insert({std::string(name), is_local});
    }
    return descriptors;
}

} // namespace

// ─── Policy loader ────────────────────────────────────────────────────────────

TEST_CASE("Router policy loader parses reflected JSON schema", "[routing][config]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "demo",
        "policies": {
            "demo": {
                "strategy": "load_balance",
                "defaults": [
                    { "provider": "alpha", "model": "alpha-fast", "weight": 90 },
                    { "provider": "beta",  "model": "beta-pro",  "weight": 10 }
                ],
                "rules": [
                    {
                        "name": "debug",
                        "priority": 5,
                        "strategy": "fallback",
                        "when": { "any_keywords": ["debug", "root cause"] },
                        "candidates": [
                            { "provider": "beta", "model": "beta-pro", "retries": 1 }
                        ]
                    }
                ]
            }
        }
    })";

    const auto config = parse_router_from_json(json);
    REQUIRE(config.enabled);
    REQUIRE(config.default_policy == "demo");
    REQUIRE(config.policies.contains("demo"));

    const auto& policy = config.policies.at("demo");
    REQUIRE(policy.defaults.size() == 2);
    REQUIRE(policy.defaults[0].provider == "alpha");
    REQUIRE(policy.rules.size() == 1);
    REQUIRE(policy.rules[0].when.any_keywords.size() == 2);
    REQUIRE(policy.rules[0].candidates[0].retries == 1);
}

TEST_CASE("Router policy loader accepts models alias for defaults", "[routing][config]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "policies": {
            "p": {
                "strategy": "fallback",
                "models": [
                    { "provider": "x", "model": "x-fast" }
                ]
            }
        }
    })";

    const auto config = parse_router_from_json(json);
    REQUIRE(config.policies.at("p").defaults.size() == 1);
    REQUIRE(config.policies.at("p").defaults[0].provider == "x");
}

TEST_CASE("Router policy loader parses tier pins and auto classifier config", "[routing][config]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "smart",
        "auto_classifier": {
            "quality_bias": 0.9,
            "fast_token_threshold": 256,
            "powerful_token_threshold": 4096,
            "escalation_turn_threshold": 4
        },
        "policies": {
            "smart": {
                "strategy": "smart",
                "defaults": [
                    { "provider": "local", "model": "qwen-local", "tier": "FAST" },
                    { "provider": "remote", "model": "gpt-5.4", "tier": "powerful" }
                ]
            }
        }
    })";

    const auto config = parse_router_from_json(json);
    REQUIRE(config.auto_classifier.quality_bias == Catch::Approx(0.9));
    REQUIRE(config.auto_classifier.fast_token_threshold == 256);
    REQUIRE(config.auto_classifier.powerful_token_threshold == 4096);
    REQUIRE(config.auto_classifier.escalation_turn_threshold == 4);
    REQUIRE(config.has_auto_classifier_overrides);

    const auto& defaults = config.policies.at("smart").defaults;
    REQUIRE(defaults.size() == 2);
    REQUIRE(defaults[0].tier == "fast");
    REQUIRE(defaults[1].tier == "powerful");
}

TEST_CASE("merge_router_config overlays policies and toggles enabled", "[routing][config]") {
    using namespace core::llm::routing;

    RouterConfig base;
    base.enabled = false;
    base.default_policy = "base-policy";
    PolicyDefinition base_pol;
    base_pol.name = "base-policy";
    base.policies["base-policy"] = std::move(base_pol);

    RouterConfig overlay;
    overlay.enabled = true;
    overlay.default_policy = "new-policy";
    PolicyDefinition new_pol;
    new_pol.name = "new-policy";
    overlay.policies["new-policy"] = std::move(new_pol);

    merge_router_config(base, overlay);

    REQUIRE(base.enabled);
    REQUIRE(base.default_policy == "new-policy");
    REQUIRE(base.policies.contains("base-policy")); // original kept
    REQUIRE(base.policies.contains("new-policy"));  // new one added
}

TEST_CASE("merge_router_config overlays auto classifier only when explicitly configured", "[routing][config]") {
    auto base = parse_router_from_json(R"({
        "enabled": true,
        "default_policy": "base",
        "auto_classifier": { "quality_bias": 0.2 },
        "policies": {
            "base": { "strategy": "fallback", "defaults": [{ "provider": "a", "model": "a" }] }
        }
    })");

    const auto overlay_without_classifier = parse_router_from_json(R"({
        "enabled": true,
        "default_policy": "base",
        "policies": {
            "base": { "strategy": "fallback", "defaults": [{ "provider": "b", "model": "b" }] }
        }
    })");
    merge_router_config(base, overlay_without_classifier);
    REQUIRE(base.auto_classifier.quality_bias == Catch::Approx(0.2));

    const auto overlay_with_classifier = parse_router_from_json(R"({
        "enabled": true,
        "default_policy": "base",
        "auto_classifier": { "quality_bias": 0.75 },
        "policies": {
            "base": { "strategy": "fallback", "defaults": [{ "provider": "b", "model": "b" }] }
        }
    })");
    merge_router_config(base, overlay_with_classifier);
    REQUIRE(base.auto_classifier.quality_bias == Catch::Approx(0.75));
}

TEST_CASE("Router policy loader parses guardrails and clamps ratios", "[routing][config]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "guardrails": {
            "max_session_cost_usd": 2.5,
            "min_requests_remaining_ratio": 0.20,
            "min_tokens_remaining_ratio": -1.0,
            "min_window_remaining_ratio": 1.25,
            "enforce_on_local": true
        },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [{ "provider": "x", "model": "x" }]
            }
        }
    })";

    const auto config = parse_router_from_json(json);
    REQUIRE(config.guardrails.has_value());
    REQUIRE(config.guardrails->max_session_cost_usd == Catch::Approx(2.5));
    REQUIRE(config.guardrails->min_requests_remaining_ratio == Catch::Approx(0.20f));
    REQUIRE(config.guardrails->min_tokens_remaining_ratio == Catch::Approx(0.0f));
    REQUIRE(config.guardrails->min_window_remaining_ratio == Catch::Approx(1.0f));
    REQUIRE(config.guardrails->enforce_on_local);
}

TEST_CASE("merge_router_config preserves guardrails when overlay omits them", "[routing][config]") {
    using namespace core::llm::routing;

    RouterConfig base;
    base.enabled = true;
    base.default_policy = "base";
    base.guardrails = RouterGuardrails{
        .max_session_cost_usd = 3.0,
        .min_requests_remaining_ratio = 0.10f,
    };
    PolicyDefinition base_pol;
    base_pol.name = "base";
    base.policies["base"] = std::move(base_pol);

    RouterConfig overlay;
    overlay.enabled = true;
    overlay.default_policy = "base";
    PolicyDefinition overlay_pol;
    overlay_pol.name = "base";
    overlay.policies["base"] = std::move(overlay_pol);

    merge_router_config(base, overlay);

    REQUIRE(base.guardrails.has_value());
    REQUIRE(base.guardrails->max_session_cost_usd == Catch::Approx(3.0));
    REQUIRE(base.guardrails->min_requests_remaining_ratio == Catch::Approx(0.10f));
}

TEST_CASE("Router policy loader accepts spend_limits alias for guardrails", "[routing][config]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "spend_limits": {
            "max_session_cost_usd": 4.2,
            "min_tokens_remaining_ratio": 0.15
        },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [{ "provider": "x", "model": "x" }]
            }
        }
    })";

    const auto config = parse_router_from_json(json);
    REQUIRE(config.guardrails.has_value());
    REQUIRE(config.guardrails->max_session_cost_usd == Catch::Approx(4.2));
    REQUIRE(config.guardrails->min_tokens_remaining_ratio == Catch::Approx(0.15f));
}

TEST_CASE("Router policy loader accepts limits alias for guardrails", "[routing][config]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "limits": {
            "min_window_remaining_ratio": 0.25
        },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [{ "provider": "x", "model": "x" }]
            }
        }
    })";

    const auto config = parse_router_from_json(json);
    REQUIRE(config.guardrails.has_value());
    REQUIRE(config.guardrails->min_window_remaining_ratio == Catch::Approx(0.25f));
}

TEST_CASE("make_default_router_config returns valid smart-code policy", "[routing][config]") {
    using namespace core::llm::routing;
    const auto config = make_default_router_config();
    REQUIRE(config.enabled);
    REQUIRE(config.default_policy == "smart-code");
    REQUIRE(config.policies.contains("smart-code"));
    const auto& policy = config.policies.at("smart-code");
    REQUIRE(!policy.defaults.empty());
    REQUIRE(!policy.rules.empty());
}

TEST_CASE("RouterEngine exposes configured guardrails", "[routing][engine]") {
    auto config = parse_router_from_json(R"({
        "enabled": true,
        "default_policy": "p",
        "guardrails": {
            "max_session_cost_usd": 2.0,
            "min_requests_remaining_ratio": 0.20
        },
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [{ "provider": "fast", "model": "fast-model" }]
            }
        }
    })");

    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"fast"}));

    const auto guardrails = engine.guardrails();
    REQUIRE(guardrails.has_value());
    REQUIRE(guardrails->max_session_cost_usd == Catch::Approx(2.0));
    REQUIRE(guardrails->min_requests_remaining_ratio == Catch::Approx(0.20f));
}

// ─── Rule conditions ──────────────────────────────────────────────────────────

TEST_CASE("Router engine applies rule matching before defaults", "[routing][engine]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "smart",
        "policies": {
            "smart": {
                "strategy": "load_balance",
                "defaults": [
                    { "provider": "fast", "model": "fast-model", "weight": 1 }
                ],
                "rules": [
                    {
                        "name": "deep-debug",
                        "priority": 10,
                        "strategy": "fallback",
                        "when": { "any_keywords": ["debug", "trace"] },
                        "candidates": [
                            { "provider": "pro", "model": "pro-model" }
                        ]
                    }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"fast", "pro"}));

    const auto deep = engine.route({
        .prompt = "Please debug this failing integration trace",
        .has_tool_messages = false,
    });
    REQUIRE(deep.provider == "pro");
    REQUIRE(deep.rule == "deep-debug");

    const auto simple = engine.route({
        .prompt = "Format this json",
        .has_tool_messages = false,
    });
    REQUIRE(simple.provider == "fast");
}

TEST_CASE("Rule condition: all_keywords requires every keyword to be present", "[routing][engine]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [{ "provider": "default", "model": "default-model" }],
                "rules": [
                    {
                        "name": "both",
                        "priority": 10,
                        "strategy": "fallback",
                        "when": { "all_keywords": ["migrate", "postgres"] },
                        "candidates": [{ "provider": "specialist", "model": "spec-model" }]
                    }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"default", "specialist"}));

    // Only one keyword present → default
    const auto r1 = engine.route({.prompt = "how do I migrate this code?", .has_tool_messages = false});
    REQUIRE(r1.provider == "default");

    // Both keywords present → specialist
    const auto r2 = engine.route({.prompt = "migrate the postgres database schema", .has_tool_messages = false});
    REQUIRE(r2.provider == "specialist");
}

TEST_CASE("Rule condition: needs_tool_history matches only when tool messages present", "[routing][engine]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [{ "provider": "default", "model": "default-model" }],
                "rules": [
                    {
                        "name": "tool-mode",
                        "priority": 10,
                        "strategy": "fallback",
                        "when": { "needs_tool_history": true },
                        "candidates": [{ "provider": "tool-specialist", "model": "ts-model" }]
                    }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"default", "tool-specialist"}));

    const auto no_tools = engine.route({.prompt = "hello world", .has_tool_messages = false});
    REQUIRE(no_tools.provider == "default");

    const auto with_tools = engine.route({.prompt = "hello world", .has_tool_messages = true});
    REQUIRE(with_tools.provider == "tool-specialist");
}

TEST_CASE("Rule condition: min_prompt_chars triggers on long prompts", "[routing][engine]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [{ "provider": "fast", "model": "fast-model" }],
                "rules": [
                    {
                        "name": "long",
                        "priority": 10,
                        "strategy": "fallback",
                        "when": { "min_prompt_chars": 50 },
                        "candidates": [{ "provider": "powerful", "model": "pow-model" }]
                    }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"fast", "powerful"}));

    const auto short_p = engine.route({.prompt = "short", .has_tool_messages = false});
    REQUIRE(short_p.provider == "fast");

    const std::string long_prompt(60, 'x');
    const auto long_p = engine.route({.prompt = long_prompt, .has_tool_messages = false});
    REQUIRE(long_p.provider == "powerful");
}

TEST_CASE("Rule condition: max_prompt_chars triggers on short prompts", "[routing][engine]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [{ "provider": "default", "model": "default-model" }],
                "rules": [
                    {
                        "name": "quick",
                        "priority": 10,
                        "strategy": "fallback",
                        "when": { "max_prompt_chars": 20 },
                        "candidates": [{ "provider": "nano", "model": "nano-model" }]
                    }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"default", "nano"}));

    const auto short_p = engine.route({.prompt = "hi", .has_tool_messages = false});
    REQUIRE(short_p.provider == "nano");

    const std::string long_prompt(30, 'x');
    const auto long_p = engine.route({.prompt = long_prompt, .has_tool_messages = false});
    REQUIRE(long_p.provider == "default");
}

TEST_CASE("Rule priority: lower number wins when multiple rules match", "[routing][engine]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [{ "provider": "default", "model": "default-model" }],
                "rules": [
                    {
                        "name": "low-priority",
                        "priority": 50,
                        "strategy": "fallback",
                        "when": { "any_keywords": ["code"] },
                        "candidates": [{ "provider": "code-generic", "model": "cg-model" }]
                    },
                    {
                        "name": "high-priority",
                        "priority": 5,
                        "strategy": "fallback",
                        "when": { "any_keywords": ["code"] },
                        "candidates": [{ "provider": "code-specialist", "model": "cs-model" }]
                    }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"default", "code-generic", "code-specialist"}));

    const auto r = engine.route({.prompt = "write some code", .has_tool_messages = false});
    REQUIRE(r.provider == "code-specialist");
    REQUIRE(r.rule == "high-priority");
}

// ─── Strategy tests ───────────────────────────────────────────────────────────

TEST_CASE("Router load_balance decisions are deterministic per prompt", "[routing][engine]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "dist",
        "policies": {
            "dist": {
                "strategy": "load_balance",
                "defaults": [
                    { "provider": "a", "model": "a-fast", "weight": 70 },
                    { "provider": "b", "model": "b-fast", "weight": 30 }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"a", "b"}));

    const auto r1 = engine.route({.prompt = "small task", .has_tool_messages = false});
    const auto r2 = engine.route({.prompt = "small task", .has_tool_messages = false});
    REQUIRE(r1.provider == r2.provider);
    REQUIRE(r1.model == r2.model);
}

TEST_CASE("Router latency strategy prefers lower-latency candidate", "[routing][engine]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "lat",
        "policies": {
            "lat": {
                "strategy": "latency",
                "defaults": [
                    { "provider": "slow", "model": "slow-model" },
                    { "provider": "fast", "model": "fast-model" }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"slow", "fast"}));

    engine.record_latency({.policy = "lat", .provider = "slow", .model = "slow-model"}, 240);
    engine.record_latency({.policy = "lat", .provider = "fast", .model = "fast-model"}, 45);

    const auto decision = engine.route({.prompt = "quick check", .has_tool_messages = false});
    REQUIRE(decision.provider == "fast");
}

TEST_CASE("Router smart strategy: complex prompt routes to quality model", "[routing][engine]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "sm",
        "policies": {
            "sm": {
                "strategy": "smart",
                "defaults": [
                    { "provider": "nano", "model": "nano-mini-fast" },
                    { "provider": "pro",  "model": "pro-reasoning" }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"nano", "pro"}));

    // Long prompt with complexity keywords → should pick "pro-reasoning" (high quality score)
    const std::string complex_prompt =
        "Please do a full architecture review and root cause analysis of the production "
        "incident affecting our multi-region database migration. We need a detailed reasoning "
        "chain covering each component, the cascading failures, and a remediation design.";

    const auto r = engine.route({.prompt = complex_prompt, .has_tool_messages = true});
    REQUIRE(r.provider == "pro");
}

TEST_CASE("Router smart strategy: simple prompt routes to fast model via latency", "[routing][engine]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "sm",
        "policies": {
            "sm": {
                "strategy": "smart",
                "defaults": [
                    { "provider": "nano", "model": "nano-mini" },
                    { "provider": "pro",  "model": "pro-reasoning" }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"nano", "pro"}));

    // Seed latency: nano is faster
    engine.record_latency({.policy = "sm", .provider = "nano", .model = "nano-mini"}, 80);
    engine.record_latency({.policy = "sm", .provider = "pro",  .model = "pro-reasoning"}, 600);

    // Short, simple prompt → iterative path → picks lowest latency
    const auto r = engine.route({.prompt = "rename this variable", .has_tool_messages = false});
    REQUIRE(r.provider == "nano");
}

TEST_CASE("Router smart strategy: fast-tier prompts prefer local candidates", "[routing][engine]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "sm",
        "policies": {
            "sm": {
                "strategy": "smart",
                "defaults": [
                    { "provider": "local",  "model": "qwen-local" },
                    { "provider": "remote", "model": "gpt-5.4" }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers_with_locality({{"local", true}, {"remote", false}}));

    // Even if remote has better measured latency, fast-tier requests should stay local.
    engine.record_latency({.policy = "sm", .provider = "local",  .model = "qwen-local"}, 450);
    engine.record_latency({.policy = "sm", .provider = "remote", .model = "gpt-5.4"}, 30);

    const auto decision = engine.route({.prompt = "hi", .has_tool_messages = false});
    REQUIRE(decision.provider == "local");
    REQUIRE(decision.reason.find("local-first") != std::string::npos);
}

// ─── Unavailable providers ────────────────────────────────────────────────────

TEST_CASE("Router skips unavailable candidates and falls back to next", "[routing][engine]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "unavailable", "model": "u-model" },
                    { "provider": "available",   "model": "a-model" }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    // Only "available" is registered
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"available"}));

    const auto r = engine.route({.prompt = "do something", .has_tool_messages = false});
    REQUIRE(r.provider == "available");
}

// ─── Policy switching ─────────────────────────────────────────────────────────

TEST_CASE("set_active_policy switches routing behaviour", "[routing][engine]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "alpha-policy",
        "policies": {
            "alpha-policy": {
                "strategy": "fallback",
                "defaults": [{ "provider": "alpha", "model": "alpha-model" }]
            },
            "beta-policy": {
                "strategy": "fallback",
                "defaults": [{ "provider": "beta", "model": "beta-model" }]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"alpha", "beta"}));

    REQUIRE(engine.active_policy() == "alpha-policy");
    auto r1 = engine.route({.prompt = "task", .has_tool_messages = false});
    REQUIRE(r1.provider == "alpha");

    REQUIRE(engine.set_active_policy("beta-policy"));
    REQUIRE(engine.active_policy() == "beta-policy");
    auto r2 = engine.route({.prompt = "task", .has_tool_messages = false});
    REQUIRE(r2.provider == "beta");

    REQUIRE_FALSE(engine.set_active_policy("non-existent-policy"));
    REQUIRE(engine.active_policy() == "beta-policy"); // unchanged
}

// ─── route_chain ──────────────────────────────────────────────────────────────

TEST_CASE("route_chain returns all candidates in fallback order", "[routing][chain]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "primary",   "model": "p-model",   "retries": 2 },
                    { "provider": "secondary", "model": "s-model",   "retries": 1 },
                    { "provider": "tertiary",  "model": "t-model",   "retries": 0 }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"primary", "secondary", "tertiary"}));

    const auto chain = engine.route_chain({.prompt = "any task", .has_tool_messages = false});
    REQUIRE(chain.size() == 3);
    REQUIRE(chain[0].provider == "primary");
    REQUIRE(chain[0].retries  == 2);
    REQUIRE(chain[1].provider == "secondary");
    REQUIRE(chain[1].retries  == 1);
    REQUIRE(chain[2].provider == "tertiary");
    REQUIRE(chain[2].retries  == 0);
}

TEST_CASE("route_chain: selected candidate is first, rest are fallback tail", "[routing][chain]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "policies": {
            "p": {
                "strategy": "latency",
                "defaults": [
                    { "provider": "slow",   "model": "slow-model" },
                    { "provider": "medium", "model": "med-model"  },
                    { "provider": "fast",   "model": "fast-model" }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"slow", "medium", "fast"}));

    engine.record_latency({.policy = "p", .provider = "slow",   .model = "slow-model"}, 500);
    engine.record_latency({.policy = "p", .provider = "medium", .model = "med-model"},  200);
    engine.record_latency({.policy = "p", .provider = "fast",   .model = "fast-model"}, 50);

    const auto chain = engine.route_chain({.prompt = "any", .has_tool_messages = false});
    REQUIRE(!chain.empty());
    // The latency winner should be first
    REQUIRE(chain[0].provider == "fast");
    // All three should appear in the chain
    REQUIRE(chain.size() == 3);
}

TEST_CASE("route_chain: policy defaults appended after rule candidates", "[routing][chain]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [{ "provider": "default-prov", "model": "d-model" }],
                "rules": [
                    {
                        "name": "keyword-rule",
                        "priority": 10,
                        "strategy": "fallback",
                        "when": { "any_keywords": ["special"] },
                        "candidates": [{ "provider": "specialist", "model": "spec-model" }]
                    }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"specialist", "default-prov"}));

    const auto chain = engine.route_chain({.prompt = "do something special", .has_tool_messages = false});
    // chain[0] = specialist (matched rule), chain[1] = default-prov (last-resort default)
    REQUIRE(chain.size() == 2);
    REQUIRE(chain[0].provider == "specialist");
    REQUIRE(chain[1].provider == "default-prov");
}

TEST_CASE("route_chain: retries field propagated from candidate config", "[routing][chain]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [
                    { "provider": "a", "model": "a-model", "retries": 3 },
                    { "provider": "b", "model": "b-model", "retries": 0 }
                ]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"a", "b"}));

    const auto chain = engine.route_chain({.prompt = "task", .has_tool_messages = false});
    REQUIRE(chain.size() == 2);
    REQUIRE(chain[0].retries == 3);
    REQUIRE(chain[1].retries == 0);
}

TEST_CASE("route_chain: empty when no providers available", "[routing][chain]") {
    const std::string json = R"({
        "enabled": true,
        "default_policy": "p",
        "policies": {
            "p": {
                "strategy": "fallback",
                "defaults": [{ "provider": "ghost", "model": "ghost-model" }]
            }
        }
    })";

    auto config = parse_router_from_json(json);
    // "ghost" is not registered
    core::llm::routing::RouterEngine engine(
        std::move(config),
        providers({"some-other-provider"}));

    const auto chain = engine.route_chain({.prompt = "task", .has_tool_messages = false});
    REQUIRE(chain.empty());
}
