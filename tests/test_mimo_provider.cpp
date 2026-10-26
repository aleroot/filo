#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/auth/AuthenticationManager.hpp"
#include "core/auth/ICredentialSource.hpp"
#include "core/auth/MimoAuthenticationStrategy.hpp"
#include "core/auth/MimoOAuthFlow.hpp"
#include "core/budget/BudgetTracker.hpp"
#include "core/budget/TokenAccounting.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/MimoModelTraits.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/ProviderCatalogGrouping.hpp"
#include "core/llm/ProviderDefinition.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/ReasoningEffortResolution.hpp"
#include "core/llm/StrictToolPolicy.hpp"
#include "core/llm/ToolCallAssembly.hpp"
#include "core/llm/protocols/MimoProtocol.hpp"
#include "core/utils/Base64.hpp"
#include "core/utils/Crypto.hpp"
#include "core/version/Version.hpp"
#include "core/tools/WebBackendAdapters.hpp"
#include "TestSessionContext.hpp"

#include <cpr/cpr.h>

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

using namespace core::llm;
using namespace core::llm::protocols;

TEST_CASE("MiMo web search recognizes pay-as-you-go and regional Token Plan providers",
          "[mimo][web]") {
    const auto backend = core::tools::web::make_mimo_web_search_backend();
    for (const auto name : {"mimo", "mimo-token-plan", "mimo-token-plan-sgp", "mimo-token-plan-cn"}) {
        CAPTURE(name);
        core::config::ProviderConfig config;
        config.api_key = "test-key";
        config.model = "mimo-v2.6-pro";
        const auto provider = ProviderFactory::create_provider(name, config);
        REQUIRE(provider);
        CHECK(backend->supports({.session_context = test_support::make_workspace_session_context(),
                                 .provider = provider}));
    }
    CHECK_FALSE(backend->supports({.session_context = test_support::make_workspace_session_context()}));
    core::config::ProviderConfig config;
    config.api_key = "test-key";
    config.model = "gpt-5.5";
    CHECK_FALSE(backend->supports({.session_context = test_support::make_workspace_session_context(),
        .provider = ProviderFactory::create_provider("openai", config)}));
}

namespace {

ChatRequest make_mimo_request(std::string model = "mimo-v2.6-pro",
                              std::string user_text = "Hello") {
    ChatRequest req;
    req.model = std::move(model);
    req.stream = true;
    req.messages.push_back(Message{.role = "user", .content = std::move(user_text)});
    return req;
}

cpr::Header mimo_headers() {
    core::auth::AuthInfo auth;
    auth.headers["api-key"] = "tp-test";
    return MimoProtocol().build_headers(auth);
}

std::string format_error(int status, std::string_view body) {
    const cpr::Header headers;
    const HttpResponse response{
        .status_code = status,
        .body = body,
        .headers = headers,
    };
    return MimoProtocol().format_error_message(response);
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Provider definitions — endpoints, billing, and credential lookup
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("MiMo Token Plan presets resolve their regional gateways",
          "[mimo][provider_definition]") {
    const auto* europe = find_builtin_provider_definition("mimo-token-plan");
    REQUIRE(europe != nullptr);
    CHECK(europe->base_url == "https://token-plan-ams.xiaomimimo.com/v1");
    CHECK(europe->billing_kind == core::auth::BillingKind::Subscription);
    CHECK(europe->env_var() == "MIMO_TOKEN_PLAN_API_KEY");
    CHECK(europe->auth_style == ProviderAuthStyle::ApiKey);
    CHECK(europe->registry_provider == "mimo");
    CHECK(europe->catalog_group == "xiaomi");
    CHECK(europe->usage_dashboard_url
          == "https://platform.xiaomimimo.com/console/plan-manage");
    CHECK(europe->has_usage_dashboard());
    CHECK(europe->display_name_or_prefix() == "mimo-token-plan");
    CHECK(core::llm::provider_display_name("mimo-token-plan-sgp")
          == "mimo-token-plan");

    const auto* singapore = find_builtin_provider_definition("mimo-token-plan-sgp");
    REQUIRE(singapore != nullptr);
    CHECK(singapore->base_url == "https://token-plan-sgp.xiaomimimo.com/v1");
    CHECK(singapore->usage_dashboard_url
          == "https://platform.xiaomimimo.com/console/plan-manage");

    const auto* china = find_builtin_provider_definition("mimo-token-plan-cn");
    REQUIRE(china != nullptr);
    CHECK(china->base_url == "https://token-plan-cn.xiaomimimo.com/v1");
    CHECK(china->usage_dashboard_url
          == "https://platform.xiaomimimo.com/console/plan-manage");
}

TEST_CASE("The bare mimo preset stays pay-as-you-go", "[mimo][provider_definition]") {
    const auto* definition = find_builtin_provider_definition("mimo");
    REQUIRE(definition != nullptr);
    CHECK(definition->base_url == "https://api.xiaomimimo.com/v1");
    CHECK(definition->billing_kind == core::auth::BillingKind::Metered);
    CHECK(definition->env_var() == "XIAOMI_API_KEY");
    // The reference client's variable name is accepted as an alias.
    CHECK(definition->env_vars[1] == "MIMO_API_KEY");
}

TEST_CASE("mimo-token-plan-ams falls through to the Europe preset",
          "[mimo][provider_definition]") {
    const auto* definition = find_builtin_provider_definition("mimo-token-plan-ams");
    REQUIRE(definition != nullptr);
    CHECK(definition->base_url == "https://token-plan-ams.xiaomimimo.com/v1");
}

TEST_CASE("ProviderFactory creates MiMo providers", "[mimo][factory]") {
    core::config::ProviderConfig config;
    config.model = "mimo-v2.6-pro";
    CHECK(ProviderFactory::create_provider("mimo", config) != nullptr);
    CHECK(ProviderFactory::create_provider("mimo-token-plan", config) != nullptr);
    CHECK(ProviderFactory::create_provider("mimo-token-plan-cn", config) != nullptr);
}

TEST_CASE("A user-defined provider on a MiMo host speaks the MiMo contract",
          "[mimo][factory]") {
    core::config::ProviderConfig config;
    config.api_type = core::config::ApiType::OpenAI;
    config.base_url = "https://token-plan-sgp.xiaomimimo.com/v1";
    config.api_key = "tp-test";
    config.model = "mimo-v2.6-pro";
    const auto provider = ProviderFactory::create_provider("my-mimo", config);
    REQUIRE(provider);
    CHECK(provider->reasoning_capabilities("mimo-v2.6-pro")
              .supports(ReasoningCapability::ToggleOnly));
}

TEST_CASE("Xiaomi is the login id; mimo remains an alias", "[mimo][auth]") {
    auto manager = core::auth::AuthenticationManager::create_with_defaults("/tmp");
    const auto providers = manager.available_login_providers();
    REQUIRE(std::ranges::find(providers, std::string("xiaomi")) != providers.end());
    CHECK(std::ranges::find(providers, std::string("mimo")) == providers.end());
}

TEST_CASE("MiMo authentication strategy accepts regional login aliases",
          "[mimo][auth]") {
    const auto strategy = core::auth::make_mimo_authentication_strategy();
    CHECK(strategy->login_provider() == "xiaomi");

    const auto aliases = strategy->login_aliases();
    CHECK(std::ranges::find(aliases, std::string_view("mimo")) != aliases.end());
    CHECK(std::ranges::find(aliases, std::string_view("mimo-token-plan-ams"))
          != aliases.end());
    CHECK(std::ranges::find(aliases, std::string_view("mimocode"))
          != aliases.end());
}

// ─────────────────────────────────────────────────────────────────────────────
// Region traits
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("MiMo regions map to gateways and presets", "[mimo][traits]") {
    CHECK(mimo_token_plan_endpoint(MimoRegion::Europe)
          == "https://token-plan-ams.xiaomimimo.com/v1");
    CHECK(mimo_region_provider_name(MimoRegion::Singapore) == "mimo-token-plan-sgp");
    CHECK(mimo_region_provider_name(MimoRegion::Europe) == "mimo-token-plan");

    CHECK(mimo_region_for_endpoint("https://token-plan-cn.xiaomimimo.com/v1")
          == MimoRegion::China);
    CHECK(is_mimo_token_plan_endpoint("https://token-plan-sgp.xiaomimimo.com/v1"));
    // Pay-as-you-go is a MiMo host but not a plan gateway.
    CHECK_FALSE(is_mimo_token_plan_endpoint("https://api.xiaomimimo.com/v1"));
    CHECK(is_mimo_endpoint("https://api.xiaomimimo.com/v1"));
    CHECK_FALSE(is_mimo_endpoint("https://api.openai.com/v1"));
    CHECK_FALSE(is_mimo_endpoint("https://evilxiaomimimo.com/v1"));
}

TEST_CASE("MiMo model detection requires the family separator", "[mimo][traits]") {
    CHECK(is_mimo_model("mimo-v2.6-pro"));
    CHECK(is_mimo_model("MiMo-V2-Flash"));
    CHECK_FALSE(is_mimo_model("mimosa-7b"));
    CHECK_FALSE(is_mimo_model("gpt-5.6-sol"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Protocol — headers, temperature, timeouts, reasoning
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("MiMo sends its client source header", "[mimo][headers]") {
    const cpr::Header headers = mimo_headers();
    REQUIRE(headers.count("X-Mimo-Source") == 1);
    CHECK(headers.at("X-Mimo-Source") == "mimocode-cli");
    CHECK(headers.at("User-Agent") == std::string(core::version::user_agent));
    CHECK(headers.at("api-key") == "tp-test");
}

TEST_CASE("MiMo pins reasoning models to temperature 1", "[mimo][serializer]") {
    const std::string payload = MimoProtocol().serialize(make_mimo_request());
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("temperature":1)"));
}

TEST_CASE("An explicit temperature overrides the MiMo default",
          "[mimo][serializer]") {
    ChatRequest req = make_mimo_request();
    req.temperature = 0.25F;
    const std::string payload = MimoProtocol().serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("temperature":0.25)"));
}

TEST_CASE("MiMo requests stream usage", "[mimo][serializer]") {
    REQUIRE_THAT(MimoProtocol().serialize(make_mimo_request()),
                 Catch::Matchers::ContainsSubstring(
                     R"("stream_options":{"include_usage":true})"));
}

TEST_CASE("MiMo exposes thinking as an on/off switch", "[mimo][serializer][reasoning]") {
    const auto capabilities = MimoProtocol().reasoning_capabilities("mimo-v2.6-pro");
    CHECK(capabilities.supports(ReasoningCapability::ToggleOnly));
    CHECK(capabilities.supports(ReasoningCapability::Disable));
    // There is no effort scale, so no level may be offered as one.
    CHECK_FALSE(capabilities.supports_effort());
    CHECK(MimoProtocol().reasoning_capabilities("gpt-5.6-sol").empty());
    CHECK(resolve_reasoning_effort("high", capabilities).effective == "on");
    CHECK(resolve_reasoning_effort("off", capabilities).effective == "off");
}

TEST_CASE("MiMo sends the thinking opt-out only when thinking is off",
          "[mimo][serializer][reasoning]") {
    constexpr std::string_view kDisabled = R"("thinking":{"type":"disabled"})";
    ChatRequest req = make_mimo_request();

    SECTION("the server default stays implicit") {
        CHECK_THAT(MimoProtocol().serialize(req), !Catch::Matchers::ContainsSubstring("thinking"));
    }
    SECTION("an enabled level never becomes reasoning_effort") {
        req.effort = "high";
        const std::string payload = MimoProtocol().serialize(req);
        CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring("thinking"));
        CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring("reasoning_effort"));
    }
    SECTION("effort off disables thinking") {
        req.effort = "off";
        CHECK_THAT(MimoProtocol().serialize(req), Catch::Matchers::ContainsSubstring(std::string(kDisabled)));
    }
    SECTION("a provider default of off applies to requests that name no effort") {
        CHECK_THAT(MimoProtocol("none").serialize(req),
                   Catch::Matchers::ContainsSubstring(std::string(kDisabled)));
        req.effort = "high";
        CHECK_THAT(MimoProtocol("none").serialize(req), !Catch::Matchers::ContainsSubstring("thinking"));
    }
    SECTION("JSON helper calls do not spend their budget thinking") {
        req.response_format = {.type = ResponseFormat::Type::JsonObject};
        req.max_tokens = 1400;
        CHECK_THAT(MimoProtocol("high").serialize(req),
                   Catch::Matchers::ContainsSubstring(std::string(kDisabled)));
        // An explicit request level still wins.
        req.effort = "high";
        CHECK_THAT(MimoProtocol().serialize(req), !Catch::Matchers::ContainsSubstring("thinking"));
    }
}

TEST_CASE("MiMo caps output with the documented max_completion_tokens field",
          "[mimo][serializer]") {
    ChatRequest req = make_mimo_request();
    req.max_tokens = 1400;
    const std::string payload = MimoProtocol().serialize(req);
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(R"("max_completion_tokens":1400)"));
    CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("max_tokens")"));
}

TEST_CASE("MiMo keeps per-turn state behind the cached prefix",
          "[mimo][serializer][prompt-cache]") {
    const auto request_with_state = [](std::string_view state) {
        ChatRequest req = make_mimo_request();
        req.prompt_plan.append({
            .kind = core::context::ContextLayerKind::RuntimeInstructions,
            .stability = core::context::PromptStability::Stable,
            .name = "runtime", .content = "STABLE_RUNTIME",
        });
        req.prompt_plan.append({
            .kind = core::context::ContextLayerKind::ConversationState,
            .stability = core::context::PromptStability::Dynamic,
            .name = "conversation_state", .content = std::string(state),
        });
        return req;
    };
    const std::string first = MimoProtocol().serialize(request_with_state("TASK_ONE"));
    const std::string second = MimoProtocol().serialize(request_with_state("TASK_TWO"));

    const auto tail_at = [](const std::string& payload) {
        return payload.find(R"({"role":"system","content":"TASK_)");
    };
    REQUIRE(tail_at(first) != std::string::npos);
    // Byte-identical up to the dynamic tail, so the automatic prefix cache hits.
    CHECK(first.substr(0, tail_at(first)) == second.substr(0, tail_at(second)));
    CHECK(first.find(R"({"role":"system","content":"STABLE_RUNTIME"})") < first.find(R"("role":"user")"));
    CHECK(first.find(R"("role":"user")") < tail_at(first));
}

TEST_CASE("MiMo tool schemas can opt into strict mode", "[mimo][strict]") {
    CHECK(model_supports_strict_tools(ToolSchemaWire::OpenAI, "mimo-v2.6-pro"));
    CHECK(model_supports_strict_tools(ToolSchemaWire::OpenAI, "MiMo-V2.6-Flash"));
    CHECK_FALSE(model_supports_strict_tools(ToolSchemaWire::OpenAI, "mimosa-7b"));
}

TEST_CASE("MiMo widens the stream idle budget", "[mimo][timeouts]") {
    const auto timeouts = MimoProtocol().stream_timeouts();
    CHECK(timeouts.response_start == std::chrono::seconds(300));
    CHECK(timeouts.inactivity == std::chrono::seconds(480));
}

TEST_CASE("MiMo streams reasoning_content through the OpenAI base",
          "[mimo][parse]") {
    MimoProtocol protocol;
    const auto result = protocol.parse_event(
        R"(data: {"choices":[{"delta":{"reasoning_content":"thinking…"}}]})");
    REQUIRE_FALSE(result.chunks.empty());
    CHECK(result.chunks.front().reasoning_content == "thinking…");
    // Provenance tag lets the serializer replay it on later turns.
    CHECK(result.chunks.front().reasoning_protocol == "mimo");
}

TEST_CASE("MiMo replays assistant reasoning on follow-up turns",
          "[mimo][serializer][reasoning_replay]") {
    ChatRequest req = make_mimo_request("mimo-v2.6-pro", "Continue");

    Message assistant;
    assistant.role = "assistant";
    assistant.content = "I will inspect the file first.";
    assistant.reasoning_content = "The user wants a change; start by reading.";
    assistant.reasoning_protocol = "mimo";
    ToolCall call{
        .index = 0,
        .id = "call_abc",
        .type = "function",
        .function = {.name = "read", .arguments = R"({"path":"a.cpp"})"},
    };
    assistant.tool_calls = {call};

    Message tool_result;
    tool_result.role = "tool";
    tool_result.content = R"({"content":"int main() {}"})";
    tool_result.tool_call_id = "call_abc";

    req.messages = {
        Message{.role = "user", .content = "Fix the build."},
        assistant,
        tool_result,
    };

    const std::string payload = MimoProtocol().serialize(req);
    REQUIRE_THAT(payload,
                 Catch::Matchers::ContainsSubstring(
                     R"("reasoning_content":"The user wants a change; start by reading.")"));
    REQUIRE_THAT(payload,
                 Catch::Matchers::ContainsSubstring(R"("tool_call_id":"call_abc")"));
}

TEST_CASE("MiMo emits reasoning_content even when an assistant has none",
          "[mimo][serializer][reasoning_replay]") {
    // The reference client always sets the field on assistant messages —
    // "some providers may return empty reasoning_content which still needs to
    // be sent back in subsequent requests".
    ChatRequest req = make_mimo_request();
    Message plain;
    plain.role = "assistant";
    plain.content = "Done.";
    req.messages.push_back(plain);

    REQUIRE_THAT(MimoProtocol().serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("reasoning_content":"")"));
}

TEST_CASE("MiMo does not leak reasoning produced by another protocol",
          "[mimo][serializer][reasoning_replay]") {
    ChatRequest req = make_mimo_request();
    Message foreign;
    foreign.role = "assistant";
    foreign.content = "ok";
    foreign.reasoning_content = "secret chain of thought from another wire";
    foreign.reasoning_protocol = "some-other-protocol";
    req.messages.push_back(foreign);

    const std::string payload = MimoProtocol().serialize(req);
    CHECK_THAT(payload,
               !Catch::Matchers::ContainsSubstring("secret chain of thought"));
    REQUIRE_THAT(payload,
                 Catch::Matchers::ContainsSubstring(R"("reasoning_content":"")"));
}

TEST_CASE("MiMo-style tool-call deltas assemble regardless of field order",
          "[mimo][tool_stream]") {
    // Xiaomi patches the AI SDK because their gateway streams tool calls
    // whose id and name do not arrive on the first delta. Filo's per-index
    // merge must tolerate arguments-first fragments and late ids.
    std::vector<ToolCall> accumulated;

    ToolCall args_only;
    args_only.index = 0;
    args_only.function.arguments = R"({"pa)";
    merge_tool_call_fragment(accumulated, args_only);

    ToolCall more_args;
    more_args.index = 0;
    more_args.function.arguments = R"(th":"src"})";
    merge_tool_call_fragment(accumulated, more_args);

    ToolCall late_identity;
    late_identity.index = 0;
    late_identity.id = "call_late_1";
    late_identity.function.name = "read";
    merge_tool_call_fragment(accumulated, late_identity);

    REQUIRE(accumulated.size() == 1);
    CHECK(accumulated[0].id == "call_late_1");
    CHECK(accumulated[0].function.name == "read");
    CHECK(accumulated[0].function.arguments == R"({"path":"src"})");
}

// ─────────────────────────────────────────────────────────────────────────────
// Gateway errors — moderation/risk-control arrive as HTTP 400
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("MiMo relabels moderation blocks hidden under HTTP 400",
          "[mimo][errors]") {
    const std::string message = format_error(
        400,
        R"({"error":{"code":421,"message":"Bad Request","param":"prompt"}})");
    CHECK_THAT(message,
               Catch::Matchers::ContainsSubstring("content moderation"));
    // error.param carries the actionable reason.
    CHECK_THAT(message, Catch::Matchers::ContainsSubstring("prompt"));
}

TEST_CASE("MiMo relabels risk-control blocks", "[mimo][errors]") {
    CHECK_THAT(format_error(400, R"({"error":{"code":"441","message":"Bad Request"}})"),
               Catch::Matchers::ContainsSubstring("risk control"));
}

TEST_CASE("MiMo auth failures distinguish key types and regions", "[mimo][errors]") {
    const std::string message = format_error(401, R"({"error":{"message":"no"}})");
    CHECK_THAT(message,
               Catch::Matchers::ContainsSubstring("tp-"));
    CHECK_THAT(message,
               Catch::Matchers::ContainsSubstring("Base URL"));
    CHECK_THAT(message, Catch::Matchers::ContainsSubstring("filo --auth xiaomi"));
}

TEST_CASE("MiMo quota exhaustion points at the console", "[mimo][errors]") {
    CHECK_THAT(format_error(429, "{}"),
               Catch::Matchers::ContainsSubstring("platform.xiaomimimo.com"));
}

TEST_CASE("MiMo stops retrying only when the plan quota is provably exhausted",
          "[mimo][errors][retry]") {
    const cpr::Header headers;
    const auto retryable = [&](int status, std::string_view body) {
        return MimoProtocol().is_retryable(
            HttpResponse{.status_code = status, .body = body, .headers = headers});
    };
    constexpr std::string_view kExhausted =
        R"({"error":{"code":"insufficient_quota","message":"quota exceeded"}})";
    // A rate limit and an exhausted plan share 429; only the code separates them.
    CHECK(retryable(429, R"({"error":{"code":"429","message":"Too many requests","type":"limitation"}})"));
    CHECK(retryable(429, "not json"));
    CHECK_FALSE(retryable(429, kExhausted));
    CHECK_FALSE(retryable(429, R"({"error":{"type":"insufficient_quota"}})"));
    CHECK(retryable(503, "{}"));

    const std::string message = format_error(429, kExhausted);
    CHECK_THAT(message, Catch::Matchers::ContainsSubstring("quota exhausted"));
    CHECK_THAT(message, !Catch::Matchers::ContainsSubstring("Rate limited"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Model registry and accounting
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("MiMo flagship models are registered", "[mimo][registry]") {
    auto& registry = ModelRegistry::instance();
    const auto pro = registry.lookup("mimo-v2.6-pro");
    REQUIRE(pro != nullptr);
    CHECK(pro->provider == "mimo");
    CHECK(pro->context_window == 1'048'576);
    CHECK(pro->max_output_tokens == 131'072);
    CHECK(has_capability(pro->capabilities, ModelCapability::Reasoning));
    CHECK(has_capability(pro->capabilities, ModelCapability::Vision));
    CHECK(has_capability(pro->capabilities, ModelCapability::PdfInput));

    const auto flash = registry.lookup("mimo-v2.6-flash");
    REQUIRE(flash != nullptr);
    CHECK(flash->context_window == 1'048'576);

    // The retired V2 and V2.5 lines are not offered.
    for (const auto retired : {"mimo-v2.5-pro", "mimo-v2.5", "mimo-v2-pro", "mimo-v2-omni",
                               "mimo-v2-flash", "mimo-v2.5-pro-ultraspeed"}) {
        CAPTURE(retired);
        const auto info = registry.lookup(retired);
        CHECK((info == nullptr || info->provider != "mimo"));
    }
}

TEST_CASE("MiMo context windows are known to the budget tracker",
          "[mimo][accounting]") {
    CHECK(core::budget::context_window_for_model("mimo-v2.6-pro") == 1'048'576);
    CHECK(core::budget::context_window_for_model("mimo-v2.6-flash") == 1'048'576);
}

TEST_CASE("MiMo pay-as-you-go cost estimates price cache hits", "[mimo][accounting]") {
    const TokenUsage cold{.prompt_tokens = 1'000'000, .completion_tokens = 0};
    const TokenUsage warm{.prompt_tokens = 1'000'000, .completion_tokens = 0,
                          .cached_prompt_tokens = 1'000'000};
    CHECK(core::budget::estimate_usage_cost(cold, "mimo-v2.6-pro") == Catch::Approx(0.435));
    CHECK(core::budget::estimate_usage_cost(warm, "mimo-v2.6-pro") == Catch::Approx(0.0036));
    CHECK(core::budget::rates_for_model("mimo-v2.6-pro-ultraspeed").output_per_m
          == Catch::Approx(8.70));
}

TEST_CASE("MiMo Token Plan credits follow the plan's rate card", "[mimo][accounting]") {
    using namespace std::chrono;
    // 2026-10-26 04:00 UTC is 12:00 in Beijing (peak); 18:00 UTC is 02:00 (off-peak).
    const auto peak = sys_days{2026y / October / 26} + 4h;
    const auto off_peak = sys_days{2026y / October / 26} + 18h;
    CHECK_FALSE(is_mimo_off_peak(peak));
    CHECK(is_mimo_off_peak(off_peak));
    CHECK_FALSE(is_mimo_off_peak(sys_days{2026y / October / 26} + 0h));  // 08:00 Beijing
    CHECK(is_mimo_off_peak(sys_days{2026y / October / 26} + 16h));       // 00:00 Beijing

    const TokenUsage usage{.prompt_tokens = 1'000, .completion_tokens = 100,
                           .cached_prompt_tokens = 800};
    // 800 hits x 2.5 + 200 misses x 300 + 100 output x 600.
    CHECK(mimo_token_plan_credits(usage, "mimo-v2.6-pro", peak) == Catch::Approx(122'000.0));
    CHECK(mimo_token_plan_credits(usage, "mimo-v2.6-pro", off_peak) == Catch::Approx(97'600.0));
    // 800 x 2 + 200 x 100 + 100 x 200.
    CHECK(mimo_token_plan_credits(usage, "mimo-v2.6-flash", peak) == Catch::Approx(41'600.0));
    // UltraSpeed is not served on the plan.
    CHECK(mimo_token_plan_credits(usage, "mimo-v2.6-pro-ultraspeed", peak) == 0.0);
}

TEST_CASE("Plan credits accumulate only for plan-billed calls", "[mimo][accounting]") {
    auto& tracker = core::budget::BudgetTracker::get_instance();
    const std::string session = "mimo-plan-credits-test";
    tracker.reset_session(session);
    const TokenUsage usage{.prompt_tokens = 1'000, .completion_tokens = 0, .total_tokens = 1'000};
    tracker.record_event({.session_id = session, .model = "mimo-v2.6-flash", .usage = usage,
                          .should_estimate_cost = false, .billable = false});
    tracker.record_event({.session_id = session, .model = "mimo-v2.6-flash", .usage = usage});
    const double credits = tracker.session_plan_credits(session);
    // One plan call of 1000 cache misses at 100 credits, possibly off-peak.
    CHECK((credits == Catch::Approx(100'000.0) || credits == Catch::Approx(80'000.0)));
    tracker.reset_session(session);
}

// ─────────────────────────────────────────────────────────────────────────────
// Picker grouping — Token Plan before pay-as-you-go, disjoint model sets
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("MiMo presets group under the vendor name for the picker",
          "[mimo][catalog]") {
    CHECK(provider_catalog_group_name("mimo") == "xiaomi");
    CHECK(provider_catalog_group_name("mimo-token-plan") == "xiaomi");
    CHECK(provider_catalog_group_name("mimo-token-plan-sgp") == "xiaomi");

    const std::vector<std::string> configured{"mimo", "mimo-token-plan"};
    const auto groups = provider_catalog_groups(configured);
    REQUIRE(groups.size() == 1);
    CHECK(groups.front().provider_name == "xiaomi");
}

TEST_CASE("MiMo keeps Token Plan and pay-as-you-go as separate sources", "[mimo][catalog]") {
    const std::vector<std::string> configured{"mimo", "mimo-token-plan"};
    const auto group = provider_catalog_group_for("xiaomi", configured);

    REQUIRE(group.sources.size() == 2);
    const auto* plan = group.find_source("mimo-token-plan");
    const auto* api = group.find_source("mimo");
    REQUIRE(plan != nullptr);
    REQUIRE(api != nullptr);
    CHECK(plan->category_label == "Token Plan · Europe.");

    // A plan key is rejected by api.xiaomimimo.com, so plan models must not be
    // offered under the pay-as-you-go source.
    CHECK(plan->registry_model_filter.matches("mimo-v2.6-pro"));
    CHECK_FALSE(api->registry_model_filter.matches("mimo-v2.6-pro"));
    // UltraSpeed is pay-as-you-go only.
    CHECK(api->registry_model_filter.matches("mimo-v2.6-pro-ultraspeed"));
    CHECK_FALSE(plan->registry_model_filter.matches("mimo-v2.6-pro-ultraspeed"));
}

TEST_CASE("MiMo picker keeps every Token Plan endpoint separate",
          "[mimo][catalog]") {
    const std::vector<std::string> configured{
        "mimo", "mimo-token-plan", "mimo-token-plan-sgp"};
    const auto group = provider_catalog_group_for("xiaomi", configured);

    REQUIRE(group.sources.size() == 3);
    const auto* europe = group.find_source("mimo-token-plan");
    const auto* singapore = group.find_source("mimo-token-plan-sgp");
    REQUIRE(europe != nullptr);
    REQUIRE(singapore != nullptr);
    CHECK(europe->category_label == "Token Plan · Europe.");
    CHECK(singapore->category_label == "Token Plan · Singapore.");
    CHECK(group.find_source("mimo") != nullptr);
    CHECK(group.contains_source_provider("mimo-token-plan-sgp"));
    CHECK(provider_catalog_selection_key(europe->service_id, "mimo-v2.6-pro")
          != provider_catalog_selection_key(singapore->service_id, "mimo-v2.6-pro"));

    const auto from_region = provider_catalog_group_for("mimo-token-plan-sgp", configured);
    REQUIRE(from_region.sources.size() == group.sources.size());
    CHECK(from_region.find_source("mimo-token-plan-sgp") != nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Browser login — sealed-box envelope
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("MiMo authorize URL carries an SPKI public key", "[mimo][auth]") {
    core::auth::MimoOAuthFlow flow;
    flow.set_key_name("filo-cli-key-abcd1234");

    const std::string pk = flow.public_key_parameter();
    const auto decoded = core::utils::Base64::decode_url(pk);
    REQUIRE(decoded.has_value());
    // 12-byte RFC 8410 SPKI prefix + the 32-byte X25519 key.
    REQUIRE(decoded->size() == 44);
    for (std::size_t i = 0; i < core::auth::kX25519SpkiPrefix.size(); ++i) {
        CHECK(static_cast<std::uint8_t>((*decoded)[i])
              == core::auth::kX25519SpkiPrefix[i]);
    }

    const std::string url = flow.authorize_url("http://localhost:52001/");
    CHECK_THAT(url, Catch::Matchers::ContainsSubstring("/authorize?pk="));
    CHECK_THAT(url, Catch::Matchers::ContainsSubstring("kn=mimocode"));
    CHECK_THAT(url,
               Catch::Matchers::ContainsSubstring("key_name=filo-cli-key-abcd1234"));
    CHECK_THAT(url,
               Catch::Matchers::ContainsSubstring(
                   "redirect_uri=http%3A%2F%2Flocalhost%3A52001%2F"));
}

namespace {

// Golden sealed envelope produced by an independent implementation of the
// console's scheme (ephemeral X25519 + SHA-256(ECDH) + AES-256-GCM, laid out
// as ephemeral_public(32) || nonce(12) || ciphertext || tag(16)).
//
// Keys are the RFC 7748 §6.1 pair so the derivation is externally checkable:
// the shared secret is 4a5d9d5b…1e161742.
constexpr std::string_view kGoldenClientPrivateKeyHex =
    "77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a";

constexpr std::string_view kGoldenSealedEnvelope =
    "3p7bfXt9wbTTW2HC7OQ1Nz-DQ8hbeGdNrfx-FG-IK08BAgMEBQYHCAkKCwyuxxcUQX2cILNS"
    "cLbHMwB-6u2i2BnCfKkpL2__9gCazPFvayUK_RMFC779_Z_nu59v7umzVTlODYIUMkMjP5GI"
    "E5Ye7DLlyPCYezrGVOUV3ibroZ9_-WcfAIKr0D6av7lLwXR3q3F7rktQcic";

std::array<std::uint8_t, 32> golden_client_private_key() {
    std::array<std::uint8_t, 32> key{};
    for (std::size_t i = 0; i < key.size(); ++i) {
        key[i] = static_cast<std::uint8_t>(
            std::stoul(std::string(kGoldenClientPrivateKeyHex.substr(i * 2, 2)),
                       nullptr, 16));
    }
    return key;
}

} // namespace

TEST_CASE("MiMo decrypts a sealed grant produced by the console's scheme",
          "[mimo][auth][crypto]") {
    const auto grant = core::auth::mimo_decrypt_sealed_grant(
        golden_client_private_key(), kGoldenSealedEnvelope);

    REQUIRE(grant.has_value());
    CHECK(grant->api_key == "sk-mimo-golden-key");
    CHECK(grant->user_id == "user-42");
    CHECK(grant->base_url == "https://token-plan-ams.xiaomimimo.com/v1");
}

TEST_CASE("A sealed grant does not open under the wrong key",
          "[mimo][auth][crypto]") {
    auto wrong_key = golden_client_private_key();
    // Flip a bit that survives X25519 clamping — the low three bits of byte 0
    // are cleared by `k[0] &= 248`, so touching those would be a no-op.
    wrong_key[1] ^= 0x01;
    CHECK_FALSE(
        core::auth::mimo_decrypt_sealed_grant(wrong_key, kGoldenSealedEnvelope)
            .has_value());
}

TEST_CASE("A tampered sealed grant fails authentication",
          "[mimo][auth][crypto]") {
    std::string tampered(kGoldenSealedEnvelope);
    // Flip a ciphertext byte well past the ephemeral key and nonce.
    tampered[80] = tampered[80] == 'A' ? 'B' : 'A';
    CHECK_FALSE(core::auth::mimo_decrypt_sealed_grant(
                    golden_client_private_key(), tampered)
                    .has_value());
}

TEST_CASE("MiMo rejects malformed sealed grants", "[mimo][auth]") {
    const core::auth::MimoOAuthFlow flow;
    CHECK_FALSE(flow.decrypt_grant("").has_value());
    CHECK_FALSE(flow.decrypt_grant("not-base64url!!").has_value());
    // Shorter than ephemeral key + nonce + tag.
    CHECK_FALSE(flow.decrypt_grant(core::utils::Base64::encode_url(
                                       std::vector<std::uint8_t>(32, 0x01)))
                    .has_value());
}
