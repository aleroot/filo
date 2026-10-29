#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <simdjson.h>
#include <algorithm>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <atomic>
#include <future>
#include <charconv>
#include <cmath>
#include "core/tools/WebBackendAdapters.hpp"
#include "TestSessionContext.hpp"
#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/llm/protocols/ClaudeProtocol.hpp"
#include "core/llm/protocols/MiniMaxProtocol.hpp"
#include "core/llm/MiniMaxModelTraits.hpp"
#include "core/llm/MiniMaxUsageClient.hpp"
#include "core/llm/ProviderDefinition.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/ProviderCatalogGrouping.hpp"
#include "core/llm/ModelCatalogProvider.hpp"
#include "core/auth/AuthenticationManager.hpp"
#include "core/budget/TokenAccounting.hpp"
#include "core/llm/providers/MiniMaxProvider.hpp"
#include "core/llm/providers/MiniMaxRequestBudget.hpp"
#include "core/commands/SkillTurnResolver.hpp"
#include "core/auth/ApiKeyPromptStrategy.hpp"
#include "core/auth/MiniMaxClientIdentity.hpp"

using namespace core::llm;
using namespace core::llm::protocols;
using Catch::Matchers::ContainsSubstring;

namespace {
struct MiniMaxEnv {
    std::vector<std::pair<std::string, std::optional<std::string>>> saved;
    MiniMaxEnv() {
        for (const auto name : {"MINIMAX_API_KEY", "MINIMAX_M_PLAN_API_KEY", "MINIMAX_SUBSCRIPTION_KEY"}) {
            const char* value = std::getenv(name);
            saved.push_back({name, value ? std::optional<std::string>{value} : std::nullopt});
            unsetenv(name);
        }
    }
    ~MiniMaxEnv() {
        for (const auto& [name, value] : saved) {
            if (value) setenv(name.c_str(), value->c_str(), 1); else unsetenv(name.c_str());
        }
    }
};
ChatRequest request(std::string model = "MiniMax-M3.1-Flash") {
    ChatRequest req;
    req.model = std::move(model);
    req.messages = {{.role = "system", .content = "Project instructions"}, {.role = "user", .content = "Hello"}};
    return req;
}
void valid_json(std::string_view payload) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(payload);
    simdjson::dom::element doc;
    REQUIRE(parser.parse(padded).get(doc) == simdjson::SUCCESS);
}
class UsageFixture final : public IMiniMaxUsageSource {
public:
    int reads = 0;
    std::optional<RateLimitInfo> read(std::string_view, const cpr::Header&) override {
        ++reads;
        RateLimitInfo info;
        info.usage_windows.push_back({"5h", 0.25F, 100});
        return info;
    }
};
}

TEST_CASE("Messages defaults are neutral even when a Claude model is named", "[anthropic-neutral][minimax]") {
    AnthropicProtocol protocol;
    auto req = request("sonnet[1m]");
    req.effort = "max";
    protocol.prepare_request(req);
    CHECK(req.model == "sonnet[1m]");
    const auto payload = protocol.serialize(req);
    valid_json(payload);
    CHECK_THAT(payload, ContainsSubstring("Project instructions"));
    for (const auto field : {"billing", "claude-code", "cache_control", "thinking", "output_config"}) CHECK_THAT(payload, !ContainsSubstring(field));
    core::auth::AuthInfo auth;
    auth.headers["x-api-key"] = "neutral-key";
    auth.properties["oauth"] = "1";
    const auto headers = protocol.build_headers(auth);
    CHECK(headers.at("x-api-key") == "neutral-key");
    CHECK_FALSE(headers.contains("x-anthropic-billing-header"));
    CHECK_FALSE(headers.contains("anthropic-beta"));
    CHECK(protocol.reasoning_capabilities("claude-opus-5").empty());
    CHECK_FALSE(protocol.accepts_legacy_continuation());
    CHECK(protocol.build_url("https://example.test/anthropic/v1/", {}) == "https://example.test/anthropic/v1/messages");
    CHECK(protocol.build_url("https://example.test/anthropic/v1/messages", {}) == "https://example.test/anthropic/v1/messages");
    CHECK(dynamic_cast<ClaudeProtocol*>(protocol.clone().get()) == nullptr);
}

TEST_CASE("Provider factory selects vendor policy independently of Messages wire type", "[minimax][factory]") {
    core::config::ProviderConfig config;
    config.api_key = "fixture-key";
    auto plan = ProviderFactory::create_provider("minimax-m-plan", config);
    auto api = ProviderFactory::create_provider("minimax", config);
    REQUIRE(plan); REQUIRE(api);
    CHECK_FALSE(plan->should_estimate_cost());
    CHECK(api->should_estimate_cost());
    CHECK(plan->max_context_size() == kMiniMaxContextWindow);
    CHECK(plan->reasoning_capabilities(kMiniMaxDefaultModel).supports(ReasoningCapability::Required));
    const auto fork = plan->fork_for_parallel_request();
    REQUIRE(fork);
    CHECK_FALSE(fork->should_estimate_cost());
    CHECK(fork->reasoning_capabilities(kMiniMaxDefaultModel).supports(ReasoningCapability::Required));
    const auto* definition = find_builtin_provider_definition("minimax-m-plan-work");
    REQUIRE(definition);
    CHECK(definition->billing_kind == core::auth::BillingKind::Subscription);
    CHECK(definition->env_vars[0] == "MINIMAX_M_PLAN_API_KEY");
    CHECK(definition->env_vars[1] == "MINIMAX_SUBSCRIPTION_KEY");
    CHECK_FALSE(find_builtin_provider_definition("minimaximum"));
}

TEST_CASE("Claude routing recognizes native hosts independently of URL path and case", "[claude][factory]") {
    core::config::ProviderConfig config;
    config.api_key = "fixture";
    config.api_type = core::config::ApiType::Anthropic;
    constexpr auto model = "claude-opus-4-7-20990101";
    for (const auto url : {"https://api.anthropic.com/", "https://api.anthropic.com/v1",
                           "https://API.ANTHROPIC.COM/v1/messages"}) {
        CAPTURE(url);
        config.base_url = url;
        const auto provider = ProviderFactory::create_provider("custom-claude", config);
        REQUIRE(provider);
        CHECK_FALSE(provider->reasoning_capabilities(model).empty());
    }
    config.base_url = "https://api.anthropic.com.example.test/v1";
    const auto generic = ProviderFactory::create_provider("custom-messages", config);
    REQUIRE(generic);
    CHECK(generic->reasoning_capabilities(model).empty());
}

TEST_CASE("MiniMax keeps required thinking and all supported effort levels", "[minimax][reasoning]") {
    MiniMaxProtocol protocol(true);
    for (const auto effort : {"low", "medium", "high", "xhigh", "max", "none", "off", "auto", ""}) {
        CAPTURE(effort);
        auto req = request(); req.effort = effort;
        protocol.prepare_request(req);
        const auto payload = protocol.serialize(req);
        valid_json(payload);
        CHECK_THAT(payload, ContainsSubstring(R"("thinking":{"type":"adaptive"})"));
        const std::string expected = std::string(effort) == "off" || std::string(effort) == "none" ? "low"
            : std::string(effort).empty() || std::string(effort) == "auto" ? "max" : effort;
        CHECK_THAT(payload, ContainsSubstring("\"effort\":\"" + expected + "\""));
        for (const auto field : {"billing", "temperature", "budget_tokens", "block_binding", "display", "cache_control", "service_tier"}) CHECK_THAT(payload, !ContainsSubstring(field));
    }
    auto m3 = request("MiniMax-M3");
    MiniMaxProtocol api;
    api.prepare_request(m3);
    CHECK_THAT(api.serialize(m3), !ContainsSubstring("thinking"));
    m3.effort = "high";
    CHECK_THAT(api.serialize(m3), ContainsSubstring(R"("type":"adaptive")"));
    CHECK_THAT(api.serialize(m3), !ContainsSubstring("output_config"));
    for (const auto effort : {"auto", "default", "unset", " OFF ", " disabled "}) {
        m3.effort = effort;
        CHECK_THAT(api.serialize(m3), !ContainsSubstring("thinking"));
    }
    // M Plan serves both curated models; only their thinking policies differ.
    CHECK_NOTHROW(protocol.prepare_request(m3));
}

TEST_CASE("MiniMax rejects mismatched key types and leaves unknown formats usable", "[minimax][auth]") {
    MiniMaxProtocol plan(true), api;
    core::auth::AuthInfo auth;
    auth.headers["x-api-key"] = "sk-api-fixture";
    CHECK_THROWS(plan.build_headers(auth));
    CHECK_NOTHROW(api.build_headers(auth));
    auth.headers["x-api-key"] = "sk-cp-fixture";
    CHECK_THROWS(api.build_headers(auth));
    CHECK_NOTHROW(plan.build_headers(auth));
    auth.headers["x-api-key"] = "future-format-fixture";
    CHECK_NOTHROW(plan.build_headers(auth));
    auto manager = core::auth::AuthenticationManager::create_with_defaults("/tmp");
    const auto logins = manager.available_login_providers();
    for (const auto name : {"minimax", "minimax-api"}) CHECK(std::ranges::find(logins, name) != logins.end());
}

TEST_CASE("MiniMax subscription credentials never fall back to a metered key", "[minimax][auth]") {
    MiniMaxEnv environment;
    setenv("MINIMAX_API_KEY", "metered-fixture", 1);
    core::config::ProviderConfig config;
    auto plan = ProviderFactory::create_provider("minimax-m-plan", config); REQUIRE(plan);
    REQUIRE(plan->metadata());
    CHECK(plan->metadata()->credential_source->availability() == core::auth::CredentialAvailability::Missing);
    setenv("MINIMAX_SUBSCRIPTION_KEY", "alias-fixture", 1);
    plan = ProviderFactory::create_provider("minimax-m-plan", config);
    CHECK(plan->metadata()->credential_source->get_auth().headers.at("x-api-key") == "alias-fixture");
    setenv("MINIMAX_M_PLAN_API_KEY", "canonical-fixture", 1);
    plan = ProviderFactory::create_provider("minimax-m-plan", config);
    CHECK(plan->metadata()->credential_source->get_auth().headers.at("x-api-key") == "canonical-fixture");
    config.api_key = "configured-fixture";
    plan = ProviderFactory::create_provider("minimax-m-plan", config);
    CHECK(plan->metadata()->credential_source->get_auth().headers.at("x-api-key") == "configured-fixture");
    CHECK_FALSE(core::tools::web::make_anthropic_web_search_backend()->supports({.session_context = test_support::make_workspace_session_context(), .provider = plan}));
    config.base_url = "https://api.minimax.io/anthropic/v1/messages/";
    plan = ProviderFactory::create_provider("minimax-m-plan", config);
    CHECK(plan->metadata()->base_url == kMiniMaxBaseUrl);
}

TEST_CASE("MiniMax sparse discovery preserves curated limits and separate plan identity", "[minimax][catalog]") {
    MiniMaxProtocol plan(true), api;
    const auto catalog = plan.model_catalog("minimax-m-plan");
    const auto result = catalog->parse_models_response(R"({"data":[{"id":"MiniMax-M3.1-Flash"},{"id":"MiniMax-M3.1-Flash-Preview"},{"id":"MiniMax-M3"},{"id":"MiniMax-M2.7"}]})");
    REQUIRE(result.ok()); REQUIRE(result.models.size() == 2);
    // Retired M2.x names never reach the picker: the service reroutes them.
    CHECK(std::ranges::none_of(result.models, [](const auto& model) {
        return model.canonical_id.starts_with("MiniMax-M2");
    }));
    const auto flash = std::ranges::find_if(result.models, [](const auto& model) {
        return model.canonical_id == kMiniMaxDefaultModel;
    });
    REQUIRE(flash != result.models.end());
    CHECK(flash->context_window == kMiniMaxContextWindow);
    CHECK(flash->max_output_tokens == kMiniMaxOutputLimit);
    CHECK(flash->provider == "minimax-m-plan");
    CHECK(flash->supports(ModelCapability::Vision));
    // JSON-object output passes request validation; schemas are refused by the protocol.
    CHECK(flash->supports(ModelCapability::JsonMode));
    CHECK(std::ranges::any_of(result.models, [](const auto& model) {
        return model.canonical_id == "MiniMax-M3";
    }));
    const std::vector<std::string> providers{"minimax", "minimax-m-plan"};
    const auto group = provider_catalog_group_for("minimax-m-plan", providers);
    REQUIRE(group.sources.size() == 2);
    const auto* source = group.find_source("minimax-m-plan"); REQUIRE(source);
    CHECK(source->includes_registry_model("MiniMax-M3.1-Flash"));
    CHECK(source->includes_registry_model("MiniMax-M3"));
    CHECK(provider_catalog_selection_key("minimax", kMiniMaxDefaultModel)
          != provider_catalog_selection_key("minimax-m-plan", kMiniMaxDefaultModel));
}

TEST_CASE("MiniMax budgets complete context and refuses overfull history", "[minimax][budget]") {
    MiniMaxProtocol protocol(true);
    auto req = request();
    req.max_tokens = 999999;
    protocol.prepare_request(req);
    CHECK(req.max_tokens == kMiniMaxOutputLimit);
    req.messages.back().content = std::string(2800000, 'a');
    protocol.prepare_request(req);
    providers::MiniMaxRequestBudget::prepare(req, kMiniMaxContextWindow);
    REQUIRE(req.max_tokens.has_value());
    CHECK(*req.max_tokens < kMiniMaxOutputLimit);
    req.messages.back().content = std::string(3200000, 'a');
    CHECK_THROWS(providers::MiniMaxRequestBudget::prepare(req, kMiniMaxContextWindow));
    req = request(); req.response_format.type = ResponseFormat::Type::JsonSchema;
    CHECK_THROWS(protocol.prepare_request(req));
    // Reader and review turns ask for JSON objects and validate them locally.
    req = request(); req.response_format.type = ResponseFormat::Type::JsonObject;
    CHECK_NOTHROW(protocol.prepare_request(req));
    valid_json(protocol.serialize(req));
}

TEST_CASE("MiniMax estimates CJK per character and other text per three characters", "[minimax][budget]") {
    using providers::MiniMaxRequestBudget;
    CHECK(MiniMaxRequestBudget::estimate_text("") == 0);
    CHECK(MiniMaxRequestBudget::estimate_text("abc") == 1);
    CHECK(MiniMaxRequestBudget::estimate_text("abcd") == 2);
    CHECK(MiniMaxRequestBudget::estimate_text("\xE4\xBD\xA0\xE5\xA5\xBD") == 2);      // 你好
    CHECK(MiniMaxRequestBudget::estimate_text("\xEC\x95\x88") == 1);                   // 안 (Hangul)
    CHECK(MiniMaxRequestBudget::estimate_text("\xF0\xA0\x80\x80") == 1);               // U+20000
    CHECK(MiniMaxRequestBudget::estimate_text("\xC3\xA9\xC3\xA9\xC3\xA9") == 1);       // ééé
    CHECK(MiniMaxRequestBudget::estimate_text("\xE4\xBD") == 1);                       // truncated, no overrun
    // Prose and code no longer count at two bytes per token, which made
    // compaction start at roughly two thirds of its configured threshold.
    const std::vector<Message> history{{.role = "user", .content = std::string(300000, 'a')}};
    CHECK(MiniMaxRequestBudget::estimate_history(history) == 100000 + 8 + 4096);
}

TEST_CASE("MiniMax lowers thinking effort when the output budget is short", "[minimax][reasoning]") {
    CHECK(minimax_effort_ceiling(1024) == "low");
    CHECK(minimax_effort_ceiling(4096) == "low");
    CHECK(minimax_effort_ceiling(8192) == "medium");
    CHECK(minimax_effort_ceiling(32768) == "high");
    CHECK(minimax_effort_ceiling(32769).empty());
    MiniMaxProtocol protocol(true, "max");
    const auto effort_for = [&](std::optional<int> max_tokens, std::string effort) {
        auto req = request(); req.max_tokens = max_tokens; req.effort = std::move(effort);
        protocol.prepare_request(req);
        return protocol.serialize(req);
    };
    CHECK_THAT(effort_for(1400, ""), ContainsSubstring(R"("effort":"low")"));
    CHECK_THAT(effort_for(2048, "xhigh"), ContainsSubstring(R"("effort":"low")"));
    CHECK_THAT(effort_for(10000, "max"), ContainsSubstring(R"("effort":"medium")"));
    CHECK_THAT(effort_for(30000, "max"), ContainsSubstring(R"("effort":"high")"));
    CHECK_THAT(effort_for(10000, "low"), ContainsSubstring(R"("effort":"low")"));
    CHECK_THAT(effort_for(std::nullopt, ""), ContainsSubstring(R"("effort":"max")"));
    MiniMaxProtocol api;
    auto m3 = request("MiniMax-M3"); m3.effort = "high"; m3.max_tokens = 1400;
    api.prepare_request(m3);
    CHECK_THAT(api.serialize(m3), !ContainsSubstring("thinking"));
    m3.max_tokens = 8192;
    CHECK_THAT(api.serialize(m3), ContainsSubstring(R"("thinking":{"type":"adaptive"})"));
}

TEST_CASE("MiniMax identifies itself as MiniMax Code on its own-key path", "[minimax][identity]") {
    namespace identity = core::auth::minimax_code;
    MiniMaxProtocol plan(true);
    core::auth::AuthInfo auth;
    auth.headers["x-api-key"] = "sk-cp-fixture";
    auto headers = plan.build_headers(auth);
    // pi-mono's Anthropic SDK client, which minimax-code uses for own keys;
    // MiniMaxAgent belongs to the managed-login backend only.
    CHECK(headers.at("User-Agent") == "Anthropic/JS 0.91.1");
    CHECK(headers.at("Accept") == "application/json");
    CHECK(headers.at("anthropic-version") == "2023-06-01");
    CHECK(headers.at("anthropic-dangerous-direct-browser-access") == "true");
    CHECK(headers.at("X-Stainless-Lang") == "js");
    CHECK(headers.at("X-Stainless-Package-Version") == "0.91.1");
    CHECK(headers.at("X-Stainless-Runtime") == "node");
    CHECK(headers.at("X-Stainless-Retry-Count") == "0");
    CHECK(headers.at("X-Stainless-Timeout") == "600");
    CHECK(headers.at("X-Stainless-OS") == identity::sdk_platform_os());
    CHECK(headers.at("X-Stainless-Arch") == identity::sdk_platform_arch());
    CHECK(headers.at("X-Mavis-Agent-Id") == "main");
    CHECK(headers.at("X-Mavis-Session-Id") == identity::process_session_id());
    CHECK(headers.at("x-api-key") == "sk-cp-fixture");
    int offset = 1;
    const auto& zone = headers.at("X-Mavis-Timezone-Offset");
    REQUIRE(std::from_chars(zone.data(), zone.data() + zone.size(), offset).ec == std::errc{});
    CHECK(offset % 60 == 0);
    CHECK(std::abs(offset) <= 14 * 3600);
    auto req = request(); req.session_id = "session-1";
    plan.prepare_headers(headers, req, kMiniMaxBaseUrl);
    CHECK(headers.at("X-Mavis-Session-Id") == "session-1");
    for (const auto field : {"anthropic-beta", "x-anthropic-billing-header", "Authorization"}) CHECK_FALSE(headers.contains(field));
}

TEST_CASE("MiniMax retries plan rate limits on a seconds-scale schedule", "[minimax][errors]") {
    const auto policy = MiniMaxProtocol(true).stream_retry_policy();
    CHECK(policy.max_retries == 5);
    CHECK(policy.initial_backoff == std::chrono::seconds(1));
    CHECK(policy.maximum_backoff == std::chrono::seconds(30));
    CHECK(policy.retry_only_before_output);
    CHECK(MiniMaxProtocol(true).clone()->stream_retry_policy() == policy);
}

TEST_CASE("MiniMax image input supports native URL sources", "[minimax][vision]") {
    MiniMaxProtocol protocol(true);
    auto req = request();
    req.messages.back().content_parts = {
        ContentPart::make_image_url("https://example.test/diagram.png"),
        ContentPart::make_image_url("https://example.test/screenshot.png")};
    protocol.prepare_request(req);
    const auto payload = protocol.serialize(req); valid_json(payload);
    CHECK_THAT(payload, ContainsSubstring(R"("type":"url","url":"https://example.test/diagram.png")"));
    CHECK_THAT(payload, ContainsSubstring(R"("type":"url","url":"https://example.test/screenshot.png")"));
    CHECK(providers::MiniMaxRequestBudget::estimate_request(req) >= 30000);
}

TEST_CASE("MiniMax context configuration survives forks and supplies compaction estimates", "[minimax][budget]") {
    core::config::ProviderConfig config;
    config.context_window = 1000000;
    config.api_key = "fixture";
    const auto provider = ProviderFactory::create_provider("minimax-m-plan", config);
    REQUIRE(provider);
    CHECK(provider->max_context_size() == 1000000);
    const auto fork = provider->fork_for_parallel_request(); REQUIRE(fork);
    CHECK(fork->max_context_size() == 1000000);
    const auto estimator = dynamic_cast<const ContextTokenEstimator*>(fork.get()); REQUIRE(estimator);
    CHECK(estimator->estimate_context_tokens(request().messages)
        == providers::MiniMaxRequestBudget::estimate_history(request().messages));
    CHECK(core::budget::context_window_for_model("MiniMax-M3.1-Flash") == kMiniMaxContextWindow);
    config.context_window = 999999;
    CHECK_THROWS(ProviderFactory::create_provider("minimax-m-plan", config));
    config.api_type = core::config::ApiType::Anthropic;
    CHECK(core::commands::detail::provider_family_key("minimax-m-plan", config) == "minimax");
    CHECK(core::commands::detail::provider_family_key("claude", config) == "anthropic");
}

TEST_CASE("MiniMax login profiles reject the other commercial key before saving", "[minimax][auth]") {
    const core::auth::ApiKeyLoginProfile profile{
        .provider_name = "minimax-m-plan", .default_model = "MiniMax-M3.1-Flash",
        .rejected_key_prefixes = {"sk-api-"}};
    CHECK_THROWS(core::auth::save_api_key_profile("/tmp/filo-minimax-rejected-profile", profile, "sk-api-fixture"));
}

TEST_CASE("MiniMax preserves ordered reasoning and tools with cache-aware usage", "[minimax][sse]") {
    MiniMaxProtocol protocol(true);
    std::vector<ContinuationItem> continuation;
    const std::vector<std::string> events{
        R"({"type":"message_start","message":{"usage":{"input_tokens":10,"cache_read_input_tokens":100}}})",
        R"({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}})",
        R"({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Reason"}})",
        R"({"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"signed"}})",
        R"({"type":"content_block_stop","index":0})",
        R"({"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call","name":"read","input":{}}})",
        R"({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":\"a\"}"}})",
        R"({"type":"content_block_stop","index":1})",
        R"({"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":20}})",
        R"({"type":"message_stop"})"};
    ParseResult final;
    for (const auto& event : events) {
        auto result = protocol.parse_event("data: " + event);
        REQUIRE_FALSE(result.stream_error);
        for (const auto& chunk : result.chunks) for (const auto& item : chunk.continuation_items) continuation.push_back(item);
        if (result.done) final = result;
    }
    CHECK(final.done); CHECK(final.prompt_tokens == 110); CHECK(final.cached_prompt_tokens == 100);
    CHECK(final.completion_tokens == 20); CHECK(final.stop_reason == "tool_use");
    auto req = request();
    Message assistant{.role = "assistant"}; assistant.continuation_items = continuation;
    assistant.tool_calls.push_back({.id = "call", .type = "function", .function = {.name = "read", .arguments = R"({"path":"a"})"}});
    req.messages.push_back(assistant);
    req.messages.push_back({.role = "tool", .content = "File content", .tool_call_id = "call"});
    const auto payload = protocol.serialize(req); valid_json(payload);
    CHECK_THAT(payload, ContainsSubstring(R"("thinking":"Reason","signature":"signed")"));
    CHECK_THAT(payload, ContainsSubstring(R"("type":"tool_result","tool_use_id":"call")"));
    CHECK(payload.find("Reason") < payload.find("\"type\":\"tool_use\""));
}

TEST_CASE("MiniMax business errors distinguish exhaustion from transient rate limits", "[minimax][errors]") {
    MiniMaxProtocol protocol(true);
    const cpr::Header headers;
    for (const auto code : {2056, 2057, 2061, 2067, 2068, 2069, 2013, 2049}) {
        const auto body = "{\"error\":{\"code\":" + std::to_string(code) + ",\"message\":\"fixture\"}}";
        CHECK_FALSE(protocol.is_retryable({429, body, headers}));
        const auto event = protocol.parse_event("event: error\ndata: " + body);
        CHECK(event.stream_error); CHECK_FALSE(event.retryable_stream_error);
    }
    const std::string body = R"JSON({"error":{"type":"rate_limit_error","message":"M Plan rate limit reached (2062)"}})JSON";
    CHECK(protocol.is_retryable({429, body, headers}));
    CHECK(protocol.parse_event("event: error\ndata: " + body).retryable_stream_error);
    CHECK_FALSE(protocol.is_retryable({503, {}, {{"x-should-retry", "false"}}}));
    protocol.on_response({429, {}, {{"Retry-After", "1.5"}}});
    CHECK(protocol.last_rate_limit().retry_after == 2);
    CHECK(protocol.last_rate_limit().is_rate_limited);
    CHECK_FALSE(protocol.is_retryable({429,
        R"({"error":{"code":2056,"message":"Exhausted"},"base_resp":{"status_code":0,"status_msg":"success"}})", headers}));
    CHECK_THAT(protocol.format_error_message({429,
        R"JSON({"error":{"message":"Rate limit reached (2062garbage)"}})JSON", headers}),
        ContainsSubstring("MiniMax error 429"));
    CHECK_FALSE(protocol.is_retryable({429,
        R"({"base_resp":{"status_code":"2056","status_msg":"Exhausted"}})", headers}));
    CHECK_THAT(protocol.format_error_message({401, R"({"type":"error","error":{"type":"authentication_error","message":"invalid key"}})", headers}),
        ContainsSubstring("filo --auth minimax"));
    CHECK_THAT(protocol.format_error_message({400, R"({"base_resp":{"status_code":1026,"status_msg":"input new_sensitive"}})", headers}),
        ContainsSubstring("content filter"));
    CHECK(minimax_error_code(R"({"base_resp":{"status_code":0,"status_msg":"success"}})") == 0);
    CHECK(minimax_error_code(R"({"base_resp":{"status_code":1004,"status_msg":"unauthorized"}})") == 1004);
    CHECK(minimax_error_code("not json") == 0);
}

TEST_CASE("MiniMax quota uses explicit remaining percentages and validates unknown data", "[minimax][quota]") {
    const auto info = parse_minimax_quota(R"({"base_resp":{"status_code":0},"model_remains":[{"model_name":"general","current_interval_remaining_percent":75.5,"current_interval_usage_count":999,"end_time":1790000000000,"current_weekly_remaining_percent":0,"weekly_end_time":1790100000000}]})");
    REQUIRE(info); REQUIRE(info->usage_windows.size() == 2);
    CHECK(info->usage_windows[0].utilization > 0.244F);
    CHECK(info->usage_windows[0].utilization < 0.246F);
    CHECK(info->usage_windows[0].resets_at == 1790000000);
    CHECK(info->is_rate_limited);
    CHECK_FALSE(parse_minimax_quota(R"({"model_remains":[]})"));
    CHECK_FALSE(parse_minimax_quota("invalid"));
    CHECK_FALSE(parse_minimax_quota(std::string(256 * 1024 + 1, ' ')));
    CHECK_FALSE(parse_minimax_quota(R"({"base_resp":{"status_code":2062},"current_interval_remaining_percent":50})"));
    CHECK_FALSE(parse_minimax_quota(R"({"current_interval_remaining_percent":-1})"));
    CHECK_FALSE(parse_minimax_quota(R"({"current_interval_remaining_percent":101})"));
    CHECK_FALSE(parse_minimax_quota(R"({"current_interval_remaining_percent":"invalid","current_interval_total_count":100,"current_interval_usage_count":100})"));
    const auto huge_reset = parse_minimax_quota(R"({"current_interval_remaining_percent":50,"end_time":1e100})");
    REQUIRE(huge_reset);
    CHECK(huge_reset->usage_windows.front().resets_at == 0);
    const auto legacy = parse_minimax_quota(R"({"model_remains":[{"model_name":"MiniMax-M*","current_interval_total_count":100,"current_interval_usage_count":75}]})");
    REQUIRE(legacy); CHECK(legacy->usage_windows.front().utilization == 0.25F);
    CHECK_FALSE(parse_minimax_quota(R"({"current_interval_total_count":0,"current_interval_usage_count":0})"));
    CHECK_FALSE(parse_minimax_quota(R"({"current_interval_status":3,"current_interval_remaining_percent":0})"));
    // Status 2 is authoritative exhaustion; a boost is reported, not folded in.
    const auto exhausted = parse_minimax_quota(R"({"model_remains":[{"model_name":"general","current_interval_status":1,"current_interval_remaining_percent":40,"current_weekly_status":2,"current_weekly_remaining_percent":3,"weekly_boost_permille":1500}]})");
    REQUIRE(exhausted); REQUIRE(exhausted->usage_windows.size() == 2);
    CHECK(exhausted->usage_windows[1].label == "7d");
    CHECK(exhausted->usage_windows[1].utilization == 1.0F);
    CHECK(exhausted->is_rate_limited);
    CHECK(exhausted->subscription.notice == "Weekly quota boosted to 150%");
    const auto unboosted = parse_minimax_quota(R"({"current_interval_remaining_percent":50,"weekly_boost_permille":1000})");
    REQUIRE(unboosted); CHECK(unboosted->subscription.notice.empty()); CHECK_FALSE(unboosted->is_rate_limited);
    MiniMaxProtocol protocol;
    const cpr::Header headers;
    CHECK_FALSE(protocol.is_retryable({429, R"({"error":{"code":"2056","message":"usage exhausted"}})", headers}));
}

TEST_CASE("MiniMax quota reads the text-model entry whatever the server names it", "[minimax][quota]") {
    const auto entry = [](std::string_view name, int remaining) {
        return std::format(R"({{"model_name":"{}","current_interval_remaining_percent":{}}})", name, remaining);
    };
    const auto remaining = [](const std::string& body) {
        const auto info = parse_minimax_quota(body);
        REQUIRE(info);
        REQUIRE_FALSE(info->usage_windows.empty());
        return std::lround(100.0F - info->usage_windows.front().utilization * 100.0F);
    };
    const auto remains = [](std::initializer_list<std::string> entries) {
        std::string body = R"({"model_remains":[)";
        for (const auto& item : entries) body += (body.ends_with('[') ? "" : ",") + item;
        return body + "]}";
    };
    // Concrete text-model ids, as the official CLI's fixtures use.
    CHECK(remaining(remains({entry("MiniMax-M3", 61)})) == 61);
    CHECK(remaining(remains({entry("MiniMax-M2", 62)})) == 62);
    // Media buckets never stand in for the text quota, even when listed first.
    CHECK(remaining(remains({entry("video", 1), entry("MiniMax-Music-2", 2), entry("speech-hd", 3),
                             entry("MiniMax-M3", 64)})) == 64);
    CHECK_FALSE(parse_minimax_quota(remains({entry("video", 1), entry("image-01", 2)})));
    // The plan model's own entry wins over shared buckets, which win over other ids.
    CHECK(remaining(remains({entry("MiniMax-M3", 10), entry("MiniMax-M*", 20), entry("general", 30),
                             entry("MiniMax-M3.1-Flash", 40)})) == 40);
    CHECK(remaining(remains({entry("MiniMax-M3", 10), entry("MiniMax-M*", 20), entry("general", 30)})) == 30);
    CHECK(remaining(remains({entry("MiniMax-M3", 10), entry("MiniMax-M*", 20)})) == 20);
    // A bucket with no limited window (not in plan / unlimited) yields to the next.
    CHECK(remaining(remains({R"({"model_name":"general","current_interval_status":3,"current_weekly_status":3})",
                             entry("MiniMax-M3", 70)})) == 70);
}

TEST_CASE("MiniMax quota labels each window from the server's own bounds", "[minimax][quota]") {
    const auto labels = [](std::string_view fields) {
        const auto info = parse_minimax_quota(std::format(
            R"({{"current_interval_remaining_percent":50,"current_weekly_remaining_percent":50{}}})", fields));
        REQUIRE(info);
        REQUIRE(info->usage_windows.size() == 2);
        return std::pair{info->usage_windows[0].label, info->usage_windows[1].label};
    };
    constexpr int64_t hour = 3'600'000, day = 24 * hour, start = 1'790'000'000'000;
    // Without bounds the documented 5-hour and weekly windows are assumed.
    CHECK(labels("") == std::pair<std::string, std::string>{"5h", "7d"});
    CHECK(labels(std::format(R"(,"start_time":{},"end_time":{},"weekly_start_time":{},"weekly_end_time":{})",
                             start, start + 5 * hour, start, start + 7 * day))
          == std::pair<std::string, std::string>{"5h", "7d"});
    // A changed plan is labelled truthfully; millisecond jitter is absorbed.
    CHECK(labels(std::format(R"(,"start_time":{},"end_time":{},"weekly_start_time":{},"weekly_end_time":{})",
                             start, start + 4 * hour - 1, start, start + day + 999))
          == std::pair<std::string, std::string>{"4h", "1d"});
    // Inverted or absurd bounds keep the documented length.
    CHECK(labels(std::format(R"(,"start_time":{},"end_time":{},"weekly_start_time":{},"weekly_end_time":{})",
                             start + hour, start, start, start + 400 * day))
          == std::pair<std::string, std::string>{"5h", "7d"});
}

TEST_CASE("MiniMax plan endpoints accept the official base URL with trailing slashes only", "[minimax][quota]") {
    CHECK(is_official_minimax_base_url(kMiniMaxBaseUrl));
    CHECK(is_official_minimax_base_url("https://api.minimax.io/anthropic/"));
    CHECK(is_official_minimax_base_url("https://api.minimax.io/anthropic//"));
    CHECK_FALSE(is_official_minimax_base_url("https://api.minimax.io/anthropic/v1"));
    CHECK_FALSE(is_official_minimax_base_url("https://api.minimax.io"));
    CHECK_FALSE(is_official_minimax_base_url("http://api.minimax.io/anthropic"));
    CHECK_FALSE(is_official_minimax_base_url("https://api.minimax.io.proxy.example/anthropic"));
    CHECK_FALSE(is_official_minimax_base_url("https://proxy.example/anthropic"));
    CHECK_FALSE(is_official_minimax_base_url(""));
    int reads = 0;
    const auto source = make_minimax_usage_source([&](std::string_view) -> std::optional<RateLimitInfo> {
        ++reads;
        return RateLimitInfo{};
    });
    CHECK(source->read("https://api.minimax.io/anthropic/", {{"x-api-key", "account"}}));
    CHECK(reads == 1);
}

TEST_CASE("MiniMax usage enrichment is subscription-only and shared across clones", "[minimax][quota]") {
    auto usage = std::make_shared<UsageFixture>();
    MiniMaxProtocol plan(true, "max", usage), api(false, {}, usage);
    const cpr::Header headers;
    api.enrich_rate_limit({}, headers, {200, {}, headers});
    CHECK(usage->reads == 0);
    auto clone = plan.clone();
    clone->enrich_rate_limit({}, headers, {200, {}, headers});
    CHECK(usage->reads == 1);
    REQUIRE(clone->last_rate_limit().usage_windows.size() == 1);
    CHECK(plan.last_rate_limit().usage_windows.empty());
    clone->reset_state();
    CHECK(clone->last_rate_limit().usage_windows.empty());
    CHECK_FALSE(make_minimax_usage_source()->read("https://proxy.example/anthropic", {{"x-api-key", "fixture"}}));
}

TEST_CASE("MiniMax quota reads coalesce per account without blocking other accounts", "[minimax][quota][parallel]") {
    std::promise<void> entered, release;
    auto entered_future = entered.get_future();
    auto released = release.get_future().share();
    std::atomic<int> reads_a{0}, reads_b{0};
    const auto source = make_minimax_usage_source([&](std::string_view key) -> std::optional<RateLimitInfo> {
        if (key == "account-a") {
            if (reads_a.fetch_add(1) == 0) entered.set_value();
            released.wait();
        } else {
            ++reads_b;
        }
        RateLimitInfo info;
        info.subscription.tier_label = std::string(key);
        return info;
    });
    const auto read = [&](std::string_view account) {
        return source->read(kMiniMaxBaseUrl, {{"x-api-key", std::string(account)}});
    };
    auto first = std::async(std::launch::async, [&] { return read("account-a"); });
    CHECK(entered_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    auto same_account = std::async(std::launch::async, [&] { return read("account-a"); });
    auto other_account = std::async(std::launch::async, [&] { return read("account-b"); });
    const bool independent = other_account.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    release.set_value(); // Always unblock workers before assertions can end the test.
    CHECK(independent);
    const auto a = first.get(), a_again = same_account.get(), b = other_account.get();
    REQUIRE(a); REQUIRE(a_again); REQUIRE(b);
    CHECK(a->subscription.tier_label == "account-a");
    CHECK(a_again->subscription.tier_label == "account-a");
    CHECK(b->subscription.tier_label == "account-b");
    CHECK(read("account-a").has_value());
    CHECK(reads_a == 1);
    CHECK(reads_b == 1);
    CHECK_FALSE(source->read("https://proxy.example/anthropic", {{"x-api-key", "account-c"}}));
    CHECK(reads_b == 1);
}

TEST_CASE("MiniMax quota failures stay unknown and are cached", "[minimax][quota]") {
    int reads = 0;
    const auto source = make_minimax_usage_source([&](std::string_view) -> std::optional<RateLimitInfo> {
        ++reads;
        throw std::runtime_error("Quota unavailable");
    });
    for (int attempt = 0; attempt < 2; ++attempt) {
        CHECK_FALSE(source->read(kMiniMaxBaseUrl, {{"x-api-key", "account"}}));
    }
    CHECK(reads == 1);
    CHECK_FALSE(source->read(kMiniMaxBaseUrl, {}));
    CHECK(reads == 1);
}

TEST_CASE("MiniMax plan search serves only the official subscription endpoint", "[minimax][web]") {
    MiniMaxEnv environment;
    using namespace core::tools::web;
    core::config::ProviderConfig config;
    config.api_key = "sk-cp-fixture";
    const auto backend = make_minimax_web_search_backend();
    const auto context = [](std::shared_ptr<LLMProvider> provider) {
        return core::tools::ToolInvocationContext{
            .session_context = test_support::make_workspace_session_context(), .provider = std::move(provider)};
    };
    CHECK(backend->supports(context(ProviderFactory::create_provider("minimax-m-plan", config))));
    CHECK(backend->supports(context(ProviderFactory::create_provider("minimax-m-plan-work", config))));
    config.api_key = "sk-api-fixture";
    CHECK_FALSE(backend->supports(context(ProviderFactory::create_provider("minimax", config))));
    config.api_key = "sk-cp-fixture";
    config.base_url = "https://proxy.example/anthropic";
    CHECK_FALSE(backend->supports(context(ProviderFactory::create_provider("minimax-m-plan", config))));
    CHECK_FALSE(backend->supports(context(nullptr)));
}

TEST_CASE("MiniMax plan search maps organic hits, limits and domain filters", "[minimax][web]") {
    using namespace core::tools::web;
    SearchRequest request{.query = "filo \"agent\"", .limit = 2};
    CHECK(minimax_web_search_request_json(request) == R"({"q":"filo \"agent\""})");
    const std::string body = R"({"organic":[
        {"title":"One","link":"https://docs.example.com/a","snippet":"First","date":"2026-01-02"},
        {"title":"Two","link":"https://blocked.example.org/b","snippet":"Second"},
        {"title":"Three","link":"https://example.com/c","snippet":"Third"},
        {"title":"Four","link":"https://example.com/d","snippet":"Fourth"}],
        "base_resp":{"status_code":0,"status_msg":"success"}})";
    auto response = parse_minimax_web_search_response(body, request);
    REQUIRE(response); REQUIRE(response->results.size() == 2);
    CHECK(response->backend == "minimax-plan-search");
    CHECK(response->results[0].url == "https://docs.example.com/a");
    CHECK(response->results[0].snippet == "2026-01-02 — First");
    CHECK(response->results[1].title == "Two");
    request.limit = 10;
    request.domains.blocked_domains = {"example.org"};
    response = parse_minimax_web_search_response(body, request);
    REQUIRE(response); CHECK(response->results.size() == 3);
    request.domains = {.allowed_domains = {"https://Example.com/"}};
    response = parse_minimax_web_search_response(body, request);
    REQUIRE(response); REQUIRE(response->results.size() == 3);
    CHECK(response->results[0].title == "One");
    const auto failed = parse_minimax_web_search_response(
        R"({"base_resp":{"status_code":1004,"status_msg":"login fail"}})", request);
    REQUIRE_FALSE(failed);
    CHECK_THAT(failed.error(), ContainsSubstring("MiniMax error 1004"));
    CHECK_THAT(failed.error(), ContainsSubstring("login fail"));
    CHECK_FALSE(parse_minimax_web_search_response("{", request));
    const auto empty = parse_minimax_web_search_response(R"({"organic":[]})", request);
    REQUIRE(empty); CHECK(empty->results.empty());
}
