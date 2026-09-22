#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/auth/AuthenticationManager.hpp"
#include "core/auth/ICredentialSource.hpp"
#include "core/auth/MimoOAuthFlow.hpp"
#include "core/budget/TokenAccounting.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/MimoModelTraits.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/ProviderCatalogGrouping.hpp"
#include "core/llm/ProviderDefinition.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/ToolCallAssembly.hpp"
#include "core/llm/protocols/MimoProtocol.hpp"
#include "core/utils/Base64.hpp"
#include "core/utils/Crypto.hpp"
#include "core/version/Version.hpp"

#include <cpr/cpr.h>

#include <algorithm>
#include <string>
#include <vector>

using namespace core::llm;
using namespace core::llm::protocols;

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
    auth.headers["Authorization"] = "Bearer sk-test";
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
    CHECK(europe->registry_provider == "mimo");
    CHECK(europe->catalog_group == "xiaomi");

    const auto* singapore = find_builtin_provider_definition("mimo-token-plan-sgp");
    REQUIRE(singapore != nullptr);
    CHECK(singapore->base_url == "https://token-plan-sgp.xiaomimimo.com/v1");

    const auto* china = find_builtin_provider_definition("mimo-token-plan-cn");
    REQUIRE(china != nullptr);
    CHECK(china->base_url == "https://token-plan-cn.xiaomimimo.com/v1");
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

TEST_CASE("Xiaomi is the login id; mimo remains an alias", "[mimo][auth]") {
    auto manager = core::auth::AuthenticationManager::create_with_defaults("/tmp");
    const auto providers = manager.available_login_providers();
    REQUIRE(std::ranges::find(providers, std::string("xiaomi")) != providers.end());
    CHECK(std::ranges::find(providers, std::string("mimo")) == providers.end());
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
    CHECK(headers.at("Authorization") == "Bearer sk-test");
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

TEST_CASE("MiMo sends no reasoning_effort field", "[mimo][serializer]") {
    // Thinking is always on and has no request-side control, so advertising
    // effort support would make the UI offer a knob the API ignores.
    ChatRequest req = make_mimo_request();
    req.effort = "high";
    const std::string payload = MimoProtocol().serialize(req);
    CHECK_THAT(payload,
               !Catch::Matchers::ContainsSubstring("reasoning_effort"));
    CHECK_FALSE(MimoProtocol().reasoning_capabilities("mimo-v2.6-pro")
                    .supports_effort());
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

TEST_CASE("MiMo auth failures name their credentials", "[mimo][errors]") {
    const std::string message = format_error(401, R"({"error":{"message":"no"}})");
    CHECK_THAT(message,
               Catch::Matchers::ContainsSubstring("MIMO_TOKEN_PLAN_API_KEY"));
    CHECK_THAT(message, Catch::Matchers::ContainsSubstring("filo --auth xiaomi"));
}

TEST_CASE("MiMo quota exhaustion points at the console", "[mimo][errors]") {
    CHECK_THAT(format_error(429, "{}"),
               Catch::Matchers::ContainsSubstring("platform.xiaomimimo.com"));
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

    const auto flash = registry.lookup("mimo-v2-flash");
    REQUIRE(flash != nullptr);
    CHECK(flash->context_window == 262'144);
}

TEST_CASE("MiMo context windows are known to the budget tracker",
          "[mimo][accounting]") {
    CHECK(core::budget::context_window_for_model("mimo-v2.6-pro") == 1'048'576);
    CHECK(core::budget::context_window_for_model("mimo-v2-omni") == 262'144);
    CHECK(core::budget::context_window_for_model("mimo-v2-flash") == 262'144);
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

TEST_CASE("MiMo groups Token Plan ahead of pay-as-you-go", "[mimo][catalog]") {
    const std::vector<std::string> configured{"mimo", "mimo-token-plan"};
    const auto group = provider_catalog_group_for("xiaomi", configured);

    REQUIRE(group.sources.size() == 2);
    CHECK(group.sources[0].provider_name == "mimo-token-plan");
    CHECK(group.sources[0].category_label == "Token Plan endpoint.");
    CHECK(group.sources[1].provider_name == "mimo");

    // A plan key is rejected by api.xiaomimimo.com, so plan models must not be
    // offered under the pay-as-you-go source.
    CHECK(group.sources[0].registry_model_filter.matches("mimo-v2.6-pro"));
    CHECK_FALSE(group.sources[1].registry_model_filter.matches("mimo-v2.6-pro"));
    // UltraSpeed is pay-as-you-go only.
    CHECK(group.sources[1].registry_model_filter.matches("mimo-v2.6-pro-ultraspeed"));
    CHECK_FALSE(
        group.sources[0].registry_model_filter.matches("mimo-v2.6-pro-ultraspeed"));
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
