#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/auth/AuthenticationManager.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/MistralProtocol.hpp"
#include "core/version/Version.hpp"

#include <algorithm>
#include <chrono>
#include <cpr/cpr.h>
#include <memory>

using namespace core::llm;
using namespace core::llm::protocols;

// ─────────────────────────────────────────────────────────────────────────────
// ProviderFactory — creates Mistral provider
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ProviderFactory creates a Mistral provider", "[mistral][factory]") {
    core::config::ProviderConfig config;
    config.model = "codestral-latest";
    auto provider = core::llm::ProviderFactory::create_provider("mistral", config);
    REQUIRE(provider != nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// Mistral serialization — Mistral uses Serializer::serialize() (standard OpenAI
// wire format).  These tests confirm the format is correct for the Mistral API.
// ─────────────────────────────────────────────────────────────────────────────

static ChatRequest make_mistral_request(std::string model    = "devstral-small-latest",
                                        std::string user_text = "Hello") {
    ChatRequest req;
    req.model  = std::move(model);
    req.stream = true;
    req.messages.push_back(Message{.role = "user", .content = std::move(user_text)});
    return req;
}

TEST_CASE("Mistral serializer - model serialized correctly", "[mistral][serializer]") {
    REQUIRE_THAT(Serializer::serialize(make_mistral_request("devstral-small-latest")),
                 Catch::Matchers::ContainsSubstring(R"("model":"devstral-small-latest")"));
}

TEST_CASE("Mistral serializer - stream flag serialized", "[mistral][serializer]") {
    auto req   = make_mistral_request();
    req.stream = true;
    REQUIRE_THAT(Serializer::serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("stream":true)"));
}

TEST_CASE("Mistral serializer - user message appears in payload", "[mistral][serializer]") {
    auto payload = Serializer::serialize(make_mistral_request("devstral-small-latest", "Write code"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("Write code"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"user")"));
}

TEST_CASE("Mistral serializer - system message role preserved", "[mistral][serializer]") {
    ChatRequest req;
    req.model = "devstral-small-latest";
    req.messages.push_back(Message{.role = "system", .content = "Be concise."});
    req.messages.push_back(Message{.role = "user",   .content = "Hi"});
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"system")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("Be concise."));
}

TEST_CASE("Mistral serializer - temperature included when set", "[mistral][serializer]") {
    auto req        = make_mistral_request();
    req.temperature = 0.3f;
    REQUIRE_THAT(Serializer::serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("temperature")"));
}

TEST_CASE("Mistral serializer - max_tokens included when set", "[mistral][serializer]") {
    auto req       = make_mistral_request();
    req.max_tokens = 2048;
    REQUIRE_THAT(Serializer::serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("max_tokens":2048)"));
}

TEST_CASE("Mistral protocol mirrors Vibe reasoning effort semantics",
          "[mistral][serializer][effort]") {
    MistralProtocol protocol;

    auto high = make_mistral_request("mistral-vibe-cli-latest");
    high.effort = "max";
    high.temperature = 0.2F;
    const auto high_payload = protocol.serialize(high);
    REQUIRE_THAT(high_payload,
                 Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"high")"));
    REQUIRE_THAT(high_payload,
                 !Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"max")"));
    REQUIRE_THAT(high_payload,
                 Catch::Matchers::ContainsSubstring(R"("temperature":1)"));

    auto low = make_mistral_request("mistral-vibe-cli-latest");
    low.effort = "low";
    REQUIRE_THAT(protocol.serialize(low),
                 Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"none")"));

    auto off = make_mistral_request("mistral-vibe-cli-latest");
    off.effort = "off";
    REQUIRE_THAT(protocol.serialize(off),
                 !Catch::Matchers::ContainsSubstring("reasoning_effort"));

    auto non_reasoning = make_mistral_request("codestral-latest");
    non_reasoning.effort = "high";
    REQUIRE_THAT(protocol.serialize(non_reasoning),
                 !Catch::Matchers::ContainsSubstring("reasoning_effort"));
}

TEST_CASE("Mistral serializer - tool definition uses OpenAI parameters format",
          "[mistral][serializer][tools]") {
    auto req = make_mistral_request();
    Tool t;
    t.function.name        = "run_code";
    t.function.description = "Execute code";
    core::tools::ToolParameter p;
    p.name     = "code";
    p.type     = "string";
    p.required = true;
    t.function.parameters.push_back(p);
    req.tools.push_back(t);
    auto payload = Serializer::serialize(req);
    // Mistral follows OpenAI format: "parameters" not "input_schema"
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("parameters")"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("input_schema"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("required":["code"])"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Mistral SSE parsing — Mistral uses the same SSE wire format as OpenAI.
// parse_openai_sse_chunk is the shared parser for both providers.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Mistral SSE - text content extracted via OpenAI parser", "[mistral][sse]") {
    // The Mistral streaming API returns the same SSE format as OpenAI.
    auto [content, tools, reasoning] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{"content":"Hello from Mistral"},"index":0}]})");
    REQUIRE(content == "Hello from Mistral");
    REQUIRE(tools.empty());
}

TEST_CASE("Mistral SSE - Vibe thinking and text blocks are preserved",
          "[mistral][sse][thinking]") {
    MistralProtocol protocol;
    const auto result = protocol.parse_event(
        "data: "
        R"({"choices":[{"delta":{"content":[{"type":"thinking","thinking":[{"type":"text","text":"reason"}]},{"type":"text","text":"answer"}]},"index":0}]})"
        "\n\n");

    REQUIRE(result.chunks.size() == 1);
    CHECK(result.chunks[0].reasoning_content == "reason");
    CHECK(result.chunks[0].reasoning_protocol == "mistral");
    CHECK(result.chunks[0].content == "answer");
}

TEST_CASE("Mistral SSE - direct reasoning content is preserved",
          "[mistral][sse][thinking]") {
    MistralProtocol protocol;
    const auto result = protocol.parse_event(
        "data: "
        R"({"choices":[{"delta":{"reasoning_content":"reason","content":"answer"},"index":0}]})"
        "\n\n");

    REQUIRE(result.chunks.size() == 1);
    CHECK(result.chunks[0].reasoning_content == "reason");
    CHECK(result.chunks[0].reasoning_protocol == "mistral");
    CHECK(result.chunks[0].content == "answer");
}

TEST_CASE("Mistral SSE - tool call extracted via OpenAI parser", "[mistral][sse][tools]") {
    auto [content, tools, reasoning] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_m1","type":"function","function":{"name":"search","arguments":""}}]},"index":0}]})");
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].id   == "call_m1");
    REQUIRE(tools[0].function.name == "search");
}

TEST_CASE("Mistral SSE - finish_reason chunk is ignored", "[mistral][sse]") {
    auto [content, tools, reasoning] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{},"finish_reason":"stop","index":0}]})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("Mistral protocol defaults reasoning models to Vibe high effort",
          "[mistral][serializer][effort]") {
    MistralProtocol protocol;
    const auto payload = protocol.serialize(
        make_mistral_request("mistral-vibe-cli-latest"));
    REQUIRE_THAT(payload,
                 Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"high")"));
    REQUIRE_THAT(payload,
                 Catch::Matchers::ContainsSubstring(R"("temperature":1)"));
    REQUIRE_THAT(payload,
                 Catch::Matchers::ContainsSubstring(
                     R"("stream_options":{"include_usage":true})"));
}

TEST_CASE("Mistral protocol replays thinking as typed content blocks",
          "[mistral][serializer][thinking]") {
    MistralProtocol protocol;
    ChatRequest req;
    req.model = "mistral-vibe-cli-latest";
    req.effort = "high";
    req.messages.push_back(Message{
        .role = "assistant",
        .content = "answer",
        .reasoning_content = "reason",
        .reasoning_protocol = "mistral",
    });

    const auto payload = protocol.serialize(req);
    REQUIRE_THAT(payload,
                 Catch::Matchers::ContainsSubstring(R"("type":"thinking")"));
    REQUIRE_THAT(payload,
                 Catch::Matchers::ContainsSubstring(R"("text":"reason")"));
    REQUIRE_THAT(payload,
                 Catch::Matchers::ContainsSubstring(R"("type":"text")"));
    REQUIRE_THAT(payload,
                 Catch::Matchers::ContainsSubstring(R"("text":"answer")"));
    REQUIRE_THAT(payload,
                 !Catch::Matchers::ContainsSubstring("reasoning_content"));
}

TEST_CASE("Mistral protocol omits an empty text chunk when replaying reasoning",
          "[mistral][serializer][thinking]") {
    MistralProtocol protocol;
    ChatRequest req;
    req.model = "mistral-vibe-cli-latest";
    req.effort = "high";
    req.messages.push_back(Message{
        .role = "assistant",
        .reasoning_content = "Let me think step by step.",
        .reasoning_protocol = "mistral",
    });

    const auto payload = protocol.serialize(req);
    REQUIRE_THAT(
        payload,
        Catch::Matchers::ContainsSubstring(
            R"("content":[{"type":"thinking","thinking":[{"type":"text","text":"Let me think step by step."}]}])"));
}

TEST_CASE("Mistral protocol drops thinking blocks when effort is off",
          "[mistral][serializer][thinking]") {
    MistralProtocol protocol;
    ChatRequest req;
    req.model = "mistral-vibe-cli-latest";
    req.effort = "off";
    req.messages.push_back(Message{
        .role = "assistant",
        .content = "answer",
        .reasoning_content = "hidden",
        .reasoning_protocol = "mistral",
    });

    const auto payload = protocol.serialize(req);
    REQUIRE_THAT(payload,
                 !Catch::Matchers::ContainsSubstring(R"("type":"thinking")"));
    REQUIRE_THAT(payload,
                 Catch::Matchers::ContainsSubstring(R"("content":"answer")"));
}

TEST_CASE("Mistral protocol uses Vibe-length stream timeouts",
          "[mistral][timeout]") {
    MistralProtocol protocol;
    const auto timeouts = protocol.stream_timeouts();
    CHECK(timeouts.response_start == std::chrono::seconds(180));
    CHECK(timeouts.inactivity == std::chrono::seconds(720));
}

TEST_CASE("Mistral protocol stamps a Filo User-Agent",
          "[mistral][headers]") {
    MistralProtocol protocol;
    const auto headers = protocol.build_headers({});
    REQUIRE(headers.contains("User-Agent"));
    REQUIRE_THAT(headers.at("User-Agent"),
                 Catch::Matchers::ContainsSubstring(
                     std::string(core::version::user_agent)));
}

TEST_CASE("Mistral protocol formats subscription auth errors",
          "[mistral][errors]") {
    MistralProtocol protocol;
    const cpr::Header headers;
    const protocols::HttpResponse unauthorized{401, "{}", headers};
    REQUIRE_THAT(protocol.format_error_message(unauthorized),
                 Catch::Matchers::ContainsSubstring("MISTRAL_API_KEY"));

    const protocols::HttpResponse limited{
        429, R"({"message":"Rate limited"})", headers};
    REQUIRE_THAT(protocol.format_error_message(limited),
                 Catch::Matchers::ContainsSubstring("Rate limited"));
}

TEST_CASE("filo --auth mistral is a first-class login",
          "[mistral][auth]") {
    const auto providers =
        core::auth::AuthenticationManager::create_with_defaults("/tmp")
            .available_login_providers();
    REQUIRE(std::ranges::find(providers, std::string("mistral")) != providers.end());
}
