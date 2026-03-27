#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/config/ConfigManager.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"

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
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{"content":"Hello from Mistral"},"index":0}]})");
    REQUIRE(content == "Hello from Mistral");
    REQUIRE(tools.empty());
}

TEST_CASE("Mistral SSE - tool call extracted via OpenAI parser", "[mistral][sse][tools]") {
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_m1","type":"function","function":{"name":"search","arguments":""}}]},"index":0}]})");
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].id   == "call_m1");
    REQUIRE(tools[0].function.name == "search");
}

TEST_CASE("Mistral SSE - finish_reason chunk is ignored", "[mistral][sse]") {
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{},"finish_reason":"stop","index":0}]})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}
