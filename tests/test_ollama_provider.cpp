#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/OllamaProtocol.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/ProviderFactory.hpp"

#include <memory>

using namespace core::llm;
using namespace core::llm::protocols;

// ─────────────────────────────────────────────────────────────────────────────
// parse_ollama_ndjson_line — text content
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_ollama_ndjson_line - text content extracted with done=false", "[ollama][ndjson]") {
    auto r = parse_ollama_ndjson_line(
        R"({"message":{"role":"assistant","content":"Hello"},"done":false})");
    REQUIRE(r.content == "Hello");
    REQUIRE(r.tools.empty());
    REQUIRE(!r.done);
}

TEST_CASE("parse_ollama_ndjson_line - text content extracted with done=true", "[ollama][ndjson]") {
    auto r = parse_ollama_ndjson_line(
        R"({"message":{"role":"assistant","content":"Bye"},"done":true})");
    REQUIRE(r.content == "Bye");
    REQUIRE(!r.tools.empty() == false);
    REQUIRE(r.done);
}

TEST_CASE("parse_ollama_ndjson_line - empty content with done=true", "[ollama][ndjson]") {
    // Final chunk often has empty content
    auto r = parse_ollama_ndjson_line(
        R"({"message":{"role":"assistant","content":""},"done":true})");
    REQUIRE(r.content.empty());
    REQUIRE(r.done);
}

TEST_CASE("parse_ollama_ndjson_line - missing message key returns done only", "[ollama][ndjson]") {
    auto r = parse_ollama_ndjson_line(R"({"done":true})");
    REQUIRE(r.content.empty());
    REQUIRE(r.tools.empty());
    REQUIRE(r.done);
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_ollama_ndjson_line — tool calls
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_ollama_ndjson_line - tool call name extracted", "[ollama][ndjson][tools]") {
    auto r = parse_ollama_ndjson_line(
        R"({"message":{"role":"assistant","tool_calls":[{"function":{"name":"get_weather","arguments":{"location":"Paris"}}}]},"done":false})");
    REQUIRE(r.tools.size() == 1);
    REQUIRE(r.tools[0].function.name == "get_weather");
}

TEST_CASE("parse_ollama_ndjson_line - tool call arguments stringified from object", "[ollama][ndjson][tools]") {
    // Ollama sends arguments as a JSON object, not a string — we stringify it
    auto r = parse_ollama_ndjson_line(
        R"({"message":{"role":"assistant","tool_calls":[{"function":{"name":"f","arguments":{"x":1,"y":"hello"}}}]},"done":false})");
    REQUIRE(r.tools.size() == 1);
    REQUIRE_THAT(r.tools[0].function.arguments,
                 Catch::Matchers::ContainsSubstring("\"x\""));
    REQUIRE_THAT(r.tools[0].function.arguments,
                 Catch::Matchers::ContainsSubstring("1"));
}

TEST_CASE("parse_ollama_ndjson_line - multiple tool calls extracted", "[ollama][ndjson][tools]") {
    auto r = parse_ollama_ndjson_line(
        R"({"message":{"role":"assistant","tool_calls":[
            {"function":{"name":"tool_a","arguments":{}}},
            {"function":{"name":"tool_b","arguments":{}}}
        ]},"done":false})");
    REQUIRE(r.tools.size() == 2);
    REQUIRE(r.tools[0].function.name == "tool_a");
    REQUIRE(r.tools[1].function.name == "tool_b");
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_ollama_ndjson_line — error handling
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_ollama_ndjson_line - empty string returns empty result", "[ollama][ndjson]") {
    auto r = parse_ollama_ndjson_line("");
    REQUIRE(r.content.empty());
    REQUIRE(r.tools.empty());
    REQUIRE(!r.done);
}

TEST_CASE("parse_ollama_ndjson_line - malformed JSON returns empty result", "[ollama][ndjson]") {
    auto r = parse_ollama_ndjson_line("{not valid json");
    REQUIRE(r.content.empty());
    REQUIRE(r.tools.empty());
    REQUIRE(!r.done);
}

TEST_CASE("parse_ollama_ndjson_line - JSON without done field defaults to false", "[ollama][ndjson]") {
    auto r = parse_ollama_ndjson_line(
        R"({"message":{"role":"assistant","content":"Hi"}})");
    REQUIRE(r.content == "Hi");
    REQUIRE(!r.done);
}

// ─────────────────────────────────────────────────────────────────────────────
// ProviderFactory — creates Ollama provider
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ProviderFactory - creates Ollama provider", "[ollama][factory]") {
    core::config::ProviderConfig config;
    config.base_url = "http://localhost:11434";
    config.model    = "llama3";
    auto provider = core::llm::ProviderFactory::create_provider("ollama", config);
    REQUIRE(provider != nullptr);
}
