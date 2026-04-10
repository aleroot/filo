#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/OllamaProtocol.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/ProviderFactory.hpp"

#include <filesystem>
#include <fstream>
#include <memory>

using namespace core::llm;
using namespace core::llm::protocols;

namespace {

std::filesystem::path make_temp_image_file(std::string_view filename = "filo-ollama-image.png") {
    const auto path = std::filesystem::temp_directory_path() / filename;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "fake-image";
    return path;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// parse_ollama_ndjson_line — text content
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("OllamaProtocol - serialize emits images array for multimodal user input",
          "[ollama][serializer][vision]") {
    const auto image = make_temp_image_file();

    ChatRequest req;
    req.model = "gemma3";
    req.messages.push_back(Message{
        .role = "user",
        .content = describe_image_attachment(image.string()),
        .content_parts = {
            ContentPart::make_text("What is in this screenshot?"),
            ContentPart::make_image(image.string(), "image/png"),
        },
    });

    OllamaProtocol protocol;
    const auto payload = protocol.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("images":[")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("content":"What is in this screenshot?")"));
}

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

TEST_CASE("ProviderFactory - localhost Ollama provider is flagged local", "[ollama][factory]") {
    core::config::ProviderConfig config;
    config.base_url = "http://localhost:11434";
    config.model = "llama3";

    auto provider = core::llm::ProviderFactory::create_provider("ollama", config);
    REQUIRE(provider != nullptr);
    REQUIRE(provider->capabilities().is_local);
}

TEST_CASE("ProviderFactory - remote Ollama endpoint is not flagged local", "[ollama][factory]") {
    core::config::ProviderConfig config;
    config.base_url = "http://192.168.1.25:11434";
    config.model = "llama3";

    auto provider = core::llm::ProviderFactory::create_provider("ollama-remote", config);
    REQUIRE(provider != nullptr);
    REQUIRE_FALSE(provider->capabilities().is_local);
}

TEST_CASE("ProviderFactory - lookalike localhost domain is not flagged local", "[ollama][factory]") {
    core::config::ProviderConfig config;
    config.base_url = "https://localhost.run/ollama";
    config.model = "llama3";

    auto provider = core::llm::ProviderFactory::create_provider("ollama-lookalike", config);
    REQUIRE(provider != nullptr);
    REQUIRE_FALSE(provider->capabilities().is_local);
}

TEST_CASE("ProviderFactory - lookalike loopback subdomain is not flagged local", "[ollama][factory]") {
    core::config::ProviderConfig config;
    config.base_url = "https://127.0.0.1.example.com/ollama";
    config.model = "llama3";

    auto provider = core::llm::ProviderFactory::create_provider("ollama-lookalike-subdomain", config);
    REQUIRE(provider != nullptr);
    REQUIRE_FALSE(provider->capabilities().is_local);
}
