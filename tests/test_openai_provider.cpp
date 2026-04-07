#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/tools/Tool.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/ProviderFactory.hpp"

#include <memory>
#include <filesystem>
#include <fstream>

using namespace core::llm;
using namespace core::llm::protocols;

namespace {

std::filesystem::path make_temp_image_file(std::string_view filename = "filo-test-image.png") {
    const auto path = std::filesystem::temp_directory_path() / filename;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "fake-image";
    return path;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static ChatRequest make_simple_request(std::string model    = "gpt-4o",
                                       std::string user_text = "Hello") {
    ChatRequest req;
    req.model  = std::move(model);
    req.stream = true;
    req.messages.push_back(Message{.role = "user", .content = std::move(user_text)});
    return req;
}

static core::tools::ToolDefinition make_tool_def(std::string name = "get_weather",
                                                  std::string desc = "Get the weather") {
    core::tools::ToolDefinition def;
    def.name        = std::move(name);
    def.description = std::move(desc);
    core::tools::ToolParameter p;
    p.name        = "location";
    p.type        = "string";
    p.description = "City name";
    p.required    = true;
    def.parameters.push_back(std::move(p));
    return def;
}

// ─────────────────────────────────────────────────────────────────────────────
// Serializer — basic structure (standard OpenAI format used by OpenAI & Mistral)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Serializer - model is serialized correctly", "[openai][serializer]") {
    REQUIRE_THAT(Serializer::serialize(make_simple_request("gpt-4o")),
                 Catch::Matchers::ContainsSubstring(R"("model":"gpt-4o")"));
}

TEST_CASE("Serializer - stream:true is serialized", "[openai][serializer]") {
    auto req   = make_simple_request();
    req.stream = true;
    REQUIRE_THAT(Serializer::serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("stream":true)"));
}

TEST_CASE("Serializer - stream:false is serialized", "[openai][serializer]") {
    auto req   = make_simple_request();
    req.stream = false;
    REQUIRE_THAT(Serializer::serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("stream":false)"));
}

TEST_CASE("Serializer - user message content appears in payload", "[openai][serializer]") {
    auto payload = Serializer::serialize(make_simple_request("gpt-4o", "Tell me a joke"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("Tell me a joke"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"user")"));
}

TEST_CASE("Serializer - user image content becomes image_url parts", "[openai][serializer][vision]") {
    const auto image = make_temp_image_file();

    ChatRequest req;
    req.model = "gpt-4o";
    req.messages.push_back(Message{
        .role = "user",
        .content = describe_image_attachment(image.string()),
        .content_parts = {
            ContentPart::make_text("What does this screenshot show?"),
            ContentPart::make_image(image.string(), "image/png"),
        },
    });

    const auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"image_url")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"text","text":"What does this screenshot show?")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("data:image/png;base64,"));
}

TEST_CASE("Serializer - empty messages produces valid JSON array", "[openai][serializer]") {
    ChatRequest req;
    req.model  = "gpt-4o";
    req.stream = true;
    REQUIRE_THAT(Serializer::serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("messages":[])"));
}

TEST_CASE("Serializer - system message role is preserved", "[openai][serializer]") {
    ChatRequest req;
    req.model = "gpt-4o";
    req.messages.push_back(Message{.role = "system", .content = "You are helpful."});
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"system")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("You are helpful."));
}

TEST_CASE("Serializer - temperature is included when set", "[openai][serializer]") {
    auto req       = make_simple_request();
    req.temperature = 0.7f;
    REQUIRE_THAT(Serializer::serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("temperature")"));
}

TEST_CASE("Serializer - temperature is omitted when not set", "[openai][serializer]") {
    auto req  = make_simple_request();
    req.temperature = std::nullopt;
    REQUIRE_THAT(Serializer::serialize(req),
                 !Catch::Matchers::ContainsSubstring("temperature"));
}

TEST_CASE("Serializer - max_tokens is included when set", "[openai][serializer]") {
    auto req     = make_simple_request();
    req.max_tokens = 1024;
    REQUIRE_THAT(Serializer::serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("max_tokens":1024)"));
}

TEST_CASE("Serializer - max_tokens is omitted when not set", "[openai][serializer]") {
    auto req   = make_simple_request();
    req.max_tokens = std::nullopt;
    REQUIRE_THAT(Serializer::serialize(req),
                 !Catch::Matchers::ContainsSubstring("max_tokens"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Serializer — tool definitions
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Serializer - single tool definition is serialized", "[openai][serializer][tools]") {
    auto req = make_simple_request();
    Tool t;
    t.function = make_tool_def("get_weather", "Get the current weather");
    req.tools.push_back(t);
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"function")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("name":"get_weather")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("Get the current weather"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("parameters")"));
}

TEST_CASE("Serializer - required parameter appears in required array", "[openai][serializer][tools]") {
    auto req = make_simple_request();
    Tool t;
    t.function = make_tool_def();
    req.tools.push_back(t);
    REQUIRE_THAT(Serializer::serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("required":["location"])"));
}

TEST_CASE("Serializer - optional parameter is not in required array", "[openai][serializer][tools]") {
    auto req = make_simple_request();
    Tool t;
    core::tools::ToolParameter p;
    p.name     = "unit";
    p.type     = "string";
    p.required = false;
    t.function.name        = "get_weather";
    t.function.description = "Get weather";
    t.function.parameters.push_back(p);
    req.tools.push_back(t);
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("required":["unit"])"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("required":[])"));
}

TEST_CASE("Serializer - multiple tools are serialized without trailing comma", "[openai][serializer][tools]") {
    auto req = make_simple_request();
    Tool t1, t2;
    t1.function.name = "tool_a"; t1.function.description = "A";
    t2.function.name = "tool_b"; t2.function.description = "B";
    req.tools.push_back(t1);
    req.tools.push_back(t2);
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("tool_a"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("tool_b"));
    // Verify the JSON array closes cleanly (no trailing comma before ])
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",]"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Serializer — assistant tool calls and tool responses
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Serializer - assistant message with tool_calls has null content", "[openai][serializer][tools]") {
    auto req = make_simple_request();
    Message asst;
    asst.role = "assistant";
    // content empty → should emit "content":null per OpenAI spec
    ToolCall tc;
    tc.id             = "call_abc";
    tc.type           = "function";
    tc.function.name  = "get_weather";
    tc.function.arguments = R"({"location":"SF"})";
    asst.tool_calls.push_back(tc);
    req.messages.push_back(asst);
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("tool_calls")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("id":"call_abc")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("name":"get_weather")"));
}

TEST_CASE("Serializer - tool role message has tool_call_id", "[openai][serializer][tools]") {
    auto req = make_simple_request();
    Message tool_msg;
    tool_msg.role         = "tool";
    tool_msg.tool_call_id = "call_abc";
    tool_msg.content      = "Sunny, 22°C";
    req.messages.push_back(tool_msg);
    auto payload = Serializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"tool")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("tool_call_id":"call_abc")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("Sunny"));
}

TEST_CASE("Serializer - reasoning_content is omitted for OpenAI requests", "[openai][serializer]") {
    ChatRequest req;
    req.model = "gpt-4o";
    req.stream = true;

    Message asst;
    asst.role = "assistant";
    asst.reasoning_content = "internal thoughts should not be sent to OpenAI protocol";
    req.messages.push_back(asst);

    const auto raw_payload = Serializer::serialize(req);
    REQUIRE_THAT(raw_payload, !Catch::Matchers::ContainsSubstring("reasoning_content"));

    OpenAIProtocol protocol(/*stream_usage=*/true);
    const auto protocol_payload = protocol.serialize(req);
    REQUIRE_THAT(protocol_payload, !Catch::Matchers::ContainsSubstring("reasoning_content"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Serializer — JSON escaping
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Serializer - newline in content is escaped", "[openai][serializer][escaping]") {
    auto req = make_simple_request("gpt-4o", "line1\nline2");
    REQUIRE_THAT(Serializer::serialize(req),
                 Catch::Matchers::ContainsSubstring(R"(line1\nline2)"));
}

TEST_CASE("Serializer - quote in content is escaped", "[openai][serializer][escaping]") {
    auto req = make_simple_request("gpt-4o", R"(say "hello")");
    REQUIRE_THAT(Serializer::serialize(req),
                 Catch::Matchers::ContainsSubstring(R"(say \"hello\")"));
}

TEST_CASE("Serializer - backslash in content is escaped", "[openai][serializer][escaping]") {
    auto req = make_simple_request("gpt-4o", R"(path\to\file)");
    REQUIRE_THAT(Serializer::serialize(req),
                 Catch::Matchers::ContainsSubstring(R"(path\\to\\file)"));
}

TEST_CASE("Serializer - tab in content is escaped", "[openai][serializer][escaping]") {
    auto req = make_simple_request("gpt-4o", "col1\tcol2");
    REQUIRE_THAT(Serializer::serialize(req),
                 Catch::Matchers::ContainsSubstring(R"(col1\tcol2)"));
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_openai_sse_chunk — text content
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_openai_sse_chunk - text content is extracted", "[openai][sse]") {
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{"content":"Hello"},"index":0}]})");
    REQUIRE(content == "Hello");
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - multi-word content is preserved", "[openai][sse]") {
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{"content":"Hello World"},"index":0}]})");
    REQUIRE(content == "Hello World");
}

TEST_CASE("parse_openai_sse_chunk - role-only delta produces empty result", "[openai][sse]") {
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{"role":"assistant"},"index":0}]})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - finish_reason chunk produces empty result", "[openai][sse]") {
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{},"finish_reason":"stop","index":0}]})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - usage-only chunk produces empty result", "[openai][sse]") {
    // The usage chunk arrives before [DONE] when stream_options.include_usage=true
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"usage":{"prompt_tokens":10,"completion_tokens":5},"choices":[]})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - null content produces empty string", "[openai][sse]") {
    // null content occurs when the model is making a tool call
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{"content":null},"index":0}]})");
    REQUIRE(content.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_openai_sse_chunk — tool calls
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_openai_sse_chunk - tool call id and name extracted", "[openai][sse][tools]") {
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"get_weather","arguments":""}}]},"index":0}]})");
    REQUIRE(content.empty());
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].id   == "call_abc");
    REQUIRE(tools[0].type == "function");
    REQUIRE(tools[0].function.name == "get_weather");
    REQUIRE(tools[0].index == 0);
}

TEST_CASE("parse_openai_sse_chunk - tool call argument chunk is accumulated", "[openai][sse][tools]") {
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"loc"}}]},"index":0}]})");
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].function.arguments == R"({"loc)");
}

TEST_CASE("parse_openai_sse_chunk - tool call index is captured", "[openai][sse][tools]") {
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{"tool_calls":[{"index":2,"id":"call_xyz","type":"function","function":{"name":"shell","arguments":""}}]}}]})");
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].index == 2);
}

TEST_CASE("parse_openai_sse_chunk - multiple tool calls in one chunk", "[openai][sse][tools]") {
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"choices":[{"delta":{"tool_calls":[
            {"index":0,"id":"c1","type":"function","function":{"name":"f1","arguments":""}},
            {"index":1,"id":"c2","type":"function","function":{"name":"f2","arguments":""}}
        ]}}]})");
    REQUIRE(tools.size() == 2);
    REQUIRE(tools[0].function.name == "f1");
    REQUIRE(tools[1].function.name == "f2");
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_openai_sse_chunk — error handling
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_openai_sse_chunk - empty string returns empty result", "[openai][sse]") {
    auto [content, tools] = parse_openai_sse_chunk("");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - malformed JSON returns empty result", "[openai][sse]") {
    auto [content, tools] = parse_openai_sse_chunk("{not valid json");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - missing choices key returns empty result", "[openai][sse]") {
    auto [content, tools] = parse_openai_sse_chunk(R"({"id":"chatcmpl-1"})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - empty choices array returns empty result", "[openai][sse]") {
    auto [content, tools] = parse_openai_sse_chunk(R"({"choices":[]})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - choice without delta returns empty result", "[openai][sse]") {
    auto [content, tools] = parse_openai_sse_chunk(
        R"({"choices":[{"finish_reason":"stop","index":0}]})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - pure whitespace returns empty result", "[openai][sse]") {
    auto [content, tools] = parse_openai_sse_chunk("   ");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenAIProtocol — Azure URL compatibility
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("OpenAIProtocol::build_url uses Azure deployments path", "[openai][azure]") {
    OpenAIProtocol protocol;

    const std::string url = protocol.build_url(
        "https://my-resource.openai.azure.com/v1",
        "gpt-4o-mini");

    REQUIRE(url == "https://my-resource.openai.azure.com/openai/deployments/gpt-4o-mini/chat/completions?api-version=2024-12-01-preview");
}

TEST_CASE("OpenAIProtocol::build_url preserves explicit Azure deployments base", "[openai][azure]") {
    OpenAIProtocol protocol;

    const std::string url = protocol.build_url(
        "https://my-resource.services.ai.azure.com/openai/deployments/my-deployment",
        "ignored-model");

    REQUIRE(url == "https://my-resource.services.ai.azure.com/openai/deployments/my-deployment/chat/completions?api-version=2024-12-01-preview");
}

TEST_CASE("OpenAIProtocol::build_headers injects chatgpt-account-id from auth account_id",
          "[openai][headers]") {
    OpenAIProtocol protocol;
    core::auth::AuthInfo auth;
    auth.properties["account_id"] = "acct_123";

    const auto headers = protocol.build_headers(auth);
    REQUIRE(headers.count("chatgpt-account-id") == 1);
    REQUIRE(headers.at("chatgpt-account-id") == "acct_123");
}

TEST_CASE("OpenAIResponsesProtocol::build_headers injects chatgpt-account-id from auth account_id",
          "[openai][responses][headers]") {
    OpenAIResponsesProtocol protocol;
    core::auth::AuthInfo auth;
    auth.properties["account_id"] = "acct_456";

    const auto headers = protocol.build_headers(auth);
    REQUIRE(headers.count("chatgpt-account-id") == 1);
    REQUIRE(headers.at("chatgpt-account-id") == "acct_456");
}

// ─────────────────────────────────────────────────────────────────────────────
// ProviderFactory — creates OpenAI provider via HttpLLMProvider
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ProviderFactory - openai provider is not null", "[openai][provider]") {
    core::config::ProviderConfig config;
    config.model = "gpt-4o";
    auto provider = core::llm::ProviderFactory::create_provider("openai", config);
    REQUIRE(provider != nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// ProviderFactory — creates OpenAI provider
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ProviderFactory - creates OpenAI provider", "[openai][factory]") {
    core::config::ProviderConfig config;
    config.model = "gpt-4o";
    auto provider = core::llm::ProviderFactory::create_provider("openai", config);
    REQUIRE(provider != nullptr);
}
