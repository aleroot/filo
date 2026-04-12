#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/llm/protocols/GeminiCodeAssistProtocol.hpp"
#include "core/llm/protocols/GeminiProtocol.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/ProviderFactory.hpp"
#include <filesystem>
#include <fstream>
#include <simdjson.h>

using namespace core::llm;
using namespace core::llm::protocols;

namespace {

std::filesystem::path make_temp_image_file(std::string_view filename = "filo-gemini-image.png") {
    const auto path = std::filesystem::temp_directory_path() / filename;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "fake-image";
    return path;
}

} // namespace

TEST_CASE("GeminiProvider Serializes Valid JSON", "[GeminiProvider]") {
    ChatRequest req;
    req.model = "gemini-2.5-flash";
    req.temperature = 0.5;
    req.max_tokens = 2048;

    Message system_msg;
    system_msg.role = "system";
    system_msg.content = "You are a helpful assistant.";
    req.messages.push_back(system_msg);

    Message user_msg;
    user_msg.role = "user";
    user_msg.content = "What is the capital of France? Also, here's a quote: \"Paris\" \\ \n and some control chars \b \f \t.";
    req.messages.push_back(user_msg);

    Tool my_tool;
    my_tool.type = "function";
    my_tool.function.name = "get_weather";
    my_tool.function.description = "Get current weather";

    core::tools::ToolParameter param1;
    param1.name = "location";
    param1.type = "string";
    param1.description = "The city, e.g., San Francisco";
    param1.required = true;
    my_tool.function.parameters.push_back(param1);

    req.tools.push_back(my_tool);

    std::string json_payload = serialize_gemini_request(req, "gemini-2.5-flash");

    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_payload);
    simdjson::ondemand::document doc;

    auto error = parser.iterate(padded).get(doc);
    REQUIRE(error == simdjson::SUCCESS);

    // Verify System Prompt
    std::string_view system_text;
    REQUIRE(doc["systemInstruction"]["parts"].at(0)["text"].get_string().get(system_text) == simdjson::SUCCESS);
    REQUIRE(std::string(system_text) == "You are a helpful assistant.\n");

    // Verify Generation Config
    double temp;
    REQUIRE(doc["generationConfig"]["temperature"].get_double().get(temp) == simdjson::SUCCESS);
    REQUIRE(temp == 0.5);

    int64_t max_tokens;
    REQUIRE(doc["generationConfig"]["maxOutputTokens"].get_int64().get(max_tokens) == simdjson::SUCCESS);
    REQUIRE(max_tokens == 2048);

    // Verify User Message Content
    std::string_view user_text;
    REQUIRE(doc["contents"].at(0)["parts"].at(0)["text"].get_string().get(user_text) == simdjson::SUCCESS);
    REQUIRE(std::string(user_text).find("capital of France") != std::string::npos);

    // Verify Tools
    std::string_view tool_name;
    REQUIRE(doc["tools"].at(0)["functionDeclarations"].at(0)["name"].get_string().get(tool_name) == simdjson::SUCCESS);
    REQUIRE(std::string(tool_name) == "get_weather");
}

TEST_CASE("GeminiProvider Serializes Tool Calls and Responses", "[GeminiProvider]") {
    ChatRequest req;
    req.model = "gemini-2.5-flash";

    Message asst_msg;
    asst_msg.role = "assistant";
    asst_msg.content = ""; // Sometimes tool calls have empty text

    ToolCall call;
    call.id = "call_123";
    call.type = "function";
    call.function.name = "get_weather";
    call.function.arguments = "{\"location\":\"Paris\"}"; // JSON string
    asst_msg.tool_calls.push_back(call);

    req.messages.push_back(asst_msg);

    Message tool_msg;
    tool_msg.role = "tool";
    tool_msg.name = "get_weather";
    tool_msg.content = "{\"temperature\": 22}"; // tool response
    req.messages.push_back(tool_msg);

    std::string json_payload = serialize_gemini_request(req, "gemini-2.5-flash");

    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_payload);
    simdjson::ondemand::document doc;

    auto error = parser.iterate(padded).get(doc);
    REQUIRE(error == simdjson::SUCCESS);

    // Validate assistant tool call
    simdjson::ondemand::object fn_call;
    REQUIRE(doc["contents"].at(0)["parts"].at(0)["functionCall"].get_object().get(fn_call) == simdjson::SUCCESS);
    std::string_view call_name;
    REQUIRE(fn_call["name"].get_string().get(call_name) == simdjson::SUCCESS);
    REQUIRE(std::string(call_name) == "get_weather");

    simdjson::ondemand::object args_obj;
    REQUIRE(fn_call["args"].get_object().get(args_obj) == simdjson::SUCCESS);
    std::string_view loc;
    REQUIRE(args_obj["location"].get_string().get(loc) == simdjson::SUCCESS);
    REQUIRE(std::string(loc) == "Paris");

    // Validate tool response
    simdjson::ondemand::object fn_resp;
    REQUIRE(doc["contents"].at(1)["parts"].at(0)["functionResponse"].get_object().get(fn_resp) == simdjson::SUCCESS);
    std::string_view resp_name;
    REQUIRE(fn_resp["name"].get_string().get(resp_name) == simdjson::SUCCESS);
    REQUIRE(std::string(resp_name) == "get_weather");

    simdjson::ondemand::object resp_result;
    REQUIRE(fn_resp["response"]["result"].get_object().get(resp_result) == simdjson::SUCCESS);
    int64_t temp;
    REQUIRE(resp_result["temperature"].get_int64().get(temp) == simdjson::SUCCESS);
    REQUIRE(temp == 22);
}

TEST_CASE("GeminiProvider serializes inlineData image parts", "[GeminiProvider][vision]") {
    const auto image = make_temp_image_file();

    ChatRequest req;
    req.model = "gemini-2.5-flash";
    req.messages.push_back(Message{
        .role = "user",
        .content = describe_image_attachment(image.string()),
        .content_parts = {
            ContentPart::make_text("Summarize the error shown here."),
            ContentPart::make_image(image.string(), "image/png"),
        },
    });

    const std::string json_payload = serialize_gemini_request(req, "gemini-2.5-flash");
    REQUIRE_THAT(json_payload, Catch::Matchers::ContainsSubstring(R"("inlineData":{"mimeType":"image/png")"));
    REQUIRE_THAT(json_payload, Catch::Matchers::ContainsSubstring(R"("text":"Summarize the error shown here.")"));
}

TEST_CASE("GeminiProvider serializes JSON mode for structured output", "[GeminiProvider][json]") {
    ChatRequest req;
    req.model = "gemini-2.5-flash";
    req.response_format.type = ResponseFormat::Type::JsonObject;
    req.messages.push_back(Message{
        .role = "user",
        .content = "Return JSON only.",
    });

    const std::string json_payload = serialize_gemini_request(req, "gemini-2.5-flash");

    REQUIRE_THAT(json_payload, Catch::Matchers::ContainsSubstring(R"("responseMimeType":"application/json")"));
    REQUIRE(json_payload.find(R"("responseJsonSchema")") == std::string::npos);
}

TEST_CASE("GeminiProvider serializes responseJsonSchema for schema outputs", "[GeminiProvider][json]") {
    ChatRequest req;
    req.model = "gemini-2.5-flash";
    req.response_format.type = ResponseFormat::Type::JsonSchema;
    req.response_format.schema =
        R"({"type":"object","properties":{"city":{"type":"string"},"temp_c":{"type":"number"}},"required":["city"]})";
    req.messages.push_back(Message{
        .role = "user",
        .content = "Return weather JSON.",
    });

    const std::string json_payload = serialize_gemini_request(req, "gemini-2.5-flash");

    simdjson::ondemand::parser parser;
    simdjson::padded_string padded(json_payload);
    simdjson::ondemand::document doc;
    REQUIRE(parser.iterate(padded).get(doc) == simdjson::SUCCESS);

    std::string_view mime_type;
    REQUIRE(doc["generationConfig"]["responseMimeType"].get_string().get(mime_type)
            == simdjson::SUCCESS);
    REQUIRE(mime_type == "application/json");

    simdjson::ondemand::object schema;
    REQUIRE(doc["generationConfig"]["responseJsonSchema"].get_object().get(schema)
            == simdjson::SUCCESS);

    std::string_view type_value;
    REQUIRE(schema["type"].get_string().get(type_value) == simdjson::SUCCESS);
    REQUIRE(type_value == "object");

    simdjson::ondemand::object props;
    REQUIRE(schema["properties"].get_object().get(props) == simdjson::SUCCESS);
    REQUIRE(props["city"].error() == simdjson::SUCCESS);
    REQUIRE(props["temp_c"].error() == simdjson::SUCCESS);
}

TEST_CASE("GeminiProvider injects propertyOrdering for Gemini 2.0 schemas", "[GeminiProvider][json]") {
    ChatRequest req;
    req.model = "gemini-2.0-flash";
    req.response_format.type = ResponseFormat::Type::JsonSchema;
    req.response_format.schema =
        R"({"type":"object","properties":{"b":{"type":"string"},"a":{"type":"string"}}})";
    req.messages.push_back(Message{
        .role = "user",
        .content = "Return JSON in schema order.",
    });

    const std::string json_payload = serialize_gemini_request(req, "gemini-2.0-flash");
    REQUIRE_THAT(json_payload, Catch::Matchers::ContainsSubstring(R"("propertyOrdering":["b","a"])"));
}

TEST_CASE("normalize_requested_gemini_model resolves Gemini aliases by default", "[GeminiProvider][models]") {
    REQUIRE(normalize_requested_gemini_model("gemini-2.5-flash") == "gemini-2.5-flash");
    REQUIRE(normalize_requested_gemini_model("auto-gemini-2.5") == "gemini-2.5-pro");
    REQUIRE(normalize_requested_gemini_model("gemini-flash-latest") == "gemini-2.5-flash");
    REQUIRE(normalize_requested_gemini_model("auto-gemini-3") == "gemini-3.1-pro-preview");
    REQUIRE(normalize_requested_gemini_model("pro") == "gemini-3.1-pro-preview");
    REQUIRE(normalize_requested_gemini_model("flash") == "gemini-3-flash-preview");
    REQUIRE(normalize_requested_gemini_model("flash-lite") == "gemini-3.1-flash-lite-preview");
}

TEST_CASE("HttpLLMProvider normalizes Gemini shorthand aliases for metadata lookups",
          "[GeminiProvider][models]") {
    HttpLLMProvider provider(
        "https://generativelanguage.googleapis.com",
        core::auth::ApiKeyCredentialSource::as_query_param("test-key"),
        "pro",
        std::make_unique<GeminiProtocol>());

    REQUIRE(provider.max_context_size() == 1'048'576);

    const auto info = provider.get_model_info();
    REQUIRE(info.has_value());
    REQUIRE(info->canonical_id == "gemini-3.1-pro-preview");

    REQUIRE(provider.supports(ModelCapability::PromptCaching));
    REQUIRE(provider.estimate_cost(1'000'000, 1'000'000) == Catch::Approx(11.25));

    ChatRequest request;
    request.model = "flash-lite";
    request.max_tokens = 100'000;
    request.messages.push_back(Message{
        .role = "user",
        .content = "Return a short response.",
    });

    const auto errors = provider.validate_request(request);
    REQUIRE(errors.size() == 1);
    REQUIRE_THAT(errors.front(), Catch::Matchers::ContainsSubstring("65536"));
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_gemini_sse_chunk — SSE parsing
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_gemini_sse_chunk - text content extracted", "[gemini][sse]") {
    auto [content, tools] = parse_gemini_sse_chunk(
        R"({"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]})");
    REQUIRE(content == "Hello");
    REQUIRE(tools.empty());
}

TEST_CASE("parse_gemini_sse_chunk - multi-sentence text preserved", "[gemini][sse]") {
    auto [content, tools] = parse_gemini_sse_chunk(
        R"({"candidates":[{"content":{"role":"model","parts":[{"text":"Hello World!"}]}}]})");
    REQUIRE(content == "Hello World!");
}

TEST_CASE("parse_gemini_sse_chunk - multiple text parts are concatenated", "[gemini][sse]") {
    auto [content, tools] = parse_gemini_sse_chunk(
        R"({"candidates":[{"content":{"role":"model","parts":[{"text":"Hello "},{"text":"World"}]}}]})");
    REQUIRE(content == "Hello World");
    REQUIRE(tools.empty());
}

TEST_CASE("parse_gemini_sse_chunk - function call extracted with auto id", "[gemini][sse][tools]") {
    auto [content, tools] = parse_gemini_sse_chunk(
        R"({"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"location":"Paris"}}}]}}]})");
    REQUIRE(content.empty());
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].function.name == "get_weather");
    // ID is auto-generated as "call_gemini_N"
    REQUIRE_THAT(tools[0].id, Catch::Matchers::StartsWith("call_gemini_"));
    REQUIRE(tools[0].type == "function");
}

TEST_CASE("parse_gemini_sse_chunk - function call args are stringified", "[gemini][sse][tools]") {
    auto [content, tools] = parse_gemini_sse_chunk(
        R"({"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"x":42}}}]}}]})");
    REQUIRE(tools.size() == 1);
    REQUIRE_THAT(tools[0].function.arguments, Catch::Matchers::ContainsSubstring("42"));
}

TEST_CASE("parse_gemini_sse_chunk - empty candidates returns empty result", "[gemini][sse]") {
    auto [content, tools] = parse_gemini_sse_chunk(
        R"({"candidates":[]})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_gemini_sse_chunk - empty string returns empty result", "[gemini][sse]") {
    auto [content, tools] = parse_gemini_sse_chunk("");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_gemini_sse_chunk - malformed JSON returns empty result", "[gemini][sse]") {
    auto [content, tools] = parse_gemini_sse_chunk("{bad json");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_gemini_sse_chunk - missing candidates key returns empty result", "[gemini][sse]") {
    auto [content, tools] = parse_gemini_sse_chunk(
        R"({"usageMetadata":{"promptTokenCount":10}})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("extract_gemini_usage_metadata - prompt and completion tokens", "[gemini][usage]") {
    const auto usage = extract_gemini_usage_metadata(
        R"({"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":34}})");
    REQUIRE(usage.prompt_tokens == 12);
    REQUIRE(usage.completion_tokens == 34);
}

TEST_CASE("GeminiProtocol parse_event - accepts data prefix without a space",
          "[gemini][sse][parser]") {
    GeminiProtocol protocol;
    const auto result = protocol.parse_event(
        R"(data:{"candidates":[{"content":{"role":"model","parts":[{"text":"NoSpace"}]}}]})");

    REQUIRE_FALSE(result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "NoSpace");
}

TEST_CASE("GeminiProtocol parse_event - accepts event envelope with CRLF",
          "[gemini][sse][parser]") {
    GeminiProtocol protocol;
    const auto result = protocol.parse_event(
        "event: message\r\n"
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"CRLF\"}]}}]}\r\n");

    REQUIRE_FALSE(result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "CRLF");
}

TEST_CASE("GeminiProtocol parse_event - joins multiline data payloads",
          "[gemini][sse][parser]") {
    GeminiProtocol protocol;
    const auto result = protocol.parse_event(
        "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hello\"}]}\n"
        "data: }]}\n");

    REQUIRE_FALSE(result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "Hello");
}

TEST_CASE("GeminiProtocol parse_event - usage-only events still surface token counts",
          "[gemini][sse][parser][usage]") {
    GeminiProtocol protocol;
    const auto result = protocol.parse_event(
        R"(data: {"usageMetadata":{"promptTokenCount":21,"outputTokenCount":8}})");

    REQUIRE(result.chunks.empty());
    REQUIRE(result.prompt_tokens == 21);
    REQUIRE(result.completion_tokens == 8);
}

TEST_CASE("Gemini Code Assist request wraps Gemini payload with project", "[gemini][codeassist]") {
    ChatRequest req;
    req.model = "gemini-2.5-flash";
    req.auth_properties["project_id"] = "project-123";
    req.messages.push_back(Message{
        .role = "user",
        .content = "Hello"
    });

    const std::string payload =
        serialize_gemini_code_assist_request(req, "gemini-2.5-flash");

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"gemini-2.5-flash")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("project":"project-123")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("request":{)"));
}

TEST_CASE("Gemini Code Assist protocol parses wrapped response and usage", "[gemini][codeassist]") {
    GeminiCodeAssistProtocol protocol;
    const auto result = protocol.parse_event(
        "data: {\"response\":{\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hello from CA\"}]}}],\"usageMetadata\":{\"promptTokenCount\":7,\"candidatesTokenCount\":9}}}\n\n");

    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks.front().content == "Hello from CA");
    REQUIRE(result.prompt_tokens == 7);
    REQUIRE(result.completion_tokens == 9);
}

// ─────────────────────────────────────────────────────────────────────────────
// ProviderFactory — creates Gemini provider
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ProviderFactory - creates Gemini provider", "[gemini][factory]") {
    core::config::ProviderConfig config;
    config.model = "gemini-2.5-flash";
    auto provider = core::llm::ProviderFactory::create_provider("gemini", config);
    REQUIRE(provider != nullptr);
}
