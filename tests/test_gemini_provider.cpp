#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/llm/protocols/GeminiAntigravityProtocol.hpp"
#include "core/llm/protocols/AntigravityConversationState.hpp"
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

TEST_CASE("Gemini serializer keeps malformed tool data inside valid JSON",
          "[GeminiProvider][tool-call][regression]") {
    ChatRequest req;
    req.model = "gemini-2.5-flash";
    req.messages.push_back(Message{
        .role = "assistant",
        .tool_calls = {ToolCall{
            .id = "call_bad_args",
            .function = {
                .name = "edit",
                .arguments = R"({"path":)",
            },
        }},
    });
    req.messages.push_back(Message{
        .role = "tool",
        .content = "[invalid tool arguments: arguments are not valid JSON]",
        .name = "edit",
    });

    const std::string payload = serialize_gemini_request(req, req.model);
    simdjson::dom::parser parser;
    const auto document = parser.parse(payload);

    simdjson::dom::object arguments;
    REQUIRE(document["contents"].at(0)["parts"].at(0)["functionCall"]["args"]
                .get_object().get(arguments) == simdjson::SUCCESS);
    CHECK(arguments.size() == 0);
    CHECK(document["contents"].at(1)["parts"].at(0)["functionResponse"]
              ["response"]["result"].get_string().value()
          == "[invalid tool arguments: arguments are not valid JSON]");
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

TEST_CASE("Gemini protocols own canonical model alias resolution", "[GeminiProvider][models]") {
    GeminiProtocol gemini;
    GeminiCodeAssistProtocol code_assist;
    GeminiAntigravityProtocol antigravity;

    REQUIRE(gemini.model_id("flash") == "gemini-3-flash-preview");
    REQUIRE(code_assist.model_id("flash") == "gemini-3-flash-preview");
    REQUIRE(antigravity.model_id("flash") == "gemini-3-flash-preview");
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

TEST_CASE("HttpLLMProvider uses inherited protocol model canonicalization",
          "[GeminiProvider][models][antigravity]") {
    HttpLLMProvider provider(
        "https://daily-cloudcode-pa.googleapis.com",
        nullptr,
        "pro",
        std::make_unique<GeminiAntigravityProtocol>());

    const auto info = provider.get_model_info();
    REQUIRE(info.has_value());
    REQUIRE(info->canonical_id == "gemini-3.1-pro-preview");
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
// GeminiAntigravityProtocol — unofficial Antigravity OAuth variant
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GeminiAntigravityProtocol wraps request with userAgent and requestId",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    ChatRequest req;
    req.model = "gemini-2.5-flash";
    req.auth_properties["project_id"] = "project-123";
    req.messages.push_back(Message{.role = "user", .content = "Hello"});

    const std::string payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"gemini-2.5-flash")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("project":"project-123")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("request":{)"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("userAgent":"antigravity")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("requestType":"agent")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("requestId":"agent/)"));

    // Must still be a single well-formed JSON object.
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(payload).get(doc) == simdjson::SUCCESS);
}

TEST_CASE("GeminiAntigravityProtocol requestId is unique per call", "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    ChatRequest req;
    req.model = "gemini-2.5-flash";
    req.messages.push_back(Message{.role = "user", .content = "Hi"});

    const std::string first = protocol.serialize(req);
    const std::string second = protocol.serialize(req);
    REQUIRE(first != second);
}

TEST_CASE("GeminiAntigravityProtocol build_headers sets Antigravity client identity",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    core::auth::AuthInfo auth;
    auth.headers["Authorization"] = "Bearer test-token";

    const cpr::Header headers = protocol.build_headers(auth);

    REQUIRE(headers.at("Authorization") == "Bearer test-token");
    REQUIRE(headers.at("Content-Type") == "application/json");
    REQUIRE(headers.count("User-Agent") == 1);
    REQUIRE_THAT(headers.at("User-Agent"),
                 Catch::Matchers::StartsWith("antigravity/hub/" + std::string(kAntigravityClientVersion)
                                             + " (aidev_client; os_type=darwin; arch=arm64;"));
    // The real hub client sends no X-Goog-Api-Client / Client-Metadata on
    // these calls; adding them would fingerprint Filo as a third-party client.
    REQUIRE(headers.count("X-Goog-Api-Client") == 0);
    REQUIRE(headers.count("Client-Metadata") == 0);
}

TEST_CASE("GeminiAntigravityProtocol name and url match Cloud Code Assist wire format",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    REQUIRE(protocol.name() == "gemini_antigravity");
    REQUIRE(protocol.build_url("https://daily-cloudcode-pa.googleapis.com", "gemini-2.5-flash")
            == "https://daily-cloudcode-pa.googleapis.com/v1internal:streamGenerateContent?alt=sse");
}

namespace {
ChatRequest antigravity_request(std::string model, std::string prompt = "Hello") {
    ChatRequest req;
    req.model = std::move(model);
    req.messages.push_back(Message{.role = "system", .content = "Be brief."});
    req.messages.push_back(Message{.role = "user", .content = std::move(prompt)});
    return req;
}

simdjson::dom::element parse_json(simdjson::dom::parser& parser, const std::string& json) {
    simdjson::dom::element doc;
    REQUIRE(parser.parse(json).get(doc) == simdjson::SUCCESS);
    return doc;
}
} // namespace

TEST_CASE("GeminiAntigravityProtocol rotates daily and sandbox on failed attempts",
          "[gemini][antigravity][failover]") {
    GeminiAntigravityProtocol protocol;
    cpr::Header headers;
    const std::string daily = std::string(kAntigravityEndpoint);
    const std::string sandbox = std::string(kAntigravitySandboxEndpoint);
    const auto url = [&](std::string_view base) { return protocol.build_url(base, "m"); };
    const auto fail = [&](int status) { protocol.on_response(HttpResponse{status, {}, headers}); };
    const std::string tail = "/v1internal:streamGenerateContent?alt=sse";

    REQUIRE(url(daily) == daily + tail);
    fail(503);
    REQUIRE(url(daily) == sandbox + tail);
    fail(0);                                  // transport failure
    REQUIRE(url(daily) == daily + tail);      // cycles back
    fail(429);
    REQUIRE(url(daily + "/") == sandbox + tail);
    fail(400);                                // not an endpoint failure
    REQUIRE(url(daily) == sandbox + tail);
    protocol.on_response(HttpResponse{200, {}, headers});
    REQUIRE(url(daily) == sandbox + tail);
}

TEST_CASE("GeminiAntigravityProtocol never rotates a user-supplied endpoint",
          "[gemini][antigravity][failover]") {
    GeminiAntigravityProtocol protocol;
    cpr::Header headers;
    protocol.on_response(HttpResponse{503, {}, headers});
    REQUIRE(protocol.build_url("http://localhost:9999", "m")
            == "http://localhost:9999/v1internal:streamGenerateContent?alt=sse");
}

TEST_CASE("GeminiAntigravityProtocol retries only transient failures",
          "[gemini][antigravity][failover]") {
    GeminiAntigravityProtocol protocol;
    cpr::Header headers;
    for (const int status : {0, 408, 429, 500, 503}) {
        CHECK(protocol.is_retryable(HttpResponse{status, {}, headers}));
    }
    for (const int status : {400, 401, 403, 404}) {
        CHECK_FALSE(protocol.is_retryable(HttpResponse{status, {}, headers}));
    }
}

TEST_CASE("GeminiAntigravityProtocol envelope mirrors the hub client",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    simdjson::dom::parser parser;
    const auto doc = parse_json(parser, protocol.serialize(antigravity_request("gemini-3.1-pro-preview")));

    std::string_view request_id;
    REQUIRE(doc["requestId"].get(request_id) == simdjson::SUCCESS);
    REQUIRE_THAT(std::string(request_id),
                 Catch::Matchers::Matches(
                     "agent/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
                     "/[0-9]+/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/2"));
    std::string_view user_agent, request_type, session_id, role, tool_mode;
    REQUIRE(doc["userAgent"].get(user_agent) == simdjson::SUCCESS);
    REQUIRE(doc["requestType"].get(request_type) == simdjson::SUCCESS);
    REQUIRE(user_agent == "antigravity");
    REQUIRE(request_type == "agent");
    REQUIRE(doc["request"]["sessionId"].get(session_id) == simdjson::SUCCESS);
    REQUIRE(session_id.starts_with('-'));
    REQUIRE(doc["request"]["systemInstruction"]["role"].get(role) == simdjson::SUCCESS);
    REQUIRE(role == "user");
    std::string_view last_step, used_claude;
    REQUIRE(doc["request"]["labels"]["last_step_index"].get(last_step) == simdjson::SUCCESS);
    REQUIRE(last_step == "1");
    REQUIRE(doc["request"]["labels"]["used_claude"].get(used_claude) == simdjson::SUCCESS);
    REQUIRE(used_claude == "false");
    // No tools declared -> no toolConfig.
    REQUIRE(doc["request"]["toolConfig"].error() == simdjson::NO_SUCH_FIELD);
    REQUIRE(tool_mode.empty());
}

TEST_CASE("GeminiAntigravityProtocol keeps session ids stable and advances the step",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    auto first = antigravity_request("gemini-3.1-pro-preview", "Same conversation");
    auto second = first;
    second.messages.push_back(Message{.role = "assistant", .content = "Hi"});
    second.messages.push_back(Message{.role = "user", .content = "More"});
    auto other = antigravity_request("gemini-3.1-pro-preview", "Different conversation");

    simdjson::dom::parser p1, p2, p3;
    const auto d1 = parse_json(p1, protocol.serialize(first));
    const auto d2 = parse_json(p2, protocol.serialize(second));
    const auto d3 = parse_json(p3, protocol.serialize(other));

    std::string_view s1, s2, s3, l2, t1, t2;
    REQUIRE(d1["request"]["sessionId"].get(s1) == simdjson::SUCCESS);
    REQUIRE(d2["request"]["sessionId"].get(s2) == simdjson::SUCCESS);
    REQUIRE(d3["request"]["sessionId"].get(s3) == simdjson::SUCCESS);
    REQUIRE(s1 == s2);
    REQUIRE(s1 != s3);
    REQUIRE(d2["request"]["labels"]["last_step_index"].get(l2) == simdjson::SUCCESS);
    REQUIRE(l2 == "2");
    REQUIRE(d1["request"]["labels"]["trajectory_id"].get(t1) == simdjson::SUCCESS);
    REQUIRE(d2["request"]["labels"]["trajectory_id"].get(t2) == simdjson::SUCCESS);
    REQUIRE(t1 == t2);
}

TEST_CASE("GeminiAntigravityProtocol prefers the explicit session id as anchor",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    auto a = antigravity_request("gemini-3.1-pro-preview", "First prompt");
    auto b = antigravity_request("gemini-3.1-pro-preview", "Edited prompt");
    a.session_id = b.session_id = "session-42";

    simdjson::dom::parser pa, pb;
    std::string_view sa, sb;
    REQUIRE(parse_json(pa, protocol.serialize(a))["request"]["sessionId"].get(sa) == simdjson::SUCCESS);
    REQUIRE(parse_json(pb, protocol.serialize(b))["request"]["sessionId"].get(sb) == simdjson::SUCCESS);
    REQUIRE(sa == sb);
}

TEST_CASE("GeminiAntigravityProtocol validates tool calling and signs Gemini 3 calls",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    auto req = antigravity_request("gemini-3.1-pro-preview");
    core::tools::ToolDefinition def;
    def.name = "read_file";
    def.description = "Read";
    def.input_schema = R"({"type":"object","properties":{}})";
    req.tools.push_back(Tool{.function = def});
    Message call{.role = "assistant"};
    ToolCall tc;
    tc.id = "c1";
    tc.function.name = "read_file";
    tc.function.arguments = "{}";
    call.tool_calls.push_back(tc);
    call.tool_calls.push_back(tc);
    req.messages.push_back(call);

    const std::string payload = protocol.serialize(req);
    simdjson::dom::parser parser;
    const auto doc = parse_json(parser, payload);
    std::string_view mode;
    REQUIRE(doc["request"]["toolConfig"]["functionCallingConfig"]["mode"].get(mode)
            == simdjson::SUCCESS);
    REQUIRE(mode == "VALIDATED");

    // Only the first call of the turn carries the bypass signature.
    const std::string needle = R"("thoughtSignature":"skip_thought_signature_validator")";
    const auto first = payload.find(needle);
    REQUIRE(first != std::string::npos);
    REQUIRE(payload.find(needle, first + 1) == std::string::npos);
}

TEST_CASE("GeminiAntigravityProtocol caps Claude output and skips the signature bypass",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    simdjson::dom::parser parser;

    auto claude = antigravity_request("claude-sonnet-4-6");
    auto doc = parse_json(parser, protocol.serialize(claude));
    int64_t max_tokens = 0;
    REQUIRE(doc["request"]["generationConfig"]["maxOutputTokens"].get(max_tokens) == simdjson::SUCCESS);
    REQUIRE(max_tokens == 64000);
    std::string_view used_claude;
    REQUIRE(doc["request"]["labels"]["used_claude"].get(used_claude) == simdjson::SUCCESS);
    REQUIRE(used_claude == "true");

    claude.max_tokens = 100000;
    simdjson::dom::parser parser2;
    doc = parse_json(parser2, protocol.serialize(claude));
    REQUIRE(doc["request"]["generationConfig"]["maxOutputTokens"].get(max_tokens) == simdjson::SUCCESS);
    REQUIRE(max_tokens == 64000);

    claude.max_tokens = 2048;
    simdjson::dom::parser parser3;
    doc = parse_json(parser3, protocol.serialize(claude));
    REQUIRE(doc["request"]["generationConfig"]["maxOutputTokens"].get(max_tokens) == simdjson::SUCCESS);
    REQUIRE(max_tokens == 2048);

    // Gemini keeps whatever the caller asked for.
    auto gemini = antigravity_request("gemini-3.1-pro-preview");
    simdjson::dom::parser parser4;
    doc = parse_json(parser4, protocol.serialize(gemini));
    REQUIRE(doc["request"]["generationConfig"]["maxOutputTokens"].error() == simdjson::NO_SUCH_FIELD);
}

TEST_CASE("GeminiAntigravityProtocol tags wire-profile models with model_enum and a fixed cap",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    simdjson::dom::parser p1, p2, p3;

    const auto profiled =
        parse_json(p1, protocol.serialize(antigravity_request("gemini-3.1-pro-low", "Profiled wire model")));
    std::string_view model_enum;
    REQUIRE(profiled["request"]["labels"]["model_enum"].get(model_enum) == simdjson::SUCCESS);
    REQUIRE(model_enum == "MODEL_PLACEHOLDER_M36");
    int64_t max_tokens = 0;
    REQUIRE(profiled["request"]["generationConfig"]["maxOutputTokens"].get(max_tokens)
            == simdjson::SUCCESS);
    REQUIRE(max_tokens == 65535);

    // The profile cap is a ceiling: a smaller caller request passes through.
    auto clamped = antigravity_request("gemini-3.1-pro-low", "Clamped wire model");
    clamped.max_tokens = 100'000;
    const auto clamped_doc = parse_json(p2, protocol.serialize(clamped));
    REQUIRE(clamped_doc["request"]["generationConfig"]["maxOutputTokens"].get(max_tokens)
            == simdjson::SUCCESS);
    REQUIRE(max_tokens == 65535);

    // Unknown ids carry no telemetry label and no forced output cap.
    const auto unknown = parse_json(
        p3, protocol.serialize(antigravity_request("gemini-3.1-pro-preview", "Unknown wire model")));
    REQUIRE(unknown["request"]["labels"]["model_enum"].error() == simdjson::NO_SUCH_FIELD);
    REQUIRE(unknown["request"]["generationConfig"]["maxOutputTokens"].error()
            == simdjson::NO_SUCH_FIELD);
}

TEST_CASE("Antigravity responseId becomes the next request's last_execution_id",
          "[gemini][antigravity]") {
    AntigravityConversationState::instance().clear();
    GeminiAntigravityProtocol protocol;
    auto first = antigravity_request("gemini-3.1-pro-low", "Execution handle conversation");

    protocol.prepare_request(first);
    simdjson::dom::parser p1;
    const auto d1 = parse_json(p1, protocol.serialize(first));
    REQUIRE(d1["request"]["labels"]["last_execution_id"].error() == simdjson::NO_SUCH_FIELD);

    const auto parsed = protocol.parse_event(
        "data: {\"response\":{\"responseId\":\"resp-77\",\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hi\"}]}}]}}\n\n");
    REQUIRE(parsed.response_id == "resp-77");

    auto second = first;
    second.messages.push_back(Message{.role = "assistant", .content = "Hi"});
    second.messages.push_back(Message{.role = "user", .content = "More"});
    simdjson::dom::parser p2;
    const auto d2 = parse_json(p2, protocol.serialize(second));
    std::string_view execution_id;
    REQUIRE(d2["request"]["labels"]["last_execution_id"].get(execution_id) == simdjson::SUCCESS);
    REQUIRE(execution_id == "resp-77");
}

TEST_CASE("AntigravityConversationState drops the oldest conversations at capacity",
          "[gemini][antigravity]") {
    auto& state = AntigravityConversationState::instance();
    state.clear();

    constexpr std::size_t overflow = 8;
    const std::size_t total = AntigravityConversationState::kMaxTrackedConversations + overflow;
    for (std::size_t i = 0; i < total; ++i) {
        state.record_execution_id("anchor-" + std::to_string(i), "exec-" + std::to_string(i));
    }

    CHECK(state.execution_id("anchor-0").empty());
    CHECK(state.execution_id("anchor-" + std::to_string(total - 1))
          == "exec-" + std::to_string(total - 1));
}

TEST_CASE("Plain Code Assist serialization is unchanged by Antigravity extras",
          "[gemini][code-assist]") {
    GeminiCodeAssistProtocol protocol;
    auto req = antigravity_request("gemini-2.5-flash");
    const std::string payload = protocol.serialize(req);
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("sessionId"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("labels"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("toolConfig"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("systemInstruction":{"parts")"));
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
TEST_CASE("Code Assist request stays valid JSON without a project id",
          "[gemini][code-assist]") {
    GeminiCodeAssistProtocol protocol;
    simdjson::dom::parser parser;
    const auto doc = parse_json(parser, protocol.serialize(antigravity_request("gemini-2.5-flash")));
    std::string_view model;
    REQUIRE(doc["model"].get(model) == simdjson::SUCCESS);
    REQUIRE(model == "gemini-2.5-flash");
    REQUIRE(doc["project"].error() == simdjson::NO_SUCH_FIELD);
}
