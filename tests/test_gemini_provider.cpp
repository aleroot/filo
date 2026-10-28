#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/auth/GoogleCodeAssist.hpp"
#include "core/llm/protocols/GeminiAntigravityProtocol.hpp"
#include "core/llm/AntigravityUsageClient.hpp"
#include "core/llm/protocols/AntigravityConversationState.hpp"
#include "core/llm/protocols/GeminiCodeAssistProtocol.hpp"
#include "core/llm/protocols/GeminiProtocol.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/ProviderFactory.hpp"
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <simdjson.h>
#include <stdexcept>

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

TEST_CASE("Gemini tool results ride on a model turn", "[GeminiProvider][tool-call]") {
    GeminiProtocol protocol;
    ChatRequest req;
    req.model = "gemini-2.5-flash";
    req.messages.push_back(Message{.role = "user", .content = "Run it"});
    Message call{.role = "assistant"};
    ToolCall tc;
    tc.id = "call_gemini_7";
    tc.function.name = "run_command";
    tc.function.arguments = R"({"CommandLine":"echo hi"})";
    call.tool_calls.push_back(tc);
    req.messages.push_back(call);
    req.messages.push_back(Message{.role = "tool",
                                   .content = "hi\n",
                                   .name = "run_command",
                                   .tool_call_id = "call_gemini_7"});

    const std::string payload = protocol.serialize(req);

    // Cloud Code Assist rejects role "function" outright (400
    // INVALID_ARGUMENT: "Role 'function' is not supported"), and the ids pair a
    // result with its call when a turn issued several calls of the same tool.
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("role":"function")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("role":"model","parts":[{"functionResponse":{"id":"call_gemini_7")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("functionCall":{"id":"call_gemini_7","name":"run_command")"));
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
    const auto parts = parse_gemini_sse_chunk(
        R"({"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]})");
    REQUIRE(parts.content == "Hello");
    REQUIRE(parts.reasoning.empty());
    REQUIRE(parts.tools.empty());
}

TEST_CASE("parse_gemini_sse_chunk - multi-sentence text preserved", "[gemini][sse]") {
    const auto parts = parse_gemini_sse_chunk(
        R"({"candidates":[{"content":{"role":"model","parts":[{"text":"Hello World!"}]}}]})");
    REQUIRE(parts.content == "Hello World!");
}

TEST_CASE("parse_gemini_sse_chunk - multiple text parts are concatenated", "[gemini][sse]") {
    const auto parts = parse_gemini_sse_chunk(
        R"({"candidates":[{"content":{"role":"model","parts":[{"text":"Hello "},{"text":"World"}]}}]})");
    REQUIRE(parts.content == "Hello World");
    REQUIRE(parts.tools.empty());
}

TEST_CASE("parse_gemini_sse_chunk - thought parts are reasoning, not content", "[gemini][sse]") {
    const auto parts = parse_gemini_sse_chunk(
        R"({"candidates":[{"content":{"role":"model","parts":[{"thought":true,"text":"Thinking hard."},{"text":"The answer."}]}}]})");
    REQUIRE(parts.content == "The answer.");
    REQUIRE(parts.reasoning == "Thinking hard.");
    REQUIRE(parts.tools.empty());
}

TEST_CASE("parse_gemini_sse_chunk - function call extracted with auto id", "[gemini][sse][tools]") {
    const auto parts = parse_gemini_sse_chunk(
        R"({"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"get_weather","args":{"location":"Paris"}}}]}}]})");
    REQUIRE(parts.content.empty());
    REQUIRE(parts.tools.size() == 1);
    REQUIRE(parts.tools[0].function.name == "get_weather");
    // ID is auto-generated as "call_gemini_N"
    REQUIRE_THAT(parts.tools[0].id, Catch::Matchers::StartsWith("call_gemini_"));
    REQUIRE(parts.tools[0].type == "function");
}

TEST_CASE("parse_gemini_sse_chunk - function call args are stringified", "[gemini][sse][tools]") {
    const auto parts = parse_gemini_sse_chunk(
        R"({"candidates":[{"content":{"role":"model","parts":[{"functionCall":{"name":"f","args":{"x":42}}}]}}]})");
    REQUIRE(parts.tools.size() == 1);
    REQUIRE_THAT(parts.tools[0].function.arguments, Catch::Matchers::ContainsSubstring("42"));
}

TEST_CASE("parse_gemini_sse_chunk - empty candidates returns empty result", "[gemini][sse]") {
    const auto parts = parse_gemini_sse_chunk(
        R"({"candidates":[]})");
    REQUIRE(parts.content.empty());
    REQUIRE(parts.tools.empty());
}

TEST_CASE("parse_gemini_sse_chunk - empty string returns empty result", "[gemini][sse]") {
    const auto parts = parse_gemini_sse_chunk("");
    REQUIRE(parts.content.empty());
    REQUIRE(parts.tools.empty());
}

TEST_CASE("parse_gemini_sse_chunk - malformed JSON returns empty result", "[gemini][sse]") {
    const auto parts = parse_gemini_sse_chunk("{bad json");
    REQUIRE(parts.content.empty());
    REQUIRE(parts.tools.empty());
}

TEST_CASE("parse_gemini_sse_chunk - missing candidates key returns empty result", "[gemini][sse]") {
    const auto parts = parse_gemini_sse_chunk(
        R"({"usageMetadata":{"promptTokenCount":10}})");
    REQUIRE(parts.content.empty());
    REQUIRE(parts.tools.empty());
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

/// The envelope's `requestId`, as a view owned by the parsed document.
std::string_view request_id_of(simdjson::dom::element doc) {
    std::string_view value;
    REQUIRE(doc["requestId"].get(value) == simdjson::SUCCESS);
    return value;
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
    // agent/<cascadeId>/<epoch ms>/<trajectoryId>/<contents>; a first turn
    // sends exactly one content, so the request number is 1.
    REQUIRE_THAT(std::string(request_id),
                 Catch::Matchers::Matches(
                     "agent/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
                     "/[0-9]+/[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}/1"));
    std::string_view user_agent, request_type, session_id, role;
    REQUIRE(doc["userAgent"].get(user_agent) == simdjson::SUCCESS);
    REQUIRE(doc["requestType"].get(request_type) == simdjson::SUCCESS);
    REQUIRE(user_agent == "antigravity");
    REQUIRE(request_type == "agent");
    // A signed 64-bit integer, one value per installation.
    REQUIRE(doc["request"]["sessionId"].get(session_id) == simdjson::SUCCESS);
    REQUIRE(!session_id.empty());
    std::int64_t session_value = 0;
    const auto* session_end = session_id.data() + session_id.size();
    REQUIRE(std::from_chars(session_id.data(), session_end, session_value).ptr
            == session_end);
    REQUIRE(doc["request"]["systemInstruction"]["role"].get(role) == simdjson::SUCCESS);
    REQUIRE(role == "user");

    // The hub sends no toolConfig at all.
    REQUIRE(doc["request"]["toolConfig"].error() == simdjson::NO_SUCH_FIELD);
    // No credential resolved a project, so the envelope carries none.
    REQUIRE(doc["project"].error() == simdjson::NO_SUCH_FIELD);

    // labels is alphabetical and carries the full hub key set.
    const auto labels = doc["request"]["labels"];
    std::string_view cascade_id, root_cascade_id, trajectory_id, label_request_id;
    std::string_view last_step, used_claude, used_claude_conservative, used_non_gemini;
    REQUIRE(labels["cascade_id"].get(cascade_id) == simdjson::SUCCESS);
    REQUIRE(labels["root_cascade_id"].get(root_cascade_id) == simdjson::SUCCESS);
    REQUIRE(cascade_id == root_cascade_id);
    REQUIRE(labels["trajectory_id"].get(trajectory_id) == simdjson::SUCCESS);
    REQUIRE(labels["request_id"].get(label_request_id) == simdjson::SUCCESS);
    REQUIRE(label_request_id == std::string(trajectory_id) + "-0");
    REQUIRE(labels["last_step_index"].get(last_step) == simdjson::SUCCESS);
    REQUIRE(last_step == "0");
    REQUIRE(labels["used_claude"].get(used_claude) == simdjson::SUCCESS);
    REQUIRE(used_claude == "false");
    REQUIRE(labels["used_claude_conservative"].get(used_claude_conservative)
            == simdjson::SUCCESS);
    REQUIRE(used_claude_conservative == "false");
    REQUIRE(labels["used_non_gemini_model"].get(used_non_gemini) == simdjson::SUCCESS);
    REQUIRE(used_non_gemini == "false");
    // The cascade and trajectory ids in the envelope are the labelled ones.
    REQUIRE_THAT(std::string(request_id),
                 Catch::Matchers::ContainsSubstring(std::string(cascade_id)));
    REQUIRE_THAT(std::string(request_id),
                 Catch::Matchers::ContainsSubstring(std::string(trajectory_id)));
}

TEST_CASE("GeminiAntigravityProtocol sends the resolved project in the envelope",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    auto req = antigravity_request("gemini-4-argon-high");
    req.auth_properties["project_id"] = "aicode-consumers";

    simdjson::dom::parser parser;
    const auto doc = parse_json(parser, protocol.serialize(req));
    std::string_view project;
    REQUIRE(doc["project"].get(project) == simdjson::SUCCESS);
    REQUIRE(project == "aicode-consumers");
    std::string_view model;
    REQUIRE(doc["model"].get(model) == simdjson::SUCCESS);
    REQUIRE(model == "gemini-4-argon-high");
}

TEST_CASE("GeminiAntigravityProtocol keeps one session id and numbers turns like the hub",
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

    // sessionId is per installation, not per conversation.
    std::string_view s1, s2, s3, l2, t1, t2, t3;
    REQUIRE(d1["request"]["sessionId"].get(s1) == simdjson::SUCCESS);
    REQUIRE(d2["request"]["sessionId"].get(s2) == simdjson::SUCCESS);
    REQUIRE(d3["request"]["sessionId"].get(s3) == simdjson::SUCCESS);
    REQUIRE(s1 == s2);
    REQUIRE(s1 == s3);

    // The three turns are: one content, then three after a tool round.
    REQUIRE(d2["request"]["labels"]["last_step_index"].get(l2) == simdjson::SUCCESS);
    REQUIRE(l2 == "2");
    REQUIRE_THAT(std::string(request_id_of(d1)), Catch::Matchers::EndsWith("/1"));
    REQUIRE_THAT(std::string(request_id_of(d2)), Catch::Matchers::EndsWith("/3"));

    // Distinct conversations get distinct trajectories, and the request ordinal
    // inside a trajectory counts prior assistant turns.
    REQUIRE(d1["request"]["labels"]["trajectory_id"].get(t1) == simdjson::SUCCESS);
    REQUIRE(d2["request"]["labels"]["trajectory_id"].get(t2) == simdjson::SUCCESS);
    REQUIRE(d3["request"]["labels"]["trajectory_id"].get(t3) == simdjson::SUCCESS);
    REQUIRE(t1 == t2);
    REQUIRE(t1 != t3);
    std::string_view r1, r2;
    REQUIRE(d1["request"]["labels"]["request_id"].get(r1) == simdjson::SUCCESS);
    REQUIRE(d2["request"]["labels"]["request_id"].get(r2) == simdjson::SUCCESS);
    REQUIRE(r1 == std::string(t1) + "-0");
    REQUIRE(r2 == std::string(t2) + "-1");
}

TEST_CASE("GeminiAntigravityProtocol anchors its ids on the explicit session id",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    auto a = antigravity_request("gemini-3.1-pro-preview", "First prompt");
    auto b = antigravity_request("gemini-3.1-pro-preview", "Edited prompt");
    a.session_id = b.session_id = "session-42";

    simdjson::dom::parser pa, pb;
    const auto da = parse_json(pa, protocol.serialize(a));
    const auto db = parse_json(pb, protocol.serialize(b));

    // The prompts differ, so only the explicit session id can tie the two
    // requests to one trajectory.
    std::string_view ta, tb;
    REQUIRE(da["request"]["labels"]["trajectory_id"].get(ta) == simdjson::SUCCESS);
    REQUIRE(db["request"]["labels"]["trajectory_id"].get(tb) == simdjson::SUCCESS);
    REQUIRE(ta == tb);
}

TEST_CASE("GeminiAntigravityProtocol mirrors the hub tool shape and signs every call",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    auto req = antigravity_request("gemini-3.1-pro-preview");
    core::tools::ToolDefinition def;
    def.name = "read_file";
    def.description = "Read";
    def.input_schema = R"({"type":"object","properties":{}})";
    core::tools::ToolDefinition second = def;
    second.name = "run_command";
    req.tools.push_back(Tool{.function = def});
    req.tools.push_back(Tool{.function = second});
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

    // The hub sends no toolConfig, not even with tools declared.
    REQUIRE(doc["request"]["toolConfig"].error() == simdjson::NO_SUCH_FIELD);

    // One `functionDeclarations` element per tool.
    simdjson::dom::array tools;
    REQUIRE(doc["request"]["tools"].get(tools) == simdjson::SUCCESS);
    REQUIRE(tools.size() == 2);
    for (auto tool_result : tools) {
        simdjson::dom::element tool;
        REQUIRE(tool_result.get(tool) == simdjson::SUCCESS);
        simdjson::dom::array declarations;
        REQUIRE(tool["functionDeclarations"].get(declarations) == simdjson::SUCCESS);
        REQUIRE(declarations.size() == 1);
    }

    // Every replayed call part is signed: the hub carries one thoughtSignature
    // per part, so a turn that issued two calls has two of them.
    const std::string needle = R"("thoughtSignature":"skip_thought_signature_validator")";
    std::size_t signatures = 0;
    for (auto at = payload.find(needle); at != std::string::npos;
         at = payload.find(needle, at + needle.size())) {
        ++signatures;
    }
    REQUIRE(signatures == 2);
}

TEST_CASE("GeminiAntigravityProtocol caps Claude output and skips the signature bypass",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    simdjson::dom::parser parser;

    auto claude = antigravity_request("claude-sonnet-5-5-high");
    auto doc = parse_json(parser, protocol.serialize(claude));
    int64_t max_tokens = 0;
    REQUIRE(doc["request"]["generationConfig"]["maxOutputTokens"].get(max_tokens) == simdjson::SUCCESS);
    REQUIRE(max_tokens == 128000);
    std::string_view used_claude, used_non_gemini, model_enum;
    REQUIRE(doc["request"]["labels"]["used_claude"].get(used_claude) == simdjson::SUCCESS);
    REQUIRE(used_claude == "true");
    REQUIRE(doc["request"]["labels"]["used_non_gemini_model"].get(used_non_gemini)
            == simdjson::SUCCESS);
    REQUIRE(used_non_gemini == "true");
    REQUIRE(doc["request"]["labels"]["model_enum"].get(model_enum) == simdjson::SUCCESS);
    REQUIRE(model_enum == "MODEL_PLACEHOLDER_M405");

    claude.max_tokens = 200000;
    simdjson::dom::parser parser2;
    doc = parse_json(parser2, protocol.serialize(claude));
    REQUIRE(doc["request"]["generationConfig"]["maxOutputTokens"].get(max_tokens) == simdjson::SUCCESS);
    REQUIRE(max_tokens == 128000);

    claude.max_tokens = 2048;
    simdjson::dom::parser parser3;
    doc = parse_json(parser3, protocol.serialize(claude));
    REQUIRE(doc["request"]["generationConfig"]["maxOutputTokens"].get(max_tokens) == simdjson::SUCCESS);
    REQUIRE(max_tokens == 2048);

    // The signature bypass is a Gemini 3+ requirement only.
    auto claude_calls = claude;
    Message call{.role = "assistant"};
    ToolCall tc;
    tc.id = "c1";
    tc.function.name = "read_file";
    tc.function.arguments = "{}";
    call.tool_calls.push_back(tc);
    claude_calls.messages.push_back(call);
    const std::string claude_payload = protocol.serialize(claude_calls);
    REQUIRE_THAT(claude_payload, !Catch::Matchers::ContainsSubstring("thoughtSignature"));

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
        parse_json(p1, protocol.serialize(antigravity_request("gemini-4-argon-high", "Profiled wire model")));
    std::string_view model_enum;
    REQUIRE(profiled["request"]["labels"]["model_enum"].get(model_enum) == simdjson::SUCCESS);
    REQUIRE(model_enum == "MODEL_PLACEHOLDER_M417");
    int64_t max_tokens = 0;
    REQUIRE(profiled["request"]["generationConfig"]["maxOutputTokens"].get(max_tokens)
            == simdjson::SUCCESS);
    REQUIRE(max_tokens == 65536);

    // The profile cap is a ceiling: a smaller caller request passes through.
    auto clamped = antigravity_request("gemini-4-argon-high", "Clamped wire model");
    clamped.max_tokens = 100'000;
    const auto clamped_doc = parse_json(p2, protocol.serialize(clamped));
    REQUIRE(clamped_doc["request"]["generationConfig"]["maxOutputTokens"].get(max_tokens)
            == simdjson::SUCCESS);
    REQUIRE(max_tokens == 65536);

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
    auto first = antigravity_request("gemini-3.8-flash-high", "Execution handle conversation");

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

TEST_CASE("GeminiAntigravityProtocol asks for thought summaries at the SKU's level",
          "[gemini][antigravity]") {
    GeminiAntigravityProtocol protocol;
    simdjson::dom::parser p1, p2, p3, p4;

    const auto argon =
        parse_json(p1, protocol.serialize(antigravity_request("gemini-4-argon-high")));
    bool include_thoughts = false;
    REQUIRE(argon["request"]["generationConfig"]["thinkingConfig"]["includeThoughts"]
                .get(include_thoughts)
            == simdjson::SUCCESS);
    CHECK(include_thoughts);
    std::string_view level;
    REQUIRE(argon["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"].get(level)
            == simdjson::SUCCESS);
    CHECK(level == "HIGH");

    // The caller's effort selects the SKU: a configured variant id is only the
    // level used when the session does not choose one.
    auto hinted = antigravity_request("gemini-4-argon-high");
    hinted.effort = "low";
    const auto hinted_doc = parse_json(p2, protocol.serialize(hinted));
    REQUIRE(hinted_doc["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"]
                .get(level)
            == simdjson::SUCCESS);
    CHECK(level == "LOW");
    std::string_view hinted_model;
    REQUIRE(hinted_doc["model"].get(hinted_model) == simdjson::SUCCESS);
    CHECK(hinted_model == "gemini-4-argon-low");

    // `*-tiered` SKUs budget dynamically instead of pinning a level.
    const auto tiered =
        parse_json(p3, protocol.serialize(antigravity_request("gemini-3.8-flash-tiered")));
    int64_t budget = 0;
    REQUIRE(tiered["request"]["generationConfig"]["thinkingConfig"]["thinkingBudget"].get(budget)
            == simdjson::SUCCESS);
    CHECK(budget == -1);
    CHECK(tiered["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"].error()
          == simdjson::NO_SUCH_FIELD);

    // A SKU that cannot think gets no thinkingConfig.
    const auto image =
        parse_json(p4, protocol.serialize(antigravity_request("gemini-3.1-flash-image")));
    CHECK(image["request"]["generationConfig"]["thinkingConfig"].error()
          == simdjson::NO_SUCH_FIELD);
}

TEST_CASE("Antigravity thinking families are one model whose effort picks the SKU",
          "[gemini][antigravity][effort]") {
    GeminiAntigravityProtocol protocol;

    struct Wire {
        std::string model;
        std::string model_enum;
        std::string level;
    };
    const auto wire_for = [&](const ChatRequest& req) {
        simdjson::dom::parser parser;
        const auto doc = parse_json(parser, protocol.serialize(req));
        std::string_view model, model_enum, level;
        REQUIRE(doc["model"].get(model) == simdjson::SUCCESS);
        REQUIRE(doc["request"]["labels"]["model_enum"].get(model_enum) == simdjson::SUCCESS);
        REQUIRE(doc["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"].get(level)
                == simdjson::SUCCESS);
        return Wire{std::string(model), std::string(model_enum), std::string(level)};
    };
    const auto with_effort = [](std::string model, std::string effort) {
        auto req = antigravity_request(std::move(model));
        req.effort = std::move(effort);
        return req;
    };

    SECTION("the family id resolves to the SKU of the session effort") {
        const Wire low = wire_for(with_effort("claude-opus-5-5", "low"));
        CHECK(low.model == "claude-opus-5-5-low");
        CHECK(low.model_enum == "MODEL_PLACEHOLDER_M400");
        CHECK(low.level == "LOW");

        const Wire medium = wire_for(with_effort("gemini-4-argon", "medium"));
        CHECK(medium.model == "gemini-4-argon-medium");
        CHECK(medium.model_enum == "MODEL_PLACEHOLDER_M418");
        CHECK(medium.level == "MEDIUM");
    }

    SECTION("auto and off use the strongest level, levels above high clamp to it") {
        for (const std::string effort : {"", "none", "high", "xhigh", "max", "ultra"}) {
            CAPTURE(effort);
            const Wire wire = wire_for(with_effort("claude-sonnet-5-5", effort));
            CHECK(wire.model == "claude-sonnet-5-5-high");
            CHECK(wire.model_enum == "MODEL_PLACEHOLDER_M405");
            CHECK(wire.level == "HIGH");
        }
        CHECK(wire_for(with_effort("gemini-3.8-flash", "minimal")).model == "gemini-3.8-flash-low");
    }

    SECTION("a configured variant id is the default level, not a pin") {
        CHECK(wire_for(with_effort("claude-opus-5-5-medium", "")).model
              == "claude-opus-5-5-medium");
        const Wire overridden = wire_for(with_effort("claude-opus-5-5-medium", "high"));
        CHECK(overridden.model == "claude-opus-5-5-high");
        CHECK(overridden.model_enum == "MODEL_PLACEHOLDER_M402");
    }

    SECTION("the live catalog decides which levels a family is sold at") {
        auto req = with_effort("gemini-9-test", "medium");
        req.catalog_reasoning = ModelReasoningProfile{
            .effort = ReasoningCapabilities{ReasoningCapability::Effort},
            .complete = true,
            .catalog_default_effort = "low",
            .supported_efforts = {"low", "high"},
        };
        simdjson::dom::parser parser;
        const auto doc = parse_json(parser, protocol.serialize(req));
        std::string_view model, level;
        // No medium SKU: the request rounds up to the next level sold, not
        // down to the catalog default.
        REQUIRE(doc["model"].get(model) == simdjson::SUCCESS);
        CHECK(model == "gemini-9-test-high");
        REQUIRE(doc["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"].get(level)
                == simdjson::SUCCESS);
        CHECK(level == "HIGH");

        req.effort.clear();
        const auto defaulted = parse_json(parser, protocol.serialize(req));
        REQUIRE(defaulted["model"].get(model) == simdjson::SUCCESS);
        CHECK(model == "gemini-9-test-low");
    }

    SECTION("single SKUs are sent unchanged") {
        simdjson::dom::parser parser;
        const auto doc = parse_json(
            parser, protocol.serialize(with_effort("gemini-3.1-pro-preview", "medium")));
        std::string_view model, level;
        REQUIRE(doc["model"].get(model) == simdjson::SUCCESS);
        CHECK(model == "gemini-3.1-pro-preview");
        REQUIRE(doc["request"]["generationConfig"]["thinkingConfig"]["thinkingLevel"].get(level)
                == simdjson::SUCCESS);
        CHECK(level == "MEDIUM");

        simdjson::dom::parser oss_parser;
        const auto oss = parse_json(
            oss_parser, protocol.serialize(with_effort("gpt-oss-120b-medium", "high")));
        REQUIRE(oss["model"].get(model) == simdjson::SUCCESS);
        CHECK(model == "gpt-oss-120b-medium");
    }

    SECTION("only families report an effort ladder") {
        for (const std::string_view family :
             {"claude-opus-5-5", "claude-sonnet-5-5-low", "gemini-4-argon", "gemini-3.8-flash"}) {
            CAPTURE(family);
            const auto capabilities = protocol.reasoning_capabilities(family);
            CHECK(capabilities.supports_effort());
            CHECK_FALSE(capabilities.supports(ReasoningCapability::MaxEffort));
            CHECK_FALSE(capabilities.supports(ReasoningCapability::Disable));
        }
        for (const std::string_view single :
             {"gemini-3.8-flash-tiered", "gemini-3.1-flash-image", "gpt-oss-120b-medium",
              "gemini-2.5-flash"}) {
            CAPTURE(single);
            CHECK(protocol.reasoning_capabilities(single).empty());
        }
    }
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

// ─────────────────────────────────────────────────────────────────────────────
// Antigravity plan usage (retrieveUserQuota / retrieveUserQuotaSummary / loadCodeAssist)
// ─────────────────────────────────────────────────────────────────────────────

namespace {

// The summary served live on 2026-10-28 to a Google AI Pro account, with most
// descriptions dropped.
constexpr std::string_view kLiveQuotaSummary = R"({"groups":[
    {"buckets":[
        {"bucketId":"gemini-weekly","displayName":"Weekly Limit Remaining","window":"weekly",
         "resetTime":"2026-11-03T16:11:46Z",
         "description":"You have used some of your weekly limit, it will fully refresh in 6 days, 7 hours.",
         "remainingFraction":0.9789123},
        {"bucketId":"gemini-5h","displayName":"Five Hour Limit Remaining","window":"5h",
         "resetTime":"2026-10-28T11:26:23Z","remainingFraction":0.9261931}],
     "displayName":"Gemini Models","description":"Models within this group: Gemini Flash, Gemini Pro"},
    {"buckets":[
        {"bucketId":"3p-weekly","displayName":"Weekly Limit Remaining","window":"weekly",
         "resetTime":"2026-11-02T09:56:25Z","remainingFraction":0.9711332},
        {"bucketId":"3p-5h","displayName":"Five Hour Limit Remaining","window":"5h",
         "resetTime":"2026-10-28T10:13:40Z","remainingFraction":0.9665336}],
     "displayName":"Claude and GPT models"},
    {"buckets":[
        {"bucketId":"gemini-argon-wtu-weekly","displayName":"Gemini 4 Argon Weekly Limit","window":"weekly",
         "resetTime":"2026-11-03T15:45:57Z","remainingFraction":0.99953616}],
     "displayName":"Gemini 4 Argon"}],
  "description":"Within each group, models share a weekly limit and a 5-hour limit."})";

// The `retrieveUserQuota` buckets served with it, abridged: each SKU repeats
// its group's tightest window.
constexpr std::string_view kLiveQuota = R"({"buckets":[
    {"tokenType":"WTUS","modelId":"chat_20706","remainingFraction":1},
    {"resetTime":"2026-10-28T10:13:40Z","tokenType":"WTUS","modelId":"claude-opus-5-5-high","remainingFraction":0.9665336},
    {"resetTime":"2026-10-28T11:26:23Z","tokenType":"WTUS","modelId":"gemini-3.8-flash-high","remainingFraction":0.9261931},
    {"resetTime":"2026-10-28T11:26:23Z","tokenType":"WTUS","modelId":"gemini-3.8-flash-tiered","remainingFraction":0.9261931},
    {"resetTime":"2026-11-03T15:45:57Z","tokenType":"WTUS","modelId":"gemini-4-argon-high","remainingFraction":0.99953616},
    {"resetTime":"2026-10-28T10:13:40Z","tokenType":"WTUS","modelId":"gpt-oss-120b-medium","remainingFraction":0.9665336}]})";

// The account's loadCodeAssist tiers.
constexpr std::string_view kLivePlan = R"({"currentTier":{"id":"free-tier","name":"Antigravity"},
    "paidTier":{"id":"g1-pro-tier","name":"Google AI Pro","description":"Google AI Pro"}})";

/// A Code Assist host behind the usage transport: serves the live fixtures and
/// records every call. A reply reset to nothing fails that method.
struct FakeCodeAssist {
    struct Call {
        std::string url;
        std::string authorization;
        std::string body;
    };
    std::optional<std::string> quota = std::string(kLiveQuota);
    std::optional<std::string> summary = std::string(kLiveQuotaSummary);
    std::optional<std::string> plan = std::string(kLivePlan);
    std::string throwing_method; ///< Its calls throw instead.
    std::mutex mutex;            // Reads run on their own threads.
    std::vector<Call> calls;

    antigravity::UsageTransport transport() {
        return [this](std::string_view url, std::string_view authorization,
                      std::string_view body) -> std::optional<std::string> {
            std::lock_guard lock(mutex);
            calls.push_back({std::string(url), std::string(authorization), std::string(body)});
            const auto method = url.substr(url.rfind(':') + 1);
            if (method == throwing_method) throw std::runtime_error("transport failure");
            if (method == "retrieveUserQuota") return quota;
            if (method == "retrieveUserQuotaSummary") return summary;
            if (method == "loadCodeAssist") return plan;
            return std::nullopt;
        };
    }

    std::vector<Call> calls_to(std::string_view method) {
        std::lock_guard lock(mutex);
        std::vector<Call> matching;
        for (const auto& call : calls) {
            if (call.url.ends_with(":" + std::string(method))) matching.push_back(call);
        }
        return matching;
    }
};

} // namespace

TEST_CASE("Antigravity quota decodes retrieveUserQuota buckets", "[gemini][antigravity][usage]") {
    // Shape served live on 2026-10-28: an untouched bucket has no resetTime
    // and an integer fraction.
    const auto buckets = antigravity::parse_quota_buckets(R"({"buckets":[
        {"tokenType":"WTUS","modelId":"chat_20706","remainingFraction":1},
        {"resetTime":"2026-10-28T10:13:40Z","tokenType":"WTUS","modelId":"claude-opus-5-5-high","remainingFraction":0.75},
        {"modelId":"broken","remainingFraction":1.5},
        {"remainingFraction":0.5}
    ]})");
    REQUIRE(buckets.size() == 2);
    CHECK(buckets[0].model == "chat_20706");
    CHECK(buckets[0].remaining_fraction == 1.0);
    CHECK(buckets[0].resets_at == 0);
    CHECK(buckets[1].model == "claude-opus-5-5-high");
    CHECK(buckets[1].resets_at == 1793182420);

    CHECK(antigravity::parse_quota_buckets("not json").empty());
    CHECK(antigravity::parse_quota_buckets(R"({"buckets":{}})").empty());
}

TEST_CASE("Antigravity quota summary decodes each group's limits", "[gemini][antigravity][usage]") {
    const auto groups = antigravity::parse_quota_summary(kLiveQuotaSummary);
    REQUIRE(groups.size() == 3);
    CHECK(groups[0].name == "Gemini Models");
    REQUIRE(groups[0].windows.size() == 2);
    CHECK(groups[0].windows[0].window == "weekly");
    CHECK(groups[0].windows[0].remaining_fraction == Catch::Approx(0.9789123));
    CHECK(groups[0].windows[0].resets_at == 1793722306);
    CHECK(groups[0].windows[1].window == "5h");
    CHECK(groups[0].windows[1].resets_at == 1793186783);
    CHECK(groups[1].name == "Claude and GPT models");
    CHECK(groups[2].name == "Gemini 4 Argon");
    CHECK(groups[2].windows.size() == 1);

    // A disabled limit, one counted in units and one without a window are not
    // shown, and a group left with nothing is dropped.
    const auto partial = antigravity::parse_quota_summary(R"({"groups":[
        {"displayName":"Limited","buckets":[
            {"window":"5h","remainingFraction":0.5,"disabled":true},
            {"window":"weekly","remainingAmount":"12"},
            {"remainingFraction":0.5},
            {"window":"weekly","remainingFraction":1}]},
        {"displayName":"Broken","buckets":[{"window":"5h","remainingFraction":2}]}]})");
    REQUIRE(partial.size() == 1);
    REQUIRE(partial[0].windows.size() == 1);
    CHECK(partial[0].windows[0].window == "weekly");
    CHECK(partial[0].windows[0].resets_at == 0);

    CHECK(antigravity::parse_quota_summary("not json").empty());
    CHECK(antigravity::parse_quota_summary(R"({"groups":{}})").empty());
}

TEST_CASE("Antigravity plan comes from loadCodeAssist", "[gemini][antigravity][usage]") {
    // Tier shape served live on 2026-10-28.
    CHECK(antigravity::parse_plan_name(R"({
        "currentTier":{"id":"free-tier","name":"Antigravity"},
        "paidTier":{"id":"g1-pro-tier","name":"Google AI Pro","description":"Google AI Pro"}})")
          == "Google AI Pro");
    CHECK(antigravity::parse_plan_name(R"({"currentTier":{"id":"free-tier","name":"Antigravity"}})")
          == "Antigravity");
    CHECK_FALSE(antigravity::parse_plan_name("{}"));
    CHECK_FALSE(antigravity::parse_plan_name("<html>"));
}

TEST_CASE("Antigravity quota shows every limit of the SKU's group", "[gemini][antigravity][usage]") {
    const auto groups = antigravity::parse_quota_summary(kLiveQuotaSummary);
    const auto buckets = antigravity::parse_quota_buckets(kLiveQuota);
    REQUIRE(buckets.size() == 6);

    SECTION("Gemini Flash shows the Gemini group's 5-hour and weekly limits") {
        const auto info = antigravity::quota_for_model(buckets, groups, "gemini-3.8-flash-high");
        REQUIRE(info);
        REQUIRE(info->usage_windows.size() == 2);
        CHECK(info->usage_windows[0].label == "5h");
        CHECK(info->usage_windows[0].utilization == Catch::Approx(1.0 - 0.9261931));
        CHECK(info->usage_windows[0].resets_at == 1793186783);
        CHECK(info->usage_windows[1].label == "7d");
        CHECK(info->usage_windows[1].utilization == Catch::Approx(1.0 - 0.9789123));
        CHECK(info->usage_windows[1].resets_at == 1793722306);
        CHECK(info->unified_representative_claim == "5h");
        CHECK_FALSE(info->is_rate_limited);
    }
    SECTION("Claude and GPT share the third-party group") {
        for (const std::string_view model : {"claude-opus-5-5-high", "gpt-oss-120b-medium"}) {
            const auto info = antigravity::quota_for_model(buckets, groups, model);
            REQUIRE(info);
            REQUIRE(info->usage_windows.size() == 2);
            CHECK(info->usage_windows[0].resets_at == 1793182420);
            CHECK(info->usage_windows[1].utilization == Catch::Approx(1.0 - 0.9711332));
        }
    }
    SECTION("Argon's only limit is weekly") {
        const auto info = antigravity::quota_for_model(buckets, groups, "gemini-4-argon-high");
        REQUIRE(info);
        REQUIRE(info->usage_windows.size() == 1);
        CHECK(info->usage_windows[0].label == "7d");
        CHECK(info->usage_windows[0].resets_at == 1793720757);
        CHECK(info->unified_representative_claim == "7d");
    }
    SECTION("a SKU without a bucket uses its family's") {
        const auto info = antigravity::quota_for_model(buckets, groups, "gemini-3.8-flash-medium");
        REQUIRE(info);
        REQUIRE(info->usage_windows.size() == 2);
        CHECK(info->usage_windows[0].resets_at == 1793186783);
    }
    SECTION("an exhausted weekly limit governs and rate-limits the model") {
        auto exhausted = groups;
        exhausted[0].windows[0].remaining_fraction = 0.0; // gemini-weekly
        auto bound = buckets;
        bound[2] = {"gemini-3.8-flash-high", 0.0, 1793722306};
        const auto info = antigravity::quota_for_model(bound, exhausted, "gemini-3.8-flash-high");
        REQUIRE(info);
        CHECK(info->unified_representative_claim == "7d");
        CHECK(info->usage_windows[1].utilization == Catch::Approx(1.0));
        CHECK(info->is_rate_limited);
    }
    SECTION("without the summary the bucket is shown alone, its window unknown") {
        const auto info = antigravity::quota_for_model(buckets, {}, "gemini-4-argon-high");
        REQUIRE(info);
        REQUIRE(info->usage_windows.size() == 1);
        CHECK(info->usage_windows[0].label == "quota");
        CHECK(info->usage_windows[0].utilization == Catch::Approx(1.0 - 0.99953616));
        CHECK(info->usage_windows[0].resets_at == 1793720757);
        CHECK(info->unified_representative_claim.empty());
    }
    SECTION("an untouched SKU matches only an untouched group") {
        const std::vector<antigravity::QuotaBucket> fresh{{"gemini-3.8-flash-high", 1.0, 0}};
        auto untouched = groups;
        untouched[0].windows = {{"weekly", 1.0, 0}, {"5h", 1.0, 0}};
        const auto info = antigravity::quota_for_model(fresh, untouched, "gemini-3.8-flash-high");
        REQUIRE(info);
        REQUIRE(info->usage_windows.size() == 2);
        CHECK(info->usage_windows[0].label == "5h");
        CHECK(info->usage_windows[0].utilization == Catch::Approx(0.0));
        CHECK(info->unified_representative_claim.empty());

        // With two untouched groups the bucket cannot say which one is its own.
        untouched[2].windows = {{"weekly", 1.0, 0}};
        const auto ambiguous = antigravity::quota_for_model(fresh, untouched, "gemini-3.8-flash-high");
        REQUIRE(ambiguous);
        REQUIRE(ambiguous->usage_windows.size() == 1);
        CHECK(ambiguous->usage_windows[0].label == "quota");
    }
    SECTION("a model the account is not metered for has no quota") {
        CHECK_FALSE(antigravity::quota_for_model(buckets, groups, "unmetered-model"));
        CHECK_FALSE(antigravity::quota_for_model(buckets, groups, "claude-opus"));
    }
}

TEST_CASE("GeminiAntigravityProtocol reports the quota group of the request's SKU",
          "[gemini][antigravity][usage]") {
    FakeCodeAssist server;
    GeminiAntigravityProtocol protocol(antigravity::make_usage_source(server.transport()));
    ChatRequest req;
    req.model = "gemini-3.8-flash-high";
    req.auth_properties["project_id"] = "project-123";
    req.messages.push_back(Message{.role = "user", .content = "Hello"});
    protocol.prepare_request(req);

    const std::string host = "https://daily-cloudcode-pa.googleapis.com";
    const cpr::Header headers{{"Authorization", "Bearer token-a"}};
    const cpr::Header no_headers;
    const HttpResponse ok{200, "", no_headers};
    auto stream = protocol.clone();
    stream->on_response(ok);
    stream->enrich_rate_limit(host, headers, ok);

    const auto info = stream->last_rate_limit();
    REQUIRE(info.usage_windows.size() == 2);
    CHECK(info.usage_windows[0].label == "5h");
    CHECK(info.usage_windows[0].utilization == Catch::Approx(1.0 - 0.9261931));
    CHECK(info.usage_windows[1].label == "7d");
    CHECK(info.usage_windows[1].utilization == Catch::Approx(1.0 - 0.9789123));
    CHECK(info.unified_representative_claim == "5h");
    CHECK(info.subscription.tier_label == "Google AI Pro");

    // One call per method, to the request's host with its credential and project.
    const auto expect_one_call = [&](std::string_view method, const std::string& body) {
        const auto calls = server.calls_to(method);
        REQUIRE(calls.size() == 1);
        CHECK(calls[0].url == host + "/v1internal:" + std::string(method));
        CHECK(calls[0].authorization == "Bearer token-a");
        CHECK(calls[0].body == body);
    };
    expect_one_call("retrieveUserQuota", R"({"project":"project-123"})");
    expect_one_call("retrieveUserQuotaSummary", R"({"project":"project-123"})");
    expect_one_call("loadCodeAssist",
                    core::auth::google_code_assist::load_code_assist_payload("project-123", "ANTIGRAVITY"));
    CHECK(server.calls.size() == 3);

    // A second response within the minute reuses the reading.
    stream->on_response(ok);
    stream->enrich_rate_limit(host, headers, ok);
    CHECK(server.calls.size() == 3);
    CHECK(stream->last_rate_limit().usage_windows.size() == 2);

    // A 429 stays rate limited with a reading. Its Retry-After does not reach
    // the retry scheduler, which moves to the next host instead.
    const cpr::Header retry_after{{"Retry-After", "3600"}};
    const HttpResponse limited{429, "", retry_after};
    stream->on_response(limited);
    stream->enrich_rate_limit(host, headers, limited);
    CHECK(stream->last_rate_limit().is_rate_limited);
    CHECK(stream->last_rate_limit().retry_after == 0);

    // Nothing is read when the request never reached the host.
    FakeCodeAssist unreachable;
    GeminiAntigravityProtocol offline(antigravity::make_usage_source(unreachable.transport()));
    offline.prepare_request(req);
    const HttpResponse no_response{0, "", no_headers};
    offline.on_response(no_response);
    offline.enrich_rate_limit(host, headers, no_response);
    CHECK(unreachable.calls.empty());
    CHECK(offline.last_rate_limit().usage_windows.empty());
}

TEST_CASE("Antigravity usage shows what it could read", "[gemini][antigravity][usage]") {
    FakeCodeAssist server;
    const auto usage = antigravity::make_usage_source(server.transport());
    const cpr::Header headers{{"Authorization", "Bearer token-a"}};
    const auto read = [&](std::string_view model) {
        return usage->read("https://daily-cloudcode-pa.googleapis.com/", headers, "project-123", model);
    };

    SECTION("a host given with a trailing slash still gets one call per method") {
        REQUIRE(read("gemini-3.8-flash-high"));
        REQUIRE(server.calls.size() == 3);
        for (const auto& call : server.calls) {
            CHECK(call.url.starts_with("https://daily-cloudcode-pa.googleapis.com/v1internal:"));
        }
    }
    SECTION("a failed summary leaves the SKU's own window, unnamed") {
        server.summary.reset();
        const auto info = read("gemini-4-argon-high");
        REQUIRE(info);
        REQUIRE(info->usage_windows.size() == 1);
        CHECK(info->usage_windows[0].label == "quota");
        CHECK(info->subscription.tier_label == "Google AI Pro");
    }
    SECTION("a throwing summary read counts as a failed one") {
        server.throwing_method = "retrieveUserQuotaSummary";
        const auto info = read("gemini-4-argon-high");
        REQUIRE(info);
        REQUIRE(info->usage_windows.size() == 1);
        CHECK(info->usage_windows[0].label == "quota");
    }
    SECTION("a failed plan read costs only the tier label") {
        server.plan.reset();
        const auto info = read("gemini-3.8-flash-high");
        REQUIRE(info);
        CHECK(info->usage_windows.size() == 2);
        CHECK(info->subscription.tier_label.empty());
    }
    SECTION("a quota naming no bucket shows nothing") {
        server.quota = R"({"buckets":[]})";
        CHECK_FALSE(read("gemini-3.8-flash-high"));
    }
    SECTION("a failed quota read shows nothing") {
        server.quota.reset();
        CHECK_FALSE(read("gemini-3.8-flash-high"));
    }
    SECTION("nothing is read without a credential") {
        CHECK_FALSE(usage->read("https://daily-cloudcode-pa.googleapis.com", {}, "project-123",
                                "gemini-3.8-flash-high"));
        CHECK(server.calls.empty());
    }
}

TEST_CASE("Antigravity usage reads the quota, its summary and the plan side by side",
          "[gemini][antigravity][usage]") {
    // Each call waits for the other two, so calls made one after another time
    // out instead: a cold refresh must cost the slowest read, not the sum.
    std::mutex mutex;
    std::condition_variable arrivals;
    int arrived = 0;
    std::atomic<bool> overlapped = true;
    FakeCodeAssist server;
    const auto serve = server.transport();
    const auto usage = antigravity::make_usage_source(
        [&](std::string_view url, std::string_view authorization, std::string_view body) {
            {
                std::unique_lock lock(mutex);
                ++arrived;
                arrivals.notify_all();
                if (!arrivals.wait_for(lock, std::chrono::seconds(5), [&] { return arrived == 3; })) {
                    overlapped = false;
                }
            }
            return serve(url, authorization, body);
        });
    const cpr::Header headers{{"Authorization", "Bearer token-a"}};
    const auto info = usage->read("https://daily-cloudcode-pa.googleapis.com", headers, "project-123",
                                  "gemini-3.8-flash-high");
    REQUIRE(info);
    CHECK(info->usage_windows.size() == 2);
    CHECK(info->subscription.tier_label == "Google AI Pro");
    CHECK(overlapped);
}
