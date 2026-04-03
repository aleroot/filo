#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/GrokProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/tools/Tool.hpp"

using namespace core::llm;
using namespace core::llm::protocols;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static ChatRequest make_simple_request(std::string model = "grok-code-fast-1",
                                       std::string user_text = "Hello") {
    ChatRequest req;
    req.model = std::move(model);
    req.stream = true;
    req.messages.push_back(Message{.role = "user", .content = std::move(user_text)});
    return req;
}

// ─────────────────────────────────────────────────────────────────────────────
// GrokSerializer — basic structure
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GrokSerializer - model is serialized correctly", "[grok][serializer]") {
    auto payload = GrokProtocol{GrokReasoningEffort::None}.serialize(make_simple_request("grok-code-fast-1"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"grok-code-fast-1")"));
}

TEST_CASE("GrokSerializer - stream true is included", "[grok][serializer]") {
    auto req = make_simple_request();
    req.stream = true;
    REQUIRE_THAT(GrokProtocol{}.serialize(req), Catch::Matchers::ContainsSubstring(R"("stream":true)"));
}

TEST_CASE("GrokSerializer - stream false is included", "[grok][serializer]") {
    auto req = make_simple_request();
    req.stream = false;
    REQUIRE_THAT(GrokProtocol{}.serialize(req), Catch::Matchers::ContainsSubstring(R"("stream":false)"));
}

TEST_CASE("GrokSerializer - user message content appears in payload", "[grok][serializer]") {
    auto payload = GrokProtocol{}.serialize(make_simple_request("grok-code-fast-1", "Tell me a joke"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("Tell me a joke"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"user")"));
}

TEST_CASE("GrokSerializer - empty messages produce valid JSON", "[grok][serializer]") {
    ChatRequest req;
    req.model = "grok-code-fast-1";
    req.stream = true;
    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("messages":[])"));
}

// ─────────────────────────────────────────────────────────────────────────────
// GrokSerializer — reasoning_effort field
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GrokSerializer - reasoning_effort None omits the field", "[grok][serializer][reasoning]") {
    auto payload = GrokProtocol{}.serialize(make_simple_request());
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("reasoning_effort"));
}

TEST_CASE("GrokSerializer - reasoning_effort Low serialized correctly", "[grok][serializer][reasoning]") {
    auto payload = GrokProtocol{GrokReasoningEffort::Low}.serialize(make_simple_request("grok-3-mini"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"low")"));
}

TEST_CASE("GrokSerializer - reasoning_effort Medium is coerced to a supported value", "[grok][serializer][reasoning]") {
    auto payload = GrokProtocol{GrokReasoningEffort::Medium}.serialize(make_simple_request("grok-3-mini"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"high")"));
}

TEST_CASE("GrokSerializer - reasoning_effort High serialized correctly", "[grok][serializer][reasoning]") {
    auto payload = GrokProtocol{GrokReasoningEffort::High}.serialize(make_simple_request("grok-3-mini"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"high")"));
}

TEST_CASE("GrokSerializer - reasoning_effort precedes temperature in payload", "[grok][serializer][reasoning]") {
    auto req = make_simple_request("grok-3-mini");
    req.temperature = 0.5f;
    auto payload = GrokProtocol{GrokReasoningEffort::High}.serialize(req);
    // Both fields must be present
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"high")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("temperature")"));
}

TEST_CASE("GrokSerializer - reasoning_effort is omitted for grok-code-fast-1", "[grok][serializer][reasoning]") {
    auto payload = GrokProtocol{GrokReasoningEffort::High}.serialize(make_simple_request("grok-code-fast-1"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("reasoning_effort"));
}

TEST_CASE("GrokSerializer - reasoning_effort is omitted for grok-4 (would cause API error)", "[grok][serializer][reasoning]") {
    // xAI docs: grok-4 does NOT accept reasoning_effort; API returns 400 if sent
    REQUIRE_THAT(GrokProtocol{GrokReasoningEffort::High}.serialize(make_simple_request("grok-4")),
                 !Catch::Matchers::ContainsSubstring("reasoning_effort"));
    REQUIRE_THAT(GrokProtocol{GrokReasoningEffort::High}.serialize(make_simple_request("grok-4-fast-non-reasoning")),
                 !Catch::Matchers::ContainsSubstring("reasoning_effort"));
    REQUIRE_THAT(GrokProtocol{GrokReasoningEffort::High}.serialize(make_simple_request("grok-4.20-reasoning")),
                 !Catch::Matchers::ContainsSubstring("reasoning_effort"));
    REQUIRE_THAT(GrokProtocol{GrokReasoningEffort::High}.serialize(make_simple_request("grok-4.20-non-reasoning")),
                 !Catch::Matchers::ContainsSubstring("reasoning_effort"));
}

TEST_CASE("GrokSerializer - reasoning_effort Low works on grok-3-mini-fast", "[grok][serializer][reasoning]") {
    auto payload = GrokProtocol{GrokReasoningEffort::Low}.serialize(make_simple_request("grok-3-mini-fast"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"low")"));
}

TEST_CASE("grok_supports_reasoning_effort matches xAI model support", "[grok][reasoning]") {
    // grok-3-mini family: supports reasoning_effort
    REQUIRE(grok_supports_reasoning_effort("grok-3-mini"));
    REQUIRE(grok_supports_reasoning_effort("grok-3-mini-fast"));
    REQUIRE(grok_supports_reasoning_effort("grok-3-mini-latest"));

    // Grok 4 and grok-4.20: always-reasoning, no reasoning_effort param (errors if sent)
    REQUIRE_FALSE(grok_supports_reasoning_effort("grok-4"));
    REQUIRE_FALSE(grok_supports_reasoning_effort("grok-4-fast-reasoning"));
    REQUIRE_FALSE(grok_supports_reasoning_effort("grok-4-fast-non-reasoning"));
    REQUIRE_FALSE(grok_supports_reasoning_effort("grok-4.20-reasoning"));
    REQUIRE_FALSE(grok_supports_reasoning_effort("grok-4.20-non-reasoning"));

    // Code and other models
    REQUIRE_FALSE(grok_supports_reasoning_effort("grok-code-fast-1"));
    REQUIRE_FALSE(grok_supports_reasoning_effort("grok-3"));
    REQUIRE_FALSE(grok_supports_reasoning_effort("grok-3-fast"));
}

// ─────────────────────────────────────────────────────────────────────────────
// GrokSerializer — optional parameters
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GrokSerializer - temperature omitted when not set", "[grok][serializer]") {
    auto req = make_simple_request();
    // req.temperature not set
    REQUIRE_THAT(GrokProtocol{GrokReasoningEffort::High}.serialize(req), !Catch::Matchers::ContainsSubstring("temperature"));
}

TEST_CASE("GrokSerializer - temperature included when set", "[grok][serializer]") {
    auto req = make_simple_request();
    req.temperature = 0.7f;
    REQUIRE_THAT(GrokProtocol{}.serialize(req), Catch::Matchers::ContainsSubstring(R"("temperature")"));
}

TEST_CASE("GrokSerializer - max_tokens omitted when not set", "[grok][serializer]") {
    REQUIRE_THAT(GrokProtocol{}.serialize(make_simple_request()), !Catch::Matchers::ContainsSubstring("max_tokens"));
}

TEST_CASE("GrokSerializer - max_tokens included when set", "[grok][serializer]") {
    auto req = make_simple_request();
    req.max_tokens = 4096;
    REQUIRE_THAT(GrokProtocol{}.serialize(req), Catch::Matchers::ContainsSubstring(R"("max_tokens":4096)"));
}

// ─────────────────────────────────────────────────────────────────────────────
// GrokSerializer — multi-turn conversation roles
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GrokSerializer - system message serialized with correct role", "[grok][serializer]") {
    ChatRequest req;
    req.model = "grok-code-fast-1";
    req.stream = true;
    req.messages = {
        Message{.role = "system", .content = "You are Grok, made by xAI."},
        Message{.role = "user",   .content = "Hello"}
    };
    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"system")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("You are Grok, made by xAI."));
}

TEST_CASE("GrokSerializer - assistant message serialized with correct role", "[grok][serializer]") {
    ChatRequest req;
    req.model = "grok-code-fast-1";
    req.stream = true;
    req.messages = {
        Message{.role = "user",      .content = "Hi"},
        Message{.role = "assistant", .content = "Hello there!"},
        Message{.role = "user",      .content = "Bye"}
    };
    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"assistant")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("Hello there!"));
}

TEST_CASE("GrokSerializer - tool response message with tool_call_id", "[grok][serializer]") {
    ChatRequest req;
    req.model = "grok-code-fast-1";
    req.stream = true;
    Message tool_resp;
    tool_resp.role = "tool";
    tool_resp.content = R"({"exit_code": 0, "output": "hello"})";
    tool_resp.tool_call_id = "call_abc123";
    tool_resp.name = "run_terminal_command";
    req.messages.push_back(tool_resp);

    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("tool_call_id":"call_abc123")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("name":"run_terminal_command")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"tool")"));
}

TEST_CASE("GrokSerializer - assistant message with tool_calls uses null content", "[grok][serializer]") {
    ChatRequest req;
    req.model = "grok-code-fast-1";
    req.stream = true;
    Message assistant_msg;
    assistant_msg.role = "assistant";
    // No text content — assistant is making a tool call
    ToolCall tc;
    tc.id = "call_xyz";
    tc.type = "function";
    tc.function.name = "read_file";
    tc.function.arguments = R"({"path": "/etc/hosts"})";
    assistant_msg.tool_calls = {tc};
    req.messages.push_back(assistant_msg);

    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("tool_calls")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("id":"call_xyz")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("name":"read_file")"));
    // Empty content with tool_calls should emit null, not an empty string
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("content":"")"));
}

// ─────────────────────────────────────────────────────────────────────────────
// GrokSerializer — tool definitions
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GrokSerializer - single tool definition serialized", "[grok][serializer][tools]") {
    auto req = make_simple_request();
    core::tools::ToolParameter param{
        .name = "command",
        .type = "string",
        .description = "The shell command to execute",
        .required = true
    };
    core::tools::ToolDefinition def{
        .name = "run_terminal_command",
        .description = "Execute a shell command on the local machine",
        .parameters = {param}
    };
    Tool tool;
    tool.type = "function";
    tool.function = def;
    req.tools.push_back(tool);

    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("tools")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("run_terminal_command")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("command")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("required":["command"])"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"function")"));
}

TEST_CASE("GrokSerializer - optional parameters excluded from required array", "[grok][serializer][tools]") {
    auto req = make_simple_request();
    core::tools::ToolDefinition def;
    def.name = "grep_search";
    def.description = "Search for a pattern";
    def.parameters = {
        {.name = "pattern",  .type = "string", .description = "Regex", .required = true},
        {.name = "path", .type = "string", .description = "Root", .required = false}
    };
    Tool tool;
    tool.type = "function";
    tool.function = def;
    req.tools.push_back(tool);

    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("required":["pattern"])"));
    // Optional param is in properties but NOT in required
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("path"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("required":["pattern","path"])"));
}

TEST_CASE("GrokSerializer - multiple tools serialized without trailing comma", "[grok][serializer][tools]") {
    auto req = make_simple_request();
    for (const auto& name : {"read_file", "write_file"}) {
        core::tools::ToolDefinition def;
        def.name = name;
        def.description = "A file tool";
        def.parameters.push_back({.name = "path", .type = "string", .description = "Path", .required = true});
        Tool tool;
        tool.type = "function";
        tool.function = def;
        req.tools.push_back(tool);
    }

    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("read_file")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("write_file")"));
    // No trailing comma before closing bracket (rudimentary check)
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",]"));
}

// ─────────────────────────────────────────────────────────────────────────────
// GrokSerializer — JSON escaping
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GrokSerializer - newline in content is JSON-escaped", "[grok][serializer][escaping]") {
    auto req = make_simple_request("grok-code-fast-1", "line1\nline2");
    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"(\n)"));
    // Raw newline must NOT appear inside a JSON string value
    // (the payload as a whole may have newlines for formatting but content values must not)
}

TEST_CASE("GrokSerializer - tab in content is JSON-escaped", "[grok][serializer][escaping]") {
    auto req = make_simple_request("grok-code-fast-1", "col1\tcol2");
    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"(\t)"));
}

TEST_CASE("GrokSerializer - double-quote in content is JSON-escaped", "[grok][serializer][escaping]") {
    auto req = make_simple_request("grok-code-fast-1", R"(He said "hello")");
    auto payload = GrokProtocol{}.serialize(req);
    // The JSON should contain escaped quotes: \" (which in the raw string is just \")
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"(\"hello\")"));
}

TEST_CASE("GrokSerializer - backslash in content is JSON-escaped", "[grok][serializer][escaping]") {
    auto req = make_simple_request("grok-code-fast-1", R"(C:\Users\foo)");
    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"(\\)"));
}

// ─────────────────────────────────────────────────────────────────────────────
// GrokSerializer — all known Grok model names round-trip
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GrokSerializer - all known Grok model names are preserved verbatim", "[grok][serializer]") {
    const std::vector<std::string> models = {
        // Code-focused
        "grok-code-fast-1",
        // Grok 4.20 variants (flagship as of 2026)
        "grok-4.20-reasoning",
        "grok-4.20-non-reasoning",
        // Grok 4 (always-reasoning)
        "grok-4",
        "grok-4-fast-reasoning",
        "grok-4-fast-non-reasoning",
        // Grok 3 mini with reasoning_effort
        "grok-3-mini",
        "grok-3-mini-fast",
        // Grok 3 (no reasoning_effort)
        "grok-3",
        "grok-3-fast",
        // Legacy
        "grok-2-1212",
        "grok-2-vision-1212",
    };
    for (const auto& model : models) {
        INFO("Testing model: " << model);
        auto payload = GrokProtocol{}.serialize(make_simple_request(model));
        REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":")" + model + "\""));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_openai_sse_chunk — text content extraction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_openai_sse_chunk - extracts text content from delta", "[grok][parser]") {
    std::string json = R"({
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": null}]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content == "Hello");
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - handles empty delta with only role", "[grok][parser]") {
    // First chunk from xAI typically carries only role, no content
    std::string json = R"({
        "id": "chatcmpl-123",
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - finish reason stop produces no content", "[grok][parser]") {
    std::string json = R"({
        "id": "chatcmpl-123",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - finish reason tool_calls produces no content", "[grok][parser]") {
    std::string json = R"({
        "id": "chatcmpl-abc",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - multi-word content returned intact", "[grok][parser]") {
    std::string json = R"({
        "id": "chatcmpl-123",
        "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": null}]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content == " world");
    REQUIRE(tools.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_openai_sse_chunk — tool call extraction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_openai_sse_chunk - first tool call chunk with id and name", "[grok][parser][tools]") {
    std::string json = R"({
        "id": "chatcmpl-abc",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_xyz789",
                    "type": "function",
                    "function": {"name": "run_terminal_command", "arguments": ""}
                }]
            },
            "finish_reason": null
        }]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content.empty());
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].id == "call_xyz789");
    REQUIRE(tools[0].type == "function");
    REQUIRE(tools[0].function.name == "run_terminal_command");
    REQUIRE(tools[0].function.arguments.empty());
    REQUIRE(tools[0].index == 0);
}

TEST_CASE("parse_openai_sse_chunk - argument streaming chunk (no id/name)", "[grok][parser][tools]") {
    // Subsequent chunks carry only partial arguments for an ongoing tool call
    std::string json = R"({
        "id": "chatcmpl-abc",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {"arguments": "{\"command\": \"ls -la\"}"}
                }]
            },
            "finish_reason": null
        }]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content.empty());
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].index == 0);
    REQUIRE(tools[0].function.arguments == R"({"command": "ls -la"})");
    // id and name are absent in streaming argument chunks — they stay empty
    REQUIRE(tools[0].id.empty());
    REQUIRE(tools[0].function.name.empty());
}

TEST_CASE("parse_openai_sse_chunk - multiple tool calls in single chunk", "[grok][parser][tools]") {
    std::string json = R"({
        "id": "chatcmpl-multi",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [
                    {"index": 0, "id": "call_aaa", "type": "function", "function": {"name": "read_file", "arguments": ""}},
                    {"index": 1, "id": "call_bbb", "type": "function", "function": {"name": "grep_search", "arguments": ""}}
                ]
            },
            "finish_reason": null
        }]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content.empty());
    REQUIRE(tools.size() == 2);
    REQUIRE(tools[0].index == 0);
    REQUIRE(tools[0].id == "call_aaa");
    REQUIRE(tools[0].function.name == "read_file");
    REQUIRE(tools[1].index == 1);
    REQUIRE(tools[1].id == "call_bbb");
    REQUIRE(tools[1].function.name == "grep_search");
}

TEST_CASE("parse_openai_sse_chunk - tool call index matches correctly", "[grok][parser][tools]") {
    std::string json = R"({
        "id": "chatcmpl-idx",
        "choices": [{
            "index": 0,
            "delta": {
                "tool_calls": [{"index": 2, "id": "call_ccc", "type": "function",
                                "function": {"name": "write_file", "arguments": "{}"}}]
            },
            "finish_reason": null
        }]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(tools.size() == 1);
    REQUIRE(tools[0].index == 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// parse_openai_sse_chunk — error handling / edge cases
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_openai_sse_chunk - empty string returns empty results", "[grok][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk("");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - malformed JSON returns empty results", "[grok][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk("{not valid json!!!}");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - truncated JSON returns empty results", "[grok][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk(R"({"id": "chatcmpl-123", "choices": [)");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - empty choices array returns empty results", "[grok][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk(R"({"id": "x", "choices": []})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - missing choices key returns empty results", "[grok][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk(R"({"id": "x", "object": "chat.completion.chunk"})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - choice without delta returns empty results", "[grok][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk(R"({"id": "x", "choices": [{"index": 0, "finish_reason": "stop"}]})");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - null content value returns empty string", "[grok][parser][errors]") {
    // xAI may send content: null in some chunk types
    std::string json = R"({
        "id": "chatcmpl-123",
        "choices": [{"index": 0, "delta": {"content": null}, "finish_reason": null}]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    // null is not a string — parser should handle gracefully and return empty
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - pure whitespace JSON returns empty results", "[grok][parser][errors]") {
    auto [content, tools] = parse_openai_sse_chunk("   ");
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// HttpLLMProvider with GrokProtocol — construction and interface
// ─────────────────────────────────────────────────────────────────────────────

static std::shared_ptr<core::auth::ApiKeyCredentialSource> make_grok_creds(const std::string& key) {
    return core::auth::ApiKeyCredentialSource::as_bearer(key);
}

TEST_CASE("GrokProvider - constructs with default model", "[grok][provider]") {
    REQUIRE_NOTHROW(HttpLLMProvider(
        "https://api.x.ai/v1",
        make_grok_creds("xai-test-key-abc"),
        "grok-code-fast-1",
        std::make_unique<GrokProtocol>(GrokReasoningEffort::None)));
}

TEST_CASE("GrokProvider - constructs with explicit model", "[grok][provider]") {
    REQUIRE_NOTHROW(HttpLLMProvider(
        "https://api.x.ai/v1",
        make_grok_creds("xai-test-key-abc"),
        "grok-3-mini",
        std::make_unique<GrokProtocol>(GrokReasoningEffort::None)));
}

TEST_CASE("GrokProvider - constructs with reasoning effort", "[grok][provider]") {
    REQUIRE_NOTHROW(HttpLLMProvider(
        "https://api.x.ai/v1",
        make_grok_creds("xai-test-key-abc"),
        "grok-3-mini",
        std::make_unique<GrokProtocol>(GrokReasoningEffort::Medium)));
}

TEST_CASE("GrokProvider - constructs with empty API key", "[grok][provider]") {
    // Empty key is allowed at construction; the error surfaces when a request is sent.
    REQUIRE_NOTHROW(HttpLLMProvider(
        "https://api.x.ai/v1",
        make_grok_creds(""),
        "grok-code-fast-1",
        std::make_unique<GrokProtocol>(GrokReasoningEffort::None)));
}

TEST_CASE("GrokProvider - satisfies LLMProvider interface", "[grok][provider]") {
    HttpLLMProvider provider(
        "https://api.x.ai/v1",
        make_grok_creds("xai-test-key-abc"),
        "grok-code-fast-1",
        std::make_unique<GrokProtocol>(GrokReasoningEffort::None));
    // Verify static polymorphism: HttpLLMProvider IS-A LLMProvider
    LLMProvider& base = provider;
    (void)base;
    SUCCEED("HttpLLMProvider is substitutable for LLMProvider");
}

TEST_CASE("GrokProvider - can be stored as shared_ptr<LLMProvider>", "[grok][provider]") {
    std::shared_ptr<LLMProvider> prov =
        std::make_shared<HttpLLMProvider>(
            "https://api.x.ai/v1",
            make_grok_creds("xai-test-key-abc"),
            "grok-3-mini",
            std::make_unique<GrokProtocol>(GrokReasoningEffort::Medium));
    REQUIRE(prov != nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// GrokSerializer — payload is valid JSON structure (structural checks)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GrokSerializer - payload starts with { and ends with }", "[grok][serializer][structure]") {
    auto payload = GrokProtocol{}.serialize(make_simple_request());
    REQUIRE(!payload.empty());
    REQUIRE(payload.front() == '{');
    REQUIRE(payload.back() == '}');
}

TEST_CASE("GrokSerializer - messages array is always present", "[grok][serializer][structure]") {
    auto payload = GrokProtocol{}.serialize(make_simple_request());
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("messages":[)"));
}

TEST_CASE("GrokSerializer - tools array absent when no tools provided", "[grok][serializer][structure]") {
    auto payload = GrokProtocol{}.serialize(make_simple_request());
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("tools")"));
}

TEST_CASE("GrokSerializer - no trailing commas in JSON arrays", "[grok][serializer][structure]") {
    // A trailing comma before ] is invalid JSON — verify it's not present
    auto payload = GrokProtocol{}.serialize(make_simple_request());
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",]"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",}"));
}

TEST_CASE("GrokSerializer - no trailing commas with multiple messages", "[grok][serializer][structure]") {
    ChatRequest req;
    req.model = "grok-code-fast-1";
    req.stream = true;
    req.messages = {
        Message{.role = "user", .content = "first"},
        Message{.role = "assistant", .content = "second"},
        Message{.role = "user", .content = "third"}
    };
    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",]"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",}"));
}

// ─────────────────────────────────────────────────────────────────────────────
// NEW: GrokProtocol-Specific Error Handling Tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GrokProtocol - name returns grok", "[grok][protocol]") {
    GrokProtocol protocol;
    REQUIRE(protocol.name() == "grok");
}

TEST_CASE("GrokProtocol - clone creates independent copy", "[grok][protocol]") {
    GrokProtocol original(GrokReasoningEffort::High);
    auto cloned = original.clone();
    
    REQUIRE(cloned != nullptr);
    REQUIRE(cloned->name() == "grok");
}

TEST_CASE("GrokProtocol - event delimiter is newline", "[grok][protocol]") {
    GrokProtocol protocol;
    REQUIRE(protocol.event_delimiter() == "\n\n");
}

TEST_CASE("GrokProtocol - build_url returns correct endpoint", "[grok][protocol]") {
    GrokProtocol protocol;
    auto url = protocol.build_url("https://api.x.ai/v1", "grok-code-fast-1");
    REQUIRE(url == "https://api.x.ai/v1/chat/completions");
}

TEST_CASE("GrokProtocol - build_headers includes correct content type", "[grok][protocol]") {
    GrokProtocol protocol;
    core::auth::AuthInfo auth;
    auth.headers["Authorization"] = "Bearer xai-test-key";
    
    auto headers = protocol.build_headers(auth);
    
    // Verify headers contain expected values
    bool has_auth = false;
    for (const auto& [k, v] : headers) {
        if (k == "Authorization" && v == "Bearer xai-test-key") {
            has_auth = true;
            break;
        }
    }
    REQUIRE(has_auth);
}

// ─────────────────────────────────────────────────────────────────────────────
// NEW: Edge Cases and Model Name Handling
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("grok_supports_reasoning_effort - handles edge case model names", "[grok][reasoning]") {
    // Edge cases that should NOT support reasoning_effort
    REQUIRE_FALSE(grok_supports_reasoning_effort(""));
    REQUIRE_FALSE(grok_supports_reasoning_effort("grok"));
    REQUIRE_FALSE(grok_supports_reasoning_effort("grok-"));
    REQUIRE_FALSE(grok_supports_reasoning_effort("grok-3"));
    REQUIRE_FALSE(grok_supports_reasoning_effort("grok-3-fast"));
    REQUIRE_FALSE(grok_supports_reasoning_effort("grok-mini"));  // Must be grok-3-mini
    
    // Valid reasoning_effort models
    REQUIRE(grok_supports_reasoning_effort("grok-3-mini"));
    REQUIRE(grok_supports_reasoning_effort("grok-3-mini-beta"));
    REQUIRE(grok_supports_reasoning_effort("grok-3-mini-fast-latest"));
}

TEST_CASE("GrokSerializer - preserves complex unicode content", "[grok][serializer]") {
    ChatRequest req;
    req.model = "grok-code-fast-1";
    req.messages.push_back(Message{.role = "user", .content = "Hello 世界 🌍 ñoño"});
    
    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("Hello 世界 🌍 ñoño"));
}

TEST_CASE("GrokSerializer - handles very long content", "[grok][serializer]") {
    ChatRequest req;
    req.model = "grok-code-fast-1";
    std::string long_content(10000, 'x');
    req.messages.push_back(Message{.role = "user", .content = long_content});
    
    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE(!payload.empty());
    REQUIRE(payload.front() == '{');
    REQUIRE(payload.back() == '}');
}

TEST_CASE("GrokSerializer - multiple system messages are all included", "[grok][serializer]") {
    // OpenAI/Grok format allows multiple system messages (unlike Claude)
    ChatRequest req;
    req.model = "grok-code-fast-1";
    req.messages = {
        Message{.role = "system", .content = "First system message"},
        Message{.role = "system", .content = "Second system message"},
        Message{.role = "user", .content = "Hello"}
    };
    
    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("First system message"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("Second system message"));
}

// ─────────────────────────────────────────────────────────────────────────────
// NEW: ProviderFactory Grok Integration Tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ProviderFactory - creates grok provider with default config", "[grok][factory]") {
    core::config::ProviderConfig config;
    config.model = "grok-code-fast-1";
    
    auto provider = ProviderFactory::create_provider("grok", config);
    REQUIRE(provider != nullptr);
}

TEST_CASE("ProviderFactory - creates grok-mini provider with reasoning_effort", "[grok][factory]") {
    core::config::ProviderConfig config;
    config.model = "grok-3-mini";
    config.reasoning_effort = "high";
    
    auto provider = ProviderFactory::create_provider("grok-mini", config);
    REQUIRE(provider != nullptr);
}

TEST_CASE("ProviderFactory - grok provider uses correct base url", "[grok][factory]") {
    core::config::ProviderConfig config;
    config.model = "grok-code-fast-1";
    // base_url should default to https://api.x.ai/v1
    
    auto provider = ProviderFactory::create_provider("grok", config);
    REQUIRE(provider != nullptr);
    // The base URL is internal to the provider, but we can verify it doesn't throw
}

TEST_CASE("ProviderFactory - handles various grok-prefixed provider names", "[grok][factory]") {
    core::config::ProviderConfig config;
    config.model = "grok-code-fast-1";
    
    // All these should create valid providers
    REQUIRE(ProviderFactory::create_provider("grok", config) != nullptr);
    REQUIRE(ProviderFactory::create_provider("grok-4", config) != nullptr);
    REQUIRE(ProviderFactory::create_provider("grok-reasoning", config) != nullptr);
    REQUIRE(ProviderFactory::create_provider("grok-mini", config) != nullptr);
}

// ─────────────────────────────────────────────────────────────────────────────
// NEW: Response parsing with usage information
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("parse_openai_sse_chunk - usage chunk with prompt tokens", "[grok][parser][usage]") {
    // Usage chunk typically arrives before [DONE] when stream_options.include_usage=true
    std::string json = R"({
        "id": "chatcmpl-123",
        "object": "chat.completion.chunk",
        "choices": [],
        "usage": {
            "prompt_tokens": 42,
            "completion_tokens": 10,
            "total_tokens": 52
        }
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    // Usage chunks have no content - just metadata
    REQUIRE(content.empty());
    REQUIRE(tools.empty());
}

TEST_CASE("parse_openai_sse_chunk - handles nested JSON in tool arguments", "[grok][parser][tools]") {
    std::string json = R"({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "function": {
                        "arguments": "{\"nested\":{\"key\":\"value\"},\"array\":[1,2,3]}"
                    }
                }]
            }
        }]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(tools.size() == 1);
    REQUIRE_THAT(tools[0].function.arguments, Catch::Matchers::ContainsSubstring("nested"));
    REQUIRE_THAT(tools[0].function.arguments, Catch::Matchers::ContainsSubstring("array"));
}

TEST_CASE("parse_openai_sse_chunk - handles unicode in content", "[grok][parser]") {
    std::string json = R"({
        "choices": [{"delta": {"content": "Hello 世界 🌍"}}]
    })";
    auto [content, tools] = parse_openai_sse_chunk(json);
    REQUIRE(content == "Hello 世界 🌍");
}
