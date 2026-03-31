#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/tools/Tool.hpp"

using namespace core::llm;
using namespace core::llm::protocols;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static ChatRequest make_simple_request(std::string model    = "claude-sonnet-4-6",
                                       std::string user_text = "Hello") {
    ChatRequest req;
    req.model  = std::move(model);
    req.stream = true;
    req.messages.push_back(Message{.role = "user", .content = std::move(user_text)});
    return req;
}

static core::tools::ToolParameter make_param(std::string name,
                                                std::string type,
                                                std::string desc,
                                                bool required = true) {
    core::tools::ToolParameter p;
    p.name        = std::move(name);
    p.type        = std::move(type);
    p.description = std::move(desc);
    p.required    = required;
    return p;
}

// ─────────────────────────────────────────────────────────────────────────────
// ClaudeSerializer — basic structure
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ClaudeSerializer - model is serialized correctly", "[claude][serializer]") {
    auto payload = AnthropicSerializer::serialize(make_simple_request("claude-opus-4-6"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"claude-opus-4-6")"));
}

TEST_CASE("ClaudeSerializer - stream true is included", "[claude][serializer]") {
    auto req = make_simple_request();
    req.stream = true;
    REQUIRE_THAT(AnthropicSerializer::serialize(req), Catch::Matchers::ContainsSubstring(R"("stream":true)"));
}

TEST_CASE("ClaudeSerializer - stream false is included", "[claude][serializer]") {
    auto req = make_simple_request();
    req.stream = false;
    REQUIRE_THAT(AnthropicSerializer::serialize(req), Catch::Matchers::ContainsSubstring(R"("stream":false)"));
}

TEST_CASE("ClaudeSerializer - user message content appears in payload", "[claude][serializer]") {
    auto payload = AnthropicSerializer::serialize(make_simple_request("claude-sonnet-4-6", "Tell me a joke"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("Tell me a joke"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"user")"));
}

TEST_CASE("ClaudeSerializer - max_tokens always present with default", "[claude][serializer]") {
    auto payload = AnthropicSerializer::serialize(make_simple_request());
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("max_tokens":8096)"));
}

TEST_CASE("ClaudeSerializer - max_tokens honours explicit value", "[claude][serializer]") {
    auto req = make_simple_request();
    req.max_tokens = 2048;
    REQUIRE_THAT(AnthropicSerializer::serialize(req), Catch::Matchers::ContainsSubstring(R"("max_tokens":2048)"));
}

TEST_CASE("ClaudeSerializer - custom default_max_tokens respected", "[claude][serializer]") {
    auto payload = AnthropicSerializer::serialize(make_simple_request(), 4096);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("max_tokens":4096)"));
}

TEST_CASE("ClaudeSerializer - temperature omitted when not set", "[claude][serializer]") {
    REQUIRE_THAT(AnthropicSerializer::serialize(make_simple_request()),
                 !Catch::Matchers::ContainsSubstring("temperature"));
}

TEST_CASE("ClaudeSerializer - temperature present when set", "[claude][serializer]") {
    auto req = make_simple_request();
    req.temperature = 0.7f;
    REQUIRE_THAT(AnthropicSerializer::serialize(req), Catch::Matchers::ContainsSubstring(R"("temperature")"));
}

// ─────────────────────────────────────────────────────────────────────────────
// ClaudeSerializer — system message extraction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ClaudeSerializer - system message becomes top-level field", "[claude][serializer][system]") {
    ChatRequest req;
    req.model = "claude-sonnet-4-6";
    req.messages.push_back(Message{.role = "system", .content = "You are helpful."});
    req.messages.push_back(Message{.role = "user",   .content = "Hi"});
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("system":"You are helpful.")"));
}

TEST_CASE("ClaudeSerializer - system message is NOT in messages array", "[claude][serializer][system]") {
    ChatRequest req;
    req.model = "claude-sonnet-4-6";
    req.messages.push_back(Message{.role = "system", .content = "You are helpful."});
    req.messages.push_back(Message{.role = "user",   .content = "Hi"});
    auto payload = AnthropicSerializer::serialize(req);
    // Only one system occurrence should appear (in top-level field, not in messages array)
    auto first = payload.find(R"("system":)");
    REQUIRE(first != std::string::npos);
    // Should NOT appear inside the "messages" array section
    auto messages_pos = payload.find(R"("messages":)");
    REQUIRE(messages_pos != std::string::npos);
    // The "role":"system" should not appear in the messages list
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("role":"system")"));
}

TEST_CASE("ClaudeSerializer - no system field when no system message", "[claude][serializer][system]") {
    REQUIRE_THAT(AnthropicSerializer::serialize(make_simple_request()),
                 !Catch::Matchers::ContainsSubstring(R"("system":)"));
}

TEST_CASE("ClaudeSerializer - system message with special chars is escaped", "[claude][serializer][system]") {
    ChatRequest req;
    req.model = "claude-sonnet-4-6";
    req.messages.push_back(Message{.role = "system", .content = "Say \"hello\"."});
    req.messages.push_back(Message{.role = "user",   .content = "Hi"});
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"(\"hello\")"));
}

// ─────────────────────────────────────────────────────────────────────────────
// ClaudeSerializer — extended thinking
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ClaudeSerializer - thinking disabled omits field", "[claude][serializer][thinking]") {
    AnthropicThinkingConfig cfg{.enabled = false};
    REQUIRE_THAT(AnthropicSerializer::serialize(make_simple_request(), 8096, cfg),
                 !Catch::Matchers::ContainsSubstring("thinking"));
}

TEST_CASE("ClaudeSerializer - thinking enabled emits type enabled", "[claude][serializer][thinking]") {
    AnthropicThinkingConfig cfg{.enabled = true, .budget_tokens = 10000};
    auto payload = AnthropicSerializer::serialize(make_simple_request(), 8096, cfg);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("thinking":{"type":"enabled","budget_tokens":10000})"));
}

TEST_CASE("ClaudeSerializer - thinking budget_tokens is correct", "[claude][serializer][thinking]") {
    AnthropicThinkingConfig cfg{.enabled = true, .budget_tokens = 5000};
    auto payload = AnthropicSerializer::serialize(make_simple_request(), 8096, cfg);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("5000"));
}

// ─────────────────────────────────────────────────────────────────────────────
// ClaudeSerializer — tools (input_schema format)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ClaudeSerializer - tools absent when no tools", "[claude][serializer][tools]") {
    REQUIRE_THAT(AnthropicSerializer::serialize(make_simple_request()),
                 !Catch::Matchers::ContainsSubstring(R"("tools")"));
}

TEST_CASE("ClaudeSerializer - tool uses input_schema not parameters", "[claude][serializer][tools]") {
    ChatRequest req = make_simple_request();
    Tool t;
    t.function.name        = "get_weather";
    t.function.description = "Get the weather";
    t.function.parameters.push_back(make_param("location", "string", "City name"));
    req.tools.push_back(t);
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("input_schema")"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("parameters")"));
}

TEST_CASE("ClaudeSerializer - tool name and description appear", "[claude][serializer][tools]") {
    ChatRequest req = make_simple_request();
    Tool t;
    t.function.name        = "read_file";
    t.function.description = "Read a file from disk";
    req.tools.push_back(t);
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("name":"read_file")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("description":"Read a file from disk")"));
}

TEST_CASE("ClaudeSerializer - tool required parameters listed", "[claude][serializer][tools]") {
    ChatRequest req = make_simple_request();
    Tool t;
    t.function.name = "shell";
    t.function.parameters.push_back(make_param("command", "string", "Shell command", true));
    t.function.parameters.push_back(make_param("timeout", "integer", "Timeout seconds", false));
    req.tools.push_back(t);
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("required":["command"])"));
}

TEST_CASE("ClaudeSerializer - tool with no required params has empty required array", "[claude][serializer][tools]") {
    ChatRequest req = make_simple_request();
    Tool t;
    t.function.name = "ping";
    t.function.parameters.push_back(make_param("host", "string", "Host", false));
    req.tools.push_back(t);
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("required":[])"));
}

TEST_CASE("ClaudeSerializer - multiple tools no trailing comma", "[claude][serializer][tools]") {
    ChatRequest req = make_simple_request();
    Tool t1; t1.function.name = "tool_a";
    Tool t2; t2.function.name = "tool_b";
    req.tools.push_back(t1);
    req.tools.push_back(t2);
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",]"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",}"));
}

// ─────────────────────────────────────────────────────────────────────────────
// ClaudeSerializer — tool role messages (tool_result format)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ClaudeSerializer - tool role becomes user message with tool_result", "[claude][serializer][tool_result]") {
    ChatRequest req;
    req.model = "claude-sonnet-4-6";
    req.messages.push_back(Message{.role = "user", .content = "What time is it?"});
    req.messages.push_back(Message{
        .role         = "assistant",
        .content      = "",
        .tool_calls   = {ToolCall{.id = "tc_001", .type = "function",
                                   .function = {.name = "get_time", .arguments = "{}"}}}
    });
    req.messages.push_back(Message{
        .role         = "tool",
        .content      = "12:00 UTC",
        .tool_call_id = "tc_001"
    });
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"tool_result")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("tool_use_id":"tc_001")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("12:00 UTC"));
}

TEST_CASE("ClaudeSerializer - tool role maps to user role in output", "[claude][serializer][tool_result]") {
    ChatRequest req;
    req.model = "claude-sonnet-4-6";
    req.messages.push_back(Message{.role = "tool", .content = "result", .tool_call_id = "id1"});
    auto payload = AnthropicSerializer::serialize(req);
    // The wrapper object should have role:user
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"user")"));
    // The original tool role should not appear
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("role":"tool")"));
}

// ─────────────────────────────────────────────────────────────────────────────
// ClaudeSerializer — assistant messages with tool_calls (tool_use format)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ClaudeSerializer - assistant tool_calls become tool_use content blocks", "[claude][serializer][tool_use]") {
    ChatRequest req = make_simple_request();
    req.messages.push_back(Message{
        .role       = "assistant",
        .content    = "",
        .tool_calls = {ToolCall{.id = "toolu_01", .type = "function",
                                 .function = {.name = "shell", .arguments = R"({"command":"ls"})"}}}
    });
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"tool_use")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("id":"toolu_01")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("name":"shell")"));
}

TEST_CASE("ClaudeSerializer - tool_use input is raw JSON not string", "[claude][serializer][tool_use]") {
    ChatRequest req = make_simple_request();
    req.messages.push_back(Message{
        .role       = "assistant",
        .tool_calls = {ToolCall{.id = "tc", .function = {.name = "f", .arguments = R"({"k":"v"})"}}}
    });
    auto payload = AnthropicSerializer::serialize(req);
    // The JSON object must be embedded directly, not as a quoted string
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("input":{"k":"v"})"));
}

TEST_CASE("ClaudeSerializer - empty tool call arguments become empty object", "[claude][serializer][tool_use]") {
    ChatRequest req = make_simple_request();
    req.messages.push_back(Message{
        .role       = "assistant",
        .tool_calls = {ToolCall{.id = "tc", .function = {.name = "noop", .arguments = ""}}}
    });
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("input":{})"));
}

TEST_CASE("ClaudeSerializer - assistant text + tool_call both appear", "[claude][serializer][tool_use]") {
    ChatRequest req = make_simple_request();
    req.messages.push_back(Message{
        .role       = "assistant",
        .content    = "I will call the tool.",
        .tool_calls = {ToolCall{.id = "tc", .function = {.name = "f", .arguments = "{}"}}}
    });
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"text")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("I will call the tool."));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"tool_use")"));
}

// ─────────────────────────────────────────────────────────────────────────────
// ClaudeSerializer — JSON escaping
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ClaudeSerializer - newlines in content are escaped", "[claude][serializer][escape]") {
    auto payload = AnthropicSerializer::serialize(make_simple_request("claude-sonnet-4-6", "line1\nline2"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"(line1\nline2)"));
}

TEST_CASE("ClaudeSerializer - tabs in content are escaped", "[claude][serializer][escape]") {
    auto payload = AnthropicSerializer::serialize(make_simple_request("claude-sonnet-4-6", "col1\tcol2"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"(col1\tcol2)"));
}

TEST_CASE("ClaudeSerializer - quotes in content are escaped", "[claude][serializer][escape]") {
    auto payload = AnthropicSerializer::serialize(make_simple_request("claude-sonnet-4-6", R"(say "hi")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"(say \"hi\")"));
}

TEST_CASE("ClaudeSerializer - backslash in content is escaped", "[claude][serializer][escape]") {
    auto payload = AnthropicSerializer::serialize(make_simple_request("claude-sonnet-4-6", R"(C:\path)"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"(C:\\path)"));
}

// ─────────────────────────────────────────────────────────────────────────────
// ClaudeSerializer — structural validity
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ClaudeSerializer - payload starts with { and ends with }", "[claude][serializer][structure]") {
    auto payload = AnthropicSerializer::serialize(make_simple_request());
    REQUIRE(payload.front() == '{');
    REQUIRE(payload.back() == '}');
}

TEST_CASE("ClaudeSerializer - messages array always present", "[claude][serializer][structure]") {
    ChatRequest req;
    req.model = "claude-sonnet-4-6";
    REQUIRE_THAT(AnthropicSerializer::serialize(req), Catch::Matchers::ContainsSubstring(R"("messages":[)"));
}

TEST_CASE("ClaudeSerializer - empty messages produce valid messages array", "[claude][serializer][structure]") {
    ChatRequest req;
    req.model = "claude-sonnet-4-6";
    REQUIRE_THAT(AnthropicSerializer::serialize(req), Catch::Matchers::ContainsSubstring(R"("messages":[])"));
}

TEST_CASE("ClaudeSerializer - multiple messages no trailing comma", "[claude][serializer][structure]") {
    ChatRequest req;
    req.model = "claude-sonnet-4-6";
    req.messages.push_back(Message{.role = "user", .content = "A"});
    req.messages.push_back(Message{.role = "assistant", .content = "B"});
    req.messages.push_back(Message{.role = "user", .content = "C"});
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",]"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(",}"));
}

TEST_CASE("ClaudeSerializer - known model names preserved verbatim", "[claude][serializer]") {
    for (const auto* model : {"claude-opus-4-6", "claude-sonnet-4-6",
                               "claude-haiku-4-5", "claude-3-5-sonnet-20241022",
                               "claude-3-5-haiku-20241022"}) {
        auto payload = AnthropicSerializer::serialize(make_simple_request(model));
        REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(model));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicProtocol headers — OAuth beta and merge behavior
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("AnthropicProtocol::build_headers adds oauth beta when oauth auth property is set", "[claude][headers]") {
    AnthropicProtocol protocol;
    core::auth::AuthInfo auth;
    auth.properties["oauth"] = "1";
    auth.headers["Authorization"] = "Bearer oauth-token";

    auto headers = protocol.build_headers(auth);

    REQUIRE(headers.count("anthropic-beta") == 1);
    REQUIRE_THAT(headers.at("anthropic-beta"), Catch::Matchers::ContainsSubstring("oauth-2025-04-20"));
    REQUIRE(headers.at("Authorization") == "Bearer oauth-token");
}

TEST_CASE("AnthropicProtocol::build_headers always includes claude-code beta header", "[claude][headers]") {
    AnthropicProtocol protocol;
    core::auth::AuthInfo auth;

    auto headers = protocol.build_headers(auth);
    REQUIRE(headers.count("anthropic-beta") == 1);
    REQUIRE_THAT(headers.at("anthropic-beta"), Catch::Matchers::ContainsSubstring("claude-code-20250219"));
}

TEST_CASE("AnthropicProtocol::build_headers merges caller anthropic-beta with internal betas", "[claude][headers]") {
    AnthropicProtocol protocol({.enabled = true, .budget_tokens = 10000});
    core::auth::AuthInfo auth;
    auth.properties["oauth"] = "1";
    auth.headers["anthropic-beta"] = "custom-beta-1";

    auto headers = protocol.build_headers(auth);
    REQUIRE(headers.count("anthropic-beta") == 1);

    const auto& merged = headers.at("anthropic-beta");
    REQUIRE_THAT(merged, Catch::Matchers::ContainsSubstring("interleaved-thinking-2025-05-14"));
    REQUIRE_THAT(merged, Catch::Matchers::ContainsSubstring("oauth-2025-04-20"));
    REQUIRE_THAT(merged, Catch::Matchers::ContainsSubstring("custom-beta-1"));
}

TEST_CASE("AnthropicProtocol::build_headers de-duplicates repeated beta names", "[claude][headers]") {
    AnthropicProtocol protocol;
    core::auth::AuthInfo auth;
    auth.properties["oauth"] = "1";
    auth.headers["anthropic-beta"] = "oauth-2025-04-20, oauth-2025-04-20";

    auto headers = protocol.build_headers(auth);
    const std::string merged = headers.at("anthropic-beta");

    std::size_t hits = 0;
    std::size_t pos = 0;
    const std::string needle = "oauth-2025-04-20";
    while ((pos = merged.find(needle, pos)) != std::string::npos) {
        ++hits;
        pos += needle.size();
    }
    REQUIRE(hits == 1);
}

TEST_CASE("AnthropicProtocol::prepare_request strips [1m] suffix and adds context beta", "[claude][headers]") {
    AnthropicProtocol protocol;
    ChatRequest req = make_simple_request("claude-sonnet-4-6[1m]");
    protocol.prepare_request(req);

    REQUIRE(req.model == "claude-sonnet-4-6");

    core::auth::AuthInfo auth;
    auto headers = protocol.build_headers(auth);
    REQUIRE(headers.count("anthropic-beta") == 1);
    REQUIRE_THAT(headers.at("anthropic-beta"), Catch::Matchers::ContainsSubstring("context-1m-2025-08-07"));
}

TEST_CASE("AnthropicProtocol::prepare_request resolves Claude alias + [1m]", "[claude][headers]") {
    AnthropicProtocol protocol;
    ChatRequest req = make_simple_request("sonnet[1m]");
    protocol.prepare_request(req);

    REQUIRE(req.model == "claude-sonnet-4-6");

    core::auth::AuthInfo auth;
    auto headers = protocol.build_headers(auth);
    REQUIRE(headers.count("anthropic-beta") == 1);
    REQUIRE_THAT(headers.at("anthropic-beta"), Catch::Matchers::ContainsSubstring("context-1m-2025-08-07"));
}

TEST_CASE("AnthropicProtocol::prepare_request keeps custom model case while removing [1m]", "[claude][headers]") {
    AnthropicProtocol protocol;
    ChatRequest req = make_simple_request("MyAzureDeployment[1M]");
    protocol.prepare_request(req);
    REQUIRE(req.model == "MyAzureDeployment");

    core::auth::AuthInfo auth;
    auto headers = protocol.build_headers(auth);
    REQUIRE(headers.count("anthropic-beta") == 1);
    REQUIRE_THAT(headers.at("anthropic-beta"), Catch::Matchers::ContainsSubstring("context-1m-2025-08-07"));
}

TEST_CASE("AnthropicProtocol::prepare_request does not add context beta without [1m]", "[claude][headers]") {
    AnthropicProtocol protocol;
    ChatRequest req = make_simple_request("sonnet");
    protocol.prepare_request(req);
    REQUIRE(req.model == "claude-sonnet-4-6");

    core::auth::AuthInfo auth;
    auto headers = protocol.build_headers(auth);
    REQUIRE(headers.count("anthropic-beta") == 1);
    REQUIRE_THAT(headers.at("anthropic-beta"), !Catch::Matchers::ContainsSubstring("context-1m-2025-08-07"));
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicSSEParser — text events
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("AnthropicSSEParser - text_delta returns text", "[claude][sse]") {
    AnthropicSSEParser p;
    auto r = p.process_event("content_block_delta",
        R"({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}})");
    REQUIRE(r.text == "Hello");
    REQUIRE(r.completed_tools.empty());
    REQUIRE(!r.done);
}

TEST_CASE("AnthropicSSEParser - empty text_delta returns empty text", "[claude][sse]") {
    AnthropicSSEParser p;
    auto r = p.process_event("content_block_delta",
        R"({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}})");
    REQUIRE(r.text.empty());
}

TEST_CASE("AnthropicSSEParser - ping produces no output", "[claude][sse]") {
    AnthropicSSEParser p;
    auto r = p.process_event("ping", R"({"type":"ping"})");
    REQUIRE(r.text.empty());
    REQUIRE(r.completed_tools.empty());
    REQUIRE(!r.done);
}

TEST_CASE("AnthropicSSEParser - message_start produces no output", "[claude][sse]") {
    AnthropicSSEParser p;
    auto r = p.process_event("message_start",
        R"({"type":"message_start","message":{"id":"msg_01","model":"claude-sonnet-4-6"}})");
    REQUIRE(r.text.empty());
    REQUIRE(!r.done);
}

TEST_CASE("AnthropicSSEParser - message_delta produces no output", "[claude][sse]") {
    AnthropicSSEParser p;
    auto r = p.process_event("message_delta",
        R"({"type":"message_delta","delta":{"stop_reason":"end_turn"}})");
    REQUIRE(r.text.empty());
    REQUIRE(!r.done);
}

TEST_CASE("AnthropicSSEParser - message_stop sets done=true", "[claude][sse]") {
    AnthropicSSEParser p;
    auto r = p.process_event("message_stop", R"({"type":"message_stop"})");
    REQUIRE(r.done);
    REQUIRE(r.text.empty());
    REQUIRE(r.completed_tools.empty());
}

TEST_CASE("AnthropicSSEParser - content_block_start for text produces no output", "[claude][sse]") {
    AnthropicSSEParser p;
    auto r = p.process_event("content_block_start",
        R"({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})");
    REQUIRE(r.text.empty());
    REQUIRE(r.completed_tools.empty());
}

TEST_CASE("AnthropicSSEParser - content_block_stop without tool produces no output", "[claude][sse]") {
    AnthropicSSEParser p;
    auto r = p.process_event("content_block_stop", R"({"type":"content_block_stop","index":0})");
    REQUIRE(r.text.empty());
    REQUIRE(r.completed_tools.empty());
    REQUIRE(!r.done);
}

TEST_CASE("AnthropicSSEParser - empty json_str produces no output", "[claude][sse]") {
    AnthropicSSEParser p;
    auto r = p.process_event("content_block_delta", "");
    REQUIRE(r.text.empty());
    REQUIRE(r.completed_tools.empty());
    REQUIRE(!r.done);
}

TEST_CASE("AnthropicSSEParser - malformed JSON produces no output", "[claude][sse]") {
    AnthropicSSEParser p;
    auto r = p.process_event("content_block_delta", "{not json}");
    REQUIRE(r.text.empty());
    REQUIRE(r.completed_tools.empty());
}

TEST_CASE("AnthropicSSEParser - unknown event type produces no output", "[claude][sse]") {
    AnthropicSSEParser p;
    auto r = p.process_event("some_future_event", R"({"type":"some_future_event"})");
    REQUIRE(r.text.empty());
    REQUIRE(!r.done);
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicSSEParser — thinking delta (must be ignored)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("AnthropicSSEParser - thinking_delta is silently ignored", "[claude][sse][thinking]") {
    AnthropicSSEParser p;
    // Simulate extended thinking block
    p.process_event("content_block_start",
        R"({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}})");
    auto r = p.process_event("content_block_delta",
        R"({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"Let me analyze..."}})");
    REQUIRE(r.text.empty());
    REQUIRE(r.completed_tools.empty());
    REQUIRE(!r.done);
}

TEST_CASE("AnthropicSSEParser - text follows thinking block correctly", "[claude][sse][thinking]") {
    AnthropicSSEParser p;
    // Thinking block
    p.process_event("content_block_start",
        R"({"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}})");
    p.process_event("content_block_delta",
        R"({"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"..."}})");
    p.process_event("content_block_stop", R"({"type":"content_block_stop","index":0})");
    // Text block
    p.process_event("content_block_start",
        R"({"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}})");
    auto r = p.process_event("content_block_delta",
        R"({"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Answer"}})");
    REQUIRE(r.text == "Answer");
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicSSEParser — tool call reassembly
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("AnthropicSSEParser - tool_use block start captures id and name", "[claude][sse][tool]") {
    AnthropicSSEParser p;
    auto r = p.process_event("content_block_start",
        R"({"type":"content_block_start","index":1,)"
        R"("content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}})");
    // No output yet — tool not complete
    REQUIRE(r.completed_tools.empty());
    REQUIRE(r.text.empty());
}

TEST_CASE("AnthropicSSEParser - input_json_delta accumulates without output", "[claude][sse][tool]") {
    AnthropicSSEParser p;
    p.process_event("content_block_start",
        R"({"type":"content_block_start","index":1,)"
        R"("content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}})");
    auto r = p.process_event("content_block_delta",
        R"({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}})");
    REQUIRE(r.completed_tools.empty());
    REQUIRE(r.text.empty());
}

TEST_CASE("AnthropicSSEParser - content_block_stop emits complete tool call", "[claude][sse][tool]") {
    AnthropicSSEParser p;
    p.process_event("content_block_start",
        R"({"type":"content_block_start","index":1,)"
        R"("content_block":{"type":"tool_use","id":"toolu_01","name":"get_weather","input":{}}})");
    p.process_event("content_block_delta",
        R"({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"location\":"}})");
    p.process_event("content_block_delta",
        R"({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"SF\"}"}})");
    auto r = p.process_event("content_block_stop", R"({"type":"content_block_stop","index":1})");
    REQUIRE(r.completed_tools.size() == 1);
    REQUIRE(r.completed_tools[0].id == "toolu_01");
    REQUIRE(r.completed_tools[0].function.name == "get_weather");
    REQUIRE(r.completed_tools[0].function.arguments == R"({"location":"SF"})");
    REQUIRE(r.completed_tools[0].type == "function");
}

TEST_CASE("AnthropicSSEParser - tool call index is captured", "[claude][sse][tool]") {
    AnthropicSSEParser p;
    p.process_event("content_block_start",
        R"({"type":"content_block_start","index":2,)"
        R"("content_block":{"type":"tool_use","id":"toolu_02","name":"shell","input":{}}})");
    auto r = p.process_event("content_block_stop", R"({"type":"content_block_stop","index":2})");
    REQUIRE(r.completed_tools.size() == 1);
    REQUIRE(r.completed_tools[0].index == 2);
}

TEST_CASE("AnthropicSSEParser - tool with empty arguments emits empty string", "[claude][sse][tool]") {
    AnthropicSSEParser p;
    p.process_event("content_block_start",
        R"({"type":"content_block_start","index":0,)"
        R"("content_block":{"type":"tool_use","id":"tc","name":"noop","input":{}}})");
    auto r = p.process_event("content_block_stop", R"({"type":"content_block_stop","index":0})");
    REQUIRE(r.completed_tools.size() == 1);
    REQUIRE(r.completed_tools[0].function.arguments.empty());
}

TEST_CASE("AnthropicSSEParser - two sequential tool calls reassembled correctly", "[claude][sse][tool]") {
    AnthropicSSEParser p;

    // First tool
    p.process_event("content_block_start",
        R"({"type":"content_block_start","index":1,)"
        R"("content_block":{"type":"tool_use","id":"id1","name":"tool_a","input":{}}})");
    p.process_event("content_block_delta",
        R"({"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"x\":1}"}})");
    auto r1 = p.process_event("content_block_stop", R"({"type":"content_block_stop","index":1})");
    REQUIRE(r1.completed_tools.size() == 1);
    REQUIRE(r1.completed_tools[0].function.name == "tool_a");

    // Second tool
    p.process_event("content_block_start",
        R"({"type":"content_block_start","index":2,)"
        R"("content_block":{"type":"tool_use","id":"id2","name":"tool_b","input":{}}})");
    p.process_event("content_block_delta",
        R"({"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"y\":2}"}})");
    auto r2 = p.process_event("content_block_stop", R"({"type":"content_block_stop","index":2})");
    REQUIRE(r2.completed_tools.size() == 1);
    REQUIRE(r2.completed_tools[0].function.name == "tool_b");
    REQUIRE(r2.completed_tools[0].function.arguments == R"({"y":2})");
}

TEST_CASE("AnthropicSSEParser - state cleared after tool emitted", "[claude][sse][tool]") {
    AnthropicSSEParser p;
    p.process_event("content_block_start",
        R"({"type":"content_block_start","index":0,)"
        R"("content_block":{"type":"tool_use","id":"tc","name":"f","input":{}}})");
    p.process_event("content_block_stop", R"({"type":"content_block_stop","index":0})");
    // Second stop with no tool in progress — should produce nothing
    auto r = p.process_event("content_block_stop", R"({"type":"content_block_stop","index":0})");
    REQUIRE(r.completed_tools.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicSSEParser — full event sequence
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("AnthropicSSEParser - full text streaming sequence", "[claude][sse][integration]") {
    AnthropicSSEParser p;
    std::string full_text;

    p.process_event("message_start",
        R"({"type":"message_start","message":{"id":"m1","model":"claude-sonnet-4-6"}})");
    p.process_event("content_block_start",
        R"({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})");

    for (const char* chunk : {"He", "llo", " World"}) {
        std::string json = R"({"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":")" +
                           std::string(chunk) + R"("}})";
        full_text += p.process_event("content_block_delta", json).text;
    }

    p.process_event("content_block_stop", R"({"type":"content_block_stop","index":0})");
    p.process_event("message_delta",
        R"({"type":"message_delta","delta":{"stop_reason":"end_turn"}})");
    auto done = p.process_event("message_stop", R"({"type":"message_stop"})");

    REQUIRE(full_text == "Hello World");
    REQUIRE(done.done);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bug-fix: Bug 1 — extended thinking forces temperature=1
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ClaudeSerializer - thinking forces temperature to 1 when request sets other value", "[claude][serializer][thinking]") {
    AnthropicThinkingConfig cfg{.enabled = true, .budget_tokens = 10000};
    auto req = make_simple_request();
    req.temperature = 0.7f;
    auto payload = AnthropicSerializer::serialize(req, 8096, cfg);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("temperature":1)"));
    // The caller-supplied 0.7 must not appear in any form.
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("0.7"));
}

TEST_CASE("ClaudeSerializer - thinking with no temperature still emits temperature 1", "[claude][serializer][thinking]") {
    AnthropicThinkingConfig cfg{.enabled = true, .budget_tokens = 5000};
    auto payload = AnthropicSerializer::serialize(make_simple_request(), 8096, cfg);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("temperature":1)"));
}

TEST_CASE("ClaudeSerializer - temperature without thinking is passed through unchanged", "[claude][serializer][thinking]") {
    auto req = make_simple_request();
    req.temperature = 0.5f;
    auto payload = AnthropicSerializer::serialize(req);
    // Not forced to 1.
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("temperature":1)"));
    // But temperature must still be present.
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("temperature")"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Bug-fix: Bug 2 — ISO 8601 rate-limit reset timestamps are parsed correctly
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("AnthropicProtocol - ISO 8601 requests_reset parses to nonzero unix timestamp", "[claude][ratelimit]") {
    AnthropicProtocol p;
    cpr::Header hdrs{
        {"anthropic-ratelimit-requests-limit",     "100"},
        {"anthropic-ratelimit-requests-remaining", "80"},
        {"anthropic-ratelimit-requests-reset",     "2025-01-15T12:00:30Z"},
    };
    p.on_response(HttpResponse{200, "", hdrs});
    auto rl = p.last_rate_limit();
    REQUIRE(rl.requests_limit == 100);
    REQUIRE(rl.requests_remaining == 80);
    // Must be a real Unix timestamp, not the broken 0 returned by the old code.
    REQUIRE(rl.requests_reset > 0);
}

TEST_CASE("AnthropicProtocol - ISO 8601 tokens_reset parses to nonzero unix timestamp", "[claude][ratelimit]") {
    AnthropicProtocol p;
    cpr::Header hdrs{
        {"anthropic-ratelimit-tokens-limit",     "50000"},
        {"anthropic-ratelimit-tokens-remaining", "40000"},
        {"anthropic-ratelimit-tokens-reset",     "2025-06-01T00:00:00Z"},
    };
    p.on_response(HttpResponse{200, "", hdrs});
    auto rl = p.last_rate_limit();
    REQUIRE(rl.tokens_limit == 50000);
    REQUIRE(rl.tokens_remaining == 40000);
    REQUIRE(rl.tokens_reset > 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Bug-fix: Bug 3 — items_schema forwarded for array tool parameters
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ClaudeSerializer - array tool parameter emits items schema", "[claude][serializer][tools]") {
    ChatRequest req = make_simple_request();
    Tool t;
    t.function.name        = "list_files";
    t.function.description = "List files";
    auto p = make_param("extensions", "array", "File extensions to filter", false);
    p.items_schema = R"({"type":"string"})";
    t.function.parameters.push_back(std::move(p));
    req.tools.push_back(t);
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"array")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("items":{"type":"string"})"));
}

TEST_CASE("ClaudeSerializer - non-array parameter without items_schema emits no items field", "[claude][serializer][tools]") {
    ChatRequest req = make_simple_request();
    Tool t;
    t.function.name = "get_value";
    t.function.parameters.push_back(make_param("key", "string", "The key"));
    req.tools.push_back(t);
    auto payload = AnthropicSerializer::serialize(req);
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("items")"));
}

// ─────────────────────────────────────────────────────────────────────────────
// ProviderFactory — creates Claude provider
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ProviderFactory creates a Claude provider", "[claude][factory]") {
    core::config::ProviderConfig config;
    config.model = "claude-sonnet-4-6";
    auto provider = core::llm::ProviderFactory::create_provider("claude", config);
    REQUIRE(provider != nullptr);
}
