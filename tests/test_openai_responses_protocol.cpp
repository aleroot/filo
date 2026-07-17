#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"
#include "core/llm/Models.hpp"

#include <filesystem>
#include <fstream>

using namespace core::llm;
using namespace core::llm::protocols;

namespace {

std::filesystem::path make_temp_image_file(std::string_view filename = "filo-responses-image.png") {
    const auto path = std::filesystem::temp_directory_path() / filename;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "fake-image";
    return path;
}

} // namespace

TEST_CASE("OpenAIResponsesProtocol - build_url uses /responses endpoint",
          "[openai][responses][url]") {
    OpenAIResponsesProtocol protocol;
    REQUIRE(protocol.build_url("https://api.openai.com/v1", "gpt-5")
            == "https://api.openai.com/v1/responses");
}

TEST_CASE("OpenAIResponsesProtocol - build_url supports xAI Grok Responses endpoint",
          "[openai][responses][url][grok]") {
    OpenAIResponsesProtocol protocol;
    REQUIRE(protocol.build_url("https://api.x.ai/v1", "grok-4.5")
            == "https://api.x.ai/v1/responses");
}

TEST_CASE("OpenAIResponsesProtocol - serializer emits responses request fields",
          "[openai][responses][serializer]") {
    OpenAIResponsesProtocol protocol;

    ChatRequest req;
    req.model = "gpt-5";
    req.stream = true;
    req.messages.push_back(Message{
        .role = "system",
        .content = "You are a coding assistant."
    });
    req.messages.push_back(Message{
        .role = "user",
        .content = "Say hello."
    });
    req.max_tokens = 200;

    const std::string payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"gpt-5")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("instructions":"You are a coding assistant.")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("input":[{"type":"message","role":"user")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("stream":true)"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("max_output_tokens":200)"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("tool_choice":"auto")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("parallel_tool_calls":false)"));
}

TEST_CASE("OpenAIResponsesProtocol - GPT-5.6 preserves max reasoning effort",
          "[openai][responses][serializer][effort]") {
    OpenAIResponsesProtocol protocol;

    ChatRequest req;
    req.model = "gpt-5.6-terra";
    req.effort = "max";
    req.messages.push_back(Message{.role = "user", .content = "Solve this."});

    const std::string payload = protocol.serialize(req);
    REQUIRE_THAT(payload,
                 Catch::Matchers::ContainsSubstring(R"("reasoning":{"effort":"max"})"));
}

TEST_CASE("OpenAIResponsesProtocol - serializer emits input_image items",
          "[openai][responses][serializer][vision]") {
    OpenAIResponsesProtocol protocol;
    const auto image = make_temp_image_file();

    ChatRequest req;
    req.model = "gpt-5";
    req.messages.push_back(Message{
        .role = "user",
        .content = describe_image_attachment(image.string()),
        .content_parts = {
            ContentPart::make_text("Read the error in this screenshot."),
            ContentPart::make_image(image.string(), "image/png"),
        },
    });

    const std::string payload = protocol.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"input_text","text":"Read the error in this screenshot.")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"input_image")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("data:image/png;base64,"));
}

TEST_CASE("OpenAIResponsesProtocol - serializer includes continuation/cache/tier fields",
          "[openai][responses][serializer][continuation]") {
    OpenAIResponsesProtocol protocol(false, "priority");

    ChatRequest req;
    req.model = "gpt-5";
    req.messages.push_back(Message{
        .role = "user",
        .content = "Continue the previous turn."
    });
    req.previous_response_id = "resp_prev_1";
    req.prompt_cache_key = "filo-session-1";

    const std::string payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("previous_response_id":"resp_prev_1")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("prompt_cache_key":"filo-session-1")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("service_tier":"priority")"));
}

TEST_CASE("OpenAIResponsesProtocol - continuity state is shared across clones",
          "[openai][responses][continuity]") {
    OpenAIResponsesProtocol owner;

    auto stream1 = owner.clone();
    ChatRequest req1;
    req1.model = "gpt-5";
    req1.messages.push_back(Message{
        .role = "user",
        .content = "First turn."
    });

    stream1->prepare_request(req1);
    REQUIRE_FALSE(req1.prompt_cache_key.empty());
    REQUIRE(req1.previous_response_id.empty());

    const auto completed = stream1->parse_event(
        "event: response.completed\n"
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_shared_1\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}");
    REQUIRE(completed.done);

    cpr::Header headers;
    stream1->on_response(HttpResponse{200, "", headers});

    auto stream2 = owner.clone();
    ChatRequest req2;
    req2.model = "gpt-5";
    req2.messages.push_back(Message{
        .role = "user",
        .content = "Second turn."
    });

    stream2->prepare_request(req2);
    REQUIRE(req2.prompt_cache_key == req1.prompt_cache_key);
    REQUIRE(req2.previous_response_id == "resp_shared_1");
}

TEST_CASE("OpenAIResponsesProtocol - session id is used as prompt cache key",
          "[openai][responses][continuity]") {
    OpenAIResponsesProtocol owner;

    auto stream = owner.clone();
    ChatRequest req;
    req.model = "gpt-5";
    req.session_id = "session-cache-key";
    req.messages.push_back(Message{
        .role = "user",
        .content = "Use the session as cache scope."
    });

    stream->prepare_request(req);

    REQUIRE(req.prompt_cache_key == "session-cache-key");
}

TEST_CASE("OpenAIResponsesProtocol - continuity is isolated by session id",
          "[openai][responses][continuity]") {
    OpenAIResponsesProtocol owner;

    auto session_a = owner.clone();
    ChatRequest req_a;
    req_a.model = "gpt-5";
    req_a.session_id = "session-a";
    req_a.messages.push_back(Message{.role = "user", .content = "First session."});
    session_a->prepare_request(req_a);

    const auto completed_a = session_a->parse_event(
        "event: response.completed\n"
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_session_a\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}");
    REQUIRE(completed_a.done);
    session_a->on_response(HttpResponse{200, "", {}});

    auto session_b = owner.clone();
    ChatRequest req_b;
    req_b.model = "gpt-5";
    req_b.session_id = "session-b";
    req_b.messages.push_back(Message{.role = "user", .content = "Second session."});
    session_b->prepare_request(req_b);

    REQUIRE(req_b.prompt_cache_key == "session-b");
    REQUIRE(req_b.previous_response_id.empty());

    auto session_a_next = owner.clone();
    ChatRequest req_a_next;
    req_a_next.model = "gpt-5";
    req_a_next.session_id = "session-a";
    req_a_next.messages.push_back(Message{.role = "user", .content = "Back to first session."});
    session_a_next->prepare_request(req_a_next);

    REQUIRE(req_a_next.prompt_cache_key == "session-a");
    REQUIRE(req_a_next.previous_response_id == "resp_session_a");
}

TEST_CASE("CodexResponsesProtocol - builds Responses websocket request",
          "[openai][responses][codex][websocket]") {
    CodexResponsesProtocol protocol;

    ChatRequest req;
    req.model = "gpt-5";
    req.session_id = "thread-123";
    req.transport_turn_id = "turn-1";
    req.messages.push_back(Message{.role = "user", .content = "hello"});
    protocol.prepare_request(req);

    cpr::Header headers{{"Content-Type", "application/json"}, {"Accept", "text/event-stream"}};
    protocol.prepare_websocket_headers(headers, req, "https://chatgpt.com/backend-api/codex");

    const auto ws_url = protocol.build_websocket_url(
        "https://chatgpt.com/backend-api/codex", req.model);
    const auto payload = protocol.serialize_websocket_request(req);
    const auto initial_connection_key =
        protocol.websocket_connection_key(ws_url, headers, req);

    REQUIRE(ws_url == "wss://chatgpt.com/backend-api/codex/responses");
    REQUIRE(headers.count("Content-Type") == 0);
    REQUIRE(headers.count("Accept") == 0);
    REQUIRE(headers.at("OpenAI-Beta") == "responses_websockets=2026-02-06");
    REQUIRE(headers.at("x-client-request-id") == "thread-123");
    REQUIRE(headers.at("x-codex-window-id") == "thread-123:0");
    REQUIRE_THAT(payload, Catch::Matchers::StartsWith(R"({"type":"response.create",)"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"gpt-5")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("client_metadata")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("x-codex-installation-id")"));

    protocol.observe_response_headers(
        cpr::Header{{"x-codex-turn-state", "sticky-route"}}, req);
    cpr::Header reconnect_headers{{"Content-Type", "application/json"}, {"Accept", "text/event-stream"}};
    protocol.prepare_websocket_headers(
        reconnect_headers, req, "https://chatgpt.com/backend-api/codex");

    REQUIRE(reconnect_headers.at("x-codex-turn-state") == "sticky-route");
    REQUIRE(protocol.websocket_connection_key(ws_url, reconnect_headers, req)
            == protocol.websocket_connection_key(ws_url, reconnect_headers, req));
    REQUIRE(protocol.websocket_connection_key(ws_url, reconnect_headers, req)
            == initial_connection_key);
}

TEST_CASE("CodexResponsesProtocol - compresses incremental websocket input",
          "[openai][responses][codex][websocket][continuity]") {
    CodexResponsesProtocol protocol;

    ChatRequest first;
    first.model = "gpt-5";
    first.session_id = "thread-123";
    first.transport_turn_id = "turn-1";
    first.messages.push_back(Message{.role = "user", .content = "run tests"});
    protocol.prepare_request(first);

    const auto first_payload = protocol.serialize_websocket_request(first);
    REQUIRE_THAT(first_payload, Catch::Matchers::ContainsSubstring(R"("run tests")"));
    REQUIRE_THAT(first_payload, Catch::Matchers::ContainsSubstring(R"("input":[{"type":"message")"));

    auto tool_call = protocol.parse_event(
        "event: response.output_item.done\n"
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_123\",\"name\":\"exec_command\",\"arguments\":\"{\\\"cmd\\\":\\\"ctest\\\"}\"}}");
    REQUIRE_FALSE(tool_call.done);

    auto completed = protocol.parse_event(
        "event: response.completed\n"
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_ws_1\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}");
    REQUIRE(completed.done);

    ChatRequest second = first;
    Message assistant;
    assistant.role = "assistant";
    assistant.tool_calls.push_back(ToolCall{
        .index = 0,
        .id = "call_123",
        .type = "function",
        .function = ToolCall::Function{
            .name = "exec_command",
            .arguments = R"({"cmd":"ctest"})",
        },
    });
    second.messages.push_back(assistant);
    second.messages.push_back(Message{
        .role = "tool",
        .content = R"({"ok":true})",
        .name = "exec_command",
        .tool_call_id = "call_123",
    });
    protocol.prepare_request(second);

    const auto incremental_payload = protocol.serialize_websocket_request(second);
    REQUIRE_THAT(incremental_payload, Catch::Matchers::ContainsSubstring(R"("previous_response_id":"resp_ws_1")"));
    REQUIRE_THAT(incremental_payload, Catch::Matchers::ContainsSubstring(R"("type":"function_call_output","call_id":"call_123")"));
    REQUIRE_THAT(incremental_payload, !Catch::Matchers::ContainsSubstring(R"("run tests")"));
    REQUIRE_THAT(incremental_payload, !Catch::Matchers::ContainsSubstring(R"("type":"function_call","call_id":"call_123")"));
}

TEST_CASE("CodexResponsesProtocol - prewarm websocket request sends generate false",
          "[openai][responses][codex][websocket][prewarm]") {
    CodexResponsesProtocol protocol;

    ChatRequest req;
    req.model = "gpt-5";
    req.session_id = "thread-123";
    req.transport_turn_id = "turn-1";
    req.messages.push_back(Message{.role = "user", .content = "run tests"});
    protocol.prepare_request(req);

    const auto prewarm_frame = protocol.initial_websocket_request_frame(req);
    const auto& prewarm_payload = prewarm_frame.payload;

    REQUIRE(prewarm_frame.suppress_output);
    REQUIRE_THAT(prewarm_payload, Catch::Matchers::StartsWith(R"({"type":"response.create",)"));
    REQUIRE_THAT(prewarm_payload, Catch::Matchers::ContainsSubstring(R"("generate":false)"));
    REQUIRE_THAT(prewarm_payload, Catch::Matchers::ContainsSubstring(R"("run tests")"));
}

TEST_CASE("CodexResponsesProtocol - real websocket request can reuse prewarm response id",
          "[openai][responses][codex][websocket][prewarm][continuity]") {
    CodexResponsesProtocol protocol;

    ChatRequest req;
    req.model = "gpt-5";
    req.session_id = "thread-123";
    req.transport_turn_id = "turn-1";
    req.messages.push_back(Message{.role = "user", .content = "run tests"});
    protocol.prepare_request(req);

    [[maybe_unused]] const auto prewarm_frame =
        protocol.initial_websocket_request_frame(req);

    auto completed = protocol.parse_event(
        "event: response.completed\n"
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_prewarm_1\",\"usage\":{\"input_tokens\":1,\"output_tokens\":0}}}");
    REQUIRE(completed.done);

    const auto real_payload = protocol.serialize_websocket_request(req);

    REQUIRE_THAT(real_payload, Catch::Matchers::ContainsSubstring(R"("previous_response_id":"resp_prewarm_1")"));
    REQUIRE_THAT(real_payload, Catch::Matchers::ContainsSubstring(R"("input":[])"));
    REQUIRE_THAT(real_payload, !Catch::Matchers::ContainsSubstring(R"("generate":false)"));
    REQUIRE_THAT(real_payload, !Catch::Matchers::ContainsSubstring(R"("run tests")"));
}

TEST_CASE("CodexResponsesProtocol - abandoned websocket request does not poison incremental state",
          "[openai][responses][codex][websocket][fallback]") {
    CodexResponsesProtocol protocol;

    ChatRequest first;
    first.model = "gpt-5";
    first.session_id = "thread-123";
    first.transport_turn_id = "turn-1";
    first.messages.push_back(Message{.role = "user", .content = "run tests"});
    protocol.prepare_request(first);

    [[maybe_unused]] const auto first_payload = protocol.serialize_websocket_request(first);
    protocol.abandon_websocket_request(first);

    auto tool_call = protocol.parse_event(
        "event: response.output_item.done\n"
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_123\",\"name\":\"exec_command\",\"arguments\":\"{\\\"cmd\\\":\\\"ctest\\\"}\"}}");
    REQUIRE_FALSE(tool_call.done);

    auto completed = protocol.parse_event(
        "event: response.completed\n"
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_ws_1\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}");
    REQUIRE(completed.done);

    ChatRequest second = first;
    Message assistant;
    assistant.role = "assistant";
    assistant.tool_calls.push_back(ToolCall{
        .index = 0,
        .id = "call_123",
        .type = "function",
        .function = ToolCall::Function{
            .name = "exec_command",
            .arguments = R"({"cmd":"ctest"})",
        },
    });
    second.messages.push_back(assistant);
    second.messages.push_back(Message{
        .role = "tool",
        .content = R"({"ok":true})",
        .name = "exec_command",
        .tool_call_id = "call_123",
    });
    protocol.prepare_request(second);

    const auto fallback_payload = protocol.serialize_websocket_request(second);
    REQUIRE_THAT(fallback_payload, Catch::Matchers::ContainsSubstring(R"("run tests")"));
    REQUIRE_THAT(fallback_payload, Catch::Matchers::ContainsSubstring(R"("type":"function_call","call_id":"call_123")"));
    REQUIRE_THAT(fallback_payload, Catch::Matchers::ContainsSubstring(R"("type":"function_call_output","call_id":"call_123")"));
    REQUIRE_THAT(fallback_payload, !Catch::Matchers::ContainsSubstring(R"("previous_response_id":"resp_ws_1")"));
}

TEST_CASE("CodexResponsesProtocol - failed websocket terminal event clears pending state",
          "[openai][responses][codex][websocket][fallback]") {
    CodexResponsesProtocol protocol;

    ChatRequest first;
    first.model = "gpt-5";
    first.session_id = "thread-123";
    first.transport_turn_id = "turn-1";
    first.messages.push_back(Message{.role = "user", .content = "run tests"});
    protocol.prepare_request(first);

    [[maybe_unused]] const auto first_payload = protocol.serialize_websocket_request(first);

    const auto failed = protocol.parse_event(
        "event: response.failed\n"
        "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"server failed\"}}}");
    REQUIRE(failed.done);

    auto tool_call = protocol.parse_event(
        "event: response.output_item.done\n"
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_123\",\"name\":\"exec_command\",\"arguments\":\"{\\\"cmd\\\":\\\"ctest\\\"}\"}}");
    REQUIRE_FALSE(tool_call.done);

    auto completed = protocol.parse_event(
        "event: response.completed\n"
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_ws_1\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}");
    REQUIRE(completed.done);

    ChatRequest second = first;
    Message assistant;
    assistant.role = "assistant";
    assistant.tool_calls.push_back(ToolCall{
        .index = 0,
        .id = "call_123",
        .type = "function",
        .function = ToolCall::Function{
            .name = "exec_command",
            .arguments = R"({"cmd":"ctest"})",
        },
    });
    second.messages.push_back(assistant);
    second.messages.push_back(Message{
        .role = "tool",
        .content = R"({"ok":true})",
        .name = "exec_command",
        .tool_call_id = "call_123",
    });
    protocol.prepare_request(second);

    const auto fallback_payload = protocol.serialize_websocket_request(second);
    REQUIRE_THAT(fallback_payload, Catch::Matchers::ContainsSubstring(R"("run tests")"));
    REQUIRE_THAT(fallback_payload, Catch::Matchers::ContainsSubstring(R"("type":"function_call","call_id":"call_123")"));
    REQUIRE_THAT(fallback_payload, Catch::Matchers::ContainsSubstring(R"("type":"function_call_output","call_id":"call_123")"));
    REQUIRE_THAT(fallback_payload, !Catch::Matchers::ContainsSubstring(R"("previous_response_id":"resp_ws_1")"));
}

TEST_CASE("OpenAIResponsesProtocol - failed HTTP response does not advance continuation id",
          "[openai][responses][continuity]") {
    OpenAIResponsesProtocol owner;

    auto stream1 = owner.clone();
    ChatRequest req1;
    req1.model = "gpt-5";
    req1.messages.push_back(Message{
        .role = "user",
        .content = "First turn."
    });
    stream1->prepare_request(req1);

    const auto completed = stream1->parse_event(
        "event: response.completed\n"
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_should_not_persist\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}");
    REQUIRE(completed.done);

    cpr::Header headers;
    stream1->on_response(HttpResponse{500, "internal", headers});

    auto stream2 = owner.clone();
    ChatRequest req2;
    req2.model = "gpt-5";
    req2.messages.push_back(Message{
        .role = "user",
        .content = "Second turn."
    });
    stream2->prepare_request(req2);

    REQUIRE(req2.prompt_cache_key == req1.prompt_cache_key);
    REQUIRE(req2.previous_response_id.empty());
}

TEST_CASE("OpenAIResponsesProtocol - reset_state clears shared continuity",
          "[openai][responses][continuity]") {
    OpenAIResponsesProtocol owner;

    auto stream1 = owner.clone();
    ChatRequest req1;
    req1.model = "gpt-5";
    req1.messages.push_back(Message{
        .role = "user",
        .content = "First turn."
    });

    stream1->prepare_request(req1);
    REQUIRE_FALSE(req1.prompt_cache_key.empty());
    REQUIRE(req1.previous_response_id.empty());

    const auto completed = stream1->parse_event(
        "event: response.completed\n"
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_reset_1\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}");
    REQUIRE(completed.done);

    cpr::Header headers;
    stream1->on_response(HttpResponse{200, "", headers});

    owner.reset_state();

    auto stream2 = owner.clone();
    ChatRequest req2;
    req2.model = "gpt-5";
    req2.messages.push_back(Message{
        .role = "user",
        .content = "Second turn."
    });

    stream2->prepare_request(req2);
    REQUIRE_FALSE(req2.prompt_cache_key.empty());
    REQUIRE(req2.prompt_cache_key != req1.prompt_cache_key);
    REQUIRE(req2.previous_response_id.empty());
}

TEST_CASE("OpenAIResponsesProtocol - serializer maps assistant tool calls and tool outputs",
          "[openai][responses][serializer][tools]") {
    OpenAIResponsesProtocol protocol;

    ChatRequest req;
    req.model = "gpt-5";
    req.messages.push_back(Message{
        .role = "user",
        .content = "run test"
    });

    Message asst;
    asst.role = "assistant";
    asst.content = "";
    asst.tool_calls.push_back(ToolCall{
        .index = 0,
        .id = "call_123",
        .type = "function",
        .function = ToolCall::Function{
            .name = "exec_command",
            .arguments = R"({"cmd":"ctest"})"
        },
    });
    req.messages.push_back(asst);

    req.messages.push_back(Message{
        .role = "tool",
        .content = R"({"ok":true})",
        .name = "exec_command",
        .tool_call_id = "call_123",
    });

    const std::string payload = protocol.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"function_call","call_id":"call_123")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("name":"exec_command")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"function_call_output","call_id":"call_123")"));
}

TEST_CASE("OpenAIResponsesProtocol - parse output_text.delta event",
          "[openai][responses][sse]") {
    OpenAIResponsesProtocol protocol;

    auto result = protocol.parse_event(
        "event: response.output_text.delta\n"
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"Hello\"}");

    REQUIRE_FALSE(result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "Hello");
    REQUIRE_FALSE(result.chunks[0].is_final);
}

TEST_CASE("OpenAIResponsesProtocol - accepts no-space SSE separators",
          "[openai][responses][sse]") {
    OpenAIResponsesProtocol protocol;

    auto result = protocol.parse_event(
        "event:response.output_text.delta\n"
        "data:{\"type\":\"response.output_text.delta\",\"delta\":\"NoSpace\"}");

    REQUIRE_FALSE(result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "NoSpace");
}

TEST_CASE("OpenAIResponsesProtocol - joins multiline data payloads",
          "[openai][responses][sse]") {
    OpenAIResponsesProtocol protocol;

    auto result = protocol.parse_event(
        "event: response.output_text.delta\n"
        "data: {\"type\":\"response.output_text.delta\",\n"
        "data: \"delta\":\"Multi\"}");

    REQUIRE_FALSE(result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "Multi");
}

TEST_CASE("OpenAIResponsesProtocol - parse function_call item event",
          "[openai][responses][sse][tools]") {
    OpenAIResponsesProtocol protocol;

    auto result = protocol.parse_event(
        "event: response.output_item.done\n"
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_abc\",\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}");

    REQUIRE_FALSE(result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].tools.size() == 1);
    REQUIRE(result.chunks[0].tools[0].id == "call_abc");
    REQUIRE(result.chunks[0].tools[0].function.name == "read_file");
}

TEST_CASE("OpenAIResponsesProtocol preserves and replays encrypted reasoning items",
          "[openai][responses][continuation]") {
    OpenAIResponsesProtocol protocol(true);
    auto parsed = protocol.parse_event(
        "event: response.output_item.done\n"
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"id\":\"rs_1\",\"type\":\"reasoning\",\"encrypted_content\":\"opaque-token\",\"summary\":[]}}");

    REQUIRE(parsed.chunks.size() == 1);
    REQUIRE(parsed.chunks[0].continuation_items.size() == 1);
    const auto& item = parsed.chunks[0].continuation_items[0];
    CHECK(item.provider == "openai");
    CHECK(item.kind == "reasoning");
    CHECK_THAT(item.payload, Catch::Matchers::ContainsSubstring("opaque-token"));

    ChatRequest request;
    request.model = "gpt-5.6";
    request.messages = {
        {.role = "assistant", .continuation_items = {item}},
        {.role = "user", .content = "Continue"},
    };
    const std::string payload = protocol.serialize(request);
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(R"("encrypted_content":"opaque-token")"));
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(R"("include":["reasoning.encrypted_content"])"));
}

TEST_CASE("OpenAIResponsesProtocol - parse assistant message from output_item.done without deltas",
          "[openai][responses][sse]") {
    OpenAIResponsesProtocol protocol;

    auto result = protocol.parse_event(
        "event: response.output_item.done\n"
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello from done\"}]}}");

    REQUIRE_FALSE(result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "Hello from done");
}

TEST_CASE("OpenAIResponsesProtocol - parse completed usage and done",
          "[openai][responses][sse][usage]") {
    OpenAIResponsesProtocol protocol;

    auto created = protocol.parse_event(
        "event: response.created\n"
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\"}}");
    REQUIRE(created.chunks.empty());

    auto result = protocol.parse_event(
        "event: response.completed\n"
        "data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"usage\":{\"input_tokens\":12,\"output_tokens\":5}}}");

    REQUIRE(result.done);
    REQUIRE(result.prompt_tokens == 12);
    REQUIRE(result.completion_tokens == 5);
    REQUIRE(protocol.last_response_id() == "resp_1");
}

TEST_CASE("OpenAIResponsesProtocol - completed falls back to created response id",
          "[openai][responses][sse][usage]") {
    OpenAIResponsesProtocol protocol;

    const auto created = protocol.parse_event(
        "event: response.created\n"
        "data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_created\"}}");
    REQUIRE_FALSE(created.done);

    auto result = protocol.parse_event(
        "event: response.completed\n"
        "data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}");

    REQUIRE(result.done);
    REQUIRE(protocol.last_response_id() == "resp_created");
}

TEST_CASE("OpenAIResponsesProtocol - parse output_text.done event",
          "[openai][responses][sse]") {
    OpenAIResponsesProtocol protocol;

    auto result = protocol.parse_event(
        "event: response.output_text.done\n"
        "data: {\"type\":\"response.output_text.done\",\"text\":\" world\"}");

    REQUIRE_FALSE(result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == " world");
}

TEST_CASE("OpenAIResponsesProtocol - parse failed event as terminal error chunk",
          "[openai][responses][sse][errors]") {
    OpenAIResponsesProtocol protocol;

    auto result = protocol.parse_event(
        "event: response.failed\n"
        "data: {\"type\":\"response.failed\",\"response\":{\"error\":{\"message\":\"Rate limit hit\"}}}");

    REQUIRE(result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].is_error);
    REQUIRE(result.chunks[0].is_final);
    REQUIRE_THAT(result.chunks[0].content, Catch::Matchers::ContainsSubstring("Rate limit hit"));
}

TEST_CASE("OpenAIResponsesProtocol - parse incomplete event as terminal error chunk",
          "[openai][responses][sse][errors]") {
    OpenAIResponsesProtocol protocol;

    auto result = protocol.parse_event(
        "event: response.incomplete\n"
        "data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"}}}");

    REQUIRE(result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].is_error);
    REQUIRE(result.chunks[0].is_final);
    REQUIRE_THAT(result.chunks[0].content, Catch::Matchers::ContainsSubstring("max_output_tokens"));
}
