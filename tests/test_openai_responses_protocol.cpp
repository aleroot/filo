#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <httplib.h>

#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"
#include "core/llm/Models.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <future>
#include <fstream>
#include <format>
#include <thread>

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

TEST_CASE("HttpLLMProvider - Codex Responses transport headers are scoped to Codex backend",
          "[openai][responses][codex][transport]") {
    httplib::Server server;

    std::atomic<int> codex_requests{0};
    std::string first_installation_id;
    std::string first_client_request_id;
    std::string first_window_id;
    std::string first_timing_header;
    std::string first_turn_state;
    std::string second_turn_state;
    std::string regular_installation_id;

    auto completed_sse = [](std::string_view id) {
        return std::format(
            "event: response.completed\n"
            "data: {{\"type\":\"response.completed\",\"response\":{{\"id\":\"{}\",\"usage\":{{\"input_tokens\":1,\"output_tokens\":1}}}}}}\n\n",
            id);
    };

    server.Post("/backend-api/codex/responses", [&](const httplib::Request& req,
                                                    httplib::Response& res) {
        const int n = ++codex_requests;
        if (n == 1) {
            first_installation_id = req.get_header_value("x-codex-installation-id");
            first_client_request_id = req.get_header_value("x-client-request-id");
            first_window_id = req.get_header_value("x-codex-window-id");
            first_timing_header = req.get_header_value("x-responsesapi-include-timing-metrics");
            first_turn_state = req.get_header_value("x-codex-turn-state");
            res.set_header("x-codex-turn-state", "turn-state-one");
            res.set_content(completed_sse("resp_codex_1"), "text/event-stream");
            return;
        }

        second_turn_state = req.get_header_value("x-codex-turn-state");
        res.set_header("x-codex-turn-state", "turn-state-two");
        res.set_content(completed_sse("resp_codex_2"), "text/event-stream");
    });

    server.Post("/v1/responses", [&](const httplib::Request& req, httplib::Response& res) {
        regular_installation_id = req.get_header_value("x-codex-installation-id");
        res.set_content(completed_sse("resp_regular_1"), "text/event-stream");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() {
        server.listen_after_bind();
    });
    for (int i = 0; i < 50 && !server.is_running(); ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    REQUIRE(server.is_running());

    auto codex_provider = std::make_shared<HttpLLMProvider>(
        std::format("http://127.0.0.1:{}/backend-api/codex", port),
        core::auth::ApiKeyCredentialSource::as_bearer("test-key"),
        "gpt-5",
        std::make_unique<CodexResponsesProtocol>(),
        core::config::ApiType::OpenAI,
        "openai");

    ChatRequest req;
    req.model = "gpt-5";
    req.session_id = "session-transport";
    req.transport_turn_id = "session-transport:1";
    req.messages.push_back(Message{.role = "user", .content = "first"});

    auto stream_once = [](const std::shared_ptr<HttpLLMProvider>& provider, const ChatRequest& request) {
        auto done = std::make_shared<std::promise<void>>();
        auto completed = std::make_shared<std::atomic<bool>>(false);
        auto future = done->get_future();

        provider->stream_response(request, [done, completed](const StreamChunk& chunk) {
            if (chunk.is_final && !completed->exchange(true)) {
                done->set_value();
            }
        });

        REQUIRE(future.wait_for(std::chrono::seconds(3)) == std::future_status::ready);
    };

    stream_once(codex_provider, req);

    ChatRequest req2 = req;
    req2.messages[0].content = "second";
    stream_once(codex_provider, req2);

    CHECK_FALSE(first_installation_id.empty());
    CHECK(first_client_request_id == "session-transport");
    CHECK(first_window_id == "session-transport:0");
    CHECK(first_timing_header == "true");
    CHECK(first_turn_state.empty());
    CHECK(second_turn_state == "turn-state-one");

    auto regular_provider = std::make_shared<HttpLLMProvider>(
        std::format("http://127.0.0.1:{}/v1", port),
        core::auth::ApiKeyCredentialSource::as_bearer("test-key"),
        "gpt-5",
        std::make_unique<OpenAIResponsesProtocol>(),
        core::config::ApiType::OpenAI,
        "openai-compatible");

    ChatRequest regular_req = req;
    regular_req.transport_turn_id = "session-transport:regular";
    stream_once(regular_provider, regular_req);
    CHECK(regular_installation_id.empty());

    server.stop();
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
