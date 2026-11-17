#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"
#include "core/llm/CodexUsageClient.hpp"
#include "core/llm/Models.hpp"
#include "core/utils/ClientPlatform.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <simdjson.h>

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

TEST_CASE("OpenAIResponsesProtocol omits reasoning.summary on the public API",
          "[openai][responses][serializer][reasoning]") {
    OpenAIResponsesProtocol protocol;
    ChatRequest req;
    req.model = "gpt-5.6-sol";
    req.messages.push_back(Message{.role = "user", .content = "Hello."});

    const std::string payload = protocol.serialize(req);
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("reasoning")"));
}

TEST_CASE("CodexResponsesProtocol honors catalog reasoning summaries",
          "[openai][responses][codex][serializer][reasoning]") {
    CodexResponsesProtocol protocol;
    ChatRequest req;
    req.model = "gpt-5.6-sol";
    req.messages.push_back(Message{.role = "user", .content = "Hello."});

    const std::string payload = protocol.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("reasoning":{"summary":"auto"})"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("effort")"));
    req.catalog_reasoning.emplace();
    req.catalog_reasoning->summary_supported = false;
    CHECK_THAT(protocol.serialize(req), !Catch::Matchers::ContainsSubstring(R"("summary")"));
    req.catalog_reasoning->summary_supported = true;
    req.catalog_reasoning->default_summary = "detailed";
    CHECK_THAT(protocol.serialize(req), Catch::Matchers::ContainsSubstring(R"("summary":"detailed")"));
    req.catalog_reasoning->default_summary = "none";
    CHECK_THAT(protocol.serialize(req), !Catch::Matchers::ContainsSubstring(R"("summary")"));
}

TEST_CASE("Codex Responses Lite carries tools in input and advertises its wire mode",
          "[openai][responses][codex][serializer]") {
    CodexResponsesProtocol protocol;
    ChatRequest req;
    req.model = "test-model";
    req.catalog_wire_protocol = "responses_lite";
    req.messages.push_back(Message{.role = "user", .content = "Hello"});
    req.tools.push_back(Tool{.function = {.name = "lookup", .description = "Look up data"}});
    const auto payload = protocol.serialize(req);
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(simdjson::padded_string(payload)).get(doc) == simdjson::SUCCESS);
    simdjson::dom::array input;
    REQUIRE(doc["input"].get(input) == simdjson::SUCCESS);
    REQUIRE(input.size() == 2);
    auto first = *input.begin();
    CHECK(std::string_view(first["type"]) == "additional_tools");
    CHECK(std::string_view(first["tools"].at(0)["name"]) == "functions");
    CHECK(std::string_view(first["tools"].at(0)["tools"].at(0)["name"]) == "lookup");
    CHECK(doc["tools"].error() == simdjson::NO_SUCH_FIELD);
    CHECK(std::string_view(doc["reasoning"]["context"]) == "all_turns");
    cpr::Header headers;
    protocol.prepare_headers(headers, req, "https://chatgpt.com/backend-api/codex");
    CHECK(headers.at("x-openai-internal-codex-responses-lite") == "true");
    req.transport_turn_id = "turn";
    const auto ws_payload = protocol.serialize_websocket_request(req);
    CHECK_THAT(ws_payload, Catch::Matchers::ContainsSubstring(
        R"("ws_request_header_x_openai_internal_codex_responses_lite":"true")"));
    simdjson::dom::element ws_doc;
    REQUIRE(parser.parse(simdjson::padded_string(ws_payload)).get(ws_doc) == simdjson::SUCCESS);
    ToolCall call;
    call.id = "call_1";
    call.function.name = "lookup";
    call.function.arguments = "{}";
    req.messages.push_back(Message{.role = "assistant", .tool_calls = {call}});
    req.messages.push_back(Message{.role = "tool", .content = "result", .tool_call_id = "call_1"});
    const auto replay = protocol.serialize(req);
    REQUIRE_THAT(replay, Catch::Matchers::ContainsSubstring(
        R"("name":"lookup","namespace":"functions","arguments":"{}")"));
}

TEST_CASE("CodexResponsesProtocol matches Codex stream idle and retry budgets",
          "[openai][responses][codex][timeout][retry]") {
    using namespace std::chrono_literals;
    const auto generic = OpenAIResponsesProtocol{}.stream_timeouts();
    CHECK(generic.response_start == 120s);
    CHECK(generic.inactivity == 240s);
    CHECK(OpenAIResponsesProtocol{}.stream_retry_policy().retry_only_before_output);

    const auto timeouts = CodexResponsesProtocol{}.stream_timeouts();
    CHECK(timeouts.response_start == 300s);
    CHECK(timeouts.inactivity == 300s);

    const auto retries = CodexResponsesProtocol{}.stream_retry_policy();
    CHECK(retries.max_retries == 5);
    CHECK_FALSE(retries.retry_only_before_output);
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

TEST_CASE("OpenAIResponsesProtocol - clamps private Codex effort tiers",
          "[openai][responses][serializer][effort]") {
    OpenAIResponsesProtocol protocol;
    ChatRequest req;
    req.model = "gpt-5.6-sol";
    req.effort = "ultra";
    req.messages.push_back(Message{.role = "user", .content = "Solve this."});
    CHECK_THAT(protocol.serialize(req),
               Catch::Matchers::ContainsSubstring(R"("reasoning":{"effort":"max"})"));

    req.model = "gpt-5.6-luna";
    CHECK_THAT(protocol.serialize(req),
               Catch::Matchers::ContainsSubstring(R"("reasoning":{"effort":"max"})"));

    CodexResponsesProtocol codex;
    req.model = "gpt-5.6-sol";
    CHECK_THAT(codex.serialize(req),
               Catch::Matchers::ContainsSubstring(
                   R"("reasoning":{"effort":"max","summary":"auto"})"));

    req.model = "gpt-6.1-terra";
    CHECK_THAT(codex.serialize(req),
               Catch::Matchers::ContainsSubstring(
                   R"("reasoning":{"effort":"max","summary":"auto"})"));
    req.effort = "off";
    CHECK_THAT(codex.serialize(req),
               Catch::Matchers::ContainsSubstring(R"("reasoning":{"summary":"auto"})"));
}

TEST_CASE("OpenAIResponsesProtocol - GPT-6.2 drops rejected reasoning efforts",
          "[openai][responses][serializer][effort]") {
    OpenAIResponsesProtocol protocol;
    ChatRequest req;
    req.model = "gpt-6.2-sol";
    req.messages.push_back(Message{.role = "user", .content = "Solve this."});

    req.effort = "none";
    CHECK_THAT(protocol.serialize(req),
               !Catch::Matchers::ContainsSubstring(R"("effort")"));
    req.effort = "off";
    CHECK_THAT(protocol.serialize(req),
               !Catch::Matchers::ContainsSubstring(R"("effort")"));
    req.effort = "minimal";
    CHECK_THAT(protocol.serialize(req),
               Catch::Matchers::ContainsSubstring(R"("reasoning":{"effort":"low"})"));
    req.effort = "max";
    CHECK_THAT(protocol.serialize(req),
               Catch::Matchers::ContainsSubstring(R"("reasoning":{"effort":"max"})"));
    req.effort = "ultra";
    const std::string ultra = protocol.serialize(req);
    CHECK_THAT(ultra, Catch::Matchers::ContainsSubstring(R"("reasoning":{"effort":"max"})"));
    CHECK_THAT(ultra, !Catch::Matchers::ContainsSubstring("ultra"));

    req.model = "gpt-6.2-astra";
    req.effort = "none";
    CHECK_THAT(protocol.serialize(req),
               !Catch::Matchers::ContainsSubstring(R"("effort")"));
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
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("detail")"));

    // codex-rs DEFAULT_IMAGE_DETAIL is `high`; Responses Lite clears it
    // (client_common.rs normalize_image_detail).
    CodexResponsesProtocol codex;
    REQUIRE_THAT(codex.serialize(req), Catch::Matchers::ContainsSubstring(R"(","detail":"high"})"));
    req.catalog_wire_protocol = "responses_lite";
    REQUIRE_THAT(codex.serialize(req), Catch::Matchers::ContainsSubstring(R"("type":"input_image")")
                                           && !Catch::Matchers::ContainsSubstring(R"("detail")"));
}

TEST_CASE("OpenAIResponsesProtocol - serializer includes continuation/cache/tier fields",
          "[openai][responses][serializer][continuation]") {
    OpenAIResponsesProtocol protocol(false, "priority");

    ChatRequest req;
    req.model = "gpt-5";
    req.messages.push_back(Message{
        .role = "system",
        .content = "Keep OpenAI instructions."
    });
    req.messages.push_back(Message{
        .role = "user",
        .content = "Continue the previous turn."
    });
    req.previous_response_id = "resp_prev_1";
    req.prompt_cache_key = "filo-session-1";

    const std::string payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("previous_response_id":"resp_prev_1")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("instructions":"Keep OpenAI instructions.")"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(
        R"("role":"system")"));
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
    REQUIRE(headers.at("originator") == "codex_cli_rs");
    REQUIRE(headers.at("session_id") == "thread-123");
    REQUIRE(headers.at("session-id") == "thread-123");
    REQUIRE(headers.at("thread-id") == "thread-123");
    REQUIRE(headers.at("version") == "0.169.0");
    // codex-rs get_codex_user_agent: the originator and version, then the
    // platform the client runs on.
    const auto& agent = headers.at("User-Agent");
    REQUIRE_THAT(agent, Catch::Matchers::StartsWith("codex_cli_rs/0.169.0 ("));
    REQUIRE_THAT(agent, Catch::Matchers::ContainsSubstring(
                            core::utils::client_platform().architecture));
    REQUIRE_THAT(agent, Catch::Matchers::ContainsSubstring(
                            core::utils::client_platform().os));
    REQUIRE_THAT(payload, Catch::Matchers::StartsWith(R"({"type":"response.create",)"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"gpt-5")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("client_metadata")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("x-codex-installation-id")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("session_id":"thread-123")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("thread_id":"thread-123")"));

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

TEST_CASE("CodexResponsesProtocol - HTTP replay includes system context and encrypted reasoning",
          "[openai][responses][codex][continuity]") {
    CodexResponsesProtocol protocol;
    ChatRequest req;
    req.model = "gpt-5.6-sol";
    req.previous_response_id = "stale-http-response";
    req.messages = {
        {.role = "system", .content = "Preserve this policy."},
        {.role = "user", .content = "Continue safely."},
    };

    const std::string payload = protocol.serialize(req);
    CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("instructions")"));
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("role":"developer","content":[{"type":"input_text","text":"Preserve this policy."}])"));
    CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("role":"system")"));
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("include":["reasoning.encrypted_content"])"));
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(R"("store":false)"));
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("reasoning":{"summary":"auto"})"));
    CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring("previous_response_id"));
}

TEST_CASE("CodexResponsesProtocol - parses subscription quota headers",
          "[openai][responses][codex][rate-limit]") {
    CodexResponsesProtocol protocol;
    cpr::Header headers{
        {"X-Codex-Primary-Used-Percent", "42.5"},
        {"x-codex-primary-window-minutes", "300"},
        {"x-codex-primary-reset-at", "2000000100"},
        {"x-codex-secondary-used-percent", "7"},
        {"x-codex-secondary-window-minutes", "10080"},
        {"x-codex-other-primary-used-percent", "12"},
        {"x-codex-other-primary-window-minutes", "60"},
        {"x-codex-credits-has-credits", "true"},
        {"x-codex-credits-unlimited", "false"},
        {"x-codex-credits-balance", "25.50"},
        {"x-codex-active-limit", "codex_other"},
        {"x-codex-promo-message", "Extra capacity is available"},
        {"x-codex-rate-limit-reached-type", "rate_limit_reached"},
    };

    protocol.on_response(HttpResponse{200, "", headers});
    const auto info = protocol.last_rate_limit();
    REQUIRE(info.usage_windows.size() == 3);
    CHECK(info.usage_windows[0].label == "5h");
    CHECK(info.usage_windows[0].utilization == Catch::Approx(0.425f));
    CHECK(info.usage_windows[0].resets_at == 2'000'000'100);
    CHECK(info.usage_windows[1].label == "7d");
    CHECK(info.usage_windows[2].label == "codex_other 1h");
    REQUIRE(info.subscription.supplemental_credits.has_value());
    CHECK(info.subscription.supplemental_credits->available);
    CHECK_FALSE(info.subscription.supplemental_credits->unlimited);
    CHECK(info.subscription.supplemental_credits->balance == "25.50");
    CHECK(info.subscription.notice == "Extra capacity is available");
    CHECK(info.subscription.limit_reached_reason == "rate_limit_reached");

    cpr::Header clearing_headers{
        {"x-codex-promo-message", ""},
        {"x-codex-rate-limit-reached-type", ""},
    };
    protocol.on_response(HttpResponse{200, "", clearing_headers});
    const auto cleared = protocol.last_rate_limit();
    CHECK(cleared.subscription.notice.empty());
    CHECK(cleared.subscription.limit_reached_reason.empty());
    REQUIRE(cleared.usage_windows.size() == 3);
}

TEST_CASE("CodexResponsesProtocol - merges sparse websocket quota updates",
          "[openai][responses][codex][rate-limit][websocket]") {
    CodexResponsesProtocol protocol;
    const auto parsed = protocol.parse_event(
        "event: codex.rate_limits\n"
        "data: {\"type\":\"codex.rate_limits\",\"plan_type\":\"pro\","
        "\"rate_limits\":{\"primary\":{\"used_percent\":18.5,\"window_minutes\":300,\"reset_at\":2000000200}},"
        "\"credits\":{\"has_credits\":true,\"unlimited\":false,\"balance\":\"12.00\"}}"
    );
    CHECK_FALSE(parsed.done);
    auto info = protocol.last_rate_limit();
    REQUIRE(info.usage_windows.size() == 1);
    CHECK(info.usage_windows[0].label == "5h");
    CHECK(info.subscription.tier_label == "ChatGPT pro");
    REQUIRE(info.subscription.supplemental_credits.has_value());
    CHECK(info.subscription.supplemental_credits->balance == "12.00");

    (void)protocol.parse_event(
        "event: codex.rate_limits\n"
        "data: {\"type\":\"codex.rate_limits\",\"metered_limit_name\":\"codex_other\","
        "\"rate_limits\":{\"primary\":{\"used_percent\":8,\"window_minutes\":60}}}"
    );
    info = protocol.last_rate_limit();
    REQUIRE(info.usage_windows.size() == 2);
    CHECK(info.usage_windows[0].label == "5h");
    CHECK(info.usage_windows[1].label == "codex_other 1h");
    CHECK(info.subscription.tier_label == "ChatGPT pro");
    REQUIRE(info.subscription.supplemental_credits.has_value());
    CHECK(info.subscription.supplemental_credits->balance == "12.00");

    // The HTTP completion often has no Codex quota headers. It must not erase
    // the event-delivered subscription snapshot.
    protocol.on_response(HttpResponse{200, "", {}});
    info = protocol.last_rate_limit();
    REQUIRE(info.usage_windows.size() == 2);
    CHECK(info.subscription.tier_label == "ChatGPT pro");
    REQUIRE(info.subscription.supplemental_credits.has_value());
    CHECK(info.subscription.supplemental_credits->balance == "12.00");
}

namespace {
// The shape `GET /wham/usage` returned live for a business workspace metered
// by credits (2026-10-28): no rate-limit windows, only a monthly spend limit.
constexpr std::string_view kCreditWorkspaceUsage = R"({
  "plan_type": "business",
  "rate_limit": null,
  "code_review_rate_limit": null,
  "additional_rate_limits": null,
  "credits": {"has_credits": true, "unlimited": false, "overage_limit_reached": false,
              "balance": null},
  "spend_control": {"reached": false, "individual_limit": {
      "source": "workspace_spend_controls", "unit": "credit", "limit": "2000",
      "used": "479.59", "remaining": "1520.41", "used_percent": 24,
      "remaining_percent": 76, "reset_after_seconds": 305719, "reset_at": 1793491200}},
  "rate_limit_reached_type": null,
  "promo": null
})";
} // namespace

TEST_CASE("Codex usage decodes a credit-metered workspace's monthly limit",
          "[openai][codex][usage]") {
    const auto info = codex::parse_usage(kCreditWorkspaceUsage);
    REQUIRE(info.has_value());
    REQUIRE(info->usage_windows.size() == 1);
    CHECK(info->usage_windows[0].label == "30d");
    CHECK(info->usage_windows[0].utilization == Catch::Approx(0.24f));
    CHECK(info->usage_windows[0].resets_at == 1'793'491'200);
    CHECK_FALSE(info->is_rate_limited);
    CHECK(info->subscription.tier_label == "ChatGPT business");
    REQUIRE(info->subscription.supplemental_credits.has_value());
    CHECK(info->subscription.supplemental_credits->available);
    CHECK(info->subscription.supplemental_credits->balance.empty());
}

TEST_CASE("Codex usage decodes plan windows, feature limits and limit state",
          "[openai][codex][usage]") {
    const auto info = codex::parse_usage(R"({
      "plan_type": "pro",
      "rate_limit": {"allowed": false, "limit_reached": true,
        "primary_window": {"used_percent": 100, "limit_window_seconds": 18000,
                           "reset_after_seconds": 60, "reset_at": 2000000000},
        "secondary_window": {"used_percent": 40, "limit_window_seconds": 604800,
                             "reset_after_seconds": 600, "reset_at": 2000000600}},
      "additional_rate_limits": [{"limit_name": "GPT-5.3-Codex-Spark",
        "metered_feature": "codex_bengalfox",
        "rate_limit": {"allowed": true, "limit_reached": false,
          "primary_window": {"used_percent": 5, "limit_window_seconds": 18000,
                             "reset_after_seconds": 1, "reset_at": 2000000001}}}],
      "spend_control": {"reached": false, "individual_limit": null},
      "rate_limit_reached_type": {"type": "rate_limit_reached"}
    })");
    REQUIRE(info.has_value());
    REQUIRE(info->usage_windows.size() == 3);
    CHECK(info->usage_windows[0].label == "5h");
    CHECK(info->usage_windows[0].utilization == Catch::Approx(1.0f));
    CHECK(info->usage_windows[1].label == "7d");
    CHECK(info->usage_windows[1].resets_at == 2'000'000'600);
    CHECK(info->usage_windows[2].label == "codex_bengalfox 5h");
    CHECK(info->is_rate_limited);
    CHECK(info->subscription.limit_reached_reason == "rate_limit_reached");

    CHECK_FALSE(codex::parse_usage("not json").has_value());
}

TEST_CASE("Codex usage is read only next to a ChatGPT backend",
          "[openai][codex][usage]") {
    CHECK(codex::usage_url("https://chatgpt.com/backend-api/codex/")
          == "https://chatgpt.com/backend-api/wham/usage");
    CHECK(codex::usage_url("https://chat.openai.com/backend-api/codex")
          == "https://chat.openai.com/backend-api/wham/usage");
    CHECK_FALSE(codex::usage_url("https://api.openai.com/v1").has_value());
    CHECK_FALSE(codex::usage_url("https://proxy.example/backend-api/codex").has_value());
}

TEST_CASE("CodexResponsesProtocol adds the account usage to streamed quota",
          "[openai][responses][codex][rate-limit][usage]") {
    int reads = 0;
    std::string read_url;
    cpr::Header read_headers;
    auto usage = codex::make_usage_source([&](std::string_view url, const cpr::Header& headers) {
        ++reads;
        read_url = std::string(url);
        read_headers = headers;
        return std::optional<std::string>(kCreditWorkspaceUsage);
    });
    CodexResponsesProtocol protocol(true, {}, {}, usage);
    const cpr::Header request_headers{
        {"Authorization", "Bearer token"},
        {"chatgpt-account-id", "account"},
        {"Content-Type", "application/json"},
    };

    protocol.on_response(HttpResponse{200, "", {}});
    protocol.enrich_rate_limit("https://chatgpt.com/backend-api/codex", request_headers,
                               HttpResponse{200, "", {}});
    auto info = protocol.last_rate_limit();
    CHECK(reads == 1);
    CHECK(read_url == "https://chatgpt.com/backend-api/wham/usage");
    CHECK(read_headers.at("Authorization") == "Bearer token");
    CHECK(read_headers.at("chatgpt-account-id") == "account");
    CHECK_FALSE(read_headers.contains("Content-Type"));
    REQUIRE(info.usage_windows.size() == 1);
    CHECK(info.usage_windows[0].label == "30d");
    CHECK(info.subscription.tier_label == "ChatGPT business");

    // A streamed update cannot erase the reading; its own windows come first,
    // and a clone shares the state.
    auto clone = protocol.clone();
    (void)clone->parse_event(
        "event: codex.rate_limits\n"
        "data: {\"type\":\"codex.rate_limits\",\"plan_type\":\"business\",\"rate_limits\":"
        "{\"primary\":{\"used_percent\":10,\"window_minutes\":300}}}");
    info = protocol.last_rate_limit();
    REQUIRE(info.usage_windows.size() == 2);
    CHECK(info.usage_windows[0].label == "5h");
    CHECK(info.usage_windows[1].label == "30d");

    // Readings are cached per account.
    clone->enrich_rate_limit("https://chatgpt.com/backend-api/codex", request_headers,
                             HttpResponse{200, "", {}});
    CHECK(reads == 1);
    // A transport failure skips the read.
    protocol.enrich_rate_limit("https://chatgpt.com/backend-api/codex",
                               {{"Authorization", "Bearer other"}}, HttpResponse{0, "", {}});
    CHECK(reads == 1);
}

TEST_CASE("OpenAIResponsesProtocol ignores private Codex quota headers",
          "[openai][responses][rate-limit][boundary]") {
    OpenAIResponsesProtocol protocol;
    cpr::Header headers{
        {"x-codex-primary-used-percent", "50"},
        {"x-codex-promo-message", "private"},
    };

    protocol.on_response(HttpResponse{200, "", headers});
    const auto info = protocol.last_rate_limit();
    CHECK(info.usage_windows.empty());
    CHECK_FALSE(info.subscription.has_data());
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

TEST_CASE("Codex continues a conversation across turns on one websocket",
          "[openai][responses][codex][websocket][continuity]") {
    // codex-rs core/src/client.rs keeps the websocket session (connection,
    // last request, last response) for the whole thread: a new turn sends
    // only its new items with previous_response_id.
    CodexResponsesProtocol protocol;
    const std::string url = "wss://chatgpt.com/backend-api/codex/responses";
    const cpr::Header headers{{"Authorization", "Bearer token"}};

    ChatRequest first;
    first.model = "gpt-6.1-sol";
    first.session_id = "thread-123";
    first.continuation_scope = "thread-123";
    first.transport_turn_id = "thread-123:0";
    first.messages.push_back(Message{.role = "user", .content = "first question"});
    protocol.prepare_request(first);
    const auto first_key = protocol.websocket_connection_key(url, headers, first);

    REQUIRE(protocol.initial_websocket_request_frame(first).suppress_output);
    REQUIRE(protocol.parse_event(
        R"({"type":"response.completed","response":{"id":"resp_prewarm"}})").done);
    const auto real = protocol.next_websocket_request_frame(
        first, WebSocketRequestFrame{.payload = "x", .suppress_output = true});
    REQUIRE_THAT(real.payload, Catch::Matchers::ContainsSubstring(R"("previous_response_id":"resp_prewarm")"));
    const auto answer = protocol.parse_event(
        R"({"type":"response.output_item.done","item":{"type":"message","role":"assistant","phase":"final_answer","content":[{"type":"output_text","text":"first answer"}]}})");
    REQUIRE(protocol.parse_event(
        R"({"type":"response.completed","response":{"id":"resp_turn_1"}})").done);

    Message assistant{.role = "assistant", .content = "first answer"};
    for (const auto& chunk : answer.chunks) {
        assistant.continuation_items.insert(assistant.continuation_items.end(),
            chunk.continuation_items.begin(), chunk.continuation_items.end());
    }
    REQUIRE_FALSE(assistant.continuation_items.empty());

    ChatRequest second = first;
    second.transport_turn_id = "thread-123:1";
    second.messages.push_back(assistant);
    second.messages.push_back(Message{.role = "user", .content = "second question"});
    protocol.prepare_request(second);

    // Same socket, no second prewarm, only the new user message.
    CHECK(protocol.websocket_connection_key(url, headers, second) == first_key);
    const auto next_turn = protocol.initial_websocket_request_frame(second);
    CHECK_FALSE(next_turn.suppress_output);
    CHECK_THAT(next_turn.payload, Catch::Matchers::ContainsSubstring(R"("previous_response_id":"resp_turn_1")"));
    CHECK_THAT(next_turn.payload, Catch::Matchers::ContainsSubstring("second question"));
    CHECK_THAT(next_turn.payload, !Catch::Matchers::ContainsSubstring("first question"));
    CHECK_THAT(next_turn.payload, !Catch::Matchers::ContainsSubstring("first answer"));

    // A request sent beside the conversation (keep-warm replay, compaction)
    // has no continuation scope: its own socket and no inherited baseline.
    ChatRequest beside = second;
    beside.continuation_scope.clear();
    CHECK(protocol.websocket_connection_key(url, headers, beside) != first_key);
    const auto beside_frame = protocol.serialize_websocket_request(beside);
    CHECK_THAT(beside_frame, Catch::Matchers::ContainsSubstring("first question"));
    CHECK_THAT(beside_frame, !Catch::Matchers::ContainsSubstring("previous_response_id"));
}

TEST_CASE("Codex continues a conversation after a tool call",
          "[openai][responses][codex][websocket][continuity]") {
    // codex-rs core/src/client.rs get_incremental_items: the turn that
    // carries a tool result continues the previous response, so only the
    // result is new. The endpoint's copy of the call and the conversation's
    // copy differ by the tool namespace, which the comparison ignores.
    CodexResponsesProtocol protocol;
    const std::string url = "wss://chatgpt.com/backend-api/codex/responses";
    const cpr::Header headers{{"Authorization", "Bearer token"}};

    ChatRequest first;
    first.model = "gpt-6.1-sol";
    first.session_id = "thread-tools";
    first.continuation_scope = "thread-tools";
    first.transport_turn_id = "thread-tools:0";
    first.catalog_wire_protocol = "responses_lite";
    first.messages.push_back(Message{.role = "user", .content = "run the lookup"});
    protocol.prepare_request(first);
    const auto first_key = protocol.websocket_connection_key(url, headers, first);

    // The socket is warmed up, then the turn runs for real.
    const auto prewarm = protocol.initial_websocket_request_frame(first);
    REQUIRE(prewarm.suppress_output);
    REQUIRE(protocol.parse_event(
        R"({"type":"response.completed","response":{"id":"resp_prewarm"}})").done);
    const auto real = protocol.next_websocket_request_frame(first, prewarm);
    REQUIRE_THAT(real.payload,
                 Catch::Matchers::ContainsSubstring(R"("previous_response_id":"resp_prewarm")"));

    // The endpoint answers with a tool call.
    (void)protocol.parse_event(
        R"({"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}})");
    REQUIRE(protocol.parse_event(
        R"({"type":"response.completed","response":{"id":"resp_tool"}})").done);

    // The conversation carries the call and its result.
    ToolCall call;
    call.id = "call_1";
    call.function.name = "lookup";
    call.function.arguments = "{}";
    ChatRequest second = first;
    second.transport_turn_id = "thread-tools:1";
    second.messages.push_back(Message{.role = "assistant", .tool_calls = {call}});
    second.messages.push_back(
        Message{.role = "tool", .content = "result", .tool_call_id = "call_1"});
    protocol.prepare_request(second);

    // Same socket, and the turn continues the previous response: only the
    // tool result is new.
    CHECK(protocol.websocket_connection_key(url, headers, second) == first_key);
    const auto next_turn = protocol.initial_websocket_request_frame(second);
    CHECK_FALSE(next_turn.suppress_output);
    CHECK_THAT(next_turn.payload,
               Catch::Matchers::ContainsSubstring(R"("previous_response_id":"resp_tool")"));
    CHECK_THAT(next_turn.payload,
               Catch::Matchers::ContainsSubstring(R"("type":"function_call_output")"));
    CHECK_THAT(next_turn.payload,
               !Catch::Matchers::ContainsSubstring(R"("type":"function_call")"));
}

TEST_CASE("Codex compacts natively with a compaction_trigger and replays the checkpoint",
          "[openai][responses][codex][native-checkpoint]") {
    CodexResponsesProtocol protocol;
    CHECK(protocol.supports_native_compaction());
    CHECK_FALSE(OpenAIResponsesProtocol{}.supports_native_compaction());

    ChatRequest req;
    req.model = "gpt-6.1-sol";
    req.session_id = "thread-compact";
    req.transport_turn_id = "thread-compact:3";
    req.messages.push_back(Message{.role = "user", .content = "fix the bug"});
    req.messages.push_back(Message{.role = "assistant", .content = "fixed"});
    req.native_compaction = true;
    protocol.prepare_request(req);

    // codex-rs compact_remote_v2_attempt.rs: the conversation, then the trigger.
    const std::string http = protocol.serialize(req);
    CHECK_THAT(http, Catch::Matchers::ContainsSubstring(
        R"("type":"output_text","text":"fixed"}]},{"type":"compaction_trigger"}])"));
    // Over the socket it is one generating request: no prewarm, no delta.
    const auto frame = protocol.initial_websocket_request_frame(req);
    CHECK_FALSE(frame.suppress_output);
    CHECK_THAT(frame.payload, Catch::Matchers::ContainsSubstring(R"({"type":"compaction_trigger"}])"));
    CHECK_THAT(frame.payload, !Catch::Matchers::ContainsSubstring("generate"));

    // The answer: one encrypted item, alias accepted, prefixed id kept.
    const auto parsed = protocol.parse_event(
        R"({"type":"response.output_item.done","item":{"type":"compaction_summary","id":"cmp_123","status":"completed","encrypted_content":"ENC=="}})");
    REQUIRE(parsed.chunks.size() == 1);
    REQUIRE(parsed.chunks[0].continuation_items.size() == 1);
    const auto checkpoint = parsed.chunks[0].continuation_items[0];
    CHECK(checkpoint.provider == "openai");
    CHECK(checkpoint.kind == kCheckpointContinuationKind);
    CHECK(checkpoint.payload == R"({"type":"compaction","id":"cmp_123","encrypted_content":"ENC=="})");
    const auto bare = protocol.parse_event(
        R"({"type":"response.output_item.done","item":{"type":"compaction","id":"opaque","encrypted_content":"E"}})");
    REQUIRE(bare.chunks.size() == 1);
    CHECK(bare.chunks[0].continuation_items[0].payload
          == R"({"type":"compaction","encrypted_content":"E"})");

    // Replayed right after the user message that carries it.
    ChatRequest next;
    next.model = req.model;
    next.session_id = req.session_id;
    Message retained{.role = "user", .content = "fix the bug"};
    retained.continuation_items.push_back(checkpoint);
    next.messages.push_back(retained);
    next.messages.push_back(Message{.role = "user", .content = "and add a test"});
    const std::string replay = protocol.serialize(next);
    CHECK_THAT(replay, Catch::Matchers::ContainsSubstring(
        R"("text":"fix the bug"}]},{"type":"compaction","id":"cmp_123","encrypted_content":"ENC=="},{"type":"message","role":"user")"));
    CHECK_THAT(replay, !Catch::Matchers::ContainsSubstring("compaction_trigger"));
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

TEST_CASE("OpenAIResponsesProtocol - ignores provisional function call and parses completed item",
          "[openai][responses][sse][tools]") {
    OpenAIResponsesProtocol protocol;

    const auto added = protocol.parse_event(
        "event: response.output_item.added\n"
        "data: {\"type\":\"response.output_item.added\",\"output_index\":0,\"item\":{\"type\":\"function_call\",\"call_id\":\"call_abc\",\"name\":\"read\",\"arguments\":\"{}\",\"status\":\"in_progress\"}}");
    REQUIRE_FALSE(added.done);
    REQUIRE(added.chunks.empty());

    auto result = protocol.parse_event(
        "event: response.output_item.done\n"
        "data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_abc\",\"name\":\"read\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}");

    REQUIRE_FALSE(result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].tools.size() == 1);
    REQUIRE(result.chunks[0].tools[0].id == "call_abc");
    REQUIRE(result.chunks[0].tools[0].function.name == "read");
    REQUIRE(result.chunks[0].tools[0].function.arguments
            == R"({"path":"README.md"})");
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

TEST_CASE("OpenAIResponsesProtocol - ignores full text snapshot after deltas",
          "[openai][responses][sse]") {
    OpenAIResponsesProtocol protocol;

    const auto delta = protocol.parse_event(
        "event: response.output_text.delta\n"
        "data: {\"type\":\"response.output_text.delta\",\"delta\":\"QWEN TOOL OK\"}");
    REQUIRE(delta.chunks.size() == 1);

    const auto done = protocol.parse_event(
        "event: response.output_text.done\n"
        "data: {\"type\":\"response.output_text.done\",\"text\":\"QWEN TOOL OK\"}");
    REQUIRE(done.chunks.empty());
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

TEST_CASE("Responses require completion and classify transient stream failures",
          "[openai][responses][regression][retry]") {
    CodexResponsesProtocol protocol;
    CHECK(protocol.requires_terminal_event());
    const auto started = protocol.parse_event(
        R"({"type":"response.created","response":{"id":"resp_start"}})");
    CHECK(started.stream_started);
    for (const auto code : {"server_error", "rate_limit_exceeded", "previous_response_not_found",
                             "websocket_connection_limit_reached", "policy_capacity_unavailable"}) {
        const auto result = protocol.parse_event(std::string{
            R"({"type":"response.failed","headers":{"Retry-After":"1.2"},"response":{"error":{"code":")"}
            + code + R"(","message":"try again"}}})");
        CHECK(result.stream_error);
        CHECK(result.retryable_stream_error);
        CHECK(result.rate_limit.retry_after == 2);
    }
    for (const auto code : {"insufficient_quota", "context_length_exceeded", "invalid_request_error",
                             "cyber_policy", "content_filter"}) {
        const auto result = protocol.parse_event(std::string{
            R"({"type":"response.failed","response":{"error":{"code":")"}
            + code + R"(","message":"do not retry"}}})");
        CHECK(result.stream_error);
        CHECK_FALSE(result.retryable_stream_error);
    }
    protocol.observe_response_headers({{"x-should-retry", "false"}}, {});
    CHECK_FALSE(protocol.parse_event(
        R"({"type":"error","error":{"code":"server_error","message":"veto"}})").retryable_stream_error);
    protocol.reset_state();
    CHECK(protocol.parse_event(
        R"({"type":"error","error":{"code":"server_error","message":"recover"}})").retryable_stream_error);

    // The HTTP stream reports errors flat: the event's own "code" is the error
    // class and its "type" is the event name. This shape used to index an
    // unbound simdjson element and crash.
    const auto flat = protocol.parse_event(
        R"({"type":"error","code":"server_error","message":"flat retry","param":null,"sequence_number":3})");
    CHECK(flat.stream_error);
    CHECK(flat.stream_error_type == "server_error");
    CHECK(flat.stream_error_message == "flat retry");
    CHECK(flat.retryable_stream_error);
    const auto flat_permanent = protocol.parse_event(
        R"({"type":"error","code":"invalid_prompt","message":"flat stop","param":null})");
    CHECK(flat_permanent.stream_error_type == "invalid_prompt");
    CHECK_FALSE(flat_permanent.retryable_stream_error);
    for (const auto* event : {R"({"type":"error","message":"no code"})",
                              R"({"type":"response.failed","response":{"status":"failed"}})",
                              R"({"type":"response.failed","response":{"error":null}})"}) {
        const auto bare = protocol.parse_event(event);
        CHECK(bare.stream_error);
        CHECK(bare.stream_error_type.empty());
        CHECK_FALSE(bare.retryable_stream_error);
        CHECK(bare.done);
    }
}

TEST_CASE("Codex replay preserves phases and interleaved encrypted reasoning",
          "[openai][responses][regression][continuation][phase]") {
    CodexResponsesProtocol protocol;
    ChatRequest request;
    request.model = "gpt-5.6-sol";
    request.session_id = "phase-replay";
    request.transport_turn_id = "phase-turn";
    request.messages = {{.role = "user", .content = "inspect the project"}};
    protocol.prepare_request(request);
    (void)protocol.serialize_websocket_request(request);
    Message assistant{.role = "assistant"};
    const auto consume = [&](std::string_view event) {
        auto parsed = protocol.parse_event(event);
        for (const auto& chunk : parsed.chunks) {
            assistant.content += chunk.content;
            assistant.tool_calls.insert(assistant.tool_calls.end(), chunk.tools.begin(), chunk.tools.end());
            assistant.continuation_items.insert(assistant.continuation_items.end(),
                chunk.continuation_items.begin(), chunk.continuation_items.end());
        }
    };
    consume(R"({"type":"response.output_item.added","item":{"type":"message","role":"assistant","content":[]}})");
    consume(R"({"type":"response.output_text.delta","delta":"Inspecting."})");
    consume(R"({"type":"response.output_item.done","item":{"type":"message","role":"assistant","phase":"commentary","content":[{"type":"output_text","text":"Inspecting."}]}})");
    consume(R"({"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_1","summary":[],"encrypted_content":"opaque"}})");
    consume(R"({"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_read","name":"read","arguments":"{}"}})");
    consume(R"({"type":"response.output_item.added","item":{"type":"message","role":"assistant","content":[]}})");
    consume(R"({"type":"response.output_text.delta","delta":"Found it."})");
    consume(R"({"type":"response.output_item.done","item":{"type":"message","role":"assistant","phase":"partial_answer","content":[{"type":"output_text","text":"Found it."}]}})");
    consume(R"({"type":"response.completed","response":{"id":"resp_phase"}})");
    REQUIRE(assistant.content == "Inspecting.Found it.");
    REQUIRE(assistant.continuation_items.size() == 4);
    request.messages.push_back(assistant);
    request.messages.push_back({.role = "tool", .content = "contents", .tool_call_id = "call_read"});
    const auto replay = protocol.serialize(request);
    const auto commentary = replay.find(R"("phase":"commentary")");
    const auto reasoning = replay.find(R"("encrypted_content":"opaque")");
    const auto call = replay.find(R"("type":"function_call")");
    const auto partial = replay.find(R"("phase":"partial_answer")");
    REQUIRE(commentary != std::string::npos);
    REQUIRE(partial != std::string::npos);
    CHECK(commentary < reasoning);
    CHECK(reasoning < call);
    CHECK(call < partial);
    const auto incremental = protocol.serialize_websocket_request(request);
    CHECK_THAT(incremental, Catch::Matchers::ContainsSubstring(R"("previous_response_id":"resp_phase")"));
    CHECK_THAT(incremental, !Catch::Matchers::ContainsSubstring("opaque"));
    CHECK_THAT(incremental, !Catch::Matchers::ContainsSubstring("Inspecting."));
    protocol.abandon_websocket_request(request);
    const auto reconnected = protocol.serialize_websocket_request(request);
    CHECK_THAT(reconnected, !Catch::Matchers::ContainsSubstring("previous_response_id"));
    CHECK_THAT(reconnected, Catch::Matchers::ContainsSubstring("opaque"));

    SECTION("edited text invalidates stale snapshots") {
        request.messages[1].content = "Updated history";
        const auto edited = protocol.serialize(request);
        CHECK_THAT(edited, Catch::Matchers::ContainsSubstring("Updated history"));
        CHECK_THAT(edited, !Catch::Matchers::ContainsSubstring("Inspecting."));
    }
    SECTION("normalized tools invalidate stale snapshots") {
        request.messages[1].tool_calls[0].function.arguments = R"({"path":"README.md"})";
        const auto edited = protocol.serialize(request);
        CHECK_THAT(edited, Catch::Matchers::ContainsSubstring(R"(\"path\":\"README.md\")"));
    }
}

TEST_CASE("Codex request controls honor the endpoint catalog and omit unsupported sampling fields",
          "[openai][responses][codex][regression][catalog]") {
    CodexResponsesProtocol protocol;
    ChatRequest request;
    request.model = "new-catalog-model";
    request.messages = {{.role = "user", .content = "reason"}};
    request.max_tokens = 1000;
    request.temperature = 0.5;
    request.catalog_reasoning = ModelReasoningProfile{
        .effort = ReasoningCapability::Effort | ReasoningCapability::UltraEffort,
        .complete = true,
        .catalog_default_effort = "high",
        .supported_efforts = {"high", "ultra"},
    };
    auto payload = protocol.serialize(request);
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(R"("effort":"high")"));
    CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring("temperature"));
    CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring("max_output_tokens"));
    request.effort = "ultra";
    CHECK_THAT(protocol.serialize(request), Catch::Matchers::ContainsSubstring(R"("effort":"high")"));
    request.catalog_reasoning->effort = request.catalog_reasoning->effort
        | ReasoningCapability::MaxEffort;
    request.catalog_reasoning->supported_efforts = {"high", "max", "ultra"};
    CHECK_THAT(protocol.serialize(request), Catch::Matchers::ContainsSubstring(R"("effort":"max")"));
    CHECK_THAT(protocol.serialize(request), !Catch::Matchers::ContainsSubstring("ultra"));
    request.effort = "low";
    CHECK_THAT(protocol.serialize(request), Catch::Matchers::ContainsSubstring(R"("effort":"high")"));
    request.catalog_reasoning->effort = {};
    request.catalog_reasoning->supported_efforts.clear();
    request.catalog_reasoning->catalog_default_effort.reset();
    CHECK_THAT(protocol.serialize(request), !Catch::Matchers::ContainsSubstring(R"("effort")"));
    CHECK_FALSE(protocol.parse_event("data: [DONE]").done);
}

TEST_CASE("Codex ultra follows the catalog's level and proactive delegation",
          "[openai][responses][codex][catalog][ultra]") {
    CodexResponsesProtocol protocol;
    ChatRequest request;
    request.model = "gpt-6.1-sol";
    request.effort = "ultra";
    request.messages = {{.role = "user", .content = "Refactor the parser."}};
    request.catalog_reasoning = ModelReasoningProfile{
        .effort = ReasoningCapability::Effort | ReasoningCapability::XHighEffort
            | ReasoningCapability::MaxEffort | ReasoningCapability::UltraEffort,
        .complete = true,
        .catalog_default_effort = "medium",
        .supported_efforts = {"low", "medium", "high", "xhigh", "max", "ultra"},
        .ultra = UltraEffortProfile{
            .effort = "xhigh", .proactive_delegation = true, .explicit_delegation = true},
    };
    Tool task;
    task.function.name = "task";
    request.tools = {task};

    const auto prepared = [&](ChatRequest req) {
        protocol.prepare_request(req);
        return protocol.serialize(req);
    };

    SECTION("the catalog's ultra level wins over max") {
        CHECK_THAT(prepared(request), Catch::Matchers::ContainsSubstring(R"("effort":"xhigh")"));
        request.catalog_reasoning->ultra->effort = "unoffered";
        CHECK_THAT(prepared(request), Catch::Matchers::ContainsSubstring(R"("effort":"max")"));
        request.catalog_reasoning->ultra.reset();
        CHECK_THAT(prepared(request), Catch::Matchers::ContainsSubstring(R"("effort":"max")"));
    }

    SECTION("a tuned model is told to delegate with the tools it has") {
        const auto payload = prepared(request);
        CHECK_THAT(payload, Catch::Matchers::ContainsSubstring("[Delegation Mode]"));
        CHECK_THAT(payload, Catch::Matchers::ContainsSubstring("do so with `task`"));
    }

    SECTION("the delegation layer is the plan's dynamic tail") {
        request.prompt_plan.append({.kind = core::context::ContextLayerKind::RuntimeInstructions,
                                    .stability = core::context::PromptStability::Stable,
                                    .name = "runtime",
                                    .content = "You are Filo."});
        ChatRequest req = request;
        protocol.prepare_request(req);
        REQUIRE(req.prompt_plan.layers().size() == 2);
        CHECK(req.prompt_plan.layers().back().name == "delegation_mode");
        CHECK(req.prompt_plan.layers().back().stability
              == core::context::PromptStability::Dynamic);
    }

    SECTION("below ultra a tuned model delegates only on request, as in codex-rs") {
        auto other = request;
        other.effort = "max";
        const auto payload = prepared(other);
        CHECK_THAT(payload, Catch::Matchers::ContainsSubstring("Delegation only on request."));
        CHECK_THAT(payload, Catch::Matchers::ContainsSubstring("Do not call `task` unless"));
        CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring("Proactive delegation is active"));
        other.effort.clear();
        CHECK_THAT(prepared(other), Catch::Matchers::ContainsSubstring("Delegation only on request."));
        CHECK_THAT(prepared(request), !Catch::Matchers::ContainsSubstring("Delegation only on request."));
    }

    SECTION("no delegation text for an untuned model, a suppressed mode, or no delegation tool") {
        auto other = request;
        other.catalog_reasoning->ultra->proactive_delegation = false;
        CHECK_THAT(prepared(other), !Catch::Matchers::ContainsSubstring("[Delegation Mode]"));
        other = request;
        other.effort = "max";
        other.catalog_reasoning->ultra->explicit_delegation = false;
        CHECK_THAT(prepared(other), !Catch::Matchers::ContainsSubstring("[Delegation Mode]"));
        other = request;
        other.tools.clear();
        CHECK_THAT(prepared(other), !Catch::Matchers::ContainsSubstring("[Delegation Mode]"));
        other.effort = "max";
        CHECK_THAT(prepared(other), !Catch::Matchers::ContainsSubstring("[Delegation Mode]"));
        other = request;
        other.catalog_reasoning->ultra.reset();
        CHECK_THAT(prepared(other), !Catch::Matchers::ContainsSubstring("[Delegation Mode]"));
    }
}

TEST_CASE("Codex sends the catalog verbosity alongside any structured-output format",
          "[openai][responses][codex][catalog][verbosity]") {
    ChatRequest request;
    request.model = "gpt-6.1-terra";
    request.messages = {{.role = "user", .content = "hi"}};
    request.catalog_verbosity = "low";

    CodexResponsesProtocol codex;
    CHECK_THAT(codex.serialize(request),
               Catch::Matchers::ContainsSubstring(R"(,"text":{"verbosity":"low"})"));

    request.response_format.type = ResponseFormat::Type::JsonObject;
    const auto with_format = codex.serialize(request);
    CHECK_THAT(with_format,
               Catch::Matchers::ContainsSubstring(R"("text":{"verbosity":"low","format":{"type":"json_schema")"));
    simdjson::dom::parser parser;
    const simdjson::padded_string padded(with_format);
    CHECK(parser.parse(padded).error() == simdjson::SUCCESS);

    OpenAIResponsesProtocol public_openai;
    CHECK_THAT(public_openai.serialize(request), !Catch::Matchers::ContainsSubstring("verbosity"));
}

TEST_CASE("Codex names a delegated worker in the header and the client metadata",
          "[openai][responses][codex][subagent]") {
    ChatRequest request;
    request.model = "gpt-6.1-sol";
    request.messages = {{.role = "user", .content = "hi"}};
    request.subagent_label = "explore";

    CodexResponsesProtocol codex;
    // codex-rs compatibility_headers and CodexResponsesMetadata::client_metadata.
    cpr::Header headers;
    codex.prepare_headers(headers, request, "https://chatgpt.com/backend-api/codex");
    CHECK(headers.at("x-openai-subagent") == "explore");
    CHECK_THAT(codex.serialize(request),
               Catch::Matchers::ContainsSubstring(R"("x-openai-subagent":"explore")"));

    // A conversation's own request names nobody.
    ChatRequest own;
    own.model = "gpt-6.1-sol";
    own.messages = {{.role = "user", .content = "hi"}};
    cpr::Header own_headers;
    codex.prepare_headers(own_headers, own, "https://chatgpt.com/backend-api/codex");
    CHECK(own_headers.count("x-openai-subagent") == 0);
    CHECK_THAT(codex.serialize(own),
               !Catch::Matchers::ContainsSubstring("x-openai-subagent"));

    // A label that is not a safe header value is dropped, as upstream drops
    // one it cannot send.
    ChatRequest unsafe_label = own;
    unsafe_label.subagent_label = "my worker";
    cpr::Header unsafe_headers;
    codex.prepare_headers(unsafe_headers, unsafe_label,
                          "https://chatgpt.com/backend-api/codex");
    CHECK(unsafe_headers.count("x-openai-subagent") == 0);
    CHECK_THAT(codex.serialize(unsafe_label),
               !Catch::Matchers::ContainsSubstring("x-openai-subagent"));

    // The WebSocket handshake carries the same header.
    cpr::Header websocket_headers;
    codex.prepare_websocket_headers(websocket_headers, request,
                                    "https://chatgpt.com/backend-api/codex");
    CHECK(websocket_headers.at("x-openai-subagent") == "explore");
}

TEST_CASE("Codex asks the endpoint to stop the response it is serving",
          "[openai][responses][codex][interrupt]") {
    CodexResponsesProtocol codex;
    // No response is known: there is nothing to stop.
    CHECK_FALSE(codex.interrupt_request("").has_value());

    // codex-rs responses_websocket.rs: the frame names the response to stop.
    const auto created = codex.parse_event(
        R"({"type":"response.created","response":{"id":"resp_123"}})");
    // The endpoint's handle for the response it is serving travels with the
    // event, so a caller can name it.
    CHECK(created.response_id == "resp_123");

    const auto frame = codex.interrupt_request(created.response_id);
    REQUIRE(frame.has_value());
    CHECK(*frame
          == R"({"type":"response.interrupt","response_id":"resp_123","mode":"discard_partial_items"})");
}

TEST_CASE("Codex gives a delegated worker its own thread",
          "[openai][responses][codex][subagent]") {
    ChatRequest request;
    request.model = "gpt-6.1-sol";
    request.session_id = "session-parent";
    request.conversation_scope = "worker:task_1";
    request.messages = {{.role = "user", .content = "hi"}};

    CodexResponsesProtocol codex;
    // codex-rs gives every subagent session its own thread id; the session
    // stays the one the worker was spawned from.
    const auto payload = codex.serialize(request);
    CHECK_THAT(payload,
               Catch::Matchers::ContainsSubstring(R"("thread_id":"worker:task_1")"));
    CHECK_THAT(payload,
               Catch::Matchers::ContainsSubstring(R"("session_id":"session-parent")"));
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(
                            R"("x-codex-window-id":"worker:task_1:0")"));

    cpr::Header headers;
    codex.prepare_headers(headers, request, "https://chatgpt.com/backend-api/codex");
    CHECK(headers.at("thread-id") == "worker:task_1");
    CHECK(headers.at("session_id") == "session-parent");
    CHECK(headers.at("x-client-request-id") == "worker:task_1");

    // The worker keeps its own prompt cache: it never shares the one of the
    // conversation that spawned it.
    CodexResponsesProtocol state;
    ChatRequest parent;
    parent.session_id = "session-parent";
    parent.messages = {{.role = "user", .content = "hi"}};
    state.prepare_request(parent);
    CHECK(parent.prompt_cache_key == "session-parent");

    ChatRequest worker;
    worker.session_id = "session-parent";
    worker.conversation_scope = "worker:task_1";
    worker.messages = {{.role = "user", .content = "hi"}};
    state.prepare_request(worker);
    CHECK(worker.prompt_cache_key == "worker:task_1");
}

TEST_CASE("Codex names the thread that spawned a worker and the turn it serves",
          "[openai][responses][codex][subagent]") {
    ChatRequest request;
    request.model = "gpt-6.1-sol";
    request.session_id = "session-parent";
    request.conversation_scope = "worker:task_1";
    request.parent_thread_id = "session-parent";
    request.transport_turn_id = "worker:task_1:1";
    request.messages = {{.role = "user", .content = "hi"}};

    CodexResponsesProtocol codex;
    // codex-rs CodexResponsesMetadata::client_metadata and
    // compatibility_headers: the worker names the thread that owns it.
    const auto payload = codex.serialize(request);
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(
                            R"("x-codex-parent-thread-id":"session-parent")"));
    CHECK_THAT(payload,
               Catch::Matchers::ContainsSubstring(R"("turn_id":"worker:task_1:1")"));

    cpr::Header headers;
    codex.prepare_headers(headers, request, "https://chatgpt.com/backend-api/codex");
    CHECK(headers.at("x-codex-parent-thread-id") == "session-parent");

    // A conversation the user started names no parent.
    ChatRequest own;
    own.model = "gpt-6.1-sol";
    own.session_id = "session-own";
    own.messages = {{.role = "user", .content = "hi"}};
    cpr::Header own_headers;
    codex.prepare_headers(own_headers, own, "https://chatgpt.com/backend-api/codex");
    CHECK(own_headers.count("x-codex-parent-thread-id") == 0);
    CHECK_THAT(codex.serialize(own),
               !Catch::Matchers::ContainsSubstring("x-codex-parent-thread-id"));
}

TEST_CASE("Codex client metadata is a well formed object whatever it carries",
          "[openai][responses][codex][subagent]") {
    ChatRequest request;
    request.model = "gpt-6.1-sol";
    request.session_id = "session-parent";
    request.conversation_scope = "worker:task_1";
    request.parent_thread_id = "session-parent";
    request.transport_turn_id = "worker:task_1:1";
    request.subagent_label = "explore";
    request.messages = {{.role = "user", .content = "hi"}};

    CodexResponsesProtocol codex;
    const auto payload = codex.serialize(request);

    // Every field codex-rs puts in the client metadata has to arrive as a
    // member of one object: a missing separator would make the whole request
    // unparseable for the backend.
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    REQUIRE(parser.parse(payload).get(document) == simdjson::SUCCESS);
    const auto field = [&document](const char* name) -> std::string {
        std::string_view value;
        REQUIRE(document["client_metadata"][name].get_string().get(value)
                == simdjson::SUCCESS);
        return std::string(value);
    };
    CHECK(field("x-openai-subagent") == "explore");
    CHECK(field("x-codex-parent-thread-id") == "session-parent");
    CHECK(field("turn_id") == "worker:task_1:1");
    CHECK(field("thread_id") == "worker:task_1");
    CHECK(field("session_id") == "session-parent");
    CHECK(field("x-codex-window-id") == "worker:task_1:0");
}

TEST_CASE("Codex keeps the routing token the endpoint names",
          "[openai][responses][codex][turn-state]") {
    ChatRequest request;
    request.model = "gpt-6.1-sol";
    request.transport_turn_id = "turn-1";
    request.messages = {{.role = "user", .content = "hi"}};

    CodexResponsesProtocol codex;
    // codex-rs responses_websocket.rs: the endpoint names the routing token
    // of the turn in a metadata event.
    const auto parsed = codex.parse_event(
        R"({"type":"response.metadata","headers":{"x-codex-turn-state":"token-1"}})");
    CHECK(parsed.turn_state == "token-1");

    // The next request of the same turn carries it back, so the turn stays on
    // the machine that serves it.
    codex.observe_turn_state(request, parsed.turn_state);
    cpr::Header headers;
    codex.prepare_headers(headers, request, "https://chatgpt.com/backend-api/codex");
    CHECK(headers.at("x-codex-turn-state") == "token-1");

    // Another turn does not carry it: the token belongs to the turn that
    // earned it.
    ChatRequest other = request;
    other.transport_turn_id = "turn-2";
    cpr::Header other_headers;
    codex.prepare_headers(other_headers, other, "https://chatgpt.com/backend-api/codex");
    CHECK(other_headers.count("x-codex-turn-state") == 0);
}

TEST_CASE("Codex reads the routing token from either metadata event",
          "[openai][responses][codex][turn-state]") {
    CodexResponsesProtocol codex;
    // The websocket names it in its own metadata event.
    CHECK(codex.parse_event(
              R"({"type":"codex.response.metadata","headers":{"x-codex-turn-state":"token-2"}})")
              .turn_state == "token-2");
    // The header name is matched the way a header is matched.
    CHECK(codex.parse_event(
              R"({"type":"response.metadata","headers":{"X-Codex-Turn-State":"token-3"}})")
              .turn_state == "token-3");
    // A header value may be a list of values.
    CHECK(codex.parse_event(
              R"({"type":"response.metadata","headers":{"x-codex-turn-state":["token-4"]}})")
              .turn_state == "token-4");
    // An event that names no token says nothing about one.
    CHECK(codex.parse_event(
              R"({"type":"response.metadata","headers":{"x-models-etag":"etag-1"}})")
              .turn_state.empty());
    CHECK(codex.parse_event(R"({"type":"response.metadata"})").turn_state.empty());
}

TEST_CASE("Codex names the turn and the tree it belongs to",
          "[openai][responses][codex][subagent]") {
    ChatRequest request;
    request.model = "gpt-6.1-sol";
    request.session_id = "session-parent";
    request.conversation_scope = "worker:task_1";
    request.parent_thread_id = "session-parent";
    request.parent_turn_id = "session-parent:3";
    request.root_turn_id = "session-parent:1";
    request.transport_turn_id = "worker:task_1:1";
    request.subagent_label = "explore";
    request.messages = {{.role = "user", .content = "hi"}};

    CodexResponsesProtocol codex;
    const auto payload = codex.serialize(request);

    // codex-rs CodexResponsesMetadata::client_metadata names the turn that
    // spawned the worker and the turn the tree starts at.
    simdjson::dom::parser parser;
    simdjson::dom::element document;
    REQUIRE(parser.parse(payload).get(document) == simdjson::SUCCESS);
    const auto field = [&document](const char* name) -> std::string {
        std::string_view value;
        REQUIRE(document["client_metadata"][name].get_string().get(value)
                == simdjson::SUCCESS);
        return std::string(value);
    };
    CHECK(field("parent_turn_id") == "session-parent:3");
    CHECK(field("root_turn_id") == "session-parent:1");

    // The turn metadata names the same turn, in the shape the backend reads.
    const std::string blob = field("x-codex-turn-metadata");
    simdjson::dom::element turn;
    REQUIRE(parser.parse(blob).get(turn) == simdjson::SUCCESS);
    const auto turn_field = [&turn](const char* name) -> std::string {
        std::string_view value;
        REQUIRE(turn[name].get_string().get(value) == simdjson::SUCCESS);
        return std::string(value);
    };
    CHECK(turn_field("turn_id") == "worker:task_1:1");
    CHECK(turn_field("thread_id") == "worker:task_1");
    CHECK(turn_field("session_id") == "session-parent");
    CHECK(turn_field("agent_name") == "explore");
    CHECK(turn_field("parent_thread_id") == "session-parent");
    CHECK(turn_field("parent_turn_id") == "session-parent:3");
    CHECK(turn_field("root_turn_id") == "session-parent:1");
    CHECK(turn_field("request_kind") == "turn");
    CHECK(turn_field("window_id") == "worker:task_1:0");
    int64_t window_number = 0;
    REQUIRE(turn["window_number"].get(window_number) == simdjson::SUCCESS);
    CHECK(window_number == 0);

    // The header carries the same thing.
    cpr::Header headers;
    codex.prepare_headers(headers, request, "https://chatgpt.com/backend-api/codex");
    CHECK(headers.at("x-codex-turn-metadata") == blob);
}

TEST_CASE("Codex says what the request is for",
          "[openai][responses][codex][turn-kind]") {
    const auto request_kind_of = [](const std::string& payload) -> std::string {
        simdjson::dom::parser parser;
        simdjson::dom::element document;
        REQUIRE(parser.parse(payload).get(document) == simdjson::SUCCESS);
        std::string_view blob;
        REQUIRE(document["client_metadata"]["x-codex-turn-metadata"]
                    .get_string().get(blob) == simdjson::SUCCESS);
        const std::string inner(blob);
        simdjson::dom::element turn;
        REQUIRE(parser.parse(inner).get(turn) == simdjson::SUCCESS);
        std::string_view kind;
        REQUIRE(turn["request_kind"].get_string().get(kind) == simdjson::SUCCESS);
        return std::string(kind);
    };

    ChatRequest request;
    request.model = "gpt-6.1-sol";
    request.transport_turn_id = "turn-1";
    request.messages = {{.role = "user", .content = "hi"}};

    CodexResponsesProtocol codex;
    // The warm-up that opens the socket is not a turn.
    ChatRequest prewarm = request;
    codex.prepare_request(prewarm);
    CHECK(request_kind_of(codex.initial_websocket_request_frame(prewarm).payload)
          == "prewarm");
    // A compaction asks the endpoint for its own checkpoint of the history.
    ChatRequest compaction = request;
    compaction.native_compaction = true;
    CHECK(request_kind_of(codex.serialize(compaction)) == "compaction");
    // A turn is a turn.
    CHECK(request_kind_of(codex.serialize(request)) == "turn");
}

TEST_CASE("Codex resolves the service tier against the catalog and hints routing",
          "[openai][responses][codex][service-tier]") {
    const auto tier_of = [](const std::string& payload) -> std::string {
        simdjson::dom::parser parser;
        const simdjson::padded_string padded(payload);
        std::string_view tier;
        if (parser.parse(padded)["service_tier"].get(tier) != simdjson::SUCCESS) return {};
        return std::string(tier);
    };
    ChatRequest req;
    req.model = "gpt-6.1-sol";
    req.session_id = "thread-tier";
    req.messages.push_back(Message{.role = "user", .content = "hello"});

    // No catalog evidence: the configured tier is sent as is.
    CodexResponsesProtocol configured(true, "priority");
    CHECK(tier_of(configured.serialize(req)) == "priority");

    // codex-rs codex-api/src/common.rs: routing fields precede the input.
    const std::string payload = configured.serialize(req);
    CHECK(payload.find(R"("service_tier")") < payload.find(R"("input")"));
    CHECK(payload.find(R"("stream")") < payload.find(R"("service_tier")"));

    cpr::Header headers;
    configured.prepare_headers(headers, req, "https://chatgpt.com/backend-api/codex");
    CHECK(headers.at("x-codex-routing-hint") == "model=gpt-6.1-sol;tier=priority");

    // The catalog offers Fast (`priority`) for this model.
    req.catalog_service_tiers = std::vector<std::string>{"priority"};
    CHECK(tier_of(configured.serialize(req)) == "priority");
    CodexResponsesProtocol unconfigured;
    CHECK(tier_of(unconfigured.serialize(req)).empty());
    cpr::Header plain;
    unconfigured.prepare_headers(plain, req, "https://chatgpt.com/backend-api/codex");
    CHECK(plain.at("x-codex-routing-hint") == "model=gpt-6.1-sol");

    // The catalog default applies only when the user chose nothing, and
    // `default` opts out of it.
    req.catalog_default_service_tier = "priority";
    CHECK(tier_of(unconfigured.serialize(req)) == "priority");
    req.service_tier = "default";
    CHECK(tier_of(unconfigured.serialize(req)).empty());

    // A tier the catalog does not offer is dropped; flex is always accepted.
    req.service_tier = "ultrafast";
    CHECK(tier_of(unconfigured.serialize(req)).empty());
    req.service_tier = "flex";
    CHECK(tier_of(unconfigured.serialize(req)) == "flex");

    // Public OpenAI keeps sending the configured value.
    OpenAIResponsesProtocol public_openai(false, "priority");
    req.service_tier.clear();
    req.catalog_service_tiers = std::vector<std::string>{};
    CHECK(tier_of(public_openai.serialize(req)) == "priority");
}

TEST_CASE("Codex records a durable reasoning-effort update when the model accepts one",
          "[openai][responses][codex][effort]") {
    CodexResponsesProtocol protocol;
    ChatRequest req;
    req.model = "gpt-6.1-sol";
    req.effort = "high";
    req.catalog_supports_reasoning_effort_updates = true;

    // codex-rs protocol/src/models/configuration_update_tests.rs: the item
    // carries the effort and nothing else.
    const auto items = protocol.conversation_control_items(req);
    REQUIRE(items.size() == 1);
    CHECK(items[0].provider == "openai");
    CHECK(items[0].kind == kReasoningEffortContinuationKind);
    CHECK(items[0].payload
          == R"({"type":"configuration_update","reasoning":{"effort":"high"}})");

    // The public API has no such item.
    CHECK(OpenAIResponsesProtocol{}.conversation_control_items(req).empty());

    // A model that does not accept the item keeps its effort on the request.
    req.catalog_supports_reasoning_effort_updates = false;
    CHECK(protocol.conversation_control_items(req).empty());

    // The catalog resolves the effort, so the update and the request agree.
    req.catalog_supports_reasoning_effort_updates = true;
    req.effort = "ultra";
    req.catalog_reasoning = ModelReasoningProfile{};
    req.catalog_reasoning->complete = true;
    req.catalog_reasoning->effort = ReasoningCapabilities{ReasoningCapability::Effort}
        | ReasoningCapability::MaxEffort;
    req.catalog_reasoning->supported_efforts = {"low", "medium", "high", "xhigh", "max"};
    req.catalog_reasoning->catalog_default_effort = "medium";
    const auto ultra = protocol.conversation_control_items(req);
    REQUIRE(ultra.size() == 1);
    CHECK(ultra[0].payload
          == R"({"type":"configuration_update","reasoning":{"effort":"max"}})");
}

TEST_CASE("Codex replays a durable effort update where the change happened",
          "[openai][responses][codex][effort][continuity]") {
    CodexResponsesProtocol protocol;
    const auto effort_item = [](std::string_view effort) {
        return ContinuationItem{
            .provider = "openai",
            .kind = std::string(kReasoningEffortContinuationKind),
            .payload = std::string(R"({"type":"configuration_update","reasoning":{"effort":")")
                + std::string(effort) + R"("}})",
        };
    };

    ChatRequest req;
    req.model = "gpt-6.1-sol";
    req.catalog_supports_reasoning_effort_updates = true;
    Message user{.role = "user", .content = "hello"};
    user.continuation_items.push_back(effort_item("high"));
    req.messages.push_back(user);

    // The item follows the message it rides on.
    CHECK_THAT(protocol.serialize(req), Catch::Matchers::ContainsSubstring(
        R"("text":"hello"}]},{"type":"configuration_update","reasoning":{"effort":"high"}})"));

    // A model that does not accept the item does not replay it.
    req.catalog_supports_reasoning_effort_updates = false;
    CHECK_THAT(protocol.serialize(req),
        !Catch::Matchers::ContainsSubstring("configuration_update"));

    // codex-rs core/src/client.rs build_responses_request: consecutive
    // updates collapse to the last one.
    req.catalog_supports_reasoning_effort_updates = true;
    req.messages.front().continuation_items.push_back(effort_item("max"));
    const std::string collapsed = protocol.serialize(req);
    CHECK_THAT(collapsed, !Catch::Matchers::ContainsSubstring(R"("effort":"high")"));
    CHECK_THAT(collapsed, Catch::Matchers::ContainsSubstring(R"("effort":"max")"));
}

TEST_CASE("Responses retry reset preserves other session continuity and permanent quota failures",
          "[openai][responses][regression][retry][continuity]") {
    OpenAIResponsesProtocol owner;
    auto stream = owner.clone();
    ChatRequest first;
    first.model = "gpt-5";
    first.session_id = "keep-session";
    stream->prepare_request(first);
    (void)stream->parse_event(R"({"type":"response.completed","response":{"id":"keep-response"}})");
    const cpr::Header headers;
    stream->on_response({200, {}, headers});
    stream->reset_attempt_state();
    auto resumed = owner.clone();
    ChatRequest next = first;
    resumed->prepare_request(next);
    CHECK(next.previous_response_id == "keep-response");
    CHECK_FALSE(stream->is_retryable({429, R"({"error":{"code":"insufficient_quota"}})", headers}));
    CHECK(stream->is_retryable({429, R"({"error":{"code":"rate_limit_exceeded"}})", headers}));
}

TEST_CASE("ChatGPT plan limits named in error.type are final and worded like Codex",
          "[openai][responses][codex][regression][retry]") {
    // codex-rs codex-api/src/api_bridge.rs reads `error.type`; the ChatGPT
    // backend sends no `error.code` for these.
    CodexResponsesProtocol protocol;
    const cpr::Header none;
    const std::string limit =
        R"({"error":{"type":"usage_limit_reached","message":"The usage limit has been reached","plan_type":"pro"}})";
    const std::string not_included = R"({"error":{"type":"usage_not_included"}})";
    CHECK_FALSE(protocol.is_retryable({429, limit, none}));
    CHECK_FALSE(protocol.is_retryable({429, not_included, none}));
    CHECK_FALSE(protocol.is_retryable({429, R"({"error":{"code":"credit_balance_exhausted"}})", none}));
    CHECK_FALSE(protocol.is_retryable({429, R"({"error":{"type":"insufficient_quota"}})", none}));
    CHECK(protocol.is_retryable({429, R"({"error":{"type":"rate_limit_exceeded"}})", none}));

    CHECK(protocol.format_error_message({429, limit, none})
          == "[You’ve hit your usage limit. Visit https://chatgpt.com/settings/usage to purchase "
             "more credits or try again later.]");
    CHECK(protocol.format_error_message({429, not_included, none})
          == "[To use Codex with your ChatGPT plan, upgrade to Plus: https://chatgpt.com/explore/plus.]");
    CHECK(protocol.format_error_message(
              {429, R"({"error":{"type":"usage_limit_reached","plan_type":"enterprise"}})", none})
          == "[You’ve hit your usage limit. Try again later.]");
    CHECK(protocol.format_error_message(
              {429, R"({"error":{"type":"usage_limit_reached","plan_type":"free"}})", none})
          == "[You’ve hit your usage limit. Upgrade to Plus to continue using Codex "
             "(https://chatgpt.com/explore/plus), or try again later.]");
    const cpr::Header workspace{{"x-codex-rate-limit-reached-type", "workspace_member_credits_depleted"}};
    CHECK(protocol.format_error_message({429, limit, workspace})
          == "[Your workspace is out of credits. Ask your workspace owner to refill in order to continue.]");
    const cpr::Header named{{"x-codex-active-limit", "codex_other"},
                            {"x-codex-other-limit-name", "GPT-6.1-Sol"}};
    CHECK(protocol.format_error_message({429, limit, named})
          == "[You’ve hit your usage limit for GPT-6.1-Sol. Switch to another model now, or try again later.]");
    // A reset time is rendered on the local clock; a far date carries its year.
    CHECK_THAT(protocol.format_error_message(
                   {429, R"({"error":{"type":"usage_limit_reached","resets_at":4086590400}})", none}),
               Catch::Matchers::StartsWith("[You’ve hit your usage limit. Try again at ")
                   && Catch::Matchers::ContainsSubstring(", 2099 ")
                   && Catch::Matchers::EndsWith("M.]"));
    // Ordinary failures keep the generic text.
    CHECK_THAT(protocol.format_error_message({500, R"({"error":{"type":"server_error"}})", none}),
               Catch::Matchers::StartsWith("[HTTP Error: 500"));

    // The WebSocket wraps the same 429 as an `error` event (codex-rs
    // responses_websocket.rs test payload), with numeric header values.
    const auto wrapped = protocol.parse_event(
        R"({"type":"error","status":429,"error":{"type":"usage_limit_reached","message":"The usage limit has been reached","plan_type":"pro","resets_at":1738888888},"headers":{"x-codex-primary-used-percent":"100.0","x-codex-primary-window-minutes":15}})");
    CHECK(wrapped.stream_error);
    CHECK_FALSE(wrapped.retryable_stream_error);
    CHECK_THAT(wrapped.stream_error_message,
               Catch::Matchers::StartsWith("You’ve hit your usage limit. Visit https://chatgpt.com/settings/usage"));
    REQUIRE_FALSE(wrapped.chunks.empty());
    CHECK_THAT(wrapped.chunks.back().content, Catch::Matchers::ContainsSubstring("usage limit"));
    const auto limits = protocol.last_rate_limit();
    REQUIRE(limits.usage_windows.size() == 1);
    CHECK(limits.usage_windows.front().label == "15m");
    CHECK(limits.usage_windows.front().utilization == Catch::Approx(1.0f));
}

TEST_CASE("A failed response is classified by the class the backend names",
          "[openai][responses][codex][retry]") {
    CodexResponsesProtocol codex;

    // codex-rs try_parse_retry_delay: a rate limit states when to come back
    // in its message, and the delay is a lower bound.
    const auto limited = codex.parse_event(
        R"({"type":"response.failed","response":{"error":{"code":"rate_limit_exceeded","message":"Rate limit reached. Please try again in 11.054s."}}})");
    CHECK(limited.stream_error);
    CHECK(limited.retryable_stream_error);
    CHECK(limited.rate_limit.retry_after == 12);

    // The header states the delay too, and the header wins.
    const auto with_header = codex.parse_event(
        R"({"type":"response.failed","headers":{"Retry-After":"3"},"response":{"error":{"code":"slow_down","message":"Please try again in 11.054s."}}})");
    CHECK(with_header.retryable_stream_error);
    CHECK(with_header.rate_limit.retry_after == 3);

    // A class the backend names and upstream does not know is transient.
    const auto unknown = codex.parse_event(
        R"({"type":"response.failed","response":{"error":{"code":"some_new_transient_class","message":"try later"}}})");
    CHECK(unknown.stream_error);
    CHECK(unknown.retryable_stream_error);

    // The classes a retry cannot fix stay terminal.
    for (const auto* code : {"usage_limit_reached", "usage_not_included",
                              "insufficient_quota", "invalid_prompt",
                              "cyber_policy", "context_length_exceeded",
                              "invalid_request_error", "content_filter"}) {
        const auto terminal = codex.parse_event(std::string{
            R"({"type":"response.failed","response":{"error":{"code":")"
            + std::string(code) + R"(","message":"no"}}})"});
        CHECK(terminal.stream_error);
        CHECK_FALSE(terminal.retryable_stream_error);
    }

    // The generic Responses API keeps to the classes it knows.
    OpenAIResponsesProtocol generic;
    CHECK_FALSE(generic.parse_event(
        R"({"type":"response.failed","response":{"error":{"code":"some_new_transient_class","message":"try later"}}})")
        .retryable_stream_error);
    CHECK(generic.parse_event(
        R"({"type":"response.failed","response":{"error":{"code":"server_error","message":"try later"}}})")
        .retryable_stream_error);
}
