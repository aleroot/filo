#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"
#include "core/llm/CodexUsageClient.hpp"
#include "core/llm/Models.hpp"

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

TEST_CASE("CodexResponsesProtocol always sends reasoning.summary",
          "[openai][responses][codex][serializer][reasoning]") {
    CodexResponsesProtocol protocol;
    ChatRequest req;
    req.model = "gpt-5.6-sol";
    req.messages.push_back(Message{.role = "user", .content = "Hello."});

    const std::string payload = protocol.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("reasoning":{"summary":"auto"})"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("effort")"));
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
    REQUIRE(headers.at("version") == "0.164.0");
    REQUIRE(headers.at("User-Agent") == "codex_cli_rs/0.164.0");
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
                             "websocket_connection_limit_reached"}) {
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
