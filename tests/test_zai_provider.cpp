#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/config/ConfigManager.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/ZaiProtocol.hpp"

#include <array>
#include <format>
#include <string_view>
#include <utility>

using namespace core::llm;
using namespace core::llm::protocols;

TEST_CASE("ProviderFactory creates Z.ai regular API provider", "[zai][factory]") {
    core::config::ProviderConfig config;
    config.model = "glm-5.1";

    auto provider = core::llm::ProviderFactory::create_provider("zai", config);

    REQUIRE(provider != nullptr);
    REQUIRE(provider->should_estimate_cost());
    REQUIRE(provider->max_context_size() == 200000);
}

TEST_CASE("ProviderFactory creates Z.ai Coding Plan provider", "[zai][factory][coding]") {
    core::config::ProviderConfig config;
    config.model = "glm-4.7";

    auto provider = core::llm::ProviderFactory::create_provider("zai-coding", config);

    REQUIRE(provider != nullptr);
    REQUIRE_FALSE(provider->should_estimate_cost());
    REQUIRE(provider->max_context_size() == 200000);
}

TEST_CASE("ProviderFactory matches Z.ai Coding Plan before generic Z.ai prefix",
          "[zai][factory][coding]") {
    core::config::ProviderConfig config;
    config.model = "glm-5.2";

    auto provider = core::llm::ProviderFactory::create_provider("zai-coding-opus", config);

    REQUIRE(provider != nullptr);
    REQUIRE_FALSE(provider->should_estimate_cost());
    REQUIRE(provider->max_context_size() == 1000000);
}

TEST_CASE("Z.ai OpenAI-compatible protocol appends chat completions path",
          "[zai][protocol]") {
    OpenAIProtocol protocol;

    REQUIRE(protocol.build_url("https://api.z.ai/api/paas/v4", "glm-5.1")
            == "https://api.z.ai/api/paas/v4/chat/completions");
    REQUIRE(protocol.build_url("https://api.z.ai/api/coding/paas/v4", "glm-4.7")
            == "https://api.z.ai/api/coding/paas/v4/chat/completions");
}

TEST_CASE("Z.ai protocol enables preserved thinking for GLM models",
          "[zai][protocol]") {
    ZaiProtocol protocol;

    ChatRequest request;
    request.model = "glm-5.2";
    request.messages.push_back(Message{.role = "user", .content = "hi"});

    const std::string payload = protocol.serialize(request);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("thinking":{"type":"enabled","clear_thinking":false})"));
}

TEST_CASE("Z.ai protocol serializes Z.ai reasoning effort",
          "[zai][protocol][effort]") {
    ZaiProtocol protocol;

    ChatRequest request;
    request.model = "glm-5.2";
    request.effort = "max";
    request.messages.push_back(Message{.role = "user", .content = "hi"});

    const std::string payload = protocol.serialize(request);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"max")"));
}

TEST_CASE("Z.ai protocol maps effort to supported GLM-5.3 levels",
          "[zai][protocol][effort]") {
    ZaiProtocol protocol;
    const std::array<std::pair<std::string_view, std::string_view>, 5> cases{{
        {"minimal", "low"},
        {"low", "low"},
        {"medium", "high"},
        {"xhigh", "high"},
        {"ultra", "max"},
    }};

    for (const auto& [configured, expected] : cases) {
        ChatRequest request;
        request.model = "glm-5.3-flash";
        request.effort = configured;
        request.messages.push_back(Message{.role = "user", .content = "hi"});

        const std::string payload = protocol.serialize(request);
        REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
            std::format(R"("reasoning_effort":"{}")", expected)));
    }
}

TEST_CASE("Z.ai protocol maps effort to supported GLM-5.2 levels",
          "[zai][protocol][effort]") {
    ZaiProtocol protocol;
    const std::array<std::pair<std::string_view, std::string_view>, 3> cases{{
        {"low", "high"},
        {"medium", "high"},
        {"xhigh", "max"},
    }};

    for (const auto& [configured, expected] : cases) {
        ChatRequest request;
        request.model = "glm-5.2";
        request.effort = configured;
        request.messages.push_back(Message{.role = "user", .content = "hi"});

        const std::string payload = protocol.serialize(request);
        REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
            std::format(R"("reasoning_effort":"{}")", expected)));
    }

    ChatRequest minimal_request;
    minimal_request.model = "glm-5.2";
    minimal_request.effort = "minimal";
    minimal_request.messages.push_back(Message{.role = "user", .content = "hi"});
    const std::string minimal_payload = protocol.serialize(minimal_request);
    REQUIRE_THAT(minimal_payload,
                 Catch::Matchers::ContainsSubstring(R"("thinking":{"type":"disabled"})"));
    REQUIRE_THAT(minimal_payload,
                 !Catch::Matchers::ContainsSubstring("reasoning_effort"));
}

TEST_CASE("Z.ai protocol enables streamed tool deltas on supported models",
          "[zai][protocol][tools]") {
    ChatRequest request;
    request.model = "glm-5.3";
    request.stream = true;
    request.messages.push_back(Message{.role = "user", .content = "hi"});
    Tool tool;
    tool.function.name = "lookup";
    tool.function.description = "Look up a value";
    request.tools.push_back(std::move(tool));

    const std::string payload = ZaiProtocol{}.serialize(request);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("tool_stream":true)"));

    request.model = "glm-5-turbo";
    const std::string glm_5_family_payload = ZaiProtocol{}.serialize(request);
    REQUIRE_THAT(glm_5_family_payload,
                 Catch::Matchers::ContainsSubstring(R"("tool_stream":true)"));

    request.model = "glm-4.5-air";
    const std::string unsupported_model_payload = ZaiProtocol{}.serialize(request);
    REQUIRE_THAT(unsupported_model_payload,
                 !Catch::Matchers::ContainsSubstring("tool_stream"));

    request.model = "glm-5.3";
    request.stream = false;
    const std::string non_stream_payload = ZaiProtocol{}.serialize(request);
    REQUIRE_THAT(non_stream_payload, !Catch::Matchers::ContainsSubstring("tool_stream"));
}

TEST_CASE("Z.ai protocol parses cached and reasoning usage from stream events",
          "[zai][protocol][usage]") {
    const auto result = ZaiProtocol{}.parse_event(
        R"(data: {"choices":[],"usage":{"prompt_tokens":120,"completion_tokens":35,"prompt_tokens_details":{"cached_tokens":80},"completion_tokens_details":{"reasoning_tokens":22}}})");

    REQUIRE(result.prompt_tokens == 120);
    REQUIRE(result.completion_tokens == 35);
    REQUIRE(result.cached_prompt_tokens == 80);
    REQUIRE(result.reasoning_tokens == 22);
    REQUIRE(result.chunks.empty());
}

TEST_CASE("Z.ai stream usage falls back when top-level usage is empty",
          "[zai][protocol][usage]") {
    const auto result = ZaiProtocol{}.parse_event(
        R"(data: {"usage":{},"choices":[{"usage":{"prompt_tokens":7,"completion_tokens":4},"delta":{}}]})");

    REQUIRE(result.prompt_tokens == 7);
    REQUIRE(result.completion_tokens == 4);
}

TEST_CASE("Z.ai protocol can disable thinking per turn",
          "[zai][protocol][effort]") {
    ZaiProtocol protocol;

    ChatRequest request;
    request.model = "glm-4.7";
    request.effort = "off";
    request.messages.push_back(Message{.role = "user", .content = "hi"});

    const std::string payload = protocol.serialize(request);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("thinking":{"type":"disabled"})"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("reasoning_effort"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("clear_thinking"));
}

TEST_CASE("Z.ai protocol echoes assistant reasoning_content for preserved thinking",
          "[zai][protocol][thinking]") {
    ZaiProtocol protocol;

    ChatRequest request;
    request.model = "glm-4.7";
    request.messages.push_back(Message{.role = "user", .content = "hi"});

    Message assistant;
    assistant.role = "assistant";
    assistant.content = "I will call a tool.";
    assistant.reasoning_content = "Need weather first.";
    assistant.tool_calls.push_back(ToolCall{
        .id = "call_1",
        .type = "function",
        .function = {.name = "get_weather", .arguments = R"({"city":"Dubai"})"},
    });
    request.messages.push_back(std::move(assistant));
    request.messages.push_back(Message{
        .role = "tool",
        .content = R"({"weather":"sunny"})",
        .tool_call_id = "call_1",
    });

    const std::string payload = protocol.serialize(request);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("reasoning_content":"Need weather first.")"));
}

TEST_CASE("Z.ai protocol does not replay foreign reasoning state",
          "[zai][protocol][thinking][provenance]") {
    ChatRequest request;
    request.model = "glm-4.7";
    request.messages.push_back(Message{
        .role = "assistant",
        .content = "Prior answer.",
        .reasoning_content = "Kimi reasoning",
        .reasoning_protocol = "kimi",
    });

    const auto payload = ZaiProtocol{}.serialize(request);
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("Kimi reasoning"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("reasoning_content"));
}

TEST_CASE("Z.ai protocol parses streamed reasoning_content",
          "[zai][protocol][thinking]") {
    ZaiProtocol protocol;

    const auto result = protocol.parse_event(
        R"(data: {"choices":[{"delta":{"reasoning_content":"think","content":"answer"},"index":0}]})");

    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].reasoning_content == "think");
    REQUIRE(result.chunks[0].reasoning_protocol == "zai");
    REQUIRE(result.chunks[0].content == "answer");
}

TEST_CASE("Z.ai protocol distinguishes 429 business errors",
          "[zai][protocol][errors]") {
    ZaiProtocol general;

    const HttpResponse insufficient_balance{
        429,
        R"({"error":{"code":"1113","message":"Insufficient balance or no resource package."}})",
        {},
    };
    general.on_response(insufficient_balance);
    CHECK_FALSE(general.is_retryable(insufficient_balance));
    CHECK_FALSE(general.last_rate_limit().is_rate_limited);
    const std::string balance_message =
        general.format_error_message(insufficient_balance);
    CHECK(balance_message.find("Z.ai Error 1113") != std::string::npos);
    CHECK(balance_message.find("General API") != std::string::npos);
    CHECK(balance_message.find("GLM Coding Plan") != std::string::npos);

    const HttpResponse request_throttle{
        429,
        R"({"error":{"code":"1302","message":"Rate limit reached for requests"}})",
        {},
    };
    general.on_response(request_throttle);
    CHECK(general.is_retryable(request_throttle));
    CHECK(general.last_rate_limit().is_rate_limited);

    const HttpResponse overloaded{
        429,
        R"({"error":{"code":1305,"message":"The service may be temporarily overloaded"}})",
        {},
    };
    general.on_response(overloaded);
    CHECK(general.is_retryable(overloaded));
    CHECK_FALSE(general.last_rate_limit().is_rate_limited);

    const HttpResponse model_not_in_plan{
        429,
        R"({"error":{"code":"1311","message":"The current plan does not include this model"}})",
        {},
    };
    general.on_response(model_not_in_plan);
    CHECK_FALSE(general.is_retryable(model_not_in_plan));
    CHECK_FALSE(general.last_rate_limit().is_rate_limited);

    const HttpResponse resettable_limit{
        429,
        R"({"error":{"code":"1316","message":"Usage limit reached for the past 5 hours"}})",
        {},
    };
    general.on_response(resettable_limit);
    CHECK_FALSE(general.is_retryable(resettable_limit));
    CHECK(general.last_rate_limit().is_rate_limited);
}

TEST_CASE("Z.ai protocol does not invent quota state for unstructured 429 responses",
          "[zai][protocol][errors]") {
    ZaiProtocol protocol;
    const HttpResponse response{429, "upstream rejected request", {}};

    protocol.on_response(response);

    CHECK_FALSE(protocol.is_retryable(response));
    CHECK_FALSE(protocol.last_rate_limit().is_rate_limited);
    CHECK(protocol.format_error_message(response)
          == "[HTTP Error: 429 - upstream rejected request]");
}

TEST_CASE("Z.ai protocol parses unified 5h and 7d utilization headers",
          "[zai][rate_limit]") {
    ZaiProtocol protocol;
    cpr::Header headers{
        {"X-RateLimit-Limit-Requests", "100"},
        {"X-RateLimit-Remaining-Requests", "88"},
        {"X-RateLimit-Unified-5h-Utilization", "0.42"},
        {"X-RateLimit-Unified-7d-Utilization", "17"},
        {"X-RateLimit-Unified-Status", "allowed"},
        {"X-RateLimit-Unified-Representative-Claim", "five_hour"},
    };

    protocol.on_response(HttpResponse{200, "{}", headers});
    const auto info = protocol.last_rate_limit();

    REQUIRE(info.requests_limit == 100);
    REQUIRE(info.requests_remaining == 88);
    REQUIRE(info.usage_windows.size() == 2);
    REQUIRE(info.usage_windows[0].label == "5h");
    REQUIRE(info.usage_windows[0].utilization == 0.42f);
    REQUIRE(info.usage_windows[1].label == "7d");
    REQUIRE(info.usage_windows[1].utilization == 0.17f);
    REQUIRE(info.unified_status == "allowed");
    REQUIRE(info.unified_representative_claim == "five_hour");
}

TEST_CASE("Z.ai Coding Plan protocol enables Z.ai thinking config",
          "[zai][protocol][coding]") {
    ZaiCodingProtocol protocol;

    ChatRequest request;
    request.model = "glm-5.2";
    request.messages.push_back(Message{.role = "user", .content = "hi"});

    const std::string payload = protocol.serialize(request);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("thinking":{"type":"enabled","clear_thinking":false})"));
}

TEST_CASE("ModelRegistry includes Z.ai GLM coding models", "[zai][registry]") {
    auto& registry = ModelRegistry::instance();

    REQUIRE(registry.has_model("glm-5.3"));
    REQUIRE(registry.has_model("glm-5.2"));
    REQUIRE(registry.has_model("glm-5-turbo"));
    REQUIRE(registry.has_model("glm-4.7"));
    REQUIRE(registry.has_model("glm-4.5-air"));
    REQUIRE(get_max_context_size("glm-5.3") == 1000000);
    REQUIRE(registry.get_info("glm-5.3")->max_output_tokens == 128000);
    REQUIRE(get_max_context_size("glm-5.2") == 1000000);
    REQUIRE(registry.get_info("glm-5.2")->max_output_tokens == 128000);
    REQUIRE(get_max_context_size("glm-5-turbo") == 200000);
    REQUIRE(get_max_context_size("glm-4.7") == 200000);
    REQUIRE(get_max_context_size("glm-4.5-air") == 200000);
}

TEST_CASE("Z.ai Coding Plan protocol accepts future GLM models and rejects other families",
          "[zai][validation][coding]") {
    ZaiCodingProtocol protocol;

    ChatRequest valid;
    valid.model = "GLM-4.7";
    valid.messages.push_back(Message{.role = "user", .content = "hi"});
    REQUIRE_NOTHROW(protocol.prepare_request(valid));

    ChatRequest future;
    future.model = "glm-future-live";
    future.messages.push_back(Message{.role = "user", .content = "hi"});
    REQUIRE_NOTHROW(protocol.prepare_request(future));

    ChatRequest invalid;
    invalid.model = "gpt-5";
    invalid.messages.push_back(Message{.role = "user", .content = "hi"});
    REQUIRE_THROWS_WITH(
        protocol.prepare_request(invalid),
        Catch::Matchers::ContainsSubstring("requires a GLM model"));
}

TEST_CASE("Z.ai subscription list payload exposes the subscription end date",
          "[zai][subscription][rate_limit]") {
    SECTION("ISO 8601 valid-until string") {
        CHECK(parse_zai_subscription_end(R"JSON({
          "data": [{
            "productName": "GLM Coding Max",
            "status": "active",
            "purchaseTime": "2026-08-01T10:00:00Z",
            "valid": "2026-08-17T23:59:59Z"
          }]
        })JSON") == 1787011199LL);
    }

    SECTION("Space-separated datetime is tolerated") {
        CHECK(parse_zai_subscription_end(R"JSON({
          "data": [{"valid": "2026-08-17 23:59:59"}]
        })JSON") == 1787011199LL);
    }

    SECTION("Epoch milliseconds are converted to seconds") {
        CHECK(parse_zai_subscription_end(R"JSON({
          "data": [{"valid": 1787011199000}]
        })JSON") == 1787011199LL);
    }

    SECTION("Alternative expiry keys") {
        CHECK(parse_zai_subscription_end(R"JSON({
          "data": [{"expireTime": "2026-09-30T23:59:59Z"}]
        })JSON") == 1790812799LL);
    }

    SECTION("No parseable end date yields zero") {
        CHECK(parse_zai_subscription_end("not-json") == 0);
        CHECK(parse_zai_subscription_end(R"({"data": []})") == 0);
        CHECK(parse_zai_subscription_end(
            R"({"data": [{"productName": "GLM Coding Max", "status": "active"}]})") == 0);
        CHECK(parse_zai_subscription_end(R"({"data": [{"valid": "true"}]})") == 0);
    }
}
