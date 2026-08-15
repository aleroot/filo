#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/config/ConfigManager.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/ZaiProtocol.hpp"

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
    CHECK(balance_message.find("Coding endpoint") != std::string::npos);

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
