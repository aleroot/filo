#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/config/ConfigManager.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"

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

TEST_CASE("Z.ai protocol parses streamed reasoning_content",
          "[zai][protocol][thinking]") {
    ZaiProtocol protocol;

    const auto result = protocol.parse_event(
        R"(data: {"choices":[{"delta":{"reasoning_content":"think","content":"answer"},"index":0}]})");

    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].reasoning_content == "think");
    REQUIRE(result.chunks[0].content == "answer");
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

    REQUIRE(registry.has_model("glm-5.2"));
    REQUIRE(registry.has_model("glm-5-turbo"));
    REQUIRE(registry.has_model("glm-4.7"));
    REQUIRE(registry.has_model("glm-4.5-air"));
    REQUIRE(get_max_context_size("glm-5.2") == 1000000);
    REQUIRE(registry.get_info("glm-5.2")->max_output_tokens == 128000);
    REQUIRE(get_max_context_size("glm-5-turbo") == 200000);
    REQUIRE(get_max_context_size("glm-4.7") == 200000);
    REQUIRE(get_max_context_size("glm-4.5-air") == 200000);
}

TEST_CASE("Z.ai Coding Plan protocol rejects non-plan models before HTTP",
          "[zai][validation][coding]") {
    ZaiCodingProtocol protocol;

    ChatRequest valid;
    valid.model = "GLM-4.7";
    valid.messages.push_back(Message{.role = "user", .content = "hi"});
    REQUIRE_NOTHROW(protocol.prepare_request(valid));

    ChatRequest invalid;
    invalid.model = "glm-5.1";
    invalid.messages.push_back(Message{.role = "user", .content = "hi"});
    REQUIRE_THROWS_WITH(
        protocol.prepare_request(invalid),
        Catch::Matchers::ContainsSubstring("supports only glm-5.2"));
}
