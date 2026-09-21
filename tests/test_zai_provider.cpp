#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/config/ConfigManager.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/ZaiProtocol.hpp"
#include "core/auth/ICredentialSource.hpp"

#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <format>
#include <optional>
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

    const HttpResponse concurrent_throttle{
        429,
        R"({"error":{"code":"1303","message":"Rate limit reached"}})",
        {},
    };
    CHECK(general.is_retryable(concurrent_throttle));

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

TEST_CASE("Z.ai Coding Plan protocol uses Anthropic Messages thinking",
          "[zai][protocol][coding]") {
    ZaiCodingProtocol protocol;

    ChatRequest request;
    request.model = "glm-5.2";
    request.messages.push_back(Message{.role = "user", .content = "hi"});

    const std::string payload = protocol.serialize(request);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("thinking":{"type":"enabled"})"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("clear_thinking"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("budget_tokens"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("reasoning_effort"));
    REQUIRE(protocol.build_url("https://api.z.ai/api/anthropic", "glm-5.3")
            == "https://api.z.ai/api/anthropic/v1/messages");
}

TEST_CASE("ModelRegistry includes Z.ai GLM coding models", "[zai][registry]") {
    auto& registry = ModelRegistry::instance();

    REQUIRE(registry.has_model("glm-5.3"));
    REQUIRE(registry.has_model("glm-5.3-flash"));
    REQUIRE(registry.has_model("glm-5.2"));
    REQUIRE(registry.has_model("glm-5-turbo"));
    REQUIRE(registry.has_model("glm-4.7"));
    REQUIRE(registry.has_model("glm-4.5-air"));
    REQUIRE(get_max_context_size("glm-5.3") == 1000000);
    REQUIRE(registry.get_info("glm-5.3")->max_output_tokens == 128000);
    REQUIRE(get_max_context_size("glm-5.3-flash") == 1000000);
    REQUIRE(registry.get_info("glm-5.3-flash")->max_output_tokens == 128000);
    REQUIRE(registry.supports("glm-5.3-flash", ModelCapability::Vision));
    REQUIRE(registry.supports("glm-5.3-flash", ModelCapability::VideoInput));
    REQUIRE(registry.supports("glm-5.3-flash", ModelCapability::PdfInput));
    REQUIRE_FALSE(registry.supports("glm-5.3", ModelCapability::Vision));
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

TEST_CASE("Z.ai Coding Plan protocol maps GLM-5.3 effort on Anthropic wire",
          "[zai][protocol][coding][effort]") {
    ZaiCodingProtocol protocol;
    ChatRequest request;
    request.model = "glm-5.3";
    request.effort = "medium";
    request.messages.push_back(Message{.role = "user", .content = "hi"});

    const std::string payload = protocol.serialize(request);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("thinking":{"type":"enabled"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("output_config":{"effort":"high"})"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("budget_tokens"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("claude-code"));
}

TEST_CASE("Z.ai Coding Plan headers omit Claude-only betas",
          "[zai][protocol][coding][headers]") {
    ZaiCodingProtocol protocol;
    core::auth::AuthInfo auth;
    auth.headers["x-api-key"] = "test-key";
    const auto headers = protocol.build_headers(auth);
    REQUIRE(headers.at("anthropic-version") == "2023-06-01");
    REQUIRE(headers.at("x-api-key") == "test-key");
    REQUIRE(headers.find("anthropic-beta") == headers.end());
    REQUIRE(headers.find("x-anthropic-billing-header") == headers.end());
}

TEST_CASE("Z.ai protocols wait 10 minutes for long GLM thinking",
          "[zai][protocol][timeouts]") {
    using namespace std::chrono_literals;
    CHECK(ZaiProtocol{}.stream_timeouts().response_start == 180s);
    CHECK(ZaiProtocol{}.stream_timeouts().inactivity == 600s);
    CHECK(ZaiCodingProtocol{}.stream_timeouts().inactivity == 600s);
}

TEST_CASE("Z.ai Coding Plan factory uses subscription billing",
          "[zai][factory][coding][flash]") {
    core::config::ProviderConfig config;
    config.model = "glm-5.3-flash";
    config.api_key = "coding-key";

    auto provider = core::llm::ProviderFactory::create_provider("zai-coding", config);
    REQUIRE(provider != nullptr);
    REQUIRE_FALSE(provider->should_estimate_cost());
    REQUIRE(provider->max_context_size() == 1000000);
    REQUIRE(provider->metadata()->api_type == core::config::ApiType::Anthropic);
    REQUIRE(provider->metadata()->base_url == "https://api.z.ai/api/anthropic");
}

// ─────────────────────────────────────────────────────────────────────────────
// Thinking-replay hygiene (ZCode parity): cross-model guard, signature
// rejection repair, context-overflow guidance, and video blocks for flash.
// ─────────────────────────────────────────────────────────────────────────────

namespace {

Message make_assistant_with_continuation(ContinuationItem item) {
    Message assistant{.role = "assistant"};
    assistant.continuation_items.push_back(std::move(item));
    return assistant;
}

ChatRequest make_replay_request(std::string model, Message assistant) {
    ChatRequest request;
    request.model = std::move(model);
    request.messages.push_back(Message{.role = "user", .content = "question"});
    request.messages.push_back(std::move(assistant));
    request.messages.push_back(Message{.role = "user", .content = "continue"});
    return request;
}

} // namespace

TEST_CASE("Z.ai Coding Plan stamps thinking continuations with the producing model",
          "[zai][protocol][coding][thinking]") {
    zai_clear_signature_rejection_cache_for_testing();
    ZaiCodingProtocol protocol;

    ChatRequest bound;
    bound.model = "glm-5.3";
    protocol.prepare_request(bound);

    protocol.parse_event(
        "event: content_block_start\n"
        "data: {\"type\":\"content_block_start\",\"index\":0,"
        "\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n");
    protocol.parse_event(
        "event: content_block_delta\n"
        "data: {\"type\":\"content_block_delta\",\"index\":0,"
        "\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"step one\"}}\n\n");
    protocol.parse_event(
        "event: content_block_delta\n"
        "data: {\"type\":\"content_block_delta\",\"index\":0,"
        "\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig-glm53\"}}\n\n");
    const auto stopped = protocol.parse_event(
        "event: content_block_stop\n"
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n");

    std::optional<ContinuationItem> stamped;
    for (const auto& chunk : stopped.chunks) {
        for (const auto& item : chunk.continuation_items) stamped = item;
    }
    REQUIRE(stamped.has_value());
    CHECK(stamped->provider == "glm-5.3");
    CHECK_THAT(stamped->payload, Catch::Matchers::ContainsSubstring("sig-glm53"));

    // Same model: signed thinking is replayed.
    ChatRequest same_model =
        make_replay_request("glm-5.3", make_assistant_with_continuation(*stamped));
    const std::string same_payload = protocol.serialize(same_model);
    CHECK_THAT(same_payload, Catch::Matchers::ContainsSubstring(R"("signature":"sig-glm53")"));

    // Different model: signed thinking is refused; the reasoning-only
    // assistant turn is repaired with the fallback text.
    ChatRequest switched =
        make_replay_request("glm-5.2", make_assistant_with_continuation(*stamped));
    const std::string switched_payload = protocol.serialize(switched);
    CHECK_THAT(switched_payload, !Catch::Matchers::ContainsSubstring("sig-glm53"));
    CHECK_THAT(switched_payload, Catch::Matchers::ContainsSubstring("[Thinking removed]"));
}

TEST_CASE("Z.ai Coding Plan repairs history after a thinking-signature rejection",
          "[zai][protocol][coding][thinking][repair]") {
    zai_clear_signature_rejection_cache_for_testing();
    ZaiCodingProtocol protocol;

    ChatRequest bound;
    bound.model = "glm-5.3";
    protocol.prepare_request(bound);

    Message assistant{.role = "assistant"};
    assistant.continuation_items.push_back(ContinuationItem{
        .provider = "glm-5.3",
        .kind = "thinking",
        .payload = R"({"type":"thinking","thinking":"stale","signature":"old-sig"})",
    });
    assistant.continuation_items.push_back(ContinuationItem{
        .provider = "glm-5.3",
        .kind = "thinking",
        .payload = R"({"type":"thinking","thinking":"fresh","signature":""})",
    });
    const ChatRequest request = make_replay_request("glm-5.3", std::move(assistant));

    const std::string initial = protocol.serialize(request);
    CHECK_THAT(initial, Catch::Matchers::ContainsSubstring("old-sig"));
    CHECK_THAT(initial, Catch::Matchers::ContainsSubstring(R"("signature":"")"));

    // The backend rejects replayed signatures with the narrow 400 wording.
    // The failure stays terminal — no blind retry of an unrepaired payload.
    const HttpResponse rejection{
        400,
        R"({"error":{"type":"invalid_request_error",)"
        R"("message":"Messages: the signature in thinking block cannot be modified"}})",
        {},
    };
    CHECK_FALSE(protocol.is_retryable(rejection));
    protocol.on_response(rejection);
    CHECK_THAT(
        protocol.format_error_message(rejection),
        Catch::Matchers::ContainsSubstring("thinking"));

    const std::string repaired = protocol.serialize(request);
    CHECK_THAT(repaired, !Catch::Matchers::ContainsSubstring("old-sig"));
    CHECK_THAT(repaired, Catch::Matchers::ContainsSubstring("fresh"));
    CHECK_THAT(repaired, Catch::Matchers::ContainsSubstring(R"("signature":"")"));
    CHECK_THAT(repaired, !Catch::Matchers::ContainsSubstring("[Thinking removed]"));

    // One-shot: the next attempt sends the original history again.
    const std::string again = protocol.serialize(request);
    CHECK_THAT(again, Catch::Matchers::ContainsSubstring("old-sig"));

    // Unrelated 400s must not arm the repair.
    protocol.on_response(HttpResponse{
        400,
        R"({"error":{"message":"invalid request: unknown field"}})",
        {},
    });
    const std::string unaffected = protocol.serialize(request);
    CHECK_THAT(unaffected, Catch::Matchers::ContainsSubstring("old-sig"));
}

TEST_CASE("Z.ai Coding Plan emits video blocks for GLM-5.3-Flash",
          "[zai][protocol][coding][media]") {
    ZaiCodingProtocol protocol;

    const auto video_path =
        std::filesystem::temp_directory_path() / "filo_zai_test_video.mp4";
    {
        std::ofstream out(video_path, std::ios::binary);
        out << "VIDEOBYTES";
    }

    Message flash_user{.role = "user"};
    flash_user.content_parts.push_back(
        ContentPart::make_video(video_path.string(), "video/mp4"));

    ChatRequest flash;
    flash.model = "glm-5.3-flash";
    flash.messages.push_back(std::move(flash_user));
    const std::string flash_payload = protocol.serialize(flash);
    CHECK_THAT(flash_payload, Catch::Matchers::ContainsSubstring(R"("type":"video")"));
    CHECK_THAT(flash_payload, Catch::Matchers::ContainsSubstring("video/mp4"));
    CHECK_THAT(flash_payload,
               !Catch::Matchers::ContainsSubstring("[Attached video unavailable:"));

    Message base_user{.role = "user"};
    base_user.content_parts.push_back(
        ContentPart::make_video(video_path.string(), "video/mp4"));
    ChatRequest base;
    base.model = "glm-5.3";
    base.messages.push_back(std::move(base_user));
    const std::string base_payload = protocol.serialize(base);
    CHECK_THAT(base_payload,
               Catch::Matchers::ContainsSubstring("[Attached video unavailable:"));
    CHECK_THAT(base_payload, !Catch::Matchers::ContainsSubstring(R"("type":"video")"));

    std::error_code cleanup_ec;
    std::filesystem::remove(video_path, cleanup_ec);
}

TEST_CASE("Z.ai context-overflow errors carry compaction guidance and stay terminal",
          "[zai][protocol][coding][errors]") {
    ZaiCodingProtocol coding;
    const HttpResponse by_code{
        400,
        R"({"error":{"code":1261,"message":"context window exceeded"}})",
        {},
    };
    const std::string coding_message = coding.format_error_message(by_code);
    CHECK_THAT(coding_message, Catch::Matchers::ContainsSubstring("1261"));
    CHECK_THAT(coding_message, Catch::Matchers::ContainsSubstring("/compact"));
    CHECK_FALSE(coding.is_retryable(by_code));

    const HttpResponse by_wording{
        400,
        R"({"error":{"message":"Your prompt is too long for this model"}})",
        {},
    };
    CHECK_THAT(coding.format_error_message(by_wording),
               Catch::Matchers::ContainsSubstring("/compact"));

    ZaiProtocol general;
    CHECK_FALSE(general.is_retryable(by_code));
    CHECK_THAT(general.format_error_message(by_code),
               Catch::Matchers::ContainsSubstring("/compact"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Coding Plan request attribution (ZCode parity): dual auth, attribution
// headers, and body metadata.user_id.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Z.ai Coding Plan sends x-api-key and Bearer authorization together",
          "[zai][protocol][coding][auth]") {
    ZaiCodingProtocol protocol;

    core::auth::AuthInfo auth;
    auth.headers["x-api-key"] = "sk-zai-key";

    const cpr::Header headers = protocol.build_headers(auth);
    REQUIRE(headers.find("x-api-key") != headers.end());
    const auto authorization = headers.find("Authorization");
    REQUIRE(authorization != headers.end());
    CHECK(authorization->second == "Bearer sk-zai-key");

    // An explicitly configured Authorization is never overwritten.
    core::auth::AuthInfo custom_auth;
    custom_auth.headers["x-api-key"] = "sk-zai-key";
    custom_auth.headers["Authorization"] = "Bearer custom";
    const cpr::Header custom_headers = protocol.build_headers(custom_auth);
    CHECK(custom_headers.find("Authorization")->second == "Bearer custom");

    // No API key: no synthesized Authorization.
    const cpr::Header empty_headers = protocol.build_headers(core::auth::AuthInfo{});
    CHECK(empty_headers.find("Authorization") == empty_headers.end());
}

TEST_CASE("Z.ai Coding Plan stamps Coding Plan attribution headers",
          "[zai][protocol][coding][attribution]") {
    ZaiCodingProtocol protocol;

    ChatRequest request;
    request.model = "glm-5.3";
    request.session_id = "sess_abc123";
    request.transport_turn_id = "sess_abc123:7";
    request.messages.push_back(Message{.role = "user", .content = "hi"});

    cpr::Header headers;
    protocol.prepare_headers(headers, request, "https://api.z.ai/api/anthropic");

    CHECK(headers.find("x-zcode-session-type")->second == "main");
    CHECK_FALSE(headers.find("x-request-id")->second.empty());
    CHECK(headers.find("x-zcode-trace-id")->second == "sess_abc123:7");
    // Internal storage prefixes are stripped before the header leaves.
    CHECK(headers.find("x-session-id")->second == "abc123");

    // Without a session scope the header is omitted entirely.
    ChatRequest anonymous;
    anonymous.model = "glm-5.3";
    anonymous.messages.push_back(Message{.role = "user", .content = "hi"});
    cpr::Header anonymous_headers;
    protocol.prepare_headers(anonymous_headers, anonymous, "https://api.z.ai/api/anthropic");
    CHECK(anonymous_headers.find("x-session-id") == anonymous_headers.end());
    CHECK_FALSE(anonymous_headers.find("x-request-id")->second.empty());
}

TEST_CASE("Z.ai Coding Plan carries attribution in metadata.user_id",
          "[zai][protocol][coding][attribution]") {
    ZaiCodingProtocol protocol;

    ChatRequest request;
    request.model = "glm-5.3";
    request.session_id = "sess_abc123";
    request.messages.push_back(Message{.role = "user", .content = "hi"});

    const std::string payload = protocol.serialize(request);
    CHECK_THAT(payload,
               Catch::Matchers::ContainsSubstring(R"("metadata":{"user_id":")"));
    // user_id is a JSON string value, so its inner quotes are escaped on the
    // wire: {"device_id":"…","account_uuid":"","session_id":"abc123"}
    CHECK_THAT(payload,
               Catch::Matchers::ContainsSubstring("\\\"device_id\\\":\\\""));
    CHECK_THAT(payload,
               Catch::Matchers::ContainsSubstring(
                   "\\\"account_uuid\\\":\\\"\\\",\\\"session_id\\\":\\\"abc123\\\"}"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Connection resilience (ZCode parity): retry policy, business-coded stream
// errors, and terminal rate-limit classification.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Z.ai stream retry policy matches ZCode runner defaults",
          "[zai][protocol][resilience]") {
    const ZaiProtocol general;
    const ZaiCodingProtocol coding;

    const auto check_policy = [](const ApiProtocolBase& protocol) {
        const auto policy = protocol.stream_retry_policy();
        CHECK(policy.max_retries == 10);
        CHECK(policy.initial_backoff == std::chrono::seconds(2));
        CHECK(policy.maximum_backoff == std::chrono::seconds(60));
        // ZCode's retry boundary: committed (non-empty) output is never
        // retracted, so retries stop there.
        CHECK(policy.retry_only_before_output);
    };
    check_policy(general);
    check_policy(coding);
}

TEST_CASE("Z.ai Coding Plan classifies business-coded stream errors",
          "[zai][protocol][coding][resilience]") {
    ZaiCodingProtocol protocol;

    // GLM waves of transient upstream failures arrive as bracketed business
    // codes instead of the Claude error-type vocabulary.
    const auto transient = protocol.parse_event(
        "event: error\n"
        "data: {\"type\":\"error\",\"error\":{\"type\":\"upstream_error\","
        "\"message\":\"[1312][upstream busy][req-1]\"}}\n\n");
    REQUIRE(transient.stream_error);
    CHECK(transient.retryable_stream_error);
    CHECK_THAT(transient.stream_error_message,
               Catch::Matchers::ContainsSubstring("1312"));

    // Terminal codes stay terminal — no pointless retry waves against them.
    const auto terminal = protocol.parse_event(
        "event: error\n"
        "data: {\"type\":\"error\",\"error\":{\"type\":\"upstream_error\","
        "\"message\":\"[1308][quota exhausted][req-2]\"}}\n\n");
    REQUIRE(terminal.stream_error);
    CHECK_FALSE(terminal.retryable_stream_error);

    // Claude-style types keep their existing retryable classification.
    const auto claude_style = protocol.parse_event(
        "event: error\n"
        "data: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\","
        "\"message\":\"Overloaded\"}}\n\n");
    REQUIRE(claude_style.stream_error);
    CHECK(claude_style.retryable_stream_error);

    // Unknown errors without a business code remain non-retryable.
    const auto unknown = protocol.parse_event(
        "event: error\n"
        "data: {\"type\":\"error\",\"error\":{\"type\":\"server_error\","
        "\"message\":\"something broke\"}}\n\n");
    REQUIRE(unknown.stream_error);
    CHECK_FALSE(unknown.retryable_stream_error);
}

TEST_CASE("Z.ai general wire surfaces in-stream business error frames",
          "[zai][protocol][resilience]") {
    ZaiProtocol protocol;

    const auto transient = protocol.parse_event(
        "data: {\"error\":{\"code\":1302,\"message\":\""
        "Concurrent stream limit reached\"}}\n\n");
    REQUIRE(transient.stream_error);
    CHECK(transient.retryable_stream_error);
    CHECK(transient.stream_error_type == "1302");

    const auto terminal = protocol.parse_event(
        "data: {\"error\":{\"code\":1308,\"message\":\""
        "Concurrency limit reached\"}}\n\n");
    REQUIRE(terminal.stream_error);
    CHECK_FALSE(terminal.retryable_stream_error);

    // Ordinary completion chunks are never mistaken for error frames.
    const auto chunk = protocol.parse_event(
        "data: {\"id\":\"x\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n");
    CHECK_FALSE(chunk.stream_error);
    REQUIRE_FALSE(chunk.chunks.empty());
}

TEST_CASE("Z.ai terminal rate-limit codes are flagged but never retried",
          "[zai][protocol][resilience]") {
    ZaiCodingProtocol coding;
    const HttpResponse throttled{
        429,
        R"({"error":{"code":1304,"message":"Rate limit reached"}})",
        {},
    };
    CHECK_FALSE(coding.is_retryable(throttled));
    coding.on_response(throttled);
    CHECK(coding.last_rate_limit().is_rate_limited);
}
