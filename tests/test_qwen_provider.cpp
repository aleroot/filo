#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/DashScopeProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/QwenModelTraits.hpp"
#include "core/llm/providers/QwenModelCatalogSelector.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/Models.hpp"
#include "core/auth/ApiKeyCredentialSource.hpp"

#include <simdjson.h>
#include <filesystem>
#include <fstream>

using namespace core::llm;
using namespace core::llm::protocols;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

static ChatRequest make_simple_request(std::string model     = "qwen3-coder-plus",
                                       std::string user_text = "Hello") {
    ChatRequest req;
    req.model  = std::move(model);
    req.stream = true;
    req.messages.push_back(Message{.role = "user", .content = std::move(user_text)});
    return req;
}

static core::auth::AuthInfo make_auth(std::string key = "test-dashscope-key") {
    auto src = core::auth::ApiKeyCredentialSource::as_bearer(std::move(key));
    return src->get_auth();
}

// HttpResponse holds a const cpr::Header& — helper keeps the header alive
// for the duration of the call.
struct ScopedResponse {
    int         status_code;
    std::string body_str;
    cpr::Header headers;

    [[nodiscard]] protocols::HttpResponse view() const noexcept {
        return {status_code, body_str, headers};
    }
};

static ScopedResponse make_response(int code, std::string body = "",
                                     cpr::Header hdrs = {}) {
    return {code, std::move(body), std::move(hdrs)};
}

static void require_valid_json(std::string_view payload) {
    simdjson::dom::parser parser;
    simdjson::padded_string json{std::string(payload)};
    simdjson::dom::element document;
    REQUIRE(parser.parse(json).get(document) == simdjson::SUCCESS);
}

static std::filesystem::path make_temp_qwen_image_file() {
    const auto path = std::filesystem::temp_directory_path()
        / "filo-qwen-token-plan-image.png";
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << "fake-image";
    return path;
}

// ─────────────────────────────────────────────────────────────────────────────
// Token Plan quota / rate-limit surfacing
//
// The QwenCloud Token Plan inference endpoints return no rate-limit or usage
// headers, so the only provider-backed quota signal is a 429 body. These tests
// pin the honest behaviour: a 200 leaves the rate-limit snapshot empty, while a
// 429 is translated into a rate-limited snapshot (with a hard-rejection marker
// when the body indicates plan-quota exhaustion) and distinct error wording.
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Token Plan - 200 response leaves rate-limit snapshot empty",
          "[qwen][ratelimit]") {
    DashScopeTokenPlanProtocol protocol;
    const auto resp = make_response(200, R"({"usage":{"prompt_tokens":5}})");
    protocol.on_response(resp.view());

    const auto info = protocol.last_rate_limit();
    REQUIRE(!info.is_rate_limited);
    REQUIRE(info.unified_status.empty());
    REQUIRE(info.usage_windows.empty());
    REQUIRE(info.requests_limit == 0);
    REQUIRE(info.tokens_limit == 0);
    REQUIRE(info.subscription_ends_at == 0);
}

TEST_CASE("Token Plan - quota 429 surfaces rate-limited snapshot",
          "[qwen][ratelimit]") {
    DashScopeTokenPlanProtocol protocol;
    cpr::Header headers{{"retry-after", "30"}};
    const auto resp = make_response(
        429,
        R"({"code":"Throttling.AllocationQuota","message":"Allocated quota exceeded"})",
        headers);
    protocol.on_response(resp.view());

    const auto info = protocol.last_rate_limit();
    REQUIRE(info.is_rate_limited);
    REQUIRE(info.unified_status == "rate_limited");
    REQUIRE(info.unified_overage_status == "rejected");  // hard plan-quota block
    REQUIRE(info.retry_after == 30);
    // No fabricated usage windows: real utilization is unknown.
    REQUIRE(info.usage_windows.empty());
}

TEST_CASE("Token Plan - per-minute 429 surfaces rate-limited without overage",
          "[qwen][ratelimit]") {
    DashScopeTokenPlanProtocol protocol;
    const auto resp = make_response(
        429,
        R"({"error":{"message":"Requests rate limit exceeded"}})");
    protocol.on_response(resp.view());

    const auto info = protocol.last_rate_limit();
    REQUIRE(info.is_rate_limited);
    REQUIRE(info.unified_status == "rate_limited");
    // A transient throttle is not a hard plan-quota rejection.
    REQUIRE(info.unified_overage_status.empty());
}

TEST_CASE("Token Plan - successful response clears prior 429 state",
          "[qwen][ratelimit]") {
    DashScopeTokenPlanProtocol protocol;
    const auto limited = make_response(
        429,
        R"({"code":"Throttling.AllocationQuota","message":"Allocated quota exceeded"})");
    protocol.on_response(limited.view());
    REQUIRE(protocol.last_rate_limit().is_rate_limited);

    const auto success = make_response(200, R"({"usage":{"prompt_tokens":5}})");
    protocol.on_response(success.view());

    const auto info = protocol.last_rate_limit();
    CHECK_FALSE(info.is_rate_limited);
    CHECK(info.unified_status.empty());
    CHECK(info.unified_overage_status.empty());
}

TEST_CASE("Token Plan - Throttling namespace does not imply plan exhaustion",
          "[qwen][ratelimit]") {
    for (const auto body : {
             R"({"code":"Throttling.RateQuota","message":"Requests rate limit exceeded"})",
             R"({"code":"Throttling.BurstRate","message":"Request rate increased too quickly"})",
         }) {
        DashScopeTokenPlanProtocol protocol;
        const auto resp = make_response(429, body);
        protocol.on_response(resp.view());

        const auto info = protocol.last_rate_limit();
        REQUIRE(info.is_rate_limited);
        REQUIRE(info.unified_status == "rate_limited");
        REQUIRE(info.unified_overage_status.empty());
        REQUIRE_THAT(
            protocol.format_error_message(resp.view()),
            Catch::Matchers::ContainsSubstring(
                "Request rate limit exceeded"));
    }
}

TEST_CASE("Token Plan - nested allocation error has consistent state and wording",
          "[qwen][ratelimit]") {
    DashScopeTokenPlanProtocol protocol;
    const auto resp = make_response(
        429,
        R"({"error":{"code":"insufficient_quota","message":"plan unavailable"}})");
    protocol.on_response(resp.view());

    REQUIRE(protocol.last_rate_limit().unified_overage_status == "rejected");
    REQUIRE_THAT(
        protocol.format_error_message(resp.view()),
        Catch::Matchers::ContainsSubstring("Credits quota exceeded"));
}

TEST_CASE("Token Plan - unknown 429 does not invent a recovery cause",
          "[qwen][ratelimit]") {
    DashScopeTokenPlanProtocol protocol;
    const auto resp = make_response(
        429,
        R"({"code":"UnknownThrottle","message":"temporarily unavailable"})");
    protocol.on_response(resp.view());

    REQUIRE(protocol.last_rate_limit().unified_overage_status.empty());
    const std::string message = protocol.format_error_message(resp.view());
    REQUIRE_THAT(
        message,
        Catch::Matchers::ContainsSubstring("Rate limit or quota exceeded"));
    REQUIRE_THAT(
        message,
        Catch::Matchers::ContainsSubstring("Retry with backoff"));
}

TEST_CASE("Token Plan - 429 error message distinguishes quota from rate limit",
          "[qwen][ratelimit]") {
    DashScopeTokenPlanProtocol protocol;

    const auto quota_resp = make_response(
        429,
        R"({"code":"Throttling.AllocationQuota","message":"insufficient_quota"})");
    protocol.on_response(quota_resp.view());
    const std::string quota_msg = protocol.format_error_message(quota_resp.view());
    REQUIRE_THAT(quota_msg, Catch::Matchers::ContainsSubstring("Credits quota exceeded"));
    REQUIRE_THAT(quota_msg, Catch::Matchers::ContainsSubstring("5-hour or 7-day window"));
    REQUIRE_THAT(quota_msg, Catch::Matchers::ContainsSubstring("home.qwencloud.com/token-plan"));

    const auto rpm_resp = make_response(
        429,
        R"({"error":{"message":"Requests rate limit exceeded"}})");
    protocol.on_response(rpm_resp.view());
    const std::string rpm_msg = protocol.format_error_message(rpm_resp.view());
    REQUIRE_THAT(rpm_msg, Catch::Matchers::ContainsSubstring("Request rate limit exceeded"));
    REQUIRE_THAT(rpm_msg, Catch::Matchers::ContainsSubstring("Reduce request frequency"));
}

TEST_CASE("Token Plan - allocation quota SSE errors fail fast",
          "[qwen][sse][errors][ratelimit]") {
    DashScopeTokenPlanProtocol protocol;
    const auto result = protocol.parse_event(
        "event:error\n"
        ":HTTP_STATUS/429\n"
        R"(data:{"request_id":"req-123","code":"Throttling.AllocationQuota","message":"Allocated quota exceeded"})");

    REQUIRE(result.stream_error);
    CHECK_FALSE(result.retryable_stream_error);
    CHECK(result.stream_error_type == "Throttling.AllocationQuota");
    CHECK(result.stream_error_message == "Allocated quota exceeded");
    CHECK(protocol.last_rate_limit().is_rate_limited);
    CHECK(protocol.last_rate_limit().unified_overage_status == "rejected");
}

TEST_CASE("Token Plan - transient SSE throttles remain retryable",
          "[qwen][sse][errors][retry]") {
    DashScopeTokenPlanProtocol protocol;
    const auto result = protocol.parse_event(
        "event: error\n"
        R"(data: {"code":"Throttling.RateQuota","message":"Requests rate limit exceeded"})");

    REQUIRE(result.stream_error);
    CHECK(result.retryable_stream_error);
    CHECK(protocol.last_rate_limit().is_rate_limited);
    CHECK(protocol.last_rate_limit().unified_overage_status.empty());
}

TEST_CASE("Token Plan - error_finish chunks are surfaced as stream errors",
          "[qwen][sse][errors]") {
    DashScopeTokenPlanProtocol protocol;
    const auto result = protocol.parse_event(
        R"(data: {"choices":[{"delta":{"content":"Throttling: TPM limit reached"},"finish_reason":"error_finish","index":0}]})");

    REQUIRE(result.stream_error);
    CHECK(result.retryable_stream_error);
    CHECK(result.stream_error_type == "error_finish");
    CHECK(result.stream_error_message == "Throttling: TPM limit reached");
    CHECK(result.chunks.empty());
}

TEST_CASE("Qwen catalog selector chooses the highest live server generation",
          "[qwen][model-catalog]") {
    const auto selector = core::llm::providers::make_qwen_model_catalog_selector();
    const std::vector<ModelInfo> models{
        ModelInfo{.canonical_id = "qwen3.7-plus"},
        ModelInfo{.canonical_id = "deepseek-v4-pro"},
        ModelInfo{.canonical_id = "qwen3.10-max-preview"},
        ModelInfo{.canonical_id = "qwen3.9-max"},
        ModelInfo{.canonical_id = "qwen-image-2.0"},
    };

    const auto selected = selector->select(models);
    REQUIRE(selected.ok());
    REQUIRE(selected.model == "qwen3.10-max-preview");
}

TEST_CASE("Qwen catalog selector preserves server order within one generation",
          "[qwen][model-catalog]") {
    const auto selector = core::llm::providers::make_qwen_model_catalog_selector();
    const std::vector<ModelInfo> models{
        ModelInfo{.canonical_id = "qwen3.8-plus"},
        ModelInfo{.canonical_id = "qwen3.8-max"},
    };

    const auto selected = selector->select(models);
    REQUIRE(selected.ok());
    REQUIRE(selected.model == "qwen3.8-plus");
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization — basic wire format
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DashScopeProtocol - model is serialized correctly", "[qwen][serializer]") {
    auto payload = DashScopeProtocol{}.serialize(make_simple_request("qwen3-coder-plus"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"qwen3-coder-plus")"));
}

TEST_CASE("DashScopeProtocol - stream:true is emitted", "[qwen][serializer]") {
    auto req = make_simple_request();
    req.stream = true;
    REQUIRE_THAT(DashScopeProtocol{}.serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("stream":true)"));
}

TEST_CASE("DashScopeProtocol - stream:false is emitted", "[qwen][serializer]") {
    auto req = make_simple_request();
    req.stream = false;
    REQUIRE_THAT(DashScopeProtocol{}.serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("stream":false)"));
}

TEST_CASE("DashScopeProtocol - user message appears in payload", "[qwen][serializer]") {
    auto payload = DashScopeProtocol{}.serialize(
        make_simple_request("qwen3-coder-plus", "write a sort function"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("write a sort function"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("role":"user")"));
}

TEST_CASE("DashScopeProtocol - empty messages list produces valid JSON", "[qwen][serializer]") {
    ChatRequest req;
    req.model  = "qwen3-coder-plus";
    req.stream = true;
    REQUIRE_THAT(DashScopeProtocol{}.serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("messages":[])"));
}

TEST_CASE("DashScopeProtocol - temperature included when set", "[qwen][serializer]") {
    auto req = make_simple_request();
    req.temperature = 0.6f;
    REQUIRE_THAT(DashScopeProtocol{}.serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("temperature")"));
}

TEST_CASE("DashScopeProtocol - temperature omitted when not set", "[qwen][serializer]") {
    REQUIRE_THAT(DashScopeProtocol{}.serialize(make_simple_request()),
                 !Catch::Matchers::ContainsSubstring("temperature"));
}

TEST_CASE("DashScopeProtocol - max_tokens included when set", "[qwen][serializer]") {
    auto req = make_simple_request();
    req.max_tokens = 8192;
    REQUIRE_THAT(DashScopeProtocol{}.serialize(req),
                 Catch::Matchers::ContainsSubstring(R"("max_tokens":8192)"));
}

TEST_CASE("DashScopeProtocol - stream_options with include_usage emitted for streaming",
          "[qwen][serializer]") {
    // DashScope protocol enables stream_usage by default (like Kimi).
    auto req = make_simple_request();
    req.stream = true;
    auto payload = DashScopeProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("stream_options")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("include_usage":true)"));
}

TEST_CASE("DashScopeProtocol - stream_options omitted for non-streaming requests",
          "[qwen][serializer]") {
    auto req = make_simple_request();
    req.stream = false;
    REQUIRE_THAT(DashScopeProtocol{}.serialize(req),
                 !Catch::Matchers::ContainsSubstring("stream_options"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization — Qwen3 thinking mode
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DashScopeProtocol - thinking fields absent when budget is zero",
          "[qwen][serializer][thinking]") {
    auto payload = DashScopeProtocol{/*thinking_budget=*/0}.serialize(make_simple_request());
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("enable_thinking"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("thinking_budget"));
}

TEST_CASE("DashScopeProtocol - enable_thinking injected when budget > 0",
          "[qwen][serializer][thinking]") {
    auto payload = DashScopeProtocol{/*thinking_budget=*/4096}.serialize(make_simple_request());
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("enable_thinking":true)"));
}

TEST_CASE("DashScopeProtocol - thinking_budget value injected correctly",
          "[qwen][serializer][thinking]") {
    auto payload = DashScopeProtocol{8192}.serialize(make_simple_request());
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("thinking_budget":8192)"));
}

TEST_CASE("DashScopeProtocol - both thinking fields co-exist in the same payload",
          "[qwen][serializer][thinking]") {
    auto payload = DashScopeProtocol{16000}.serialize(make_simple_request("qwen3-max"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("enable_thinking":true)"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("thinking_budget":16000)"));
    // Sanity: the model is still in there.
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("model":"qwen3-max")"));
}

TEST_CASE("DashScopeProtocol - preserves assistant reasoning across tool turns",
          "[qwen][serializer][thinking][tools]") {
    auto req = make_simple_request("qwen3.7-plus");
    Message assistant;
    assistant.role = "assistant";
    assistant.reasoning_content = "Need to inspect the repository first.";
    assistant.tool_calls.push_back(ToolCall{
        .id = "call_1",
        .function = {.name = "read", .arguments = R"({"path":"README.md"})"},
    });
    req.messages.insert(req.messages.begin() + 1, std::move(assistant));

    const auto payload = DashScopeProtocol(0, "high").serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("reasoning_content":"Need to inspect the repository first.")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("preserve_thinking":true)"));
}

TEST_CASE("DashScopeProtocol - steering after reasoning-only assistant keeps required content",
          "[qwen][serializer][thinking][steering]") {
    auto req = make_simple_request("qwen3.7-plus", "Steer the unfinished turn");
    req.messages.insert(req.messages.begin(), Message{
        .role = "assistant",
        .reasoning_content = "I was planning the next edit.",
        .reasoning_protocol = "dashscope",
    });

    const auto payload = DashScopeProtocol(0, "high").serialize(req);
    require_valid_json(payload);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("role":"assistant","content":"","reasoning_content":"I was planning the next edit.")"));
}

TEST_CASE("DashScopeProtocol - tool-call-only assistant keeps nullable content field",
          "[qwen][serializer][tools]") {
    auto req = make_simple_request("qwen3.7-plus");
    Message assistant;
    assistant.role = "assistant";
    assistant.tool_calls.push_back(ToolCall{
        .id = "call_1",
        .function = {.name = "read", .arguments = R"({"path":"README.md"})"},
    });
    req.messages.insert(req.messages.begin(), std::move(assistant));

    const auto payload = DashScopeProtocol(0, "high").serialize(req);
    require_valid_json(payload);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("role":"assistant","content":null,"tool_calls")"));
}

TEST_CASE("DashScopeProtocol - does not replay foreign reasoning state",
          "[qwen][serializer][thinking][provenance]") {
    auto req = make_simple_request("qwen3.7-plus");
    req.messages.insert(req.messages.begin() + 1, Message{
        .role = "assistant",
        .content = "Prior answer.",
        .reasoning_content = "Kimi reasoning",
        .reasoning_protocol = "kimi",
    });

    const auto payload = DashScopeProtocol(0, "high").serialize(req);
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("Kimi reasoning"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("reasoning_content"));
}

TEST_CASE("DashScopeProtocol - effort can disable hybrid thinking",
          "[qwen][serializer][thinking][effort]") {
    auto req = make_simple_request("qwen3.7-plus");
    req.effort = "none";
    const auto payload = DashScopeProtocol(8192, "high").serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("enable_thinking":false)"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("thinking_budget"));
}

TEST_CASE("DashScopeProtocol - qwen3.8 sends the native effort tier alone",
          "[qwen][serializer][thinking][effort]") {
    auto req = make_simple_request("qwen3.8-max");
    req.effort = "max";

    const auto payload = DashScopeProtocol(8192, "high").serialize(req);
    require_valid_json(payload);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("reasoning_effort":"xhigh")"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("enable_thinking"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("thinking_budget"));
}

TEST_CASE("DashScopeProtocol - qwen3.8 honors explicit thinking disable",
          "[qwen][serializer][thinking][effort]") {
    auto req = make_simple_request("qwen3.8-max");
    req.effort = "none";

    const auto payload = DashScopeProtocol(8192, "high").serialize(req);
    require_valid_json(payload);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("reasoning_effort":"none")"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("enable_thinking"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("thinking_budget"));
}

TEST_CASE("Token Plan normalizes Qwen 3.8 effort for aliases and Flash",
          "[qwen][serializer][thinking][effort]") {
    for (const auto model : {"qwen3.8-max", "qwen3.8-max-preview",
                             "qwen3.8-max-0902", "qwen3.8-flash"}) {
        for (const auto& [configured, expected] :
             {std::pair{"minimal", "low"}, {"low", "low"}, {"medium", "medium"},
              {"high", "xhigh"}, {"xhigh", "xhigh"}, {"max", "xhigh"},
              {" MAX ", "xhigh"}}) {
            CAPTURE(model, configured);
            auto request = make_simple_request(model);
            request.effort = configured;
            const auto payload = DashScopeTokenPlanProtocol{}.serialize(request);
            REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
                std::string(R"("reasoning_effort":")") + expected + '"'));
            CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring("enable_thinking"));
            CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring("thinking_budget"));
        }
    }
    auto flash = make_simple_request("qwen3.8-flash");
    flash.effort = "none";
    const auto payload = DashScopeTokenPlanProtocol{}.serialize(flash);
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(R"("enable_thinking":false)"));
    CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring("reasoning_effort"));
}

TEST_CASE("Token Plan preserves a Qwen 3.8 budget without competing effort",
          "[qwen][serializer][thinking][effort]") {
    DashScopeTokenPlanProtocol protocol({.thinking_budget = 4096});
    auto request = make_simple_request("qwen3.8-max");
    const auto payload = protocol.serialize(request);
    CHECK_THAT(payload, Catch::Matchers::ContainsSubstring(R"("thinking_budget":4096)"));
    CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring("reasoning_effort"));
    request.effort = "low";
    const auto override_payload = protocol.serialize(request);
    CHECK_THAT(override_payload, Catch::Matchers::ContainsSubstring(R"("reasoning_effort":"low")"));
    CHECK_THAT(override_payload, !Catch::Matchers::ContainsSubstring("thinking_budget"));
}

TEST_CASE("DashScopeProtocol - streaming requests carry cache anchors and metadata",
          "[qwen][serializer][cache][metadata]") {
    ChatRequest req;
    req.model = "qwen3.7-plus";
    req.stream = true;
    req.session_id = "session-42";
    req.transport_turn_id = "prompt-7";
    req.messages = {
        Message{.role = "system", .content = "Stable instructions"},
        Message{.role = "user", .content = "Latest prompt"},
    };
    Tool first_tool;
    first_tool.function.name = "read";
    first_tool.function.description = "Read a file";
    Tool last_tool;
    last_tool.function.name = "run_command";
    last_tool.function.description = "Run a command";
    req.tools = {std::move(first_tool), std::move(last_tool)};

    const auto payload = DashScopeTokenPlanProtocol{}.serialize(req);
    require_valid_json(payload);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("metadata":{"sessionId":"session-42","promptId":"prompt-7"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("text":"Stable instructions","cache_control":{"type":"ephemeral"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("text":"Latest prompt","cache_control":{"type":"ephemeral"})"));
    const auto first_tool_position = payload.find(R"("name":"read")");
    const auto last_tool_position = payload.find(R"("name":"run_command")");
    const auto tool_cache_position = payload.find(
        R"("cache_control":{"type":"ephemeral"})", last_tool_position);
    const auto messages_position = payload.find(R"("messages")");
    REQUIRE(first_tool_position != std::string::npos);
    REQUIRE(last_tool_position != std::string::npos);
    REQUIRE(tool_cache_position != std::string::npos);
    CHECK(first_tool_position < last_tool_position);
    CHECK(last_tool_position < tool_cache_position);
    CHECK(tool_cache_position < messages_position);
}

TEST_CASE("DashScopeProtocol - non-streaming cache control is system-only",
          "[qwen][serializer][cache]") {
    ChatRequest req;
    req.model = "qwen3.7-plus";
    req.stream = false;
    req.messages = {
        Message{.role = "system", .content = "Stable instructions"},
        Message{.role = "user", .content = "Latest prompt"},
    };

    const auto payload = DashScopeTokenPlanProtocol{}.serialize(req);
    require_valid_json(payload);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("text":"Stable instructions","cache_control":{"type":"ephemeral"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("content":"Latest prompt")"));
}

TEST_CASE("DashScopeProtocol - omits Qwen thinking fields for third-party models",
          "[qwen][serializer][thinking][token-plan]") {
    auto req = make_simple_request("glm-5.2");
    req.effort = "high";
    const auto payload = DashScopeProtocol(8192, "high").serialize(req);
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("enable_thinking"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("thinking_budget"));
}

TEST_CASE("DashScope Responses - Token Plan payload enables native features",
          "[qwen][responses][token-plan]") {
    auto req = make_simple_request("qwen3.8-max");
    req.prompt_cache_key = "filo-session";
    req.effort = "max";

    const auto payload = DashScopeResponsesProtocol({
        .default_effort = "high",
        .enable_hosted_tools = true,
    }).serialize(req);
    require_valid_json(payload);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("reasoning":{"effort":"xhigh"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"({"type":"web_search"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"({"type":"code_interpreter"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"({"type":"web_extractor"})"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("store")"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("prompt_cache_key"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("include":[])"));
}

TEST_CASE("DashScope Responses retains its hosted/local tool serialization policy",
          "[qwen][responses][token-plan][isolation]") {
    auto req = make_simple_request("qwen3.8-max");
    Tool local_tool;
    local_tool.function.name = "read";
    local_tool.function.description = "Read a file";
    req.tools.push_back(std::move(local_tool));

    const auto payload = DashScopeResponsesProtocol({
        .enable_hosted_tools = true,
    }).serialize(req);

    // Grok's collision policy must not reorder DashScope's established payload.
    const auto hosted_pos = payload.find(R"({"type":"web_search"})");
    const auto local_pos = payload.find(R"("name":"read")");
    REQUIRE(hosted_pos != std::string::npos);
    REQUIRE(local_pos != std::string::npos);
    CHECK(hosted_pos < local_pos);
}

TEST_CASE("DashScope Responses - Token Plan omits Qwen-only features for GLM",
          "[qwen][responses][token-plan][glm]") {
    auto req = make_simple_request("glm-5.2");
    req.effort = "high";
    Tool local_tool;
    local_tool.function.name = "read";
    local_tool.function.description = "Read a file";
    local_tool.function.input_schema =
        R"({"type":"object","properties":{"path":{"type":"string"}},"required":["path"]})";
    req.tools.push_back(std::move(local_tool));

    const auto payload = DashScopeResponsesProtocol({
        .default_effort = "high",
        .enable_hosted_tools = true,
    }).serialize(req);

    require_valid_json(payload);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("model":"glm-5.2")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("name":"read")"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(
        R"("reasoning":)"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(
        R"({"type":"web_search"})"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(
        R"({"type":"code_interpreter"})"));
}

TEST_CASE("Qwen Token Plan hosted tools are selected by model generation",
          "[qwen][responses][token-plan][traits]") {
    CHECK(core::llm::qwen_model_supports_token_plan_hosted_tools(
        "qwen3.8-max"));
    CHECK(core::llm::qwen_model_supports_token_plan_hosted_tools(
        "qwen3.7-plus"));
    CHECK(core::llm::qwen_model_supports_token_plan_hosted_tools(
        "qwen3.6-flash"));
    CHECK_FALSE(core::llm::qwen_model_supports_token_plan_hosted_tools(
        "qwen3.5-plus"));
    CHECK_FALSE(core::llm::qwen_model_supports_token_plan_hosted_tools(
        "glm-5.2"));
    CHECK_FALSE(core::llm::qwen_model_supports_token_plan_hosted_tools(
        "deepseek-v4-pro"));
}

TEST_CASE("DashScope Responses - sends only incremental messages with previous response",
          "[qwen][responses][continuity]") {
    ChatRequest req;
    req.model = "qwen3.7-plus";
    req.stream = true;
    req.previous_response_id = "resp_previous";
    req.messages = {
        Message{.role = "system", .content = "Always be concise."},
        Message{.role = "user", .content = "old question"},
        Message{.role = "assistant", .content = "old answer"},
        Message{.role = "user", .content = "new question"},
    };

    const auto payload = DashScopeResponsesProtocol({
        .default_effort = "high",
        .enable_hosted_tools = false,
    }).serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("previous_response_id":"resp_previous")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("instructions":"Always be concise.")"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(
        R"("role":"system")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("new question"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("old question"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("old answer"));
}

TEST_CASE("DashScope Responses - session cache headers are enabled",
          "[qwen][responses][headers]") {
    const auto headers = DashScopeResponsesProtocol{}.build_headers(make_auth("sk-sp-test"));
    REQUIRE(headers.at("X-DashScope-Session-Cache") == "enable");
    REQUIRE(headers.at("Authorization") == "Bearer sk-sp-test");
}

// ─────────────────────────────────────────────────────────────────────────────
// Headers — DashScope-specific fields
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DashScopeProtocol - build_headers includes X-DashScope-CacheControl",
          "[qwen][headers]") {
    DashScopeProtocol proto;
    auto headers = proto.build_headers(make_auth());
    REQUIRE(headers.count("X-DashScope-CacheControl") == 1);
    REQUIRE(headers.at("X-DashScope-CacheControl") == "enable");
}

TEST_CASE("DashScopeProtocol - build_headers includes X-DashScope-UserAgent",
          "[qwen][headers]") {
    DashScopeProtocol proto;
    auto headers = proto.build_headers(make_auth());
    REQUIRE(headers.count("X-DashScope-UserAgent") == 1);
    // Value must be non-empty; exact string may change with version bumps.
    REQUIRE(!headers.at("X-DashScope-UserAgent").empty());
}

TEST_CASE("DashScopeProtocol - build_headers includes Content-Type and Accept",
          "[qwen][headers]") {
    DashScopeProtocol proto;
    auto headers = proto.build_headers(make_auth());
    REQUIRE(headers.count("Content-Type") == 1);
    REQUIRE(headers.count("Accept") == 1);
    REQUIRE_THAT(headers.at("Content-Type"),
                 Catch::Matchers::ContainsSubstring("application/json"));
}

TEST_CASE("DashScopeProtocol - build_headers includes Authorization bearer token",
          "[qwen][headers]") {
    DashScopeProtocol proto;
    auto headers = proto.build_headers(make_auth("sk-mykey"));
    REQUIRE(headers.count("Authorization") == 1);
    REQUIRE_THAT(headers.at("Authorization"),
                 Catch::Matchers::ContainsSubstring("Bearer"));
    REQUIRE_THAT(headers.at("Authorization"),
                 Catch::Matchers::ContainsSubstring("sk-mykey"));
}

// ─────────────────────────────────────────────────────────────────────────────
// URL construction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DashScopeProtocol - build_url appends /chat/completions", "[qwen][url]") {
    DashScopeProtocol proto;
    const std::string base = "https://dashscope.aliyuncs.com/compatible-mode/v1";
    REQUIRE(proto.build_url(base, "qwen3-coder-plus") == base + "/chat/completions");
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE parsing — standard content (no thinking)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DashScopeProtocol - parse_event extracts content chunk", "[qwen][sse]") {
    DashScopeProtocol proto;
    const std::string event =
        R"(data: {"choices":[{"delta":{"content":"Hello world"},"index":0}]})";
    auto result = proto.parse_event(event);
    REQUIRE(!result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "Hello world");
    REQUIRE(result.chunks[0].reasoning_content.empty());
}

TEST_CASE("DashScopeProtocol - parse_event accepts data prefix without a space", "[qwen][sse]") {
    DashScopeProtocol proto;
    const std::string event =
        R"(data:{"choices":[{"delta":{"content":"NoSpace"},"index":0}]})";
    auto result = proto.parse_event(event);
    REQUIRE(!result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "NoSpace");
}

TEST_CASE("DashScopeProtocol - parse_event accepts event envelope with CRLF", "[qwen][sse]") {
    DashScopeProtocol proto;
    const std::string event =
        "event: response.output_text.delta\r\n"
        "data: {\"choices\":[{\"delta\":{\"content\":\"CRLF\"},\"index\":0}]}\r\n";
    auto result = proto.parse_event(event);
    REQUIRE(!result.done);
    REQUIRE(result.chunks.size() == 1);
    REQUIRE(result.chunks[0].content == "CRLF");
}

TEST_CASE("DashScopeProtocol - parse_event handles [DONE] sentinel", "[qwen][sse]") {
    DashScopeProtocol proto;
    auto result = proto.parse_event("data: [DONE]");
    REQUIRE(result.done);
    REQUIRE(result.chunks.empty());
}

TEST_CASE("DashScopeProtocol - parse_event ignores malformed events", "[qwen][sse]") {
    DashScopeProtocol proto;
    auto result = proto.parse_event("not-an-sse-event");
    REQUIRE(!result.done);
    REQUIRE(result.chunks.empty());
}

TEST_CASE("DashScopeProtocol - parse_event handles empty content delta gracefully",
          "[qwen][sse]") {
    DashScopeProtocol proto;
    // Empty delta — no content, no tools, no reasoning.
    const std::string event =
        R"(data: {"choices":[{"delta":{},"index":0,"finish_reason":"stop"}]})";
    auto result = proto.parse_event(event);
    REQUIRE(!result.done);
    REQUIRE(result.chunks.empty());
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE parsing — reasoning_content (Qwen3 thinking mode)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DashScopeProtocol - parse_event extracts reasoning_content chunk",
          "[qwen][sse][thinking]") {
    DashScopeProtocol proto{4096};
    const std::string event =
        R"(data: {"choices":[{"delta":{"reasoning_content":"Let me think..."},"index":0}]})";
    auto result = proto.parse_event(event);
    REQUIRE(!result.done);
    REQUIRE(result.chunks.size() == 1);
    bool found_reasoning = false;
    for (const auto& c : result.chunks) {
        if (!c.reasoning_content.empty()) {
            REQUIRE(c.reasoning_content == "Let me think...");
            REQUIRE(c.reasoning_protocol == "dashscope");
            found_reasoning = true;
        }
    }
    REQUIRE(found_reasoning);
}

TEST_CASE("DashScopeProtocol - parse_event emits reasoning chunk independent of budget",
          "[qwen][sse][thinking]") {
    // reasoning_content in the response is parsed regardless of thinking_budget
    // — the server controls whether it sends it; we just forward what we receive.
    DashScopeProtocol proto{0};  // budget=0, but server may still send it
    const std::string event =
        R"(data: {"choices":[{"delta":{"reasoning_content":"<thinking>step</thinking>"},"index":0}]})";
    auto result = proto.parse_event(event);
    bool found = false;
    for (const auto& c : result.chunks)
        if (!c.reasoning_content.empty()) found = true;
    REQUIRE(found);
}

TEST_CASE("DashScopeProtocol - content and reasoning_content in separate chunks",
          "[qwen][sse][thinking]") {
    // The server sends thinking tokens before content tokens in separate SSE events.
    DashScopeProtocol proto{8192};

    const std::string think_event =
        R"(data: {"choices":[{"delta":{"reasoning_content":"Thinking..."},"index":0}]})";
    const std::string content_event =
        R"(data: {"choices":[{"delta":{"content":"Final answer"},"index":0}]})";

    auto think_result   = proto.parse_event(think_event);
    auto content_result = proto.parse_event(content_event);

    bool has_reasoning = false;
    for (const auto& c : think_result.chunks)
        if (!c.reasoning_content.empty()) has_reasoning = true;
    REQUIRE(has_reasoning);

    bool has_content = false;
    for (const auto& c : content_result.chunks)
        if (!c.content.empty()) has_content = true;
    REQUIRE(has_content);
}

// ─────────────────────────────────────────────────────────────────────────────
// SSE parsing — usage extraction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DashScopeProtocol - parse_event extracts token usage from stream_options chunk",
          "[qwen][sse][usage]") {
    DashScopeProtocol proto;
    const std::string event =
        R"(data: {"choices":[],"usage":{"prompt_tokens":42,"completion_tokens":17}})";
    auto result = proto.parse_event(event);
    REQUIRE(result.prompt_tokens     == 42);
    REQUIRE(result.completion_tokens == 17);
}

TEST_CASE("DashScopeProtocol - parses cached and reasoning token details",
          "[qwen][sse][usage]") {
    DashScopeProtocol proto;
    const auto result = proto.parse_event(
        R"(data: {"choices":[],"usage":{"prompt_tokens":1520,"completion_tokens":300,"prompt_tokens_details":{"cached_tokens":1480,"cache_creation_input_tokens":20},"completion_tokens_details":{"reasoning_tokens":245}}})");
    REQUIRE(result.prompt_tokens == 1520);
    REQUIRE(result.completion_tokens == 300);
    REQUIRE(result.cached_prompt_tokens == 1480);
    REQUIRE(result.cache_creation_prompt_tokens == 20);
    REQUIRE(result.reasoning_tokens == 245);
}

TEST_CASE("DashScope Responses - parses detailed Token Plan usage",
          "[qwen][responses][usage]") {
    DashScopeResponsesProtocol proto;
    const auto result = proto.parse_event(
        "event: response.completed\n"
        R"(data: {"type":"response.completed","response":{"id":"resp_1","usage":{"input_tokens":1520,"output_tokens":300,"input_tokens_details":{"cached_tokens":1480,"cache_creation_input_tokens":20},"output_tokens_details":{"reasoning_tokens":245}}}})");
    REQUIRE(result.done);
    REQUIRE(result.prompt_tokens == 1520);
    REQUIRE(result.completion_tokens == 300);
    REQUIRE(result.cached_prompt_tokens == 1480);
    REQUIRE(result.cache_creation_prompt_tokens == 20);
    REQUIRE(result.reasoning_tokens == 245);
}

TEST_CASE("DashScope Responses - does not concatenate provisional and completed tool arguments",
          "[qwen][responses][sse][tools]") {
    DashScopeResponsesProtocol protocol;

    const auto added = protocol.parse_event(
        "event: response.output_item.added\n"
        R"(data: {"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"list_directory","arguments":"{}","status":"in_progress"}})");
    REQUIRE(added.chunks.empty());

    const auto done = protocol.parse_event(
        "event: response.output_item.done\n"
        R"(data: {"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","call_id":"call_1","name":"list_directory","arguments":"{\"path\":\"Lampo/Prompter\"}","status":"completed"}})");
    REQUIRE(done.chunks.size() == 1);
    REQUIRE(done.chunks.front().tools.size() == 1);
    CHECK(done.chunks.front().tools.front().function.arguments
          == R"({"path":"Lampo/Prompter"})");
}

TEST_CASE("Token Plan follows Qwen Code's Chat Completions generation path",
          "[qwen][token-plan][routing]") {
    DashScopeTokenPlanProtocol protocol;
    const auto req = make_simple_request("qwen3.8-max");
    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("messages":[)"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("input":[)"));
    REQUIRE(protocol.build_url("https://example.test/v1", req.model)
            == "https://example.test/v1/chat/completions");
}

TEST_CASE("Token Plan sends local images as Chat Completions data URLs",
          "[qwen][token-plan][routing][vision]") {
    DashScopeTokenPlanProtocol protocol;
    const auto image = make_temp_qwen_image_file();

    ChatRequest req;
    req.model = "qwen3.8-max";
    req.messages.push_back(Message{
        .role = "user",
        .content = "Read the error in this screenshot.",
        .content_parts = {
            ContentPart::make_text("Read the error in this screenshot."),
            ContentPart::make_image(image.string(), "image/png"),
        },
    });

    const auto payload = protocol.serialize(req);
    require_valid_json(payload);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("messages":[)"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("type":"image_url")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("data:image/png;base64,"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("type":"input_image")"));
    REQUIRE(protocol.build_url("https://example.test/v1", req.model)
            == "https://example.test/v1/chat/completions");
}

TEST_CASE("Token Plan routes third-party models through Chat Completions",
          "[qwen][token-plan][routing]") {
    DashScopeTokenPlanProtocol protocol;
    const auto req = make_simple_request("glm-5.2");
    const auto payload = protocol.serialize(req);

    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("messages":[)"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("input":[)"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("enable_thinking"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("cache_control"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("content":"Hello")"));
    REQUIRE(protocol.build_url("https://example.test/v1", req.model)
            == "https://example.test/v1/chat/completions");
}

TEST_CASE("Token Plan Chat route parses fragmented GLM tool arguments",
          "[qwen][token-plan][routing][tools]") {
    DashScopeTokenPlanProtocol protocol;
    auto request = make_simple_request("glm-5.2");
    protocol.prepare_request(request);

    const auto named = protocol.parse_event(
        R"(data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"ping","arguments":"{"}}]}}]})");
    REQUIRE(named.chunks.size() == 1);
    REQUIRE(named.chunks.front().tools.size() == 1);
    REQUIRE(named.chunks.front().tools.front().function.name == "ping");
    REQUIRE(named.chunks.front().tools.front().function.arguments == "{");

    const auto args = protocol.parse_event(
        R"(data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"value\":\"ok\"}"}}]}}]})");
    REQUIRE(args.chunks.size() == 1);
    REQUIRE(args.chunks.front().tools.front().function.arguments
            == R"("value":"ok"})");
}

TEST_CASE("Token Plan adapter preserves future Qwen reasoning capabilities",
          "[qwen][token-plan][routing][traits]") {
    CHECK(DashScopeTokenPlanProtocol{}.reasoning_capabilities("qwen4-coder")
          .supports_effort());
}

// ─────────────────────────────────────────────────────────────────────────────
// Retry policy
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DashScopeProtocol - 429 is retryable", "[qwen][retry]") {
    DashScopeProtocol proto;
    REQUIRE(proto.is_retryable(make_response(429).view()));
}

TEST_CASE("Token Plan - allocation quota HTTP failures are not retried",
          "[qwen][retry][ratelimit]") {
    DashScopeTokenPlanProtocol protocol;
    const auto quota = make_response(
        429,
        R"({"code":"Throttling.AllocationQuota","message":"Allocated quota exceeded"})");
    const auto transient = make_response(
        429,
        R"({"code":"Throttling.RateQuota","message":"Requests rate limit exceeded"})");

    CHECK_FALSE(protocol.is_retryable(quota.view()));
    CHECK(protocol.is_retryable(transient.view()));
}

TEST_CASE("Token Plan recognizes weekly quota resets without an allocation code",
          "[qwen][retry][ratelimit][sse]") {
    const std::string message = "Your token-plan 1-week quota has been exhausted. "
        "The quota will reset at 09-09 09:25:00 UTC.";
    const std::string json = R"({"error":{"code":"429","message":")" + message + R"("}})";
    for (const auto& body : {message, json}) {
        DashScopeTokenPlanProtocol protocol;
        const auto response = make_response(429, body);
        CHECK_FALSE(protocol.is_retryable(response.view()));
        protocol.on_response(response.view());
        CHECK(protocol.last_rate_limit().unified_overage_status == "rejected");
        CHECK(protocol.last_rate_limit().subscription_ends_at == 0);
        CHECK_THAT(protocol.format_error_message(response.view()),
                   Catch::Matchers::ContainsSubstring("09-09 09:25:00 UTC"));
    }
    for (const auto& event : {"data: " + json, "event: error\ndata: " + json,
         R"(data: {"choices":[{"finish_reason":"error_finish","delta":{"content":")"
             + message + R"("}}]})"}) {
        DashScopeTokenPlanProtocol protocol;
        const auto result = protocol.parse_event(event);
        CHECK(result.stream_error);
        CHECK_FALSE(result.retryable_stream_error);
        CHECK(result.stream_error_message == message);
        CHECK(protocol.last_rate_limit().unified_overage_status == "rejected");
        protocol.on_response(make_response(200).view());
        CHECK(protocol.last_rate_limit().unified_overage_status == "rejected");
        protocol.reset_state();
        protocol.on_response(make_response(200).view());
        CHECK_FALSE(protocol.last_rate_limit().is_rate_limited);
    }
}

TEST_CASE("Token Plan surfaces bare SSE API errors instead of an empty success",
          "[qwen][sse][errors]") {
    DashScopeTokenPlanProtocol protocol;
    const auto result = protocol.parse_event(
        R"(data: {"error":{"code":"InvalidApiKey","message":"Invalid API-key provided."}})");
    CHECK(result.stream_error);
    CHECK_FALSE(result.retryable_stream_error);
    CHECK(result.stream_error_type == "InvalidApiKey");
    CHECK(result.chunks.empty());
}

TEST_CASE("DashScopeProtocol - 500/502/503/504 are retryable", "[qwen][retry]") {
    DashScopeProtocol proto;
    REQUIRE(proto.is_retryable(make_response(500).view()));
    REQUIRE(proto.is_retryable(make_response(502).view()));
    REQUIRE(proto.is_retryable(make_response(503).view()));
    REQUIRE(proto.is_retryable(make_response(504).view()));
}

TEST_CASE("DashScopeProtocol - 400/401/403/404 are not retryable", "[qwen][retry]") {
    DashScopeProtocol proto;
    REQUIRE(!proto.is_retryable(make_response(400).view()));
    REQUIRE(!proto.is_retryable(make_response(401).view()));
    REQUIRE(!proto.is_retryable(make_response(403).view()));
    REQUIRE(!proto.is_retryable(make_response(404).view()));
}

// ─────────────────────────────────────────────────────────────────────────────
// Error message formatting
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DashScopeProtocol - format_error_message 401 mentions API key", "[qwen][errors]") {
    DashScopeProtocol proto;
    auto msg = proto.format_error_message(make_response(401).view());
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("401"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("DASHSCOPE_API_KEY"));
}

TEST_CASE("DashScopeProtocol - format_error_message 429 mentions quota", "[qwen][errors]") {
    DashScopeProtocol proto;
    auto msg = proto.format_error_message(make_response(429).view());
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("429"));
    // Must mention rate limit / quota.
    const bool mentions_limit = msg.find("Rate limit") != std::string::npos ||
                                msg.find("rate limit") != std::string::npos ||
                                msg.find("quota")      != std::string::npos;
    REQUIRE(mentions_limit);
}

TEST_CASE("DashScopeProtocol - format_error_message 400 includes API error code from body",
          "[qwen][errors]") {
    DashScopeProtocol proto;
    const std::string body =
        R"({"code":"InvalidParameter","message":"enable_thinking requires qwen3"})";
    auto msg = proto.format_error_message(make_response(400, body).view());
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("InvalidParameter"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("enable_thinking"));
}

TEST_CASE("DashScopeProtocol - format_error_message 500 mentions retry", "[qwen][errors]") {
    DashScopeProtocol proto;
    auto msg = proto.format_error_message(make_response(500).view());
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("500"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("retry"));
}

// ─────────────────────────────────────────────────────────────────────────────
// Protocol identity
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DashScopeProtocol - name returns dashscope", "[qwen][identity]") {
    REQUIRE(DashScopeProtocol{}.name() == "dashscope");
}

TEST_CASE("DashScopeProtocol - clone produces an independent copy", "[qwen][identity]") {
    DashScopeProtocol original{4096};
    auto cloned = original.clone();
    REQUIRE(cloned != nullptr);
    REQUIRE(cloned->name() == "dashscope");
    // The clone is a distinct object.
    REQUIRE(cloned.get() != &original);
}

// ─────────────────────────────────────────────────────────────────────────────
// ApiType string conversion
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ApiType::DashScope - to_string returns dashscope", "[qwen][config]") {
    REQUIRE(core::config::to_string(core::config::ApiType::DashScope) == "dashscope");
}

TEST_CASE("ApiType - api_type_from_string dashscope round-trips", "[qwen][config]") {
    REQUIRE(core::config::api_type_from_string("dashscope") ==
            core::config::ApiType::DashScope);
}

TEST_CASE("ApiType - api_type_from_string qwen alias resolves to DashScope", "[qwen][config]") {
    REQUIRE(core::config::api_type_from_string("qwen") ==
            core::config::ApiType::DashScope);
}

// ─────────────────────────────────────────────────────────────────────────────
// ProviderFactory integration
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ProviderFactory - creates provider for qwen builtin name", "[qwen][factory]") {
    core::config::ProviderConfig cfg;
    cfg.model   = "qwen3-coder-plus";
    cfg.api_key = "sk-test";

    auto provider = core::llm::ProviderFactory::create_provider("qwen", cfg);
    REQUIRE(provider != nullptr);
}

TEST_CASE("ProviderFactory - creates DashScope provider when api_type is explicit",
          "[qwen][factory]") {
    core::config::ProviderConfig cfg;
    cfg.api_type = core::config::ApiType::DashScope;
    cfg.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1";
    cfg.model    = "qwen3-coder-plus";
    cfg.api_key  = "sk-test";

    auto provider = core::llm::ProviderFactory::create_provider("my-qwen", cfg);
    REQUIRE(provider != nullptr);
}

TEST_CASE("ProviderFactory - qwen with thinking_budget creates provider", "[qwen][factory]") {
    core::config::ProviderConfig cfg;
    cfg.model          = "qwen3-coder-plus";
    cfg.api_key        = "sk-test";
    cfg.thinking_budget = 8192;

    auto provider = core::llm::ProviderFactory::create_provider("qwen", cfg);
    REQUIRE(provider != nullptr);
}

TEST_CASE("ProviderFactory - Qwen Token Plan is subscription-backed Chat Completions",
          "[qwen][factory][token-plan]") {
    core::config::ProviderConfig cfg;
    cfg.api_key = "sk-sp-test";

    auto provider = core::llm::ProviderFactory::create_provider("qwen-token-plan", cfg);
    REQUIRE(provider != nullptr);
    REQUIRE_FALSE(provider->should_estimate_cost());
    const auto metadata = provider->metadata();
    REQUIRE(metadata.has_value());
    REQUIRE(metadata->default_model.empty());
    REQUIRE(metadata->base_url ==
        "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1");
}

TEST_CASE("ProviderFactory - custom Token Plan endpoint keeps subscription billing",
          "[qwen][factory][token-plan]") {
    core::config::ProviderConfig cfg;
    cfg.api_type = core::config::ApiType::DashScope;
    cfg.base_url =
        "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1";
    cfg.model = "qwen3.7-plus";
    cfg.api_key = "sk-sp-test";
    cfg.wire_api = "chat_completions";

    auto provider = core::llm::ProviderFactory::create_provider("company-qwen", cfg);
    REQUIRE(provider != nullptr);
    REQUIRE_FALSE(provider->should_estimate_cost());
}

TEST_CASE("ProviderFactory - Token Plan classification requires an exact URL host",
          "[qwen][factory][token-plan][security]") {
    core::config::ProviderConfig cfg;
    cfg.api_type = core::config::ApiType::DashScope;
    cfg.base_url =
        "https://example.test/token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1";
    cfg.model = "qwen3.7-plus";
    cfg.api_key = "test-key";
    cfg.wire_api = "responses";

    auto provider = core::llm::ProviderFactory::create_provider("company-qwen", cfg);
    REQUIRE(provider != nullptr);
    REQUIRE(provider->should_estimate_cost());
}

// ─────────────────────────────────────────────────────────────────────────────
// ModelRegistry — Qwen model catalog
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ModelRegistry - qwen3-coder-plus is registered", "[qwen][registry]") {
    auto& reg = ModelRegistry::instance();
    REQUIRE(reg.has_model("qwen3-coder-plus"));
}

TEST_CASE("ModelRegistry - Token Plan featured models have first-class cards",
          "[qwen][registry][token-plan]") {
    const auto& registry = ModelRegistry::instance();
    for (const auto model : {"qwen3.8-max", "qwen3.7-max",
                             "qwen3.7-plus", "qwen3.6-plus",
                             "qwen3.6-flash"}) {
        CAPTURE(model);
        const auto info = registry.get_info(model);
        REQUIRE(info.has_value());
        REQUIRE(info->context_window == 1'000'000);
        REQUIRE(info->max_output_tokens == 64'000);
        REQUIRE(info->supports(ModelCapability::Reasoning));
        REQUIRE(info->supports(ModelCapability::FunctionCalling));
    }
}

TEST_CASE("ModelRegistry - Token Plan vision and structured-output capabilities match docs",
          "[qwen][registry][token-plan]") {
    const auto& registry = ModelRegistry::instance();
    for (const auto model : {"qwen3.7-max", "qwen3.6-flash"}) {
        CAPTURE(model);
        REQUIRE_FALSE(registry.supports(model, ModelCapability::Vision));
        REQUIRE_FALSE(registry.supports(model, ModelCapability::VideoInput));
    }
    for (const auto model : {"qwen3.8-max", "qwen3.7-plus",
                             "qwen3.6-plus"}) {
        CAPTURE(model);
        REQUIRE(registry.supports(model, ModelCapability::Vision));
        REQUIRE(registry.supports(model, ModelCapability::VideoInput));
        REQUIRE(registry.supports(model, ModelCapability::JsonMode));
    }
}

TEST_CASE("ModelRegistry - qwen3-coder-plus has 1M context window", "[qwen][registry]") {
    auto& reg  = ModelRegistry::instance();
    REQUIRE(reg.get_max_context_size("qwen3-coder-plus") == 1'000'000);
}

TEST_CASE("ModelRegistry - qwen3-coder-plus has 64K max output tokens", "[qwen][registry]") {
    auto info = ModelRegistry::instance().get_info("qwen3-coder-plus");
    REQUIRE(info.has_value());
    REQUIRE(info->max_output_tokens == 64'000);
}

TEST_CASE("ModelRegistry - qwen3-coder-plus supports Reasoning capability", "[qwen][registry]") {
    REQUIRE(ModelRegistry::instance().supports("qwen3-coder-plus", ModelCapability::Reasoning));
}

TEST_CASE("ModelRegistry - qwen3-coder-plus supports FunctionCalling", "[qwen][registry]") {
    REQUIRE(ModelRegistry::instance().supports("qwen3-coder-plus", ModelCapability::FunctionCalling));
}

TEST_CASE("ModelRegistry - qwen3-coder-plus supports PromptCaching", "[qwen][registry]") {
    REQUIRE(ModelRegistry::instance().supports("qwen3-coder-plus", ModelCapability::PromptCaching));
}

TEST_CASE("ModelRegistry - qwen3-coder-flash is a Fast tier model", "[qwen][registry]") {
    auto tier = ModelRegistry::instance().get_tier("qwen3-coder-flash");
    REQUIRE(tier.has_value());
    REQUIRE(*tier == ModelTier::Fast);
}

TEST_CASE("ModelRegistry - qwen3-max is a Powerful tier model", "[qwen][registry]") {
    auto tier = ModelRegistry::instance().get_tier("qwen3-max");
    REQUIRE(tier.has_value());
    REQUIRE(*tier == ModelTier::Powerful);
}

TEST_CASE("ModelRegistry - qwen3-max has 256K context window", "[qwen][registry]") {
    REQUIRE(ModelRegistry::instance().get_max_context_size("qwen3-max") == 256'000);
}

TEST_CASE("ModelRegistry - qwen3.5-plus supports Vision capability", "[qwen][registry]") {
    REQUIRE(ModelRegistry::instance().supports("qwen3.5-plus", ModelCapability::Vision));
}

TEST_CASE("ModelRegistry - qwen3-vl-plus supports Vision capability", "[qwen][registry]") {
    REQUIRE(ModelRegistry::instance().supports("qwen3-vl-plus", ModelCapability::Vision));
}

TEST_CASE("ModelRegistry - qwen3-plus alias qwen-plus-latest resolves correctly",
          "[qwen][registry]") {
    auto& reg = ModelRegistry::instance();
    REQUIRE(reg.has_model("qwen-plus-latest"));
    // Both the alias and the canonical id should have the same context window.
    REQUIRE(reg.get_max_context_size("qwen-plus-latest") ==
            reg.get_max_context_size("qwen3-plus"));
}

TEST_CASE("ModelRegistry - all Qwen models belong to provider qwen", "[qwen][registry]") {
    auto models = ModelRegistry::instance().get_by_provider("qwen");
    REQUIRE(!models.empty());
    for (const auto& m : models) {
        REQUIRE(m.provider == "qwen");
    }
}

TEST_CASE("ModelRegistry - legacy context lookup works for unknown Qwen models",
          "[qwen][registry]") {
    // A model not in the catalog falls back to the legacy prefix table.
    int size = ModelRegistry::get_max_context_size_legacy("qwen3-coder-custom-variant");
    REQUIRE(size == 1'000'000);
}
