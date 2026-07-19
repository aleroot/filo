#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/DashScopeProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/llm/providers/QwenModelCatalogSelector.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/Models.hpp"
#include "core/auth/ApiKeyCredentialSource.hpp"

#include <simdjson.h>

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
        .function = {.name = "read_file", .arguments = R"({"path":"README.md"})"},
    });
    req.messages.insert(req.messages.begin() + 1, std::move(assistant));

    const auto payload = DashScopeProtocol(0, "high").serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("reasoning_content":"Need to inspect the repository first.")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("preserve_thinking":true)"));
}

TEST_CASE("DashScopeProtocol - effort can disable hybrid thinking",
          "[qwen][serializer][thinking][effort]") {
    auto req = make_simple_request("qwen3.7-plus");
    req.effort = "none";
    const auto payload = DashScopeProtocol(8192, "high").serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("enable_thinking":false)"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("thinking_budget"));
}

TEST_CASE("DashScope Responses - Token Plan payload enables native features",
          "[qwen][responses][token-plan]") {
    auto req = make_simple_request("qwen3.8-max-preview");
    req.prompt_cache_key = "filo-session";
    req.effort = "max";

    const auto payload = DashScopeResponsesProtocol({
        .default_effort = "high",
        .enable_hosted_tools = true,
    }).serialize(req);
    require_valid_json(payload);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(
        R"("reasoning":{"effort":"max"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"({"type":"web_search"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"({"type":"code_interpreter"})"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"({"type":"web_extractor"})"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("store")"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("prompt_cache_key"));
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("include":[])"));
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
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring("Always be concise."));
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
    // At least one chunk must contain reasoning_content.
    bool found_reasoning = false;
    for (const auto& c : result.chunks) {
        if (!c.reasoning_content.empty()) {
            REQUIRE(c.reasoning_content == "Let me think...");
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

// ─────────────────────────────────────────────────────────────────────────────
// Retry policy
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("DashScopeProtocol - 429 is retryable", "[qwen][retry]") {
    DashScopeProtocol proto;
    REQUIRE(proto.is_retryable(make_response(429).view()));
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

TEST_CASE("ProviderFactory - Qwen Token Plan is subscription-backed Responses API",
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
    cfg.wire_api = "responses";

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
    for (const auto model : {"qwen3.8-max-preview", "qwen3.7-max",
                             "qwen3.7-plus", "qwen3.6-flash"}) {
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
    for (const auto model : {"qwen3.8-max-preview", "qwen3.7-max"}) {
        CAPTURE(model);
        REQUIRE_FALSE(registry.supports(model, ModelCapability::Vision));
        REQUIRE_FALSE(registry.supports(model, ModelCapability::JsonMode));
    }
    for (const auto model : {"qwen3.7-plus", "qwen3.6-flash"}) {
        CAPTURE(model);
        REQUIRE(registry.supports(model, ModelCapability::Vision));
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
