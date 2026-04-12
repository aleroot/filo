#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/DashScopeProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/llm/ModelRegistry.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/Models.hpp"
#include "core/auth/ApiKeyCredentialSource.hpp"

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

// ─────────────────────────────────────────────────────────────────────────────
// ModelRegistry — Qwen model catalog
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ModelRegistry - qwen3-coder-plus is registered", "[qwen][registry]") {
    auto& reg = ModelRegistry::instance();
    REQUIRE(reg.has_model("qwen3-coder-plus"));
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
    for (const auto* m : models) {
        REQUIRE(m->provider == "qwen");
    }
}

TEST_CASE("ModelRegistry - legacy context lookup works for unknown Qwen models",
          "[qwen][registry]") {
    // A model not in the catalog falls back to the legacy prefix table.
    int size = ModelRegistry::get_max_context_size_legacy("qwen3-coder-custom-variant");
    REQUIRE(size == 1'000'000);
}
