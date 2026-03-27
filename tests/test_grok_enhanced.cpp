/**
 * @file test_grok_enhanced.cpp
 * @brief Enhanced tests for xAI Grok protocol with error handling, rate limits, and JSON mode.
 * 
 * These tests cover the L7-level enhancements:
 * - xAI-specific error message formatting
 * - Rate limit header extraction
 * - JSON mode / structured outputs
 * - Response lifecycle hooks
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/GrokProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/tools/Tool.hpp"

using namespace core::llm;
using namespace core::llm::protocols;

// ─────────────────────────────────────────────────────────────────────────────
// ResponseFormat / JSON Mode Tests
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ResponseFormat - Text type to_string", "[grok][json_mode]") {
    ResponseFormat rf;
    rf.type = ResponseFormat::Type::Text;
    REQUIRE(rf.to_string() == "text");
    REQUIRE_FALSE(rf.is_structured());
}

TEST_CASE("ResponseFormat - JsonObject type to_string", "[grok][json_mode]") {
    ResponseFormat rf;
    rf.type = ResponseFormat::Type::JsonObject;
    REQUIRE(rf.to_string() == "json_object");
    REQUIRE(rf.is_structured());
}

TEST_CASE("ResponseFormat - JsonSchema type to_string", "[grok][json_mode]") {
    ResponseFormat rf;
    rf.type = ResponseFormat::Type::JsonSchema;
    REQUIRE(rf.to_string() == "json_schema");
    REQUIRE(rf.is_structured());
}

TEST_CASE("GrokSerializer - response_format omitted by default", "[grok][serializer][json_mode]") {
    ChatRequest req;
    req.model = "grok-code-fast-1";
    req.messages.push_back(Message{.role = "user", .content = "Hello"});
    
    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, !Catch::Matchers::ContainsSubstring("response_format"));
}

TEST_CASE("GrokSerializer - response_format json_object is serialized", "[grok][serializer][json_mode]") {
    ChatRequest req;
    req.model = "grok-code-fast-1";
    req.response_format.type = ResponseFormat::Type::JsonObject;
    req.messages.push_back(Message{.role = "user", .content = "Generate JSON"});
    
    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("response_format":{"type":"json_object"})"));
}

TEST_CASE("GrokSerializer - response_format json_schema with schema", "[grok][serializer][json_mode]") {
    ChatRequest req;
    req.model = "grok-code-fast-1";
    req.response_format.type = ResponseFormat::Type::JsonSchema;
    req.response_format.schema = R"({"type":"object","properties":{"name":{"type":"string"}}})";
    req.messages.push_back(Message{.role = "user", .content = "Generate JSON"});
    
    auto payload = GrokProtocol{}.serialize(req);
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("response_format":{"type":"json_schema")"));
    REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("schema":{"type":"object")"));
}

TEST_CASE("GrokSerializer - JSON mode works with all Grok models", "[grok][serializer][json_mode]") {
    const std::vector<std::string> models = {
        "grok-code-fast-1",
        "grok-4",
        "grok-4.20-reasoning",
        "grok-3-mini",
        "grok-3-mini-fast"
    };
    
    for (const auto& model : models) {
        INFO("Testing JSON mode for model: " << model);
        ChatRequest req;
        req.model = model;
        req.response_format.type = ResponseFormat::Type::JsonObject;
        req.messages.push_back(Message{.role = "user", .content = "Hello"});
        
        auto payload = GrokProtocol{}.serialize(req);
        REQUIRE_THAT(payload, Catch::Matchers::ContainsSubstring(R"("response_format":{"type":"json_object"})"));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GrokProtocol Lifecycle Hooks
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GrokProtocol - on_response extracts rate limit headers", "[grok][protocol][rate_limit]") {
    GrokProtocol protocol;
    
    // Simulate response with xAI rate limit headers
    cpr::Header headers;
    headers["x-ratelimit-limit-requests"] = "100";
    headers["x-ratelimit-remaining-requests"] = "95";
    headers["x-ratelimit-limit-tokens"] = "100000";
    headers["x-ratelimit-remaining-tokens"] = "95000";
    
    const HttpResponse response{200, "{}", headers};
    protocol.on_response(response);
    
    auto info = protocol.last_rate_limit();
    REQUIRE(info.requests_limit == 100);
    REQUIRE(info.requests_remaining == 95);
    REQUIRE(info.tokens_limit == 100000);
    REQUIRE(info.tokens_remaining == 95000);
}

TEST_CASE("GrokProtocol - on_response extracts retry_after on 429", "[grok][protocol][rate_limit]") {
    GrokProtocol protocol;
    
    cpr::Header headers;
    headers["retry-after"] = "30";
    
    const HttpResponse response{429, "{}", headers};
    protocol.on_response(response);
    
    auto info = protocol.last_rate_limit();
    REQUIRE(info.retry_after == 30);
    REQUIRE(info.is_rate_limited);
}

TEST_CASE("GrokProtocol - last_rate_limit returns zero initially", "[grok][protocol][rate_limit]") {
    GrokProtocol protocol;
    auto info = protocol.last_rate_limit();
    REQUIRE(info.requests_limit == 0);
    REQUIRE(info.requests_remaining == 0);
    REQUIRE_FALSE(info.has_data());
}

// ─────────────────────────────────────────────────────────────────────────────
// GrokProtocol Error Message Formatting
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GrokProtocol - format_error_message for 400", "[grok][protocol][errors]") {
    GrokProtocol protocol;
    cpr::Header headers;
    const HttpResponse response{400, "{}", headers};
    
    auto msg = protocol.format_error_message(response);
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("xAI API Error 400"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("Grok 4 models do NOT support 'reasoning_effort'"));
}

TEST_CASE("GrokProtocol - format_error_message for 400 with xAI error body", "[grok][protocol][errors]") {
    GrokProtocol protocol;
    cpr::Header headers;
    const HttpResponse response{400, R"({"error":{"message":"Invalid model"}})", headers};
    
    auto msg = protocol.format_error_message(response);
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("Invalid model"));
}

TEST_CASE("GrokProtocol - format_error_message for 401", "[grok][protocol][errors]") {
    GrokProtocol protocol;
    cpr::Header headers;
    const HttpResponse response{401, "{}", headers};
    
    auto msg = protocol.format_error_message(response);
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("xAI API Error 401"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("XAI_API_KEY"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("console.x.ai"));
}

TEST_CASE("GrokProtocol - format_error_message for 403", "[grok][protocol][errors]") {
    GrokProtocol protocol;
    cpr::Header headers;
    const HttpResponse response{403, "{}", headers};
    
    auto msg = protocol.format_error_message(response);
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("xAI API Error 403"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("subscription"));
}

TEST_CASE("GrokProtocol - format_error_message for 404", "[grok][protocol][errors]") {
    GrokProtocol protocol;
    cpr::Header headers;
    const HttpResponse response{404, "{}", headers};
    
    auto msg = protocol.format_error_message(response);
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("xAI API Error 404"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("grok-code-fast-1"));
}

TEST_CASE("GrokProtocol - format_error_message for 429", "[grok][protocol][errors]") {
    GrokProtocol protocol;
    cpr::Header headers;
    headers["retry-after"] = "60";
    const HttpResponse response{429, "{}", headers};
    
    protocol.on_response(response);  // Cache rate limit info
    auto msg = protocol.format_error_message(response);
    
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("xAI API Error 429"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("Rate limit exceeded"));
}

TEST_CASE("GrokProtocol - format_error_message for 500", "[grok][protocol][errors]") {
    GrokProtocol protocol;
    cpr::Header headers;
    const HttpResponse response{500, "{}", headers};
    
    auto msg = protocol.format_error_message(response);
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("xAI API Error 500"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("retry"));
}

TEST_CASE("GrokProtocol - format_error_message for 529 overloaded", "[grok][protocol][errors]") {
    GrokProtocol protocol;
    cpr::Header headers;
    const HttpResponse response{529, "{}", headers};
    
    auto msg = protocol.format_error_message(response);
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("xAI API Error 529"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("overloaded"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("NOT a rate limit"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("grok-code-fast-1"));
}

TEST_CASE("GrokProtocol - format_error_message for unknown 5xx", "[grok][protocol][errors]") {
    GrokProtocol protocol;
    cpr::Header headers;
    const HttpResponse response{599, "{}", headers};
    
    auto msg = protocol.format_error_message(response);
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("xAI API Error 599"));
    REQUIRE_THAT(msg, Catch::Matchers::ContainsSubstring("Server error"));
}

// ─────────────────────────────────────────────────────────────────────────────
// GrokProtocol Retry Logic
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("GrokProtocol - is_retryable for 429", "[grok][protocol][retry]") {
    GrokProtocol protocol;
    cpr::Header headers;
    REQUIRE(protocol.is_retryable({429, "{}", headers}));
}

TEST_CASE("GrokProtocol - is_retryable for 529", "[grok][protocol][retry]") {
    GrokProtocol protocol;
    cpr::Header headers;
    REQUIRE(protocol.is_retryable({529, "{}", headers}));
}

TEST_CASE("GrokProtocol - is_retryable for 500-504", "[grok][protocol][retry]") {
    GrokProtocol protocol;
    cpr::Header headers;
    REQUIRE(protocol.is_retryable({500, "{}", headers}));
    REQUIRE(protocol.is_retryable({502, "{}", headers}));
    REQUIRE(protocol.is_retryable({503, "{}", headers}));
    REQUIRE(protocol.is_retryable({504, "{}", headers}));
}

TEST_CASE("GrokProtocol - is_not_retryable for 4xx client errors", "[grok][protocol][retry]") {
    GrokProtocol protocol;
    cpr::Header headers;
    REQUIRE_FALSE(protocol.is_retryable({400, "{}", headers}));
    REQUIRE_FALSE(protocol.is_retryable({401, "{}", headers}));
    REQUIRE_FALSE(protocol.is_retryable({403, "{}", headers}));
    REQUIRE_FALSE(protocol.is_retryable({404, "{}", headers}));
    REQUIRE_FALSE(protocol.is_retryable({422, "{}", headers}));
}

// ─────────────────────────────────────────────────────────────────────────────
// ProviderFactory and HttpLLMProvider with GrokProtocol
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("ProviderFactory - creates grok provider", "[grok][factory]") {
    core::config::ProviderConfig config;
    config.model = "grok-code-fast-1";

    auto provider = ProviderFactory::create_provider("grok", config);
    REQUIRE(provider != nullptr);
}

TEST_CASE("ProviderFactory - creates grok provider with JSON mode config", "[grok][factory]") {
    core::config::ProviderConfig config;
    config.model = "grok-code-fast-1";
    // Note: response_format would be set via ChatRequest, not ProviderConfig

    auto provider = ProviderFactory::create_provider("grok", config);
    REQUIRE(provider != nullptr);
}

static std::shared_ptr<core::auth::ApiKeyCredentialSource> make_grok_creds_enhanced(const std::string& key) {
    return core::auth::ApiKeyCredentialSource::as_bearer(key);
}

TEST_CASE("GrokProvider - constructs with JSON mode request", "[grok][provider][json_mode]") {
    REQUIRE_NOTHROW(HttpLLMProvider(
        "https://api.x.ai/v1",
        make_grok_creds_enhanced("xai-test-key-abc"),
        "grok-code-fast-1",
        std::make_unique<GrokProtocol>()));
}
