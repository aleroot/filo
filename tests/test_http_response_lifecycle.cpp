/**
 * @file test_http_response_lifecycle.cpp
 * @brief Unit tests for the ApiProtocolBase response lifecycle hooks.
 *
 * Verifies that:
 *  - `HttpResponse` is a non-owning aggregate, constructible from raw values.
 *  - The **default** hook implementations in ApiProtocolBase produce the
 *    correct generic behaviour (no-op `on_response`, generic error string,
 *    `is_retryable` → false).
 *  - `AnthropicProtocol` overrides all three hooks with Anthropic-specific
 *    logic and stores rate-limit metadata in `last_rate_limit()`.
 *
 * These tests exercise the lifecycle contract through the public interface
 * (via `OpenAIProtocol` for defaults, `AnthropicProtocol` for overrides) so
 * they remain valid regardless of internal implementation changes.
 */

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/protocols/ApiProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/AnthropicProtocol.hpp"

using namespace core::llm::protocols;
using Catch::Matchers::ContainsSubstring;

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Build a minimal cpr::Header map from an initializer list.
static cpr::Header make_headers(
    std::initializer_list<std::pair<std::string, std::string>> pairs)
{
    cpr::Header h;
    for (auto& [k, v] : pairs) {
        h[k] = v;
    }
    return h;
}

/// Returns the utilization of the first UnifiedWindow matching @p label, or 0.
static float find_window(const RateLimitInfo& info, std::string_view label) {
    for (const auto& w : info.usage_windows) {
        if (w.label == label) return w.utilization;
    }
    return 0.0f;
}

// ─────────────────────────────────────────────────────────────────────────────
// HttpResponse — aggregate construction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("HttpResponse - aggregate-initialised from raw values", "[lifecycle][HttpResponse]") {
    const std::string   body    = R"({"error":"bad request"})";
    const cpr::Header   headers = make_headers({{"content-type", "application/json"}});
    const HttpResponse  resp{400, body, headers};

    REQUIRE(resp.status_code == 400);
    REQUIRE(resp.body        == body);
    REQUIRE(resp.headers.at("content-type") == "application/json");
}

// ─────────────────────────────────────────────────────────────────────────────
// Default hook implementations (via OpenAIProtocol, which does not override)
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Default on_response - no-op, does not throw", "[lifecycle][default]") {
    OpenAIProtocol    protocol;
    const cpr::Header headers;
    const HttpResponse resp{200, "", headers};

    REQUIRE_NOTHROW(protocol.on_response(resp));
}

TEST_CASE("Default format_error_message - generic [HTTP Error: N - body] string", "[lifecycle][default]") {
    OpenAIProtocol    protocol;
    const cpr::Header headers;
    const HttpResponse resp{503, "Service Unavailable", headers};

    const std::string msg = protocol.format_error_message(resp);

    REQUIRE_THAT(msg, ContainsSubstring("503"));
    REQUIRE_THAT(msg, ContainsSubstring("Service Unavailable"));
}

TEST_CASE("Default format_error_message - empty body produces valid string", "[lifecycle][default]") {
    OpenAIProtocol    protocol;
    const cpr::Header headers;
    const HttpResponse resp{429, "", headers};

    const std::string msg = protocol.format_error_message(resp);

    REQUIRE_THAT(msg, ContainsSubstring("429"));
    REQUIRE(!msg.empty());
}

TEST_CASE("Default is_retryable - always false for unknown protocol", "[lifecycle][default]") {
    OpenAIProtocol    protocol;
    const cpr::Header headers;

    for (const int code : {400, 401, 429, 500, 502, 503, 529}) {
        const HttpResponse resp{code, "", headers};
        INFO("status_code = " << code);
        REQUIRE_FALSE(protocol.is_retryable(resp));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicProtocol::on_response — rate-limit header extraction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("AnthropicProtocol::on_response - populates last_rate_limit from headers", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers = make_headers({
        {"anthropic-ratelimit-requests-limit",     "100"},
        {"anthropic-ratelimit-requests-remaining", "42"},
        {"anthropic-ratelimit-tokens-limit",       "50000"},
        {"anthropic-ratelimit-tokens-remaining",   "12345"},
    });
    const HttpResponse resp{200, "", headers};

    protocol.on_response(resp);
    const RateLimitInfo& info = protocol.last_rate_limit();

    REQUIRE(info.requests_limit     == 100);
    REQUIRE(info.requests_remaining == 42);
    REQUIRE(info.tokens_limit       == 50000);
    REQUIRE(info.tokens_remaining   == 12345);
}

TEST_CASE("AnthropicProtocol::on_response - detects rate-limited from retry-after header", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers = make_headers({
        {"retry-after", "30"},
    });
    const HttpResponse resp{429, "", headers};

    protocol.on_response(resp);
    const RateLimitInfo& info = protocol.last_rate_limit();

    REQUIRE(info.retry_after     == 30);
    REQUIRE(info.is_rate_limited == true);
}

TEST_CASE("AnthropicProtocol::on_response - detects rate-limited from unified-status header", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers = make_headers({
        {"anthropic-ratelimit-unified-status", "rate_limited"},
    });
    const HttpResponse resp{200, "", headers};

    protocol.on_response(resp);

    REQUIRE(protocol.last_rate_limit().is_rate_limited == true);
    REQUIRE(protocol.last_rate_limit().unified_status  == "rate_limited");
}

TEST_CASE("AnthropicProtocol::on_response - treats unified-status rejected as rate-limited", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers = make_headers({
        {"anthropic-ratelimit-unified-status", "rejected"},
    });
    const HttpResponse resp{200, "", headers};

    protocol.on_response(resp);

    REQUIRE(protocol.last_rate_limit().is_rate_limited == true);
    REQUIRE(protocol.last_rate_limit().unified_status  == "rejected");
}

TEST_CASE("AnthropicProtocol::on_response - parses overage and fallback headers", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers = make_headers({
        {"anthropic-ratelimit-unified-overage-status", "allowed_warning"},
        {"anthropic-ratelimit-unified-overage-reset", "1743590400"},
        {"anthropic-ratelimit-unified-overage-disabled-reason", "out_of_credits"},
        {"anthropic-ratelimit-unified-fallback", "available"},
    });
    const HttpResponse resp{200, "", headers};

    protocol.on_response(resp);
    const RateLimitInfo& info = protocol.last_rate_limit();

    REQUIRE(info.unified_overage_status == "allowed_warning");
    REQUIRE(info.unified_overage_reset == 1743590400);
    REQUIRE(info.unified_overage_disabled_reason == "out_of_credits");
    REQUIRE(info.unified_fallback_available == true);
}

TEST_CASE("AnthropicProtocol::on_response - keeps zero utilization windows and parses overage utilization", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers = make_headers({
        {"anthropic-ratelimit-unified-5h-utilization", "0"},
        {"anthropic-ratelimit-unified-7d-utilization", "0.72"},
        {"anthropic-ratelimit-unified-overage-utilization", "0.33"},
    });
    const HttpResponse resp{200, "", headers};

    protocol.on_response(resp);
    const RateLimitInfo& info = protocol.last_rate_limit();

    auto has_window = [&](std::string_view label) {
        for (const auto& window : info.usage_windows) {
            if (window.label == label) return true;
        }
        return false;
    };

    REQUIRE(has_window("5h"));
    REQUIRE(has_window("7d"));
    REQUIRE(has_window("overage"));
    REQUIRE(find_window(info, "5h") == Catch::Approx(0.0f));
    REQUIRE(find_window(info, "7d") == Catch::Approx(0.72f));
    REQUIRE(find_window(info, "overage") == Catch::Approx(0.33f));
}

TEST_CASE("AnthropicProtocol::on_response - empty headers yield zeroed RateLimitInfo", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers;
    const HttpResponse resp{200, "", headers};

    protocol.on_response(resp);
    const RateLimitInfo& info = protocol.last_rate_limit();

    REQUIRE(info.has_data()      == false);
    REQUIRE(info.is_rate_limited == false);
}

TEST_CASE("AnthropicProtocol::on_response - fresh clone has empty rate limit", "[lifecycle][anthropic]") {
    AnthropicProtocol original;
    const cpr::Header headers = make_headers({{"retry-after", "5"}});
    const HttpResponse resp{429, "", headers};

    original.on_response(resp);
    REQUIRE(original.last_rate_limit().retry_after == 5);

    // A clone produced before on_response must start with clean state.
    auto clone = original.clone();
    auto* typed = dynamic_cast<AnthropicProtocol*>(clone.get());
    REQUIRE(typed != nullptr);
    REQUIRE(typed->last_rate_limit().retry_after     == 0);
    REQUIRE(typed->last_rate_limit().is_rate_limited == false);
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicProtocol::format_error_message — provider-specific guidance
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("AnthropicProtocol::format_error_message - 400 mentions invalid request", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers;
    const std::string msg = protocol.format_error_message({400, "", headers});

    REQUIRE_THAT(msg, ContainsSubstring("400"));
    REQUIRE_THAT(msg, ContainsSubstring("Invalid request"));
}

TEST_CASE("AnthropicProtocol::format_error_message - 401 mentions authentication", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers;
    const std::string msg = protocol.format_error_message({401, "", headers});

    REQUIRE_THAT(msg, ContainsSubstring("401"));
    REQUIRE_THAT(msg, ContainsSubstring("Authentication"));
}

TEST_CASE("AnthropicProtocol::format_error_message - 429 mentions rate limit", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers;
    const std::string msg = protocol.format_error_message({429, "error body", headers});

    REQUIRE_THAT(msg, ContainsSubstring("429"));
    REQUIRE_THAT(msg, ContainsSubstring("Rate limit"));
}

TEST_CASE("AnthropicProtocol::format_error_message - 429 suggests Haiku fallback for Sonnet/Opus",
          "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    core::llm::ChatRequest req;
    req.model = "claude-sonnet-4-6";
    protocol.prepare_request(req);

    const cpr::Header headers;
    const std::string msg = protocol.format_error_message({429, "error body", headers});

    REQUIRE_THAT(msg, ContainsSubstring("/model claude haiku"));
}

TEST_CASE("AnthropicProtocol::format_error_message - 429 does not suggest Haiku for Haiku model",
          "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    core::llm::ChatRequest req;
    req.model = "claude-haiku-4-5";
    protocol.prepare_request(req);

    const cpr::Header headers;
    const std::string msg = protocol.format_error_message({429, "error body", headers});

    REQUIRE_THAT(msg, !ContainsSubstring("/model claude haiku"));
}

TEST_CASE("AnthropicProtocol::format_error_message - 529 is distinct from 429", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers;

    const std::string msg_429 = protocol.format_error_message({429, "", headers});
    const std::string msg_529 = protocol.format_error_message({529, "", headers});

    REQUIRE_THAT(msg_529, ContainsSubstring("529"));
    REQUIRE_THAT(msg_529, ContainsSubstring("overloaded"));
    // The 529 message must be visibly different from the 429 message.
    REQUIRE(msg_429 != msg_529);
}

TEST_CASE("AnthropicProtocol::format_error_message - 500 mentions server error", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers;
    const std::string msg = protocol.format_error_message({500, "", headers});

    REQUIRE_THAT(msg, ContainsSubstring("500"));
    REQUIRE_THAT(msg, ContainsSubstring("server error"));
}

TEST_CASE("AnthropicProtocol::format_error_message - unknown 5xx mentions retry", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers;
    const std::string msg = protocol.format_error_message({599, "", headers});

    REQUIRE_THAT(msg, ContainsSubstring("599"));
    REQUIRE_THAT(msg, ContainsSubstring("retry"));
}

TEST_CASE("AnthropicProtocol::format_error_message - result has no leading newline", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers;

    // HttpLLMProvider prepends "\n"; the returned string must not include it.
    for (const int code : {400, 401, 429, 500, 529}) {
        const std::string msg = protocol.format_error_message({code, "", headers});
        INFO("status_code = " << code);
        REQUIRE(!msg.empty());
        REQUIRE(msg.front() != '\n');
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicProtocol::is_retryable — transient vs permanent errors
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("AnthropicProtocol::is_retryable - retryable status codes", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers;

    for (const int code : {429, 500, 502, 503, 504, 529}) {
        const HttpResponse resp{code, "", headers};
        INFO("status_code = " << code);
        REQUIRE(protocol.is_retryable(resp));
    }
}

TEST_CASE("AnthropicProtocol::is_retryable - non-retryable status codes", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers;

    for (const int code : {400, 401, 403, 404, 422}) {
        const HttpResponse resp{code, "", headers};
        INFO("status_code = " << code);
        REQUIRE_FALSE(protocol.is_retryable(resp));
    }
}

TEST_CASE("AnthropicProtocol::on_response - parses 5h and 7d utilization windows", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers = make_headers({
        {"anthropic-ratelimit-unified-5h-utilization", "0.63"},
        {"anthropic-ratelimit-unified-7d-utilization", "0.12"},
    });

    protocol.on_response(HttpResponse{200, "", headers});
    const RateLimitInfo& info = protocol.last_rate_limit();

    REQUIRE(info.usage_windows.size() == 2);
    REQUIRE(find_window(info, "5h") == Catch::Approx(0.63f).epsilon(0.001f));
    REQUIRE(find_window(info, "7d") == Catch::Approx(0.12f).epsilon(0.001f));
    REQUIRE(info.has_data());
    // max_window_utilization returns the highest utilization across all windows
    REQUIRE(info.max_window_utilization() == Catch::Approx(0.63f).epsilon(0.001f));
}

TEST_CASE("AnthropicProtocol::on_response - only 7d window present when 5h absent", "[lifecycle][anthropic]") {
    AnthropicProtocol protocol;
    const cpr::Header headers = make_headers({
        {"anthropic-ratelimit-unified-7d-utilization", "0.08"},
    });

    protocol.on_response(HttpResponse{200, "", headers});
    const RateLimitInfo& info = protocol.last_rate_limit();

    REQUIRE(info.usage_windows.size() == 1);
    REQUIRE(find_window(info, "5h") == 0.0f);  // not present
    REQUIRE(find_window(info, "7d") == Catch::Approx(0.08f).epsilon(0.001f));
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenAIProtocol::on_response — x-ratelimit header extraction
// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("OpenAIProtocol::on_response - populates rate limit from x-ratelimit headers", "[lifecycle][openai]") {
    OpenAIProtocol protocol;
    const cpr::Header headers = make_headers({
        {"x-ratelimit-limit-requests",     "500"},
        {"x-ratelimit-remaining-requests", "483"},
        {"x-ratelimit-limit-tokens",       "200000"},
        {"x-ratelimit-remaining-tokens",   "187500"},
    });
    const HttpResponse resp{200, "", headers};

    protocol.on_response(resp);
    const RateLimitInfo& info = protocol.last_rate_limit();

    REQUIRE(info.requests_limit     == 500);
    REQUIRE(info.requests_remaining == 483);
    REQUIRE(info.tokens_limit       == 200000);
    REQUIRE(info.tokens_remaining   == 187500);
    REQUIRE(info.is_rate_limited    == false);
}

TEST_CASE("OpenAIProtocol::on_response - detects rate-limited from retry-after header", "[lifecycle][openai]") {
    OpenAIProtocol protocol;
    const cpr::Header headers = make_headers({{"retry-after", "15"}});
    const HttpResponse resp{429, "", headers};

    protocol.on_response(resp);
    const RateLimitInfo& info = protocol.last_rate_limit();

    REQUIRE(info.retry_after     == 15);
    REQUIRE(info.is_rate_limited == true);
}

TEST_CASE("OpenAIProtocol::on_response - detects rate-limited from 429 status", "[lifecycle][openai]") {
    OpenAIProtocol protocol;
    const cpr::Header headers;
    const HttpResponse resp{429, "", headers};

    protocol.on_response(resp);

    REQUIRE(protocol.last_rate_limit().is_rate_limited == true);
}

TEST_CASE("OpenAIProtocol::on_response - empty headers yield zeroed RateLimitInfo", "[lifecycle][openai]") {
    OpenAIProtocol protocol;
    const cpr::Header headers;
    const HttpResponse resp{200, "", headers};

    protocol.on_response(resp);
    const RateLimitInfo& info = protocol.last_rate_limit();

    REQUIRE(info.has_data()      == false);
    REQUIRE(info.is_rate_limited == false);
}

TEST_CASE("OpenAIProtocol::on_response - assumes full remaining when limit known but remaining absent", "[lifecycle][openai]") {
    OpenAIProtocol protocol;
    const cpr::Header headers = make_headers({
        {"x-ratelimit-limit-tokens", "100000"},
    });
    const HttpResponse resp{200, "", headers};

    protocol.on_response(resp);
    const RateLimitInfo& info = protocol.last_rate_limit();

    REQUIRE(info.tokens_limit     == 100000);
    REQUIRE(info.tokens_remaining == 100000);  // defaulted to limit when absent
}

TEST_CASE("OpenAIProtocol::on_response - fresh clone has empty rate limit", "[lifecycle][openai]") {
    OpenAIProtocol original;
    const cpr::Header headers = make_headers({{"retry-after", "5"}});
    const HttpResponse resp{429, "", headers};

    original.on_response(resp);
    REQUIRE(original.last_rate_limit().retry_after == 5);

    auto clone = original.clone();
    auto* typed = dynamic_cast<OpenAIProtocol*>(clone.get());
    REQUIRE(typed != nullptr);
    REQUIRE(typed->last_rate_limit().retry_after     == 0);
    REQUIRE(typed->last_rate_limit().is_rate_limited == false);
}
