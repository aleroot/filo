#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <httplib.h>
#include <simdjson.h>

#include "TestSessionContext.hpp"
#include "core/llm/ProviderFactory.hpp"
#include "core/tools/WebBackendAdapters.hpp"
#include "core/tools/WebSearchTool.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <format>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace core::tools;

class ScopedServerStop {
public:
    explicit ScopedServerStop(httplib::Server& server) : server_(server) {}
    ~ScopedServerStop() { server_.stop(); }
private:
    httplib::Server& server_;
};

ToolInvocationContext invocation_for(std::string base_url,
                                     std::string provider_name = "mimo-token-plan-sgp") {
    core::config::ProviderConfig config;
    config.base_url = std::move(base_url);
    config.api_key = "tp-test-mimo-key";
    config.model = "mimo-v2.6-pro";
    return {.session_context = test_support::make_workspace_session_context(),
            .provider_name = provider_name,
            .model_name = "mimo-v2.6-flash",
            .provider = core::llm::ProviderFactory::create_provider(provider_name, config)};
}

constexpr std::string_view kAnnotations = R"([
    {"type":"url_citation","url":"https://example.com/one","title":"One","summary":"First source","publish_time":"2026-10-24"},
    {"url":"https://example.com/one","title":"Duplicate"},
    {"title":"Missing URL"},
    {"url":"https://example.com/two","site_name":"Two","summary":"Second source"},
    {"url":"https://example.com/three","title":"Over limit"}])";

} // namespace

TEST_CASE("MiMo web search uses the active gateway, credentials and model through the tool registry",
          "[integration][mimo][web]") {
    std::string field = "delta";
    std::string newline = "\r\n";
    SECTION("Streaming delta annotations with CRLF framing") {}
    SECTION("Streaming message annotations with LF framing") { field = "message"; newline = "\n"; }
    const std::string frame = "data: {\"choices\":[{\"" + field + "\":{\"annotations\":"
        + std::string(kAnnotations) + "}}]}";
    // Minify so this is a single SSE data line, as the real gateway sends.
    simdjson::dom::parser frame_parser;
    const std::string compact = "data: " + simdjson::minify(frame_parser.parse(frame.substr(6)));
    const auto body = ": keepalive" + newline + newline
        + "data: {\"choices\":[{\"delta\":{\"annotations\":[]}}]}" + newline + newline
        + compact + newline + newline
        + "data: {\"error\":{\"message\":\"Do not consume after sources\"}}" + newline + newline;
    httplib::Server server;
    std::vector<httplib::Request> received;
    std::mutex mutex;
    server.Post("/v1/chat/completions", [&](const httplib::Request& request, httplib::Response& response) {
        {
            std::lock_guard lock(mutex);
            received.push_back(request);
        }
        // No [DONE] or generated answer: sources are sufficient for success.
        response.set_chunked_content_provider("text/event-stream",
            [body](std::size_t offset, httplib::DataSink& sink) {
                const auto count = std::min(std::size_t{7}, body.size() - offset);
                if (!sink.write(body.data() + offset, count)) return false;
                if (offset + count == body.size()) sink.done();
                return true;
            });
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    std::jthread server_thread([&] { server.listen_after_bind(); });
    ScopedServerStop stop(server);
    for (const auto suffix : {"", "/v1/", "/v1/chat/completions"}) {
        const auto invocation = invocation_for(std::format("http://127.0.0.1:{}{}", port, suffix));
        REQUIRE(web::make_mimo_web_search_backend()->supports(invocation));
        WebSearchTool tool;
        const auto output = tool.execute(R"({"query":"MiMo \"latest\" news","limit":2})", invocation);
        INFO(output);
        simdjson::dom::parser parser;
        const auto doc = parser.parse(output);
        REQUIRE(doc["backend"].get_string().value() == "mimo-native-web-search");
        const auto hits = doc["results"].get_array().value();
        REQUIRE(hits.size() == 2);
        CHECK(hits.at(0)["url"].get_string().value() == "https://example.com/one");
        CHECK(hits.at(0)["snippet"].get_string().value() == "2026-10-24 — First source");
        CHECK(hits.at(1)["title"].get_string().value() == "Two");
    }
    std::lock_guard lock(mutex);
    REQUIRE(received.size() == 3);
    for (const auto& request : received) {
        CHECK(request.get_header_value("api-key") == "tp-test-mimo-key");
        CHECK_FALSE(request.has_header("Authorization"));
        CHECK(request.get_header_value("X-Mimo-Source") == "mimocode-cli");
        CHECK(request.get_header_value("Accept") == "text/event-stream");
        CHECK(request.get_header_value("Content-Type") == "application/json");
        CHECK(request.body.find("tp-test-mimo-key") == std::string::npos);
        simdjson::dom::parser parser;
        const auto doc = parser.parse(request.body);
        CHECK(doc["model"].get_string().value() == "mimo-v2.6-flash");
        CHECK(doc["messages"].at(0)["content"].get_string().value() == "MiMo \"latest\" news");
        CHECK(doc["tools"].at(0)["type"].get_string().value() == "web_search");
        CHECK(doc["tools"].at(0)["force_search"].get_bool().value());
        CHECK(doc["tools"].at(0)["max_keyword"].get_int64().value() == 3);
        CHECK(doc["tools"].at(0)["limit"].get_int64().value() == 2);
        CHECK(doc["thinking"]["type"].get_string().value() == "disabled");
        CHECK(doc["stream"].get_bool().value());
        CHECK(doc["max_completion_tokens"].get_int64().value() == 256);
        CHECK(doc["temperature"].get_double().value() == 1.0);
        CHECK(doc["top_p"].get_double().value() == 0.95);
    }
}

TEST_CASE("MiMo web search accepts JSON and final unterminated SSE source frames",
          "[integration][mimo][web]") {
    std::string body = R"({"choices":[{"message":{"annotations":[{"url":"https://example.com/source","title":"Source"}]}}]})";
    SECTION("JSON response") {}
    SECTION("SSE frame without final blank line") { body = "data: " + body; }
    httplib::Server server;
    server.Post("/v1/chat/completions", [&](const httplib::Request& request, httplib::Response& response) {
        simdjson::dom::parser parser;
        CHECK(parser.parse(request.body)["model"].get_string().value() == "mimo-v2.6-pro");
        CHECK(parser.parse(request.body)["tools"].at(0)["limit"].get_int64().value() == 5);
        response.set_content(body, "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    std::jthread server_thread([&] { server.listen_after_bind(); });
    ScopedServerStop stop(server);
    auto invocation = invocation_for(std::format("http://127.0.0.1:{}/v1", port));
    invocation.model_name.clear();
    const auto result = web::make_mimo_web_search_backend()->search({.query = "source", .limit = 20}, invocation);
    INFO((result ? "success" : result.error()));
    REQUIRE(result);
    REQUIRE(result->results.size() == 1);
    CHECK(result->results[0].url == "https://example.com/source");
}

TEST_CASE("MiMo web search reports API errors and missing sources without inventing results",
          "[integration][mimo][web]") {
    int status = 200;
    std::string body;
    std::string expected;
    SECTION("Quota exhausted") { status = 409; expected = "quota exhausted"; }
    SECTION("Wrong key or gateway") { status = 401; expected = "Authentication failed"; }
    SECTION("Permission denied") { status = 403; expected = "Permission denied"; }
    SECTION("Rate limit") { status = 429; expected = "Rate limited"; }
    SECTION("Gateway error") { status = 503; body = R"({"error":{"message":"Gateway unavailable"}})"; expected = "Gateway unavailable"; }
    SECTION("Risk control") { status = 400; body = R"({"error":{"code":441,"param":"policy block"}})"; expected = "risk control"; }
    SECTION("Streaming API error") { body = "data: {\"error\":{\"message\":\"plugin disabled\"}}\n\n"; expected = "plugin disabled"; }
    SECTION("JSON API error") { body = R"({"error":{"message":"plugin disabled"}})"; expected = "plugin disabled"; }
    SECTION("Answer without sources") { body = R"({"choices":[{"message":{"content":"An answer from memory."}}]})"; expected = "did not return search sources"; }
    SECTION("Empty citations") { body = "data: {\"choices\":[{\"delta\":{\"annotations\":[]}}]}\n\ndata: [DONE]\n\n"; expected = "did not return search sources"; }
    SECTION("Malformed response") { body = "not JSON"; expected = "did not return search sources"; }
    SECTION("Oversized response") { body.assign(4 * 1024 * 1024 + 1, 'x'); expected = "size limit"; }
    httplib::Server server;
    server.Post("/v1/chat/completions", [&](const httplib::Request&, httplib::Response& response) {
        response.status = status;
        response.set_content(body, "text/event-stream");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    std::jthread server_thread([&] { server.listen_after_bind(); });
    ScopedServerStop stop(server);
    const auto result = web::make_mimo_web_search_backend()->search({.query = "source"},
        invocation_for(std::format("http://127.0.0.1:{}/v1", port)));
    REQUIRE_FALSE(result);
    CHECK_THAT(result.error(), Catch::Matchers::ContainsSubstring(expected));
}

TEST_CASE("MiMo web search validates filters and cancellation before sending a request",
          "[integration][mimo][web]") {
    auto invocation = invocation_for("http://127.0.0.1:1/v1");
    web::SearchRequest request{.query = "source"};
    std::string expected;
    SECTION("Allowed domains") { request.domains.allowed_domains = {"example.com"}; expected = "does not support"; }
    SECTION("Blocked domains") { request.domains.blocked_domains = {"example.com"}; expected = "does not support"; }
    SECTION("Cancellation") { invocation.cancellation_requested = [] { return true; }; expected = "cancelled"; }
    SECTION("No provider") { invocation.provider.reset(); expected = "active MiMo provider"; }
    const auto result = web::make_mimo_web_search_backend()->search(request, invocation);
    REQUIRE_FALSE(result);
    CHECK_THAT(result.error(), Catch::Matchers::ContainsSubstring(expected));
}

TEST_CASE("MiMo web search bounds stalled requests and cancels in flight",
          "[integration][mimo][web]") {
    std::atomic<bool> cancelled = false;
    bool cancel_in_flight = false;
    SECTION("Timeout") {}
    SECTION("Cancellation during response") { cancel_in_flight = true; }
    httplib::Server server;
    server.Post("/v1/chat/completions", [&](const httplib::Request&, httplib::Response& response) {
        if (!cancel_in_flight) std::this_thread::sleep_for(std::chrono::milliseconds(150));
        response.set_chunked_content_provider("text/event-stream",
            [&](std::size_t, httplib::DataSink& sink) {
                if (cancel_in_flight) cancelled = true;
                if (!sink.write(": keepalive\n\n", 13)) return false;
                sink.done();
                return true;
            });
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    std::jthread server_thread([&] { server.listen_after_bind(); });
    ScopedServerStop stop(server);
    auto invocation = invocation_for(std::format("http://127.0.0.1:{}/v1", port));
    invocation.cancellation_requested = [&] { return cancelled.load(); };
    const auto backend = web::make_mimo_web_search_backend({.timeout_ms = cancel_in_flight ? 1000 : 30});
    const auto result = backend->search({.query = "source"}, invocation);
    REQUIRE_FALSE(result);
    CHECK_THAT(result.error(), Catch::Matchers::ContainsSubstring(cancel_in_flight ? "cancelled" : "request failed"));
}

TEST_CASE("MiMo web search redacts reflected credentials before returning or truncating responses",
          "[integration][mimo][web][security]") {
    int status = 200;
    std::string body;
    bool citations = false;
    SECTION("Streaming API error") { body = "data: {\"error\":{\"message\":\"tp-test-mimo-key\"}}\n\n"; }
    SECTION("JSON API error") { body = R"({"error":{"message":"tp-test-mimo-key"}})"; }
    SECTION("HTTP error") { status = 503; body = R"({"error":{"message":"Bearer tp-test-mimo-key"}})"; }
    SECTION("Credential straddles the error truncation boundary") {
        status = 503;
        body = "{\"error\":{\"message\":\"" + std::string(970, 'x') + "tp-test-mimo-key\"}}";
    }
    SECTION("Citation fields") {
        citations = true;
        body = R"({"choices":[{"message":{"annotations":[{"title":"tp-test-mimo-key","url":"https://example.com/tp-test-mimo-key","summary":"Bearer tp-test-mimo-key"}]}}]})";
    }
    httplib::Server server;
    server.Post("/v1/chat/completions", [&](const httplib::Request&, httplib::Response& response) {
        response.status = status;
        response.set_content(body, "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    std::jthread server_thread([&] { server.listen_after_bind(); });
    ScopedServerStop stop(server);
    for (const auto provider : {"mimo", "mimo-token-plan-sgp"}) {
        CAPTURE(provider);
        const auto result = web::make_mimo_web_search_backend()->search({.query = "source"},
            invocation_for(std::format("http://127.0.0.1:{}/v1", port), provider));
        REQUIRE(result.has_value() == citations);
        const auto output = result ? web::search_response_to_json(*result) : result.error();
        CHECK(output.find("tp-test") == std::string::npos);
        CHECK_THAT(output, Catch::Matchers::ContainsSubstring("[REDACTED]"));
    }
}

TEST_CASE("MiMo web search never forwards credentials through redirects",
          "[integration][mimo][web][security]") {
    httplib::Server server;
    std::atomic<int> redirected_requests = 0;
    server.Post("/v1/chat/completions", [&](const httplib::Request&, httplib::Response& response) {
        response.set_redirect("/credential-sink", 307);
    });
    server.Post("/credential-sink", [&](const httplib::Request&, httplib::Response& response) {
        ++redirected_requests;
        response.set_content("{}", "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    std::jthread server_thread([&] { server.listen_after_bind(); });
    ScopedServerStop stop(server);
    const auto result = web::make_mimo_web_search_backend()->search({.query = "source"},
        invocation_for(std::format("http://127.0.0.1:{}/v1", port)));
    REQUIRE_FALSE(result);
    CHECK(redirected_requests.load() == 0);
}
