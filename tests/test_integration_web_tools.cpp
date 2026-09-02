#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <httplib.h>
#include <simdjson.h>

#include "core/config/ConfigManager.hpp"
#include "core/context/SessionContext.hpp"
#include "core/tools/WebAccess.hpp"
#include "core/tools/WebBackendAdapters.hpp"
#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/tools/WebFetchTool.hpp"
#include "core/workspace/Workspace.hpp"
#include "TestSessionContext.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

using namespace core::tools;

namespace {

class ScopedServerStop {
public:
    explicit ScopedServerStop(httplib::Server& server)
        : server_(server) {}

    ~ScopedServerStop() {
        server_.stop();
    }

private:
    httplib::Server& server_;
};

class ScopedConfigReload {
public:
    ~ScopedConfigReload() {
        core::config::ConfigManager::get_instance().load(std::filesystem::current_path());
    }
};

[[nodiscard]] core::context::SessionContext make_tool_test_context(std::string session_id = {}) {
    core::config::ConfigManager::get_instance().load(std::filesystem::current_path());
    return test_support::make_workspace_session_context(
        core::context::SessionTransport::cli,
        std::move(session_id));
}

[[nodiscard]] std::string run_web_fetch_tool(
    WebFetchTool& tool,
    const std::string& json_args,
    const core::tools::ToolInvocationContext& invocation) {
    auto run = static_cast<std::string (WebFetchTool::*)(
        const std::string&,
        const core::tools::ToolInvocationContext&)>(&WebFetchTool::execute);
    return (tool.*run)(json_args, invocation);
}

} // namespace

TEST_CASE("Claude web search uses compatible tool choice and shared client headers",
          "[integration][web][claude][fable51]") {
    httplib::Server server;
    std::vector<httplib::Request> received;
    std::mutex mutex;
    server.Post("/v1/messages", [&](const httplib::Request& request, httplib::Response& response) {
        std::lock_guard lock(mutex);
        received.push_back(request);
        response.set_content(R"({"content":[
            {"type":"web_search_tool_result","content":[{"type":"web_search_result","title":"Reference","url":"https://example.com/reference"}]},
            {"type":"text","text":"Found a reference."}]})", "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);

    const auto backend = web::make_anthropic_web_search_backend();
    const std::vector<std::string> models{
        "fable", "claude-fable-5-1", "claude-mythos-5-1", "claude-fable-5", "sonnet[1m]"};
    for (const auto& model : models) {
        auto provider = std::make_shared<core::llm::HttpLLMProvider>(
            std::format("http://127.0.0.1:{}", port),
            core::auth::ApiKeyCredentialSource::as_bearer("test-oauth-token", true),
            model, std::make_unique<core::llm::protocols::AnthropicProtocol>(),
            core::config::ApiType::Anthropic, "claude");
        const auto response = backend->search(web::SearchRequest{.query = "reference \"quoted\""},
            ToolInvocationContext{.session_context = test_support::make_workspace_session_context(),
                                  .model_name = model, .provider = std::move(provider)});
        INFO((response ? "success" : response.error()));
        REQUIRE(response.has_value());
        REQUIRE(response->results.size() == 1);
        CHECK(response->results[0].url == "https://example.com/reference");
    }
    std::lock_guard lock(mutex);
    REQUIRE(received.size() == models.size());
    for (std::size_t i = 0; i < received.size(); ++i) {
        CAPTURE(models[i]);
        const auto& actual = received[i];
        simdjson::dom::parser parser;
        const auto doc = parser.parse(actual.body);
        const auto choice = doc["tool_choice"]["type"].get_string().value();
        CHECK(choice == (i < 3 ? "auto" : "tool"));
        if (i < 3) CHECK(doc["tool_choice"]["name"].error() == simdjson::NO_SUCH_FIELD);
        CHECK_THAT(actual.get_header_value("x-anthropic-billing-header"), Catch::Matchers::StartsWith("cc_version=2.1.255"));
        CHECK_THAT(actual.body, Catch::Matchers::ContainsSubstring(actual.get_header_value("x-anthropic-billing-header")));
        CHECK_THAT(actual.get_header_value("anthropic-beta"), Catch::Matchers::ContainsSubstring("web-search-2025-03-05"));
        CHECK_THAT(actual.get_header_value("anthropic-beta"), Catch::Matchers::ContainsSubstring("oauth-2025-04-20"));
        CHECK(actual.get_header_value("Authorization") == "Bearer test-oauth-token");
        CHECK(actual.get_header_value("Accept") == "application/json");
        CHECK_THAT(std::string(doc["messages"].at(0)["content"].get_string().value()),
                   Catch::Matchers::ContainsSubstring("Call the web_search tool"));
        if (i == 0) CHECK(doc["model"].get_string().value() == "claude-fable-5-1");
        if (i == 4) {
            CHECK(doc["model"].get_string().value() == "claude-sonnet-5");
            CHECK_THAT(actual.get_header_value("anthropic-beta"), Catch::Matchers::ContainsSubstring("context-1m-2025-08-07"));
        }
    }
}

TEST_CASE("Claude web search distinguishes missing search and tool errors from empty results",
          "[integration][web][claude][fable51]") {
    std::string response_body;
    std::string expected_error;
    SECTION("Model answered without invoking search") {
        response_body = R"({"content":[{"type":"text","text":"An answer from memory."}]})";
        expected_error = "did not complete a web search";
    }
    SECTION("Search tool returned an API error") {
        response_body = R"({"content":[{"type":"web_search_tool_result","content":{"type":"web_search_tool_result_error","error_code":"max_uses_exceeded"}}]})";
        expected_error = "max_uses_exceeded";
    }
    SECTION("A completed search found no matches") {
        response_body = R"({"content":[{"type":"web_search_tool_result","content":[]}]})";
    }
    httplib::Server server;
    server.Post("/v1/messages", [&](const httplib::Request&, httplib::Response& response) {
        response.set_content(response_body, "application/json");
    });
    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    std::jthread server_thread([&server]() { server.listen_after_bind(); });
    ScopedServerStop stop_server(server);
    auto provider = std::make_shared<core::llm::HttpLLMProvider>(
        std::format("http://127.0.0.1:{}", port),
        core::auth::ApiKeyCredentialSource::as_custom_header("test-key", "x-api-key"),
        "claude-fable-5-1", std::make_unique<core::llm::protocols::AnthropicProtocol>(),
        core::config::ApiType::Anthropic, "claude");
    const auto response = web::make_anthropic_web_search_backend()->search(
        web::SearchRequest{.query = "a test query"},
        ToolInvocationContext{.session_context = test_support::make_workspace_session_context(),
                              .model_name = "claude-fable-5-1", .provider = std::move(provider)});
    if (expected_error.empty()) {
        REQUIRE(response.has_value());
        CHECK(response->results.empty());
    } else {
        REQUIRE_FALSE(response.has_value());
        CHECK_THAT(response.error(), Catch::Matchers::ContainsSubstring(expected_error));
    }
}

TEST_CASE("WebFetchTool validates redirected URLs against trusted URL policy",
          "[integration][tools][web]") {
    httplib::Server server;
    bool private_endpoint_hit = false;
    server.Get("/redirect", [](const httplib::Request&, httplib::Response& res) {
        res.status = 302;
        res.set_header("Location", "/private");
    });
    server.Get("/private", [&](const httplib::Request&, httplib::Response& res) {
        private_endpoint_hit = true;
        res.set_content("private data", "text/plain");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() {
        server.listen_after_bind();
    });
    ScopedServerStop stop_server(server);

    const auto stamp = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto sandbox = std::filesystem::temp_directory_path()
        / ("filo_web_fetch_redirect_policy_" + stamp);
    const auto project = sandbox / "project";
    const auto config_path = project / ".filo" / "config.json";
    std::filesystem::create_directories(config_path.parent_path());

    const std::string trusted_url =
        std::format("http://127.0.0.1:{}/redirect", port);
    {
        std::ofstream config(config_path);
        config << std::format(R"({{
            "tools": {{
                "fetch_url": {{
                    "trusted_urls": ["{}"]
                }}
            }}
        }})", trusted_url);
    }

    auto& manager = core::config::ConfigManager::get_instance();
    manager.load(project);
    ScopedConfigReload restore_config;

    WebFetchTool tool;
    const auto res = run_web_fetch_tool(
        tool,
        std::format(R"({{"url":"{}"}})", trusted_url),
        core::tools::ToolInvocationContext{
            .session_context = test_support::make_session_context(
                core::workspace::WorkspaceSnapshot{
                    .primary = project,
                    .additional = {},
                    .enforce = false,
                    .version = 1,
                },
                core::context::SessionTransport::cli,
                "web-fetch-redirect-policy"),
        });

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("trusted_urls"));
    REQUIRE_FALSE(private_endpoint_hit);

    std::error_code ec;
    std::filesystem::remove_all(sandbox, ec);
}

TEST_CASE("WebFetchTool returns a usable truncated result for oversized responses",
          "[integration][tools][web]") {
    httplib::Server server;
    server.Get("/large", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(std::string(2 * 1024 * 1024 + 4096, 'x'), "text/plain");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() {
        server.listen_after_bind();
    });
    ScopedServerStop stop_server(server);

    WebFetchTool tool;
    const auto res = run_web_fetch_tool(
        tool,
        std::format(
            R"({{"url":"http://127.0.0.1:{}/large"}})",
            port),
        core::tools::ToolInvocationContext{
            .session_context = make_tool_test_context("web-fetch-max-bytes"),
        });

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring(R"("truncated":true)"));
    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring(R"("content":")"));
    REQUIRE_FALSE(res.find(R"("error":)") != std::string::npos);
    REQUIRE(res.size() < 128 * 1024);
}

TEST_CASE("WebFetchTool ignores legacy max_bytes arguments", "[integration][tools][web]") {
    httplib::Server server;
    server.Get("/small", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(std::string(4096, 'x'), "text/plain");
    });

    const int port = server.bind_to_any_port("127.0.0.1");
    if (port <= 0) {
        SKIP("Local socket bind/listen is unavailable in this environment.");
    }
    std::jthread server_thread([&server]() {
        server.listen_after_bind();
    });
    ScopedServerStop stop_server(server);

    WebFetchTool tool;
    const auto res = run_web_fetch_tool(
        tool,
        std::format(
            R"({{"url":"http://127.0.0.1:{}/small","max_bytes":1024}})",
            port),
        core::tools::ToolInvocationContext{
            .session_context = make_tool_test_context("web-fetch-legacy-max-bytes"),
        });

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring(R"("truncated":false)"));
    REQUIRE_FALSE(res.find(R"("error":)") != std::string::npos);
}

TEST_CASE("Web fetch output truncation preserves UTF-8 boundaries", "[integration][tools][web]") {
    core::tools::web::FetchResponse response{
        .text = std::string(64 * 1024 - 1, 'x') + "\xE2\x82\xACsuffix",
    };

    const auto json = core::tools::web::fetch_response_to_json(response);

    REQUIRE_THAT(json, Catch::Matchers::ContainsSubstring(R"("truncated":true)"));
    REQUIRE_FALSE(json.find("\xE2\x82\xAC") != std::string::npos);
}

TEST_CASE("Web response JSON repairs malformed external bytes",
          "[integration][tools][web][utf8]") {
    core::tools::web::FetchResponse response{
        .final_url = "https://example.com",
        .content_type = "text/plain",
        .title = std::string("bad ") + "\xc2" + " title",
        .text = std::string("bad ") + "\xc2" + " content",
        .status_code = 200,
    };

    const auto json = core::tools::web::fetch_response_to_json(response);

    REQUIRE(simdjson::validate_utf8(json));
    REQUIRE_THAT(json, Catch::Matchers::ContainsSubstring(R"(bad \ufffd title)"));
    REQUIRE_THAT(json, Catch::Matchers::ContainsSubstring(R"(bad \ufffd content)"));
}
