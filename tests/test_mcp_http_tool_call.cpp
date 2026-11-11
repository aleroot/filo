#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <httplib.h>
#include <simdjson.h>

#include "TestSessionContext.hpp"
#include "core/auth/ApiKeyCredentialSource.hpp"
#include "core/llm/HttpLLMProvider.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/tools/McpHttpToolCall.hpp"
#include "core/tools/WebBackendAdapters.hpp"

#include <atomic>
#include <format>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using Catch::Matchers::ContainsSubstring;
using core::tools::web::detail::McpHttpFailure;

namespace {

class ScopedServer {
public:
    ScopedServer() = default;
    ~ScopedServer() {
        server.stop();
        if (thread_.joinable()) thread_.join();
    }
    [[nodiscard]] int start() {
        port_ = server.bind_to_any_port("127.0.0.1");
        if (port_ > 0) thread_ = std::thread([this] { server.listen_after_bind(); });
        return port_;
    }
    [[nodiscard]] std::string origin() const { return std::format("http://127.0.0.1:{}", port_); }
    httplib::Server server;

private:
    int port_ = 0;
    std::thread thread_;
};

/// Records every request and answers by JSON-RPC method.
struct McpServer {
    ScopedServer http;
    std::mutex mutex;
    std::vector<httplib::Request> received;
    std::string protocol_version = "2025-06-18";
    std::string tool_result = R"({"content":[{"type":"text","text":"done"}]})";

    explicit McpServer(std::string path) {
        http.server.Post(path, [this](const httplib::Request& request, httplib::Response& response) {
            {
                std::lock_guard lock(mutex);
                received.push_back(request);
            }
            simdjson::dom::parser parser;
            const std::string_view method = parser.parse(request.body)["method"].get_string().value();
            if (method == "initialize") {
                response.set_header("Mcp-Session-Id", "s-1");
                response.set_content(std::format(
                    R"({{"jsonrpc":"2.0","id":1,"result":{{"protocolVersion":"{}","capabilities":{{}}}}}})",
                    protocol_version), "application/json");
            } else if (method == "notifications/initialized") {
                response.status = 202;
            } else {
                response.set_content(std::format(R"({{"jsonrpc":"2.0","id":2,"result":{}}})", tool_result),
                                     "application/json");
            }
        });
    }
};

/// A credential source whose token changes when it is asked to refresh.
class RotatingCredentials final : public core::auth::ICredentialSource {
public:
    core::auth::AuthInfo get_auth() override {
        core::auth::AuthInfo auth;
        auth.headers["Authorization"] = "Bearer " + token;
        return auth;
    }
    bool refresh_on_auth_failure() override {
        ++refreshes;
        token = "fresh";
        return true;
    }
    std::string token = "stale";
    int refreshes = 0;
};

} // namespace

TEST_CASE("An MCP tool call carries the session and only a defined version header",
          "[tools][web][mcp]") {
    McpServer mcp("/mcp");
    if (mcp.http.start() <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    const auto credentials = core::auth::ApiKeyCredentialSource::as_bearer("k");

    SECTION("2025-06-18 and later send MCP-Protocol-Version after initialize") {
        const auto text = core::tools::web::detail::call_mcp_http_tool({
            .url = mcp.http.origin() + "/mcp",
            .tool = "lookup",
            .arguments_json = R"({"q":"x"})",
            .protocol_version = "2025-11-25",
            .credentials = credentials.get(),
        });
        REQUIRE(text);
        CHECK(*text == "done");
        std::lock_guard lock(mcp.mutex);
        REQUIRE(mcp.received.size() == 3);
        CHECK_THAT(mcp.received[0].body, ContainsSubstring(R"("protocolVersion":"2025-11-25")"));
        CHECK_FALSE(mcp.received[0].has_header("Mcp-Session-Id"));
        CHECK_FALSE(mcp.received[0].has_header("MCP-Protocol-Version"));
        for (std::size_t i = 1; i < 3; ++i) {
            CHECK(mcp.received[i].get_header_value("Mcp-Session-Id") == "s-1");
            CHECK(mcp.received[i].get_header_value("MCP-Protocol-Version") == "2025-06-18");
        }
        CHECK_THAT(mcp.received[2].body, ContainsSubstring(R"("name":"lookup","arguments":{"q":"x"})"));
    }

    SECTION("an older negotiated revision gets no version header, which it does not define") {
        mcp.protocol_version = "2024-11-05";
        const auto text = core::tools::web::detail::call_mcp_http_tool({
            .url = mcp.http.origin() + "/mcp",
            .tool = "lookup",
            .arguments_json = "{}",
            .protocol_version = "2024-11-05",
            .credentials = credentials.get(),
        });
        REQUIRE(text);
        std::lock_guard lock(mcp.mutex);
        REQUIRE(mcp.received.size() == 3);
        CHECK_FALSE(mcp.received[2].has_header("MCP-Protocol-Version"));
        CHECK(mcp.received[2].get_header_value("Mcp-Session-Id") == "s-1");
    }
}

TEST_CASE("An MCP tool call refreshes credentials once on 401", "[tools][web][mcp]") {
    ScopedServer http;
    std::atomic<int> unauthorized = 0;
    http.server.Post("/mcp", [&](const httplib::Request& request, httplib::Response& response) {
        if (request.get_header_value("Authorization") != "Bearer fresh") {
            ++unauthorized;
            response.status = 401;
            return;
        }
        simdjson::dom::parser parser;
        const std::string_view method = parser.parse(request.body)["method"].get_string().value();
        const int id = method == "initialize" ? 1 : 2;
        response.set_content(std::format(R"({{"jsonrpc":"2.0","id":{},"result":{{"content":[]}}}})", id),
                             "application/json");
    });
    if (http.start() <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");

    RotatingCredentials credentials;
    const auto text = core::tools::web::detail::call_mcp_http_tool({
        .url = http.origin() + "/mcp", .tool = "t", .arguments_json = "{}",
        .protocol_version = "2025-06-18", .credentials = &credentials,
    });
    REQUIRE(text);
    CHECK(text->empty());
    CHECK(credentials.refreshes == 1);
    CHECK(unauthorized == 1);
}

TEST_CASE("MCP tool failures keep their kind for the caller's wording", "[tools][web][mcp]") {
    McpServer mcp("/mcp");
    if (mcp.http.start() <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");
    const auto credentials = core::auth::ApiKeyCredentialSource::as_bearer("k");
    const auto call = [&] {
        return core::tools::web::detail::call_mcp_http_tool({
            .url = mcp.http.origin() + "/mcp", .tool = "t", .arguments_json = "{}",
            .protocol_version = "2025-06-18", .credentials = credentials.get(),
        });
    };

    mcp.tool_result = R"({"isError":true,"content":[{"type":"text","text":"quota spent"}]})";
    auto failed = call();
    REQUIRE_FALSE(failed);
    CHECK(failed.error().kind == McpHttpFailure::Kind::Tool);
    CHECK(core::tools::web::detail::describe_mcp_failure(failed.error(), "Search")
          == "Search MCP tool call failed: quota spent");

    const auto missing = core::tools::web::detail::call_mcp_http_tool({
        .url = mcp.http.origin() + "/absent", .tool = "t", .arguments_json = "{}",
        .protocol_version = "2025-06-18", .credentials = credentials.get(),
    });
    REQUIRE_FALSE(missing);
    CHECK(missing.error().kind == McpHttpFailure::Kind::Http);
    CHECK(missing.error().status == 404);
    CHECK(missing.error().stage == "initialize");
}

TEST_CASE("Z.ai web search calls web_search_prime over the shared MCP session",
          "[tools][web][zai][mcp]") {
    McpServer mcp("/api/mcp/web_search_prime/mcp");
    mcp.protocol_version = "2024-11-05";
    mcp.tool_result = R"({"content":[{"type":"text","text":"[{\"title\":\"GLM\",\"link\":\"https://z.ai/glm\",\"content\":\"GLM-5.3\"}]"}]})";
    if (mcp.http.start() <= 0) SKIP("Local socket bind/listen is unavailable in this environment.");

    auto provider = std::make_shared<core::llm::HttpLLMProvider>(
        mcp.http.origin() + "/api/coding/paas/v4",
        core::auth::ApiKeyCredentialSource::as_bearer("zai-key"),
        "glm-5.3",
        std::make_unique<core::llm::protocols::OpenAIProtocol>(),
        core::config::ApiType::OpenAI,
        "zai-coding");
    const core::tools::ToolInvocationContext invocation{
        .session_context = test_support::make_workspace_session_context(),
        .model_name = "glm-5.3",
        .provider = std::move(provider),
    };

    const auto result = core::tools::web::make_zai_web_search_backend()->search(
        {.query = "glm", .limit = 5, .domains = {.allowed_domains = {"z.ai"}}}, invocation);
    REQUIRE(result);
    REQUIRE(result->results.size() == 1);
    CHECK(result->results[0].url == "https://z.ai/glm");

    std::lock_guard lock(mcp.mutex);
    REQUIRE(mcp.received.size() == 3);
    for (const auto& request : mcp.received) {
        CHECK(request.get_header_value("Authorization") == "Bearer zai-key");
    }
    CHECK(mcp.received[2].get_header_value("Mcp-Session-Id") == "s-1");
    CHECK_FALSE(mcp.received[2].has_header("MCP-Protocol-Version"));
    CHECK_THAT(mcp.received[2].body, ContainsSubstring(R"("name":"web_search_prime")"));
    CHECK_THAT(mcp.received[2].body, ContainsSubstring(R"("search_query":"glm")"));
    CHECK_THAT(mcp.received[2].body, ContainsSubstring(R"("search_domain_filter":"z.ai")"));
}
