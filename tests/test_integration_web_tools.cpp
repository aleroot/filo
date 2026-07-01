#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <httplib.h>

#include "core/config/ConfigManager.hpp"
#include "core/context/SessionContext.hpp"
#include "core/tools/WebFetchTool.hpp"
#include "core/workspace/Workspace.hpp"
#include "TestSessionContext.hpp"

#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
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

TEST_CASE("WebFetchTool aborts response bodies that exceed max_bytes",
          "[integration][tools][web]") {
    httplib::Server server;
    server.Get("/large", [](const httplib::Request&, httplib::Response& res) {
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
            R"({{"url":"http://127.0.0.1:{}/large","max_bytes":1024}})",
            port),
        core::tools::ToolInvocationContext{
            .session_context = make_tool_test_context("web-fetch-max-bytes"),
        });

    REQUIRE_THAT(res, Catch::Matchers::ContainsSubstring("exceeded max_bytes"));
}
