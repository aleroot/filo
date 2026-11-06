#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/mcp/McpConnectionManager.hpp"
#include "core/tools/GetTimeTool.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/utils/JsonValue.hpp"

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <format>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

using namespace std::chrono_literals;
using Catch::Matchers::StartsWith;
using core::mcp::McpConnectionManager;
using core::mcp::McpServerState;

namespace {

using Clock = std::chrono::steady_clock;

/// A Streamable HTTP MCP server with one tool whose initialize answer is held
/// back for up to `delay`, or until release().
class FakeMcpServer {
public:
    FakeMcpServer(std::string tool, std::chrono::milliseconds delay)
        : tool_(std::move(tool)), delay_(delay) {
        server_.Post("/mcp", [this](const httplib::Request& request, httplib::Response& response) {
            handle(request, response);
        });
        server_.Delete("/mcp", [this](const httplib::Request&, httplib::Response& response) {
            ++deletes_;
            response.status = 200;
        });
        port_ = server_.bind_to_any_port("127.0.0.1");
        listener_ = std::jthread([this] { server_.listen_after_bind(); });
    }

    ~FakeMcpServer() {
        release();
        server_.stop();
    }

    void release() {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        gate_.notify_all();
    }

    [[nodiscard]] std::string url() const { return std::format("http://127.0.0.1:{}/mcp", port_); }
    [[nodiscard]] int deletes() const { return deletes_.load(); }

private:
    void handle(const httplib::Request& request, httplib::Response& response) {
        const auto body = core::utils::JsonValue::parse(request.body);
        const auto* id = body ? body->find("id") : nullptr;
        if (!id) {
            response.status = 202;
            return;
        }
        const auto* method_value = body->find("method");
        const std::string method(method_value ? method_value->as_string() : "");
        std::string result = "{}";
        if (method == "initialize") {
            {
                std::unique_lock lock(mutex_);
                gate_.wait_for(lock, delay_, [&] { return released_; });
            }
            result = R"({"protocolVersion":"2025-11-25","capabilities":{"tools":{}},)"
                     R"("serverInfo":{"name":"fake","version":"1"}})";
            response.set_header("Mcp-Session-Id", "fake-session");
        } else if (method == "tools/list") {
            result = std::format(
                R"({{"tools":[{{"name":"{}","description":"d","inputSchema":{{"type":"object"}}}}]}})",
                tool_);
        }
        response.set_content(
            std::format(R"({{"jsonrpc":"2.0","id":{},"result":{}}})", id->dump(), result),
            "application/json");
    }

    std::string tool_;
    std::chrono::milliseconds delay_;
    std::mutex mutex_;
    std::condition_variable gate_;
    bool released_ = false;
    std::atomic<int> deletes_{0};
    httplib::Server server_;
    int port_ = 0;
    std::jthread listener_;  // declared last: joined after stop()
};

core::config::McpServerConfig http_server(std::string name, const FakeMcpServer& server) {
    core::config::McpServerConfig config;
    config.name = std::move(name);
    config.transport = "http";
    config.url = server.url();
    config.auth = "none";
    return config;
}

core::config::AppConfig config_of(std::vector<core::config::McpServerConfig> servers) {
    core::config::AppConfig config;
    config.mcp_servers = std::move(servers);
    return config;
}

template <class Predicate>
bool eventually(Predicate predicate, std::chrono::milliseconds timeout = 5s) {
    const auto deadline = Clock::now() + timeout;
    while (!predicate()) {
        if (Clock::now() > deadline) return false;
        std::this_thread::sleep_for(5ms);
    }
    return true;
}

McpServerState state_of(const McpConnectionManager& connections, std::size_t index) {
    return connections.server_statuses().at(index).state;
}

/// Leaves the singleton disconnected and without a listener, whatever the
/// test did. Declared after the servers, so it runs before they stop.
struct Disconnect {
    ~Disconnect() {
        auto& connections = McpConnectionManager::get_instance();
        connections.set_change_listener({});
        connections.shutdown_all();
    }
};

} // namespace

TEST_CASE("Connecting MCP servers returns before any handshake and registers tools later",
          "[mcp][connections]") {
    FakeMcpServer slow("slow_tool", 5s);
    core::tools::ToolManager tools;
    auto& connections = McpConnectionManager::get_instance();
    const Disconnect disconnect;
    std::atomic<int> changes{0};
    connections.set_change_listener([&] { ++changes; });

    const auto started = Clock::now();
    connections.connect_all_async(config_of({http_server("slow", slow)}), tools);
    CHECK(Clock::now() - started < 100ms);

    // Published as connecting before the call returned; nothing registered yet.
    CHECK(connections.configured_count() == 1);
    CHECK(state_of(connections, 0) == McpServerState::connecting);
    CHECK(connections.connected_count() == 0);
    CHECK_FALSE(tools.get_tool_definition("slow_tool").has_value());
    CHECK_FALSE(connections.wait_until_settled(50ms));

    slow.release();
    REQUIRE(connections.wait_until_settled(5s));
    CHECK(state_of(connections, 0) == McpServerState::connected);
    CHECK(connections.connected_count() == 1);
    CHECK(tools.get_tool_definition("slow_tool").has_value());
    CHECK(eventually([&] { return changes.load() >= 1; }));
}

TEST_CASE("A slow MCP server never delays a fast one", "[mcp][connections]") {
    FakeMcpServer slow("slow_tool", 5s);
    FakeMcpServer fast("fast_tool", 0ms);
    core::tools::ToolManager tools;
    auto& connections = McpConnectionManager::get_instance();
    const Disconnect disconnect;

    connections.connect_all_async(
        config_of({http_server("slow", slow), http_server("fast", fast)}), tools);

    REQUIRE(eventually([&] { return state_of(connections, 1) == McpServerState::connected; }, 2s));
    CHECK(tools.get_tool_definition("fast_tool").has_value());
    CHECK(state_of(connections, 0) == McpServerState::connecting);
    CHECK(connections.connected_count() == 1);
    CHECK_FALSE(connections.wait_until_settled(20ms));
    slow.release();
}

TEST_CASE("A newer MCP round supersedes a handshake still in flight", "[mcp][connections]") {
    FakeMcpServer stale("stale_tool", 5s);
    FakeMcpServer fresh("fresh_tool", 0ms);
    core::tools::ToolManager tools;
    auto& connections = McpConnectionManager::get_instance();
    const Disconnect disconnect;

    connections.connect_all_async(config_of({http_server("stale", stale)}), tools);
    connections.connect_all_async(config_of({http_server("fresh", fresh)}), tools);
    REQUIRE(connections.wait_until_settled(5s));

    // The superseded handshake completes now: its session is shut down and
    // it neither registers a tool nor touches the current round's status.
    stale.release();
    REQUIRE(eventually([&] { return stale.deletes() == 1; }));
    CHECK_FALSE(tools.get_tool_definition("stale_tool").has_value());
    CHECK(tools.get_tool_definition("fresh_tool").has_value());
    const auto statuses = connections.server_statuses();
    REQUIRE(statuses.size() == 1);
    CHECK(statuses[0].name == "fresh");
    CHECK(statuses[0].state == McpServerState::connected);
    CHECK(connections.connected_count() == 1);
}

TEST_CASE("Reconnecting MCP servers unregisters the old tools at once and shuts them down "
          "in the background",
          "[mcp][connections]") {
    FakeMcpServer first("first_tool", 0ms);
    FakeMcpServer second("second_tool", 5s);
    core::tools::ToolManager tools;
    auto& connections = McpConnectionManager::get_instance();
    const Disconnect disconnect;

    connections.connect_all(config_of({http_server("first", first)}), tools);
    REQUIRE(tools.get_tool_definition("first_tool").has_value());

    const auto started = Clock::now();
    connections.connect_all_async(config_of({http_server("second", second)}), tools);
    CHECK(Clock::now() - started < 100ms);
    CHECK_FALSE(tools.get_tool_definition("first_tool").has_value());
    CHECK(connections.connected_count() == 0);
    CHECK(eventually([&] { return first.deletes() == 1; }));
    second.release();
}

TEST_CASE("Shutting MCP down is bounded while a handshake hangs", "[mcp][connections]") {
    FakeMcpServer hung("hung_tool", 10s);
    core::tools::ToolManager tools;
    auto& connections = McpConnectionManager::get_instance();
    const Disconnect disconnect;

    connections.connect_all_async(config_of({http_server("hung", hung)}), tools);
    const auto started = Clock::now();
    connections.shutdown_all(200ms);
    CHECK(Clock::now() - started < 1s);
    CHECK(connections.configured_count() == 0);
    CHECK(connections.wait_until_settled(0ms));

    // When the hung handshake finally answers it is discarded, not adopted.
    hung.release();
    REQUIRE(eventually([&] { return hung.deletes() == 1; }));
    CHECK_FALSE(tools.get_tool_definition("hung_tool").has_value());
    CHECK(connections.connected_count() == 0);
}

TEST_CASE("A clashing MCP tool name goes to the server later in the configuration, "
          "whichever connects first",
          "[mcp][connections]") {
    const bool earlier_is_slow = GENERATE(true, false);
    FakeMcpServer earlier("shared", earlier_is_slow ? 300ms : 0ms);
    FakeMcpServer later("shared", earlier_is_slow ? 0ms : 300ms);
    core::tools::ToolManager tools;
    auto& connections = McpConnectionManager::get_instance();
    const Disconnect disconnect;

    connections.connect_all(
        config_of({http_server("earlier", earlier), http_server("later", later)}), tools);

    REQUIRE(connections.connected_count() == 2);
    const auto definition = tools.get_tool_definition("shared");
    REQUIRE(definition.has_value());
    CHECK_THAT(definition->description, StartsWith("[MCP:later]"));
    CHECK(connections.registered_tool_names() == std::vector<std::string>{"later/shared"});
}

TEST_CASE("An unreachable MCP server fails in the background with its error",
          "[mcp][connections]") {
    core::tools::ToolManager tools;
    auto& connections = McpConnectionManager::get_instance();
    const Disconnect disconnect;
    core::config::McpServerConfig dead;
    dead.name = "dead";
    dead.transport = "http";
    dead.url = "http://127.0.0.1:1/mcp";
    dead.auth = "none";

    connections.connect_all_async(config_of({dead}), tools);
    REQUIRE(connections.wait_until_settled(10s));
    const auto statuses = connections.server_statuses();
    REQUIRE(statuses.size() == 1);
    CHECK(statuses[0].state == McpServerState::failed);
    CHECK_FALSE(statuses[0].error.empty());
    CHECK(connections.connected_count() == 0);
}

TEST_CASE("An MCP tool never replaces a built-in tool, even across reconnects",
          "[mcp][connections]") {
    FakeMcpServer server("get_current_time", 0ms);
    core::tools::ToolManager tools;
    tools.register_tool(std::make_shared<core::tools::GetTimeTool>());
    const auto builtin = tools.get_tool_definition("get_current_time");
    REQUIRE(builtin.has_value());
    auto& connections = McpConnectionManager::get_instance();
    const Disconnect disconnect;

    for (int round = 0; round < 2; ++round) {
        connections.connect_all(config_of({http_server("clash", server)}), tools);
        REQUIRE(state_of(connections, 0) == McpServerState::connected);
        const auto current = tools.get_tool_definition("get_current_time");
        REQUIRE(current.has_value());
        CHECK(current->description == builtin->description);
        CHECK(connections.registered_tool_names().empty());
    }
    connections.shutdown_all();
    CHECK(tools.get_tool_definition("get_current_time").has_value());
}

TEST_CASE("A throwing change listener neither blocks settling nor later rounds",
          "[mcp][connections]") {
    FakeMcpServer server("listened_tool", 0ms);
    core::tools::ToolManager tools;
    auto& connections = McpConnectionManager::get_instance();
    const Disconnect disconnect;
    std::atomic<int> calls{0};
    connections.set_change_listener([&] {
        ++calls;
        throw std::runtime_error("listener failure");
    });

    for (int round = 0; round < 2; ++round) {
        connections.connect_all(config_of({http_server("listened", server)}), tools);
        CHECK(state_of(connections, 0) == McpServerState::connected);
        CHECK(tools.get_tool_definition("listened_tool").has_value());
    }
    CHECK(eventually([&] { return calls.load() >= 2; }));
}
