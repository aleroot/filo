#pragma once

#include "../config/ConfigManager.hpp"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace core::mcp {

/// What the server reported about itself in its initialize result.
struct McpServerInfo {
    std::string name;
    std::string version;
    std::string protocol_version;
};

/// Point-in-time copy of one server's tool-call activity.
struct McpServerActivitySnapshot {
    std::uint64_t calls = 0;
    std::uint64_t failures = 0;
    int in_flight = 0;
    std::string last_tool;
    bool last_failed = false;
    std::string last_error;
    std::chrono::system_clock::time_point last_call_at{};
    std::chrono::milliseconds last_duration{0};
};

/// Thread-safe tool-call counters shared by every tool of one server.
class McpServerActivity {
public:
    void begin(std::string_view tool_name);
    void finish(std::string_view tool_name,
                bool failed,
                std::string_view error,
                std::chrono::milliseconds duration);
    [[nodiscard]] McpServerActivitySnapshot snapshot() const;

private:
    mutable std::mutex mutex_;
    McpServerActivitySnapshot state_;
};

enum class McpServerState { connecting, connected, failed };

/// Everything the UI needs to explain one configured MCP server.
/// Never carries secret values: headers are listed by name only.
struct McpServerStatus {
    std::string name;
    std::string transport;
    std::string endpoint;
    std::string auth;
    McpServerState state = McpServerState::connecting;
    std::string error;
    bool needs_login = false;
    McpServerInfo server_info;
    std::vector<std::string> tools;
    bool resources = false;
    bool prompts = false;
    bool sampling = false;
    std::chrono::system_clock::time_point connected_at{};
    std::chrono::milliseconds connect_duration{0};
    McpServerActivitySnapshot activity;
};

/// URL for HTTP servers, the command line for stdio servers.
[[nodiscard]] std::string describe_mcp_endpoint(const core::config::McpServerConfig& config);

/// How requests are authenticated, e.g. "OAuth", "header: Authorization".
[[nodiscard]] std::string describe_mcp_auth(const core::config::McpServerConfig& config);

/// True when a connection error means the user should run `/mcp login`.
[[nodiscard]] bool mcp_error_needs_login(const core::config::McpServerConfig& config,
                                         std::string_view error);

} // namespace core::mcp
