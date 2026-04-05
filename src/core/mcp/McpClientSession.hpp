#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <future>
#include <atomic>
#include <functional>
#include <stdexcept>
#include <optional>
#include <cpr/cpr.h>
#include "../config/ConfigManager.hpp"
#include "../tools/Tool.hpp"

namespace core::mcp {

// ---------------------------------------------------------------------------
// McpToolDef — tool descriptor returned by tools/list.
// ---------------------------------------------------------------------------
struct McpToolParameter {
    std::string name;
    std::string type;
    std::string description;
    bool required = false;
    std::string schema;
    std::string items_schema;
};

struct McpToolDef {
    std::string name;
    std::string title;
    std::string description;
    std::vector<McpToolParameter> parameters;
    std::string input_schema;
    std::string output_schema;
    core::tools::ToolAnnotations annotations = {};
};

// ---------------------------------------------------------------------------
// IMcpClientSession — interface for both stdio and HTTP MCP sessions.
// ---------------------------------------------------------------------------
class IMcpClientSession {
public:
    virtual ~IMcpClientSession() = default;

    // Perform the initialize handshake and call tools/list.
    // Returns the list of tools exposed by this server.
    // Throws std::runtime_error on failure.
    [[nodiscard]] virtual std::vector<McpToolDef> initialize() = 0;

    // Call a specific tool.  arguments_json is a JSON object string, e.g. {"path":"/foo"}.
    // Returns the tool result as a JSON string.
    [[nodiscard]] virtual std::string call_tool(const std::string& tool_name,
                                                const std::string& arguments_json) = 0;

    // Graceful shutdown.
    virtual void shutdown() noexcept = 0;
};

// ---------------------------------------------------------------------------
// StdioMcpSession — spawns a subprocess and speaks MCP over its stdin/stdout.
// ---------------------------------------------------------------------------
class StdioMcpSession : public IMcpClientSession {
public:
    explicit StdioMcpSession(const core::config::McpServerConfig& config);
    ~StdioMcpSession() override;

    [[nodiscard]] std::vector<McpToolDef> initialize() override;
    [[nodiscard]] std::string call_tool(const std::string& tool_name,
                                        const std::string& arguments_json) override;
    void shutdown() noexcept override;

private:
    // Internal JSON-RPC helpers
    [[nodiscard]] std::string send_request(std::string_view method,
                                           std::string_view params_json);
    void send_notification(std::string_view method, std::string_view params_json = "");

    // Reader loop (runs on reader_thread_)
    void reader_loop();

    int write_fd_ = -1;
    int read_fd_  = -1;
    pid_t child_pid_ = -1;

    std::atomic<bool> running_{false};
    std::thread reader_thread_;

    std::mutex pending_mutex_;
    std::unordered_map<int, std::promise<std::string>> pending_;
    std::atomic<int> next_id_{1};

    std::mutex write_mutex_;  // serialises writes to the child's stdin
};

// ---------------------------------------------------------------------------
// HttpMcpSession — sends MCP JSON-RPC over HTTP POST.
// ---------------------------------------------------------------------------
class HttpMcpSession : public IMcpClientSession {
public:
    explicit HttpMcpSession(const core::config::McpServerConfig& config);
    ~HttpMcpSession() override;

    [[nodiscard]] std::vector<McpToolDef> initialize() override;
    [[nodiscard]] std::string call_tool(const std::string& tool_name,
                                        const std::string& arguments_json) override;
    void shutdown() noexcept override;

private:
    struct HttpJsonResponse {
        long status_code = 0;
        std::string body;
        cpr::Header headers;
    };

    [[nodiscard]] HttpJsonResponse post_json(const std::string& body,
                                             bool include_protocol_header = true);
    [[nodiscard]] std::string send_request(std::string_view method,
                                           std::string_view params_json);
    void update_negotiated_protocol_version(std::string_view initialize_result);

    std::string url_;
    std::string protocol_version_{"2025-11-25"};
    std::string session_id_;
    std::atomic<int> next_id_{1};
    std::mutex request_mutex_;
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
[[nodiscard]] std::unique_ptr<IMcpClientSession>
make_mcp_session(const core::config::McpServerConfig& config);

// ---------------------------------------------------------------------------
// parse_tools_list — converts a tools/list JSON result into McpToolDef vector.
// ---------------------------------------------------------------------------
[[nodiscard]] std::vector<McpToolDef> parse_tools_list(std::string_view json_result);

} // namespace core::mcp
