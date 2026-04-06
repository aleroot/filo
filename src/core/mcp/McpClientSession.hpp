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
// Resources and prompts descriptors.
// ---------------------------------------------------------------------------
struct McpResourceDef {
    std::string uri;
    std::string name;
    std::string title;
    std::string description;
    std::string mime_type;
};

struct McpResourceTemplateDef {
    std::string uri_template;
    std::string name;
    std::string title;
    std::string description;
    std::string mime_type;
};

struct McpPromptArgumentDef {
    std::string name;
    std::string title;
    std::string description;
    bool required = false;
};

struct McpPromptDef {
    std::string name;
    std::string title;
    std::string description;
    std::vector<McpPromptArgumentDef> arguments;
};

using McpSamplingCallback =
    std::function<std::string(std::string_view server_name, std::string_view params_json)>;

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

    // List concrete resources exposed by the server.
    [[nodiscard]] virtual std::vector<McpResourceDef> list_resources() = 0;

    // List parameterized resource templates exposed by the server.
    [[nodiscard]] virtual std::vector<McpResourceTemplateDef> list_resource_templates() = 0;

    // Read one resource by URI.
    [[nodiscard]] virtual std::string read_resource(const std::string& uri) = 0;

    // List prompt templates exposed by the server.
    [[nodiscard]] virtual std::vector<McpPromptDef> list_prompts() = 0;

    // Expand one prompt by name with optional JSON object arguments.
    [[nodiscard]] virtual std::string get_prompt(const std::string& prompt_name,
                                                 const std::string& arguments_json) = 0;

    [[nodiscard]] virtual bool supports_resources() const noexcept = 0;
    [[nodiscard]] virtual bool supports_prompts() const noexcept = 0;
    [[nodiscard]] virtual bool supports_sampling() const noexcept = 0;

    // Graceful shutdown.
    virtual void shutdown() noexcept = 0;
};

// ---------------------------------------------------------------------------
// StdioMcpSession — spawns a subprocess and speaks MCP over its stdin/stdout.
// ---------------------------------------------------------------------------
class StdioMcpSession : public IMcpClientSession {
public:
    explicit StdioMcpSession(const core::config::McpServerConfig& config,
                             McpSamplingCallback sampling_callback = {});
    ~StdioMcpSession() override;

    [[nodiscard]] std::vector<McpToolDef> initialize() override;
    [[nodiscard]] std::string call_tool(const std::string& tool_name,
                                        const std::string& arguments_json) override;
    [[nodiscard]] std::vector<McpResourceDef> list_resources() override;
    [[nodiscard]] std::vector<McpResourceTemplateDef> list_resource_templates() override;
    [[nodiscard]] std::string read_resource(const std::string& uri) override;
    [[nodiscard]] std::vector<McpPromptDef> list_prompts() override;
    [[nodiscard]] std::string get_prompt(const std::string& prompt_name,
                                         const std::string& arguments_json) override;
    [[nodiscard]] bool supports_resources() const noexcept override {
        return server_supports_resources_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool supports_prompts() const noexcept override {
        return server_supports_prompts_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool supports_sampling() const noexcept override {
        return client_sampling_enabled_.load(std::memory_order_acquire);
    }
    void shutdown() noexcept override;

private:
    // Internal JSON-RPC helpers
    [[nodiscard]] std::string send_request(std::string_view method,
                                           std::string_view params_json);
    void send_notification(std::string_view method, std::string_view params_json = "");

    // Reader loop (runs on reader_thread_)
    void reader_loop();
    void update_server_capabilities(std::string_view initialize_result);

    int write_fd_ = -1;
    int read_fd_  = -1;
    pid_t child_pid_ = -1;
    std::string server_name_;
    McpSamplingCallback sampling_callback_;

    std::atomic<bool> running_{false};
    std::thread reader_thread_;

    std::mutex pending_mutex_;
    std::unordered_map<int, std::promise<std::string>> pending_;
    std::atomic<int> next_id_{1};

    std::mutex write_mutex_;  // serialises writes to the child's stdin
    std::atomic<bool> server_supports_tools_{true};
    std::atomic<bool> server_tools_advertised_{false};
    std::atomic<bool> server_supports_resources_{false};
    std::atomic<bool> server_supports_prompts_{false};
    std::atomic<bool> client_sampling_enabled_{false};
};

// ---------------------------------------------------------------------------
// HttpMcpSession — sends MCP JSON-RPC over HTTP POST.
// ---------------------------------------------------------------------------
class HttpMcpSession : public IMcpClientSession {
public:
    explicit HttpMcpSession(const core::config::McpServerConfig& config,
                            McpSamplingCallback sampling_callback = {});
    ~HttpMcpSession() override;

    [[nodiscard]] std::vector<McpToolDef> initialize() override;
    [[nodiscard]] std::string call_tool(const std::string& tool_name,
                                        const std::string& arguments_json) override;
    [[nodiscard]] std::vector<McpResourceDef> list_resources() override;
    [[nodiscard]] std::vector<McpResourceTemplateDef> list_resource_templates() override;
    [[nodiscard]] std::string read_resource(const std::string& uri) override;
    [[nodiscard]] std::vector<McpPromptDef> list_prompts() override;
    [[nodiscard]] std::string get_prompt(const std::string& prompt_name,
                                         const std::string& arguments_json) override;
    [[nodiscard]] bool supports_resources() const noexcept override {
        return server_supports_resources_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool supports_prompts() const noexcept override {
        return server_supports_prompts_.load(std::memory_order_acquire);
    }
    [[nodiscard]] bool supports_sampling() const noexcept override {
        // Streamable HTTP sampling requires a bidirectional channel that this
        // transport adapter does not currently implement.
        return false;
    }
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
    void update_server_capabilities(std::string_view initialize_result);
    void update_negotiated_protocol_version(std::string_view initialize_result);

    std::string url_;
    std::string server_name_;
    std::string protocol_version_{"2025-11-25"};
    std::string session_id_;
    McpSamplingCallback sampling_callback_;
    std::atomic<int> next_id_{1};
    std::mutex request_mutex_;
    std::atomic<bool> server_supports_tools_{true};
    std::atomic<bool> server_tools_advertised_{false};
    std::atomic<bool> server_supports_resources_{false};
    std::atomic<bool> server_supports_prompts_{false};
};

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------
[[nodiscard]] std::unique_ptr<IMcpClientSession>
make_mcp_session(const core::config::McpServerConfig& config,
                 McpSamplingCallback sampling_callback = {});

// ---------------------------------------------------------------------------
// parse_tools_list — converts a tools/list JSON result into McpToolDef vector.
// ---------------------------------------------------------------------------
[[nodiscard]] std::vector<McpToolDef> parse_tools_list(std::string_view json_result);
[[nodiscard]] std::vector<McpResourceDef> parse_resources_list(std::string_view json_result);
[[nodiscard]] std::vector<McpResourceTemplateDef>
parse_resource_templates_list(std::string_view json_result);
[[nodiscard]] std::vector<McpPromptDef> parse_prompts_list(std::string_view json_result);

} // namespace core::mcp
