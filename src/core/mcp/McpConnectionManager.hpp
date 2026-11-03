#pragma once

#include "../tools/Tool.hpp"
#include "../tools/ToolManager.hpp"
#include "../config/ConfigManager.hpp"
#include "McpClientSession.hpp"
#include "McpServerStatus.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace core::llm {
class LLMProvider;
}

namespace core::mcp {

class McpSamplingBridge;

// ---------------------------------------------------------------------------
// McpConnectionManager — initialises all configured MCP servers and registers
// their tools in the given ToolManager.
//
// Singleton.
// ---------------------------------------------------------------------------
class McpConnectionManager {
public:
    static McpConnectionManager& get_instance() noexcept {
        static McpConnectionManager instance;
        return instance;
    }

    // Connect to every MCP server in config.mcp_servers, initialise each
    // session, discover their tools, and register them in tool_manager.
    // If MCP sessions were already connected, this call replaces the previous
    // set atomically (old sessions are shut down and old MCP tools removed).
    // Errors for individual servers are logged and skipped gracefully.
    void connect_all(const core::config::AppConfig& config,
                     core::tools::ToolManager& tool_manager,
                     std::shared_ptr<core::llm::LLMProvider> sampling_provider = nullptr,
                     std::string sampling_model = {});

    // Update the sampling backend used by already-connected MCP sessions.
    void update_sampling_backend(std::shared_ptr<core::llm::LLMProvider> sampling_provider,
                                 std::string sampling_model = {});

    // Shut down all active sessions cleanly.
    void shutdown_all() noexcept;

    // Returns the number of successfully connected MCP servers.
    [[nodiscard]] int connected_count() const noexcept;

    // Returns registered MCP tool names (for diagnostics / /help display).
    [[nodiscard]] std::vector<std::string> registered_tool_names() const;

    // One entry per configured server, in configuration order, including
    // servers that are still connecting or failed. Never blocks on a
    // connection in progress.
    [[nodiscard]] std::vector<McpServerStatus> server_statuses() const;

private:
    McpConnectionManager() noexcept = default;

    struct ServerEntry {
        std::string name;
        std::shared_ptr<IMcpClientSession> session;
        std::vector<std::string> tool_names;
    };

    struct StatusEntry {
        McpServerStatus status;
        std::shared_ptr<McpServerActivity> activity;
    };

    mutable std::mutex mutex_;
    std::vector<ServerEntry> servers_;
    std::shared_ptr<McpSamplingBridge> sampling_bridge_;
    // Separate lock: connect_all holds mutex_ for the whole handshake, while
    // the UI reads statuses on every frame.
    mutable std::mutex status_mutex_;
    std::vector<StatusEntry> statuses_;
};

} // namespace core::mcp
