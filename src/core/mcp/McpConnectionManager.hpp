#pragma once

#include "../tools/Tool.hpp"
#include "../tools/ToolManager.hpp"
#include "../config/ConfigManager.hpp"
#include "McpClientSession.hpp"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <mutex>

namespace core::mcp {

// ---------------------------------------------------------------------------
// McpConnectionManager — initialises all configured MCP servers and registers
// their tools in the given ToolManager.
//
// Singleton.  Call connect_all() once during application startup.
// ---------------------------------------------------------------------------
class McpConnectionManager {
public:
    static McpConnectionManager& get_instance() noexcept {
        static McpConnectionManager instance;
        return instance;
    }

    // Connect to every MCP server in config.mcp_servers, initialise each
    // session, discover their tools, and register them in tool_manager.
    // Errors for individual servers are logged and skipped gracefully.
    void connect_all(const core::config::AppConfig& config,
                     core::tools::ToolManager& tool_manager);

    // Shut down all active sessions cleanly.
    void shutdown_all() noexcept;

    // Returns the number of successfully connected MCP servers.
    [[nodiscard]] int connected_count() const noexcept;

    // Returns registered MCP tool names (for diagnostics / /help display).
    [[nodiscard]] std::vector<std::string> registered_tool_names() const;

private:
    McpConnectionManager() noexcept = default;

    struct ServerEntry {
        std::string name;
        std::shared_ptr<IMcpClientSession> session;
        std::vector<std::string> tool_names;
    };

    mutable std::mutex mutex_;
    std::vector<ServerEntry> servers_;
};

} // namespace core::mcp
