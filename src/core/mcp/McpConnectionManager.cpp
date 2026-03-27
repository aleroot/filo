#include "McpConnectionManager.hpp"
#include "McpDynamicTool.hpp"
#include "../logging/Logger.hpp"
#include <stdexcept>

namespace core::mcp {

void McpConnectionManager::connect_all(const core::config::AppConfig& config,
                                        core::tools::ToolManager& tool_manager) {
    std::lock_guard lock(mutex_);

    for (const auto& srv_config : config.mcp_servers) {
        try {
            std::shared_ptr<IMcpClientSession> session = make_mcp_session(srv_config);
            std::vector<McpToolDef> tools = session->initialize();

            ServerEntry entry;
            entry.name    = srv_config.name;
            entry.session = session;  // shared ownership

            for (const auto& tool_def : tools) {
                auto tool = std::make_shared<McpDynamicTool>(session, tool_def, srv_config.name);
                tool_manager.register_tool(tool);
                entry.tool_names.push_back(tool_def.name);
            }

            core::logging::info("[MCP] Connected to '{}' - {} tool(s) registered.",
                                srv_config.name,
                                tools.size());
            servers_.push_back(std::move(entry));

        } catch (const std::exception& e) {
            core::logging::warn("[MCP] Could not connect to '{}': {}",
                                srv_config.name,
                                e.what());
        }
    }
}

void McpConnectionManager::shutdown_all() noexcept {
    std::lock_guard lock(mutex_);
    for (auto& entry : servers_) {
        try { entry.session->shutdown(); }
        catch (...) {}
    }
    servers_.clear();
}

int McpConnectionManager::connected_count() const noexcept {
    std::lock_guard lock(mutex_);
    return static_cast<int>(servers_.size());
}

std::vector<std::string> McpConnectionManager::registered_tool_names() const {
    std::lock_guard lock(mutex_);
    std::vector<std::string> names;
    for (const auto& entry : servers_) {
        for (const auto& n : entry.tool_names) {
            names.push_back(entry.name + "/" + n);
        }
    }
    return names;
}

} // namespace core::mcp
