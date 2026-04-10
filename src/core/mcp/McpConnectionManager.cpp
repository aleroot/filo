#include "McpConnectionManager.hpp"
#include "McpDynamicTool.hpp"
#include "McpDynamicFeatureTools.hpp"
#include "McpSamplingBridge.hpp"
#include "../logging/Logger.hpp"
#include "../llm/LLMProvider.hpp"
#include <stdexcept>
#include <unordered_set>

namespace core::mcp {

void McpConnectionManager::connect_all(const core::config::AppConfig& config,
                                       core::tools::ToolManager& tool_manager,
                                       std::shared_ptr<core::llm::LLMProvider> sampling_provider,
                                       std::string sampling_model) {
    std::lock_guard lock(mutex_);

    // Replace any existing MCP sessions/tools to keep runtime state in sync
    // with the latest config (for example after live profile switching).
    std::unordered_set<std::string> stale_tool_names;
    for (auto& entry : servers_) {
        for (const auto& tool_name : entry.tool_names) {
            stale_tool_names.insert(tool_name);
        }
        try {
            entry.session->shutdown();
        } catch (...) {}
    }
    servers_.clear();

    if (!stale_tool_names.empty()) {
        std::vector<std::string> tool_names_to_remove;
        tool_names_to_remove.reserve(stale_tool_names.size());
        for (const auto& name : stale_tool_names) {
            const auto definition = tool_manager.get_tool_definition(name);
            if (!definition.has_value()) {
                continue;
            }
            if (!definition->description.starts_with("[MCP:")) {
                continue;
            }
            tool_names_to_remove.push_back(name);
        }
        if (!tool_names_to_remove.empty()) {
            tool_manager.unregister_tools(tool_names_to_remove);
        }
    }

    if (sampling_provider) {
        if (!sampling_bridge_) {
            sampling_bridge_ = std::make_shared<McpSamplingBridge>(
                std::move(sampling_provider),
                std::move(sampling_model));
        } else {
            sampling_bridge_->set_backend(
                std::move(sampling_provider),
                std::move(sampling_model));
        }
    } else {
        sampling_bridge_.reset();
    }

    for (const auto& srv_config : config.mcp_servers) {
        try {
            McpSamplingCallback sampling_callback;
            if (sampling_bridge_) {
                auto bridge = sampling_bridge_;
                sampling_callback = [bridge](std::string_view server_name,
                                             std::string_view params_json) {
                    return bridge->create_message(server_name, params_json);
                };
            }

            std::shared_ptr<IMcpClientSession> session =
                make_mcp_session(srv_config, sampling_callback);
            std::vector<McpToolDef> tools = session->initialize();

            ServerEntry entry;
            entry.name    = srv_config.name;
            entry.session = session;  // shared ownership

            for (const auto& tool_def : tools) {
                auto tool = std::make_shared<McpDynamicTool>(session, tool_def, srv_config.name);
                tool_manager.register_tool(tool);
                entry.tool_names.push_back(tool->get_definition().name);
            }

            int feature_tool_count = 0;
            if (session->supports_resources()) {
                auto list_resources = std::make_shared<McpListResourcesTool>(session, srv_config.name);
                auto list_templates = std::make_shared<McpListResourceTemplatesTool>(session, srv_config.name);
                auto read_resource = std::make_shared<McpReadResourceTool>(session, srv_config.name);
                tool_manager.register_tool(list_resources);
                tool_manager.register_tool(list_templates);
                tool_manager.register_tool(read_resource);

                entry.tool_names.push_back(list_resources->get_definition().name);
                entry.tool_names.push_back(list_templates->get_definition().name);
                entry.tool_names.push_back(read_resource->get_definition().name);
                feature_tool_count += 3;
            }

            if (session->supports_prompts()) {
                auto list_prompts = std::make_shared<McpListPromptsTool>(session, srv_config.name);
                auto get_prompt = std::make_shared<McpGetPromptTool>(session, srv_config.name);
                tool_manager.register_tool(list_prompts);
                tool_manager.register_tool(get_prompt);

                entry.tool_names.push_back(list_prompts->get_definition().name);
                entry.tool_names.push_back(get_prompt->get_definition().name);
                feature_tool_count += 2;
            }

            core::logging::info(
                "[MCP] Connected to '{}' - {} tool(s), {} feature tool(s), resources={}, prompts={}, sampling={}.",
                srv_config.name,
                tools.size(),
                feature_tool_count,
                session->supports_resources() ? "on" : "off",
                session->supports_prompts() ? "on" : "off",
                session->supports_sampling() ? "on" : "off");
            servers_.push_back(std::move(entry));

        } catch (const std::exception& e) {
            core::logging::warn("[MCP] Could not connect to '{}': {}",
                                srv_config.name,
                                e.what());
        }
    }
}

void McpConnectionManager::update_sampling_backend(
    std::shared_ptr<core::llm::LLMProvider> sampling_provider,
    std::string sampling_model)
{
    std::lock_guard lock(mutex_);
    if (!sampling_bridge_) return;
    sampling_bridge_->set_backend(
        std::move(sampling_provider),
        std::move(sampling_model));
}

void McpConnectionManager::shutdown_all() noexcept {
    std::lock_guard lock(mutex_);
    for (auto& entry : servers_) {
        try { entry.session->shutdown(); }
        catch (...) {}
    }
    servers_.clear();
    sampling_bridge_.reset();
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
