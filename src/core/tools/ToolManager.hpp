#pragma once

#include "Tool.hpp"
#include "LandrunToolPolicy.hpp"
#include "ToolPolicy.hpp"
#include "ToolSchema.hpp"
#include "ToolNames.hpp"
#include "../context/SessionContext.hpp"
#include "../landrun/LandrunSettings.hpp"
#include <algorithm>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <string_view>
#include <vector>
#include <stdexcept>
#include <optional>
#include <expected>
#include "../utils/JsonUtils.hpp"
#include "../llm/Models.hpp"

namespace core::tools {

class ToolManager {
public:
    static ToolManager& get_instance() {
        static ToolManager instance;
        return instance;
    }

    // Thread-safe: called from both the main (TUI) thread and the daemon thread
    // at startup. Overwrites silently if a tool with the same name already exists.
    void register_tool(std::shared_ptr<Tool> tool) {
        auto def = tool->get_definition();
        std::lock_guard lock(mutex_);
        tools_[def.name] = {std::move(tool), std::move(def.title)};
    }

    // Thread-safe remove: useful for replacing dynamic tool sets (e.g. MCP).
    [[nodiscard]] bool unregister_tool(const std::string& name) {
        std::lock_guard lock(mutex_);
        return tools_.erase(name) > 0;
    }

    // Thread-safe batch remove for efficiency when pruning many tools at once.
    void unregister_tools(const std::vector<std::string>& names) {
        std::lock_guard lock(mutex_);
        for (const auto& name : names) {
            tools_.erase(name);
        }
    }

    // Thread-safe read: called from any HTTP handler thread or the TUI agent thread.
    std::vector<core::llm::Tool> get_all_tools(
        core::context::SessionTransport transport = core::context::SessionTransport::unspecified) const {
        std::lock_guard lock(mutex_);
        std::vector<core::llm::Tool> tools;
        tools.reserve(tools_.size());
        for (const auto& [name, registered] : tools_) {
            core::llm::Tool entry;
            entry.type = "function";
            entry.function = registered.tool->get_definition();
            if (!tool_allows_transport(entry.function, transport)) continue;
            if (!landrun_allows_tool(
                    core::landrun::LandrunSettings::instance().mode(),
                    entry.function.name)) {
                continue;
            }
            tools.push_back(entry);
        }
        std::ranges::sort(tools, [](const core::llm::Tool& a, const core::llm::Tool& b) {
            return a.function.name < b.function.name;
        });
        return tools;
    }

    // Thread-safe: locks only for the map lookup, then releases before executing
    // the tool (execution can be slow — shell commands, file I/O — and must not
    // hold the mutex while it runs).
    std::string execute_tool(
        const std::string& name,
        const std::string& json_args,
        const core::context::SessionContext& context)
    {
        return execute_tool(name, json_args, ToolInvocationContext{
            .session_context = context,
        });
    }

    std::string execute_tool(
        const std::string& name,
        const std::string& json_args,
        const ToolInvocationContext& invocation)
    {
        std::shared_ptr<Tool> tool;
        ToolDefinition definition;
        {
            std::lock_guard lock(mutex_);
            const auto* registered = find_tool_unlocked(name);
            if (!registered)
                return "{\"error\": \"Tool not found: " + name + "\"}";
            tool = registered->tool;
            definition = tool->get_definition();
        }
        if (!tool_allows_transport(definition, invocation.session_context.transport)) {
            return R"({"error":"Tool is available only to Filo's internal agents."})";
        }
        const auto sandbox_mode =
            core::landrun::LandrunSettings::instance().mode();
        if (!landrun_allows_tool(sandbox_mode, definition.name)) {
            return "{\"error\":\"Tool "
                + core::utils::escape_json_string(name)
                + " is unavailable while sandbox mode is "
                + std::string(core::landrun::landrun_mode_name(sandbox_mode))
                + ".\"}";
        }
        const auto normalized = schema::normalize_arguments(definition, json_args);
        if (!normalized.has_value()) {
            return "{\"error\":\"Invalid arguments for "
                + core::utils::escape_json_string(name)
                + ": "
                + core::utils::escape_json_string(normalized.error())
                + "\"}";
        }
        try {
            return tool->execute(*normalized, invocation);
        } catch (const std::exception& e) {
            return "{\"error\": \"Exception executing tool: " + std::string(e.what()) + "\"}";
        }
    }

    [[nodiscard]] std::expected<std::string, std::string>
    normalize_tool_arguments(const std::string& name,
                             std::string_view json_args) const {
        ToolDefinition definition;
        {
            std::lock_guard lock(mutex_);
            const auto* registered = find_tool_unlocked(name);
            if (!registered) {
                return std::unexpected("tool not found");
            }
            definition = registered->tool->get_definition();
        }
        return schema::normalize_arguments(definition, json_args);
    }

    [[nodiscard]] bool has_tool(const std::string& name) const {
        std::lock_guard lock(mutex_);
        return static_cast<bool>(find_tool_unlocked(name));
    }

    [[nodiscard]] std::optional<ToolDefinition>
    get_tool_definition(const std::string& name,
        core::context::SessionTransport transport = core::context::SessionTransport::unspecified) const {
        std::lock_guard lock(mutex_);
        const auto* registered = find_tool_unlocked(name);
        if (!registered) return std::nullopt;
        auto definition = registered->tool->get_definition();
        if (!tool_allows_transport(definition, transport)) return std::nullopt;
        return definition;
    }

    [[nodiscard]] std::optional<std::string>
    get_tool_title(const std::string& name) const {
        std::lock_guard lock(mutex_);
        const auto* registered = find_tool_unlocked(name);
        if (!registered) {
            const auto agent_title = names::agent_tool_title(name);
            if (agent_title.empty()) return std::nullopt;
            return std::string(agent_title);
        }
        if (registered->title.empty()) return std::nullopt;
        return registered->title;
    }

    void clear_session_state(std::string_view session_id) {
        std::vector<std::shared_ptr<Tool>> tools;
        {
            std::lock_guard lock(mutex_);
            tools.reserve(tools_.size());
            for (const auto& [_, registered] : tools_) {
                tools.push_back(registered.tool);
            }
        }
        for (const auto& tool : tools) {
            tool->clear_session_state(session_id);
        }
    }

private:
    ToolManager() = default;

    struct RegisteredTool {
        std::shared_ptr<Tool> tool;
        std::string title;
    };

    [[nodiscard]] const RegisteredTool* find_tool_unlocked(std::string_view name) const {
        const std::string key{name};
        if (auto it = tools_.find(key); it != tools_.end()) return &it->second;
        const auto canonical = policy::canonical_tool_name(name);
        if (canonical == key) return {};
        if (auto it = tools_.find(canonical); it != tools_.end()) return &it->second;
        return {};
    }

    mutable std::mutex mutex_;
    std::unordered_map<std::string, RegisteredTool> tools_;
};

} // namespace core::tools
