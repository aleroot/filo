#pragma once

#include "Tool.hpp"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <vector>
#include <stdexcept>
#include <optional>
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
        tools_[def.name] = std::move(tool);
    }

    // Thread-safe read: called from any HTTP handler thread or the TUI agent thread.
    std::vector<core::llm::Tool> get_all_tools() const {
        std::lock_guard lock(mutex_);
        std::vector<core::llm::Tool> tools;
        tools.reserve(tools_.size());
        for (const auto& [name, impl] : tools_) {
            core::llm::Tool entry;
            entry.type = "function";
            entry.function = impl->get_definition();
            tools.push_back(entry);
        }
        return tools;
    }

    // Thread-safe: locks only for the map lookup, then releases before executing
    // the tool (execution can be slow — shell commands, file I/O — and must not
    // hold the mutex while it runs).
    std::string execute_tool(const std::string& name, const std::string& json_args) {
        std::shared_ptr<Tool> tool;
        {
            std::lock_guard lock(mutex_);
            auto it = tools_.find(name);
            if (it == tools_.end())
                return "{\"error\": \"Tool not found: " + name + "\"}";
            tool = it->second;
        }
        try {
            return tool->execute(json_args);
        } catch (const std::exception& e) {
            return "{\"error\": \"Exception executing tool: " + std::string(e.what()) + "\"}";
        }
    }

    [[nodiscard]] bool has_tool(const std::string& name) const {
        std::lock_guard lock(mutex_);
        return tools_.contains(name);
    }

    [[nodiscard]] std::optional<ToolDefinition>
    get_tool_definition(const std::string& name) const {
        std::lock_guard lock(mutex_);
        auto it = tools_.find(name);
        if (it == tools_.end()) return std::nullopt;
        return it->second->get_definition();
    }

private:
    ToolManager() = default;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<Tool>> tools_;
};

} // namespace core::tools
