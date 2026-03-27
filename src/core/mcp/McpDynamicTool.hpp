#pragma once

#include "../tools/Tool.hpp"
#include "../utils/JsonUtils.hpp"
#include "McpClientSession.hpp"
#include <memory>
#include <string>
#include <stdexcept>

namespace core::mcp {

// ---------------------------------------------------------------------------
// McpDynamicTool — adapts one tool from an external MCP server into filo's
// Tool interface.  The underlying session is shared among all tools exposed
// by the same server.
// ---------------------------------------------------------------------------
class McpDynamicTool : public core::tools::Tool {
public:
    McpDynamicTool(std::shared_ptr<IMcpClientSession> session, const McpToolDef& tool_def, std::string server_name)
        : session_(std::move(session))
        , server_name_(std::move(server_name)) {
        // Build ToolDefinition from the McpToolDef
        def_.name        = tool_def.name;
        def_.description = "[MCP:" + server_name_ + "] " + tool_def.description;
        for (const auto& p : tool_def.parameters) {
            def_.parameters.push_back({
                .name        = p.name,
                .type        = p.type.empty() ? "string" : p.type,
                .description = p.description,
                .required    = p.required
            });
        }
    }

    core::tools::ToolDefinition get_definition() const override { return def_; }

    std::string execute(const std::string& json_args) override {
        try {
            return session_->call_tool(def_.name, json_args);
        } catch (const std::exception& e) {
            return std::string(R"({"error":"MCP tool call failed: "})").insert(
                22, core::utils::escape_json_string(e.what()));
        }
    }

private:
    std::shared_ptr<IMcpClientSession>   session_;
    core::tools::ToolDefinition        def_;
    std::string                          server_name_;
};

} // namespace core::mcp
