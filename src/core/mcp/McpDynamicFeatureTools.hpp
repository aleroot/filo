#pragma once

#include "../tools/Tool.hpp"
#include "McpClientSession.hpp"

#include <memory>
#include <string>

namespace core::mcp {

[[nodiscard]] std::string sanitize_mcp_tool_token(std::string_view value);
[[nodiscard]] std::string make_mcp_feature_tool_name(std::string_view server_name,
                                                     std::string_view suffix);

class McpSessionBackedTool : public core::tools::Tool {
public:
    core::tools::ToolDefinition get_definition() const override { return def_; }

protected:
    McpSessionBackedTool(std::shared_ptr<IMcpClientSession> session,
                         std::string server_name,
                         core::tools::ToolDefinition def);

    [[nodiscard]] std::string make_error(std::string_view message) const;

    std::shared_ptr<IMcpClientSession> session_;
    std::string server_name_;
    core::tools::ToolDefinition def_;
};

class McpListResourcesTool final : public McpSessionBackedTool {
public:
    McpListResourcesTool(std::shared_ptr<IMcpClientSession> session,
                         std::string server_name);

    std::string execute(const std::string& json_args,
                        const core::context::SessionContext& context) override;
};

class McpListResourceTemplatesTool final : public McpSessionBackedTool {
public:
    McpListResourceTemplatesTool(std::shared_ptr<IMcpClientSession> session,
                                 std::string server_name);

    std::string execute(const std::string& json_args,
                        const core::context::SessionContext& context) override;
};

class McpReadResourceTool final : public McpSessionBackedTool {
public:
    McpReadResourceTool(std::shared_ptr<IMcpClientSession> session,
                        std::string server_name);

    std::string execute(const std::string& json_args,
                        const core::context::SessionContext& context) override;
};

class McpListPromptsTool final : public McpSessionBackedTool {
public:
    McpListPromptsTool(std::shared_ptr<IMcpClientSession> session,
                       std::string server_name);

    std::string execute(const std::string& json_args,
                        const core::context::SessionContext& context) override;
};

class McpGetPromptTool final : public McpSessionBackedTool {
public:
    McpGetPromptTool(std::shared_ptr<IMcpClientSession> session,
                     std::string server_name);

    std::string execute(const std::string& json_args,
                        const core::context::SessionContext& context) override;
};

} // namespace core::mcp
