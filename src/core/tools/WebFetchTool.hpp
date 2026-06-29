#pragma once

#include "Tool.hpp"

namespace core::tools {

class WebFetchTool final : public Tool {
public:
    [[nodiscard]] ToolDefinition get_definition() const override;
    [[nodiscard]] std::string execute(const std::string& json_args,
                                      const core::context::SessionContext& context) override;
    [[nodiscard]] std::string execute(const std::string& json_args,
                                      const ToolInvocationContext& invocation) override;
};

} // namespace core::tools
