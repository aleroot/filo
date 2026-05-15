#pragma once

#include "Tool.hpp"

namespace core::tools {

class TaskTool : public Tool {
public:
    static constexpr std::string_view kToolName = "delegate_task";

    ToolDefinition get_definition() const override;
    std::string execute(const std::string& json_args,
                        const core::context::SessionContext& context) override;
};

} // namespace core::tools
