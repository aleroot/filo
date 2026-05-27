#pragma once

#include "Tool.hpp"

namespace core::tools {

class ActivateSkillTool final : public Tool {
public:
    [[nodiscard]] ToolDefinition get_definition() const override;

    std::string execute(
        const std::string& json_args,
        const core::context::SessionContext& context) override;
};

} // namespace core::tools
