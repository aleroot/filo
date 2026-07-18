#pragma once

#include "Tool.hpp"
#include "../agent/ToolResultStore.hpp"

namespace core::tools {

class ReadToolResultTool final : public Tool {
public:
    explicit ReadToolResultTool(core::agent::ToolResultStore& store) noexcept;

    [[nodiscard]] ToolDefinition get_definition() const override;
    [[nodiscard]] std::string execute(
        const std::string& json_args,
        const core::context::SessionContext& context) override;

private:
    core::agent::ToolResultStore& store_;
};

} // namespace core::tools
