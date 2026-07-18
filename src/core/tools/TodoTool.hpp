#pragma once

#include "Tool.hpp"
#include "../session/TodoManager.hpp"

namespace core::tools {

class TodoTool final : public Tool {
public:
    explicit TodoTool(core::session::TodoManager& manager) noexcept;

    [[nodiscard]] ToolDefinition get_definition() const override;
    [[nodiscard]] std::string execute(
        const std::string& json_args,
        const core::context::SessionContext& context) override;

private:
    core::session::TodoManager& manager_;
};

} // namespace core::tools
