#pragma once

#include "Tool.hpp"

#include <memory>

namespace core::workspace {
class IPathVisibilityFactory;
}

namespace core::tools {

class PathVisibilityToolDecorator final : public Tool {
public:
    explicit PathVisibilityToolDecorator(std::shared_ptr<Tool> wrapped);
    PathVisibilityToolDecorator(
        std::shared_ptr<Tool> wrapped,
        std::shared_ptr<const core::workspace::IPathVisibilityFactory> visibility_factory);

    [[nodiscard]] ToolDefinition get_definition() const override;

    [[nodiscard]] std::string execute(
        const std::string& json_args,
        const core::context::SessionContext& context) override;

    [[nodiscard]] std::string execute(
        const std::string& json_args,
        const ToolInvocationContext& invocation) override;

private:
    std::shared_ptr<Tool> wrapped_;
    std::shared_ptr<const core::workspace::IPathVisibilityFactory> visibility_factory_;
};

[[nodiscard]] std::shared_ptr<Tool> with_path_visibility(
    std::shared_ptr<Tool> tool);

} // namespace core::tools
