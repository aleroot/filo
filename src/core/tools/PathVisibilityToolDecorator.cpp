#include "PathVisibilityToolDecorator.hpp"

#include "../workspace/PathVisibility.hpp"

#include <stdexcept>
#include <utility>

namespace core::tools {

PathVisibilityToolDecorator::PathVisibilityToolDecorator(std::shared_ptr<Tool> wrapped)
    : PathVisibilityToolDecorator(
          std::move(wrapped),
          std::make_shared<core::workspace::AgentIgnorePathVisibilityFactory>()) {}

PathVisibilityToolDecorator::PathVisibilityToolDecorator(
    std::shared_ptr<Tool> wrapped,
    std::shared_ptr<const core::workspace::IPathVisibilityFactory> visibility_factory)
    : wrapped_(std::move(wrapped))
    , visibility_factory_(std::move(visibility_factory)) {
    if (!wrapped_) {
        throw std::invalid_argument("PathVisibilityToolDecorator requires a wrapped tool");
    }
    if (!visibility_factory_) {
        throw std::invalid_argument("PathVisibilityToolDecorator requires a visibility factory");
    }
}

ToolDefinition PathVisibilityToolDecorator::get_definition() const {
    return wrapped_->get_definition();
}

std::string PathVisibilityToolDecorator::execute(
    const std::string& json_args,
    const core::context::SessionContext& context) {
    return execute(json_args, ToolInvocationContext{.session_context = context});
}

std::string PathVisibilityToolDecorator::execute(
    const std::string& json_args,
    const ToolInvocationContext& invocation) {
    ToolInvocationContext decorated = invocation;
    if (!decorated.session_context.path_visibility) {
        decorated.session_context.path_visibility =
            std::make_shared<core::workspace::PathVisibility>(
                visibility_factory_->for_context(decorated.session_context));
    }
    return wrapped_->execute(json_args, decorated);
}

std::shared_ptr<Tool> with_path_visibility(std::shared_ptr<Tool> tool) {
    return std::make_shared<PathVisibilityToolDecorator>(std::move(tool));
}

} // namespace core::tools
