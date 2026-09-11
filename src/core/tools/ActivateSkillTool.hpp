#pragma once

#include "Tool.hpp"

#include <filesystem>
#include <vector>

namespace core::tools {

class ActivateSkillTool final : public Tool {
public:
    /**
     * @param workspace_roots Roots skill discovery scans, in precedence order —
     *        @ref core::workspace::ordered_roots(), so index 0 is the primary and
     *        the rest are additional workspace roots. Empty means "ask the
     *        process-wide Workspace singleton when the schema is built", which
     *        keeps the tool correct across `/workspace change` and never falls
     *        back to the ambient working directory.
     */
    explicit ActivateSkillTool(std::vector<std::filesystem::path> workspace_roots = {});

    [[nodiscard]] ToolDefinition get_definition() const override;

    std::string execute(
        const std::string& json_args,
        const core::context::SessionContext& context) override;

private:
    /// Roots the tool *schema* enumerates: the injected ones when the owner knew
    /// them, otherwise the live process workspace.
    [[nodiscard]] std::vector<std::filesystem::path> schema_workspace_roots() const;

    std::vector<std::filesystem::path> workspace_roots_;
};

} // namespace core::tools
