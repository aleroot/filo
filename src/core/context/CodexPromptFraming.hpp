#pragma once

#include "CodexInstructions.hpp"
#include "PromptFraming.hpp"

#include <string>
#include <vector>

namespace core::context {

/// The prompt a Codex-trained model reads (codex-rs core/src/context): the
/// catalog's instructions template with `<permissions instructions>` and the
/// collaboration block, `<environment_context>`, project instructions as
/// `# AGENTS.md instructions for <dir>`, and skills as `## Skills`.
class CodexPromptFraming final : public PromptFraming {
public:
    /// @p experimental_tools: the catalog's `experimental_supported_tools`.
    explicit CodexPromptFraming(CodexPrompt prompt, std::vector<std::string> experimental_tools = {});

    [[nodiscard]] std::string runtime(std::string_view mode) const override;
    [[nodiscard]] std::string workspace(const SessionContext& session_context) const override;
    [[nodiscard]] std::string steering(const SteeringLoadResult& loaded,
                                       std::string_view withheld_notice) const override;
    [[nodiscard]] std::string skills(
        const std::vector<std::filesystem::path>& workspace_roots) const override;
    /// The template's "Working with the user" section: a message sent during
    /// active work steers the task.
    [[nodiscard]] bool describes_mid_turn_messages() const noexcept override { return true; }
    /// codex-rs spec_plan.rs add_core_utility_tools: the async user tools the
    /// catalog lists for the model.
    [[nodiscard]] std::vector<std::string> user_messaging_tools() const override {
        return user_messaging_tools_;
    }
    [[nodiscard]] bool same_as(const PromptFraming& other) const noexcept override;

private:
    CodexPrompt prompt_;
    std::vector<std::string> user_messaging_tools_;
};

} // namespace core::context
