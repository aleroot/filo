#pragma once

#include "CodexInstructions.hpp"
#include "PromptFraming.hpp"

#include <memory>

namespace core::context {

/// The prompt a Codex-trained model reads (codex-rs core/src/context): the
/// catalog's instructions template with `<permissions instructions>` and the
/// collaboration block, `<environment_context>`, project instructions as
/// `# AGENTS.md instructions for <dir>`, and skills as `## Skills`.
class CodexPromptFraming final : public PromptFraming {
public:
    explicit CodexPromptFraming(CodexPrompt prompt) : prompt_(std::move(prompt)) {}

    [[nodiscard]] std::string runtime(std::string_view mode) const override;
    [[nodiscard]] std::string workspace(const SessionContext& session_context) const override;
    [[nodiscard]] std::string steering(const SteeringLoadResult& loaded,
                                       std::string_view withheld_notice) const override;
    [[nodiscard]] std::string skills(
        const std::vector<std::filesystem::path>& workspace_roots) const override;

private:
    CodexPrompt prompt_;
};

/// The framing for a model: Codex's when @p codex carries instructions,
/// otherwise Filo's tailored by @p guidance.
[[nodiscard]] std::shared_ptr<const PromptFraming> make_prompt_framing(
    CodexPrompt codex, const ModelGuidance* guidance);

} // namespace core::context
