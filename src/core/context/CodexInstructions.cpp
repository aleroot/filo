#include "CodexInstructions.hpp"

#include <string>

namespace core::context {
namespace {

constexpr std::string_view kCollaborationOpen = "<collaboration_mode>";
constexpr std::string_view kCollaborationClose = "</collaboration_mode>";

#include "CodexInstructionsData.inc"

[[nodiscard]] bool plan_mode(std::string_view mode) noexcept {
    return mode == "PLAN" || mode == "RESEARCH";
}

[[nodiscard]] std::string collaboration_block(std::string_view body) {
    std::string block;
    block.reserve(kCollaborationOpen.size() + body.size() + kCollaborationClose.size());
    block.append(kCollaborationOpen);
    block.append(body);
    block.append(kCollaborationClose);
    return block;
}

} // namespace

CodexPrompt codex_prompt_for(std::string_view model_id,
                             std::string_view mode,
                             std::string_view catalog_instructions,
                             std::string_view catalog_collaboration_default,
                             std::string_view catalog_collaboration_plan,
                             bool code_mode_only) {
    (void)model_id;
    std::string instructions(catalog_instructions);
    if (instructions.empty() && code_mode_only) {
        instructions.assign(kBundledCodexInstructions);
    }
    if (instructions.empty()) return {};

    const std::string_view selected = plan_mode(mode)
        ? (catalog_collaboration_plan.empty()
               ? kBundledCollaborationPlan
               : catalog_collaboration_plan)
        : (catalog_collaboration_default.empty()
               ? kBundledCollaborationDefault
               : catalog_collaboration_default);
    return CodexPrompt{
        .instructions = std::move(instructions),
        .collaboration = collaboration_block(selected),
    };
}

} // namespace core::context
