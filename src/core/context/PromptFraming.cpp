#include "PromptFraming.hpp"

#include "ModelGuidance.hpp"
#include "SteeringLoader.hpp"
#include "WorkingRules.hpp"
#include "../landrun/LandrunSettings.hpp"
#include "../tools/SkillRegistry.hpp"

#include <string>

namespace core::context {

std::string render_workspace_paths(const SessionContext& session_context) {
    const auto& workspace = session_context.workspace_view();
    if (workspace.primary().empty()) {
        return {};
    }

    std::string section = "\n\n[Workspace]\n";
    section += "Primary: " + workspace.primary().string() + "\n";
    // Attached files are named in the message that carried them; listing them
    // here would spend tokens and invalidate the cached prefix per attachment.
    if (const auto directories = workspace.additional_directories(); !directories.empty()) {
        section += "Additional directories:\n";
        for (const auto& dir : directories) {
            section += "- " + dir.string() + "\n";
        }
    }
    section += std::string("Path enforcement: ")
        + (workspace.enforce() ? "enabled" : "disabled") + "\n";
    section += "Relative paths resolve against the primary workspace.";
    return section;
}

std::string FiloPromptFraming::runtime(std::string_view mode) const {
    std::string prompt =
        "You are Filo, an advanced AI coding assistant running in " + std::string(mode) + " mode.\n\n";
    if (mode == "AUTO") {
        prompt += kAutoContractInstructions;
        prompt += " Where the contract is silent, the defaults below apply.";
    } else if (mode == "PLAN" || mode == "RESEARCH") {
        prompt += "Analyse, research, and plan. Do NOT modify files (avoid apply_patch / write_file). "
                  "Use read and search tools to understand the codebase, then propose a plan.\n"
                  "- Explore before asking: run at least one targeted read or search pass before "
                  "any question, and never ask what one tool call can answer.\n"
                  "- Planning the work is not doing the work: no edits and no state-changing "
                  "commands; read-only builds or tests are fine.\n"
                  "- Make the plan decision-complete: approach, files to touch, verification "
                  "steps, risks, and any assumptions you could not resolve from the code.";
    } else if (mode == "EXECUTE") {
        prompt += "Execute instructions directly. Use tools to modify files and run commands "
                  "when appropriate; client-side permission prompts still apply. Close every "
                  "mutation with verification evidence you actually observed.";
    } else {
        prompt += "Build software methodically. Search, read, edit, and run commands. "
                  "Verify your changes where possible. Ask clarifying questions only when truly needed.";
    }

    if (guidance_ == nullptr) {
        prompt += render_working_rules();
    } else {
        prompt += render_working_rules(guidance_->rule_overrides);
        prompt += guidance_->instructions;
    }

    prompt += "\n\nYou can delegate well-scoped, self-contained subtasks via the `task` tool.";
    prompt += " Use the `subagent_type` values listed in the task tool schema/description.";
    prompt += " Default profiles are `general` (broad multi-step work) and";
    prompt += " `explore` (fast read-only codebase search).";
    prompt += " If the user asks with `@general` or `@explore`, map that request to a `task` call.";
    prompt += " Never predict or fabricate a pending subagent's results; treat surprising claims"
              " from workers as unverified until checked.";

    if (mode == "DEBUG") {
        prompt += "\nDebug with a reproduce -> isolate -> fix -> verify loop. "
                  "Reproduce before editing; check the evidence supports this specific fix; a "
                  "familiar-looking symptom may have a different cause. "
                  "Fix root causes, not symptoms: no deleted assertions, no masking fallbacks. "
                  "Rerun the failing command with its surrounding suite, not just one case. "
                  "If two attempts on the same path fail or the bug won't reproduce, stop and "
                  "report what you know.";
    }

    return prompt;
}

std::string FiloPromptFraming::workspace(const SessionContext& session_context) const {
    std::string section = render_workspace_paths(session_context);
    if (section.empty()) return section;

    const auto sandbox_mode = core::landrun::LandrunSettings::instance().mode();
    if (core::landrun::landrun_enabled(sandbox_mode)) {
        section += "\nOS sandbox: ";
        section += core::landrun::landrun_mode_name(sandbox_mode);
        section += core::landrun::landrun_workspace_writable(sandbox_mode)
            ? " (workspace/temp writes allowed; child network denied)."
            : " (workspace read-only; private temp writes allowed; child network denied).";
    }
    return section;
}

std::string FiloPromptFraming::steering(const SteeringLoadResult& loaded,
                                        std::string_view withheld_notice) const {
    std::string content = loaded.block;
    if (withheld_notice.empty()) return content;
    if (content.empty()) return "\n\n[Project Steering]\n" + std::string(withheld_notice);
    content += "\n";
    content += withheld_notice;
    return content;
}

bool FiloPromptFraming::same_as(const PromptFraming& other) const noexcept {
    const auto* filo = dynamic_cast<const FiloPromptFraming*>(&other);
    return filo != nullptr && filo->guidance_ == guidance_;
}

std::string FiloPromptFraming::skills(
    const std::vector<std::filesystem::path>& workspace_roots) const {
    return core::tools::SkillRegistry::build_catalog_prompt(workspace_roots);
}

} // namespace core::context
