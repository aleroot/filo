#include "ContextBuilder.hpp"

#include "SteeringLoader.hpp"
#include "../scm/ScmFactory.hpp"
#include "../tools/SkillRegistry.hpp"
#include "../utils/FileSystemUtils.hpp"

#include <exception>
#include <filesystem>
#include <utility>
#include <string>
#include <vector>

namespace core::context {

namespace {

[[nodiscard]] std::filesystem::path resolve_project_root(
    const SessionContext& session_context)
{
    return session_context.workspace_view().primary();
}

[[nodiscard]] std::string build_runtime_prompt(std::string_view mode) {
    std::string prompt =
        "You are Filo, an advanced AI coding assistant running in " + std::string(mode) + " mode.\n\n";
    if (mode == "PLAN" || mode == "RESEARCH") {
        prompt += "Analyse, research, and plan. Do NOT modify files (avoid apply_patch / write_file). "
                  "Use read and search tools to understand the codebase, then propose a plan.";
    } else if (mode == "EXECUTE") {
        prompt += "Execute instructions autonomously. Use tools to modify files and run commands "
                  "immediately without asking permission.";
    } else {
        prompt += "Build software methodically. Search, read, edit, and run commands. "
                  "Verify your changes where possible. Ask clarifying questions only when truly needed.";
    }

    prompt += "\n\nYou can delegate complex background work via the `task` tool.";
    prompt += " Use the `subagent_type` values listed in the task tool schema/description.";
    prompt += " Default profiles are `general` (broad multi-step work) and";
    prompt += " `explore` (fast read-only codebase search).";
    prompt += " If the user asks with `@general` or `@explore`, map that request to a `task` call.";

    if (mode == "DEBUG") {
        prompt += "\nRun a tight reproduce -> inspect -> fix -> verify loop. "
                  "Collect concrete diagnostics before editing, prefer the smallest fix that resolves the root cause, "
                  "and finish by rerunning the failing command or test.";
    }

    return prompt;
}

[[nodiscard]] std::string build_project_facts(const std::filesystem::path& project_root) {
    auto scm = core::scm::ScmFactory::create(project_root);

    std::string context_section = "\n\n[Project Context]\n";
    bool has_context = false;

    const std::string status = scm->get_status_summary();
    if (!status.empty()) {
        context_section += "Status:\n" + status + "\n";
        has_context = true;
    }

    const std::string tree = core::utils::get_file_tree(project_root, *scm, 2);
    if (!tree.empty()) {
        context_section += "Structure:\n" + tree + "\n";
        has_context = true;
    }

    if (has_context) {
        return context_section;
    }

    return {};
}

void append_layer(std::vector<ContextLayer>& layers,
                  ContextLayerKind kind,
                  std::string name,
                  std::string content) {
    if (content.empty()) {
        return;
    }

    layers.push_back(ContextLayer{
        .kind = kind,
        .name = std::move(name),
        .content = std::move(content),
    });
}

} // namespace

ContextBuilder::ContextBuilder(const SessionContext& session_context)
    : session_context_(session_context) {}

ContextBuilder& ContextBuilder::with_mode(std::string_view mode) {
    mode_ = mode.empty() ? "BUILD" : std::string(mode);
    return *this;
}

ContextBuilder& ContextBuilder::include_project_context(bool include) noexcept {
    include_project_context_ = include;
    return *this;
}

ContextBuilder& ContextBuilder::include_skill_catalog(bool include) noexcept {
    include_skill_catalog_ = include;
    return *this;
}

std::vector<ContextLayer> ContextBuilder::build_layers() const
{
    std::vector<ContextLayer> layers;
    append_layer(
        layers,
        ContextLayerKind::RuntimeInstructions,
        "runtime",
        build_runtime_prompt(mode_));

    if (!include_project_context_) {
        return layers;
    }

    try {
        const auto project_root = resolve_project_root(session_context_);
        if (project_root.empty()) {
            return layers;
        }

        append_layer(
            layers,
            ContextLayerKind::ProjectSteering,
            "project_steering",
            load_project_steering_block(project_root));

        if (include_skill_catalog_) {
            append_layer(
                layers,
                ContextLayerKind::SkillCatalog,
                "skill_catalog",
                core::tools::SkillRegistry::build_catalog_prompt(project_root));
        }

        append_layer(
            layers,
            ContextLayerKind::ProjectFacts,
            "project_facts",
            build_project_facts(project_root));
    } catch (const std::exception&) {
        // Context discovery must not prevent agent startup.
    }

    return layers;
}

std::string ContextBuilder::build() const {
    std::string prompt;
    for (const auto& layer : build_layers()) {
        prompt += layer.content;
    }
    return prompt;
}

} // namespace core::context
