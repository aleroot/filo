#include "ContextBuilder.hpp"

#include "SteeringLoader.hpp"
#include "../scm/ScmFactory.hpp"
#include "../memory/MemoryStore.hpp"
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
        prompt += "Execute instructions directly. Use tools to modify files and run commands "
                  "when appropriate; client-side permission prompts still apply.";
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

[[nodiscard]] std::string build_workspace_facts(const SessionContext& session_context) {
    const auto& workspace = session_context.workspace_view();
    if (workspace.primary().empty()) {
        return {};
    }

    std::string context_section = "\n\n[Workspace]\n";
    context_section += "Primary: " + workspace.primary().string() + "\n";
    if (!workspace.additional().empty()) {
        context_section += "Additional directories:\n";
        for (const auto& dir : workspace.additional()) {
            context_section += "- " + dir.string() + "\n";
        }
    }
    context_section += std::string("Path enforcement: ")
        + (workspace.enforce() ? "enabled" : "disabled") + "\n";
    context_section += "Relative paths resolve against the primary workspace.";

    return context_section;
}

void append_layer(std::vector<ContextLayer>& layers,
                  ContextLayerKind kind,
                  PromptStability stability,
                  std::string name,
                  std::string content) {
    if (content.empty()) {
        return;
    }

    layers.push_back(ContextLayer{
        .kind = kind,
        .stability = stability,
        .name = std::move(name),
        .content = std::move(content),
    });
}

} // namespace

std::optional<ProjectFactsSnapshot> capture_project_facts(
    const SessionContext& session_context) noexcept {
    try {
        const auto project_root = resolve_project_root(session_context);
        if (project_root.empty()) {
            return std::nullopt;
        }

        auto scm = core::scm::ScmFactory::create(project_root);
        return ProjectFactsSnapshot{
            .status = scm->get_status_summary(),
            .tree = core::utils::get_file_tree(project_root, *scm, 2),
        };
    } catch (const std::exception&) {
        // Repository discovery is opportunistic and must never block a turn.
        return std::nullopt;
    } catch (...) {
        return std::nullopt;
    }
}

std::string render_project_facts(const ProjectFactsSnapshot& snapshot) {
    if (snapshot.empty()) {
        return {};
    }

    std::string context_section = "\n\n[Project Context]\n";
    if (!snapshot.status.empty()) {
        context_section += "Status:\n" + snapshot.status + "\n";
    }
    if (!snapshot.tree.empty()) {
        context_section += "Structure:\n" + snapshot.tree + "\n";
    }
    return context_section;
}

std::string render_project_facts_update(
    const ProjectFactsSnapshot& previous,
    const ProjectFactsSnapshot& current) {
    if (previous == current) {
        return {};
    }

    std::string update =
        "[Project Context Update]\n"
        "Only the sections shown below changed; they supersede those sections in the prior "
        "repository snapshot.\n";
    if (previous.status != current.status) {
        update += "Status:\n";
        update += current.status.empty() ? "Working tree clean.\n" : current.status + "\n";
    }
    if (previous.tree != current.tree) {
        update += "Structure:\n";
        update += current.tree.empty() ? "(empty)\n" : current.tree + "\n";
    }
    return update;
}

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

ContextBuilder& ContextBuilder::include_project_facts(bool include) noexcept {
    include_project_facts_ = include;
    return *this;
}

std::vector<ContextLayer> ContextBuilder::build_layers() const
{
    std::vector<ContextLayer> layers;
    append_layer(
        layers,
        ContextLayerKind::RuntimeInstructions,
        PromptStability::Stable,
        "runtime",
        build_runtime_prompt(mode_));

    append_layer(
        layers,
        ContextLayerKind::WorkspaceFacts,
        PromptStability::Workspace,
        "workspace",
        build_workspace_facts(session_context_));

    if (!include_project_context_) {
        if (session_context_.memory_policy.use_memories) {
            append_layer(layers, ContextLayerKind::Memory, PromptStability::Session,
                         "memory", core::memory::build_memory_prompt_block(
                             core::memory::MemoryStore{}.load(), 24,
                             session_context_.memory_policy.generate_memories));
        }
        return layers;
    }

    std::filesystem::path project_root;
    try {
        project_root = resolve_project_root(session_context_);
    } catch (const std::exception&) {
        // Context discovery must not prevent agent startup.
    }

    try {
        if (project_root.empty()) {
            return layers;
        }

        append_layer(
            layers,
            ContextLayerKind::ProjectSteering,
            PromptStability::Workspace,
            "project_steering",
            load_project_steering_block(project_root));

        if (include_skill_catalog_) {
            append_layer(
                layers,
                ContextLayerKind::SkillCatalog,
                PromptStability::Workspace,
                "skill_catalog",
                core::tools::SkillRegistry::build_catalog_prompt(project_root));
        }

    } catch (const std::exception&) {
        // Context discovery must not prevent agent startup.
    }

    if (session_context_.memory_policy.use_memories) {
        append_layer(layers, ContextLayerKind::Memory, PromptStability::Session,
                     "memory", core::memory::build_memory_prompt_block(
                         core::memory::MemoryStore{}.load(), 24,
                         session_context_.memory_policy.generate_memories));
    }

    if (include_project_facts_) {
        if (const auto snapshot = capture_project_facts(session_context_); snapshot.has_value()) {
            append_layer(layers, ContextLayerKind::ProjectFacts, PromptStability::Dynamic,
                         "project_facts", render_project_facts(*snapshot));
        }
    }

    return layers;
}

PromptPlan ContextBuilder::build_plan() const {
    PromptPlan plan;
    for (auto& layer : build_layers()) plan.append(std::move(layer));
    return plan;
}

std::string ContextBuilder::build() const {
    return build_plan().render();
}

} // namespace core::context
