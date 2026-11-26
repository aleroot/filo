#include "ContextBuilder.hpp"
#include "../landrun/LandrunSettings.hpp"

#include "SteeringGuard.hpp"
#include "ModelGuidance.hpp"
#include "SteeringLoader.hpp"
#include "WorkingRules.hpp"
#include "../scm/ScmFactory.hpp"
#include "../tools/SkillRegistry.hpp"
#include "../utils/FileSystemUtils.hpp"
#include "../utils/StringUtils.hpp"
#include "../workspace/SessionWorkspace.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
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

[[nodiscard]] std::string build_runtime_prompt(std::string_view mode,
                                               const ModelGuidance* guidance,
                                               std::string_view codex_instructions,
                                               std::string_view codex_collaboration) {
    if (!codex_instructions.empty()) {
        // The catalog template is the whole doctrine. Filo's working rules
        // contradict it (smallest change, shortest reply, tools the model
        // cannot call directly), so they are not added beside it.
        std::string prompt(codex_instructions);
        if (!codex_collaboration.empty()) {
            prompt += "\n\n";
            prompt += codex_collaboration;
        }
        if (mode == "AUTO") {
            prompt += "\n\nLet Filo's AUTO controller choose a proportional execution contract for each "
                      "turn. Follow the generated AUTO contract exactly: keep simple work direct, "
                      "decompose complex work, fan out independent read-only investigations, "
                      "preserve repository state, and finish mutations with fresh verification "
                      "evidence.";
        }
        return prompt;
    }

    std::string prompt =
        "You are Filo, an advanced AI coding assistant running in " + std::string(mode) + " mode.\n\n";
    if (mode == "AUTO") {
        prompt += "Let Filo's AUTO controller choose a proportional execution contract for each "
                  "turn. Follow the generated AUTO contract exactly: keep simple work direct, "
                  "decompose complex work, fan out independent read-only investigations, preserve "
                  "repository state, and finish mutations with fresh verification evidence. "
                  "Where the contract is silent, the defaults below apply.";
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

    if (guidance == nullptr) {
        prompt += render_working_rules();
    } else {
        prompt += render_working_rules(guidance->rule_overrides);
        prompt += guidance->instructions;
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

/// The IANA name of the local time zone (TZ, else the /etc/localtime link),
/// or empty when neither names one.
[[nodiscard]] std::string local_timezone_name() {
    if (const char* tz = std::getenv("TZ"); tz != nullptr && *tz != '\0') {
        std::string_view name(tz);
        if (name.starts_with(':')) name.remove_prefix(1);
        if (!name.empty() && !name.starts_with('/')) return std::string(name);
    }
    std::error_code error;
    const auto target = std::filesystem::read_symlink("/etc/localtime", error);
    if (error) return {};
    const std::string path = target.generic_string();
    constexpr std::string_view kZoneInfo = "zoneinfo/";
    const auto at = path.rfind(kZoneInfo);
    return at == std::string::npos ? std::string{} : path.substr(at + kZoneInfo.size());
}

/// codex-rs context/world_state/environment.rs: the `<environment_context>`
/// a Codex-trained model reads its cwd, shell, date, and time zone from. The
/// shell is bash because Filo's terminal tool runs every command in bash.
[[nodiscard]] std::string build_environment_context(const SessionContext& session_context) {
    const auto& primary = session_context.workspace_view().primary();
    const auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm local{};
#ifdef _WIN32
    const bool have_date = localtime_s(&local, &now) == 0;
#else
    const bool have_date = localtime_r(&now, &local) != nullptr;
#endif
    std::string block = "\n\n<environment_context>\n";
    if (!primary.empty()) {
        block += "  <cwd>" + core::utils::str::escape_xml(primary.string()) + "</cwd>\n";
    }
    block += "  <shell>bash</shell>\n";
    if (have_date) {
        char date[16] = {};
        if (std::strftime(date, sizeof(date), "%Y-%m-%d", &local) > 0) {
            block += std::string("  <current_date>") + date + "</current_date>\n";
        }
    }
    if (const auto timezone = local_timezone_name(); !timezone.empty()) {
        block += "  <timezone>" + core::utils::str::escape_xml(timezone) + "</timezone>\n";
    }
    block += "</environment_context>";
    return block;
}

[[nodiscard]] std::string build_workspace_facts(const SessionContext& session_context) {
    const auto& workspace = session_context.workspace_view();
    if (workspace.primary().empty()) {
        return {};
    }

    std::string context_section = "\n\n[Workspace]\n";
    context_section += "Primary: " + workspace.primary().string() + "\n";
    // Attached files are named in the message that carried them; listing them
    // here would spend tokens and invalidate the cached prefix per attachment.
    if (const auto directories = workspace.additional_directories();
        !directories.empty()) {
        context_section += "Additional directories:\n";
        for (const auto& dir : directories) {
            context_section += "- " + dir.string() + "\n";
        }
    }
    context_section += std::string("Path enforcement: ")
        + (workspace.enforce() ? "enabled" : "disabled") + "\n";
    context_section += "Relative paths resolve against the primary workspace.";

    const auto sandbox_mode = core::landrun::LandrunSettings::instance().mode();
    if (core::landrun::landrun_enabled(sandbox_mode)) {
        context_section += "\nOS sandbox: ";
        context_section += core::landrun::landrun_mode_name(sandbox_mode);
        context_section += core::landrun::landrun_workspace_writable(sandbox_mode)
            ? " (workspace/temp writes allowed; child network denied)."
            : " (workspace read-only; private temp writes allowed; child network denied).";
    }

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

/**
 * The steering layer: whatever the policy loaded, plus an explicit statement of
 * whatever it deliberately withheld.
 *
 * Naming the unreadable files is what keeps a model from spending turns
 * rediscovering the restriction through rejected tool calls — and, worse, from
 * concluding that the instructions merely went missing and fetching them back
 * through the shell. The list comes from the same guard that enforces the rule,
 * so the prompt can never promise access the tools would deny.
 */
[[nodiscard]] std::string build_steering_layer(
    const std::vector<std::filesystem::path>& roots,
    const SteeringPolicy& policy) {
    auto content = load_workspace_steering_block(roots, policy);

    const auto notice = SteeringGuard::for_roots(roots, policy).prompt_notice();
    if (notice.empty()) {
        return content;
    }
    if (content.empty()) {
        return "\n\n[Project Steering]\n" + notice;
    }
    content += "\n" + notice;
    return content;
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

ContextBuilder& ContextBuilder::with_model_guidance(const ModelGuidance* guidance) noexcept {
    model_guidance_ = guidance;
    return *this;
}

ContextBuilder& ContextBuilder::with_codex_prompt(std::string instructions,
                                                  std::string collaboration) {
    codex_instructions_ = std::move(instructions);
    codex_collaboration_ = std::move(collaboration);
    return *this;
}

ContextBuilder& ContextBuilder::with_memory_prompt(std::string prompt) {
    memory_prompt_ = std::move(prompt);
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
        build_runtime_prompt(mode_, model_guidance_, codex_instructions_, codex_collaboration_));

    append_layer(
        layers,
        ContextLayerKind::WorkspaceFacts,
        PromptStability::Workspace,
        "workspace",
        codex_instructions_.empty()
            ? build_workspace_facts(session_context_)
            : build_workspace_facts(session_context_)
                  + build_environment_context(session_context_));

    if (!include_project_context_) {
        if (session_context_.memory_policy.use_memories
            && !core::landrun::LandrunSettings::instance().enabled()) {
            append_layer(layers, ContextLayerKind::Memory, PromptStability::Session,
                         "memory", memory_prompt_);
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
        // Both steering and the skill catalog are workspace-scoped, not
        // primary-scoped, and they walk the *same* ordered root list so the two
        // subsystems can never disagree about what "the workspace" is. Steering
        // is the first-wins consumer (see SteeringMode::Fallback); skill
        // discovery is the last-wins consumer, which is what lets the primary
        // override a secondary root's skill of the same name.
        const auto workspace_roots = core::workspace::ordered_roots(
            project_root, session_context_.workspace_view().additional_directories());
        const auto steering_mode = session_context_.steering_policy.mode;
        if (!workspace_roots.empty()
            || steering_mode == SteeringMode::CustomFile
            || steering_mode == SteeringMode::CustomDir) {
            append_layer(
                layers,
                ContextLayerKind::ProjectSteering,
                PromptStability::Workspace,
                "project_steering",
                build_steering_layer(workspace_roots, session_context_.steering_policy));
        }

        if (include_skill_catalog_ && !workspace_roots.empty()) {
            append_layer(
                layers,
                ContextLayerKind::SkillCatalog,
                PromptStability::Workspace,
                "skill_catalog",
                core::tools::SkillRegistry::build_catalog_prompt(workspace_roots));
        }

    } catch (const std::exception&) {
        // Context discovery must not prevent agent startup.
    }

    if (session_context_.memory_policy.use_memories
        && !core::landrun::LandrunSettings::instance().enabled()) {
        append_layer(layers, ContextLayerKind::Memory, PromptStability::Session,
                     "memory", memory_prompt_);
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
