#include "CodexPromptFraming.hpp"

#include "SteeringLoader.hpp"
#include "../landrun/LandrunSettings.hpp"
#include "../tools/SkillRegistry.hpp"
#include "../utils/StringUtils.hpp"

#include <chrono>
#include <cstdlib>
#include <ctime>
#include <string>
#include <system_error>

namespace core::context {
namespace {

/// codex-rs prompts/templates/permissions: the sandbox sentence for the OS
/// sandbox in effect, adjusted to what Filo's sandbox allows, and the
/// `unless-trusted` approval sentence, which is how Filo's gate behaves.
/// Codex's escalation texts are left out: they name tool parameters
/// (`sandbox_permissions`, `justification`) Filo does not have.
[[nodiscard]] std::string permissions_instructions() {
    constexpr std::string_view kSandboxIntro =
        "Filesystem sandboxing defines which files can be read or written. ";
    const auto mode = core::landrun::LandrunSettings::instance().mode();
    std::string text = "<permissions instructions>\n";
    text += kSandboxIntro;
    if (!core::landrun::landrun_enabled(mode)) {
        text += "`sandbox_mode` is `danger-full-access`: No filesystem sandboxing - all commands "
                "are permitted. Network access is enabled.";
    } else if (core::landrun::landrun_workspace_writable(mode)) {
        text += "`sandbox_mode` is `workspace-write`: The sandbox permits reading files, and "
                "editing files in `cwd`, the other workspace directories, and temporary "
                "directories. Commands cannot edit files elsewhere. Network access is restricted.";
    } else {
        text += "`sandbox_mode` is `read-only`: The sandbox only permits reading files, apart "
                "from a private temporary directory. Network access is restricted.";
    }
    text += "\n`approval_policy` is `unless-trusted`: The harness will require user approval "
            "before running commands unless an explicit exec policy rule allows them. The harness "
            "asks the user itself; a call the user denies fails with an error saying so.\n"
            "</permissions instructions>";
    return text;
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

/// codex-rs context/world_state/environment.rs. The shell is bash because
/// Filo runs every command in bash.
[[nodiscard]] std::string environment_context(const SessionContext& session_context) {
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
    if (char date[16] = {}; have_date && std::strftime(date, sizeof(date), "%Y-%m-%d", &local) > 0) {
        block += std::string("  <current_date>") + date + "</current_date>\n";
    }
    if (const auto timezone = local_timezone_name(); !timezone.empty()) {
        block += "  <timezone>" + core::utils::str::escape_xml(timezone) + "</timezone>\n";
    }
    block += "</environment_context>";
    return block;
}

} // namespace

std::string CodexPromptFraming::runtime(std::string_view mode) const {
    // The template is the whole doctrine: Filo's working rules contradict it
    // (smallest change, shortest reply), so they are not added beside it.
    std::string prompt = prompt_.instructions;
    prompt += "\n\n";
    prompt += permissions_instructions();
    if (!prompt_.collaboration.empty()) {
        prompt += "\n\n";
        prompt += prompt_.collaboration;
    }
    if (mode == "AUTO") {
        prompt += "\n\n";
        prompt += kAutoContractInstructions;
    }
    return prompt;
}

std::string CodexPromptFraming::workspace(const SessionContext& session_context) const {
    // The sandbox is described in <permissions instructions>.
    return render_workspace_paths(session_context) + environment_context(session_context);
}

std::string CodexPromptFraming::steering(const SteeringLoadResult& loaded,
                                         std::string_view withheld_notice) const {
    // codex-rs context/user_instructions.rs.
    if (loaded.included.empty() && withheld_notice.empty()) return {};
    std::string block = "\n\n# AGENTS.md instructions";
    if (!loaded.root.empty()) block += " for " + loaded.root.string();
    block += "\n\n<INSTRUCTIONS>\n";
    for (std::size_t i = 0; i < loaded.included.size(); ++i) {
        if (i > 0) block += '\n';
        block += loaded.included[i].content;
        if (!block.ends_with('\n')) block += '\n';
    }
    if (!withheld_notice.empty()) {
        if (!loaded.included.empty()) block += '\n';
        block += withheld_notice;
        if (!block.ends_with('\n')) block += '\n';
    }
    block += "</INSTRUCTIONS>";
    return block;
}

std::string CodexPromptFraming::skills(
    const std::vector<std::filesystem::path>& workspace_roots) const {
    // codex-rs ext/skills catalog_prompt.rs. Every locator is activate_skill,
    // which reads a skill wherever it lives.
    const auto skills = core::tools::SkillRegistry::discover_instruction_skills(workspace_roots);
    if (skills.empty()) return {};
    std::string out = "\n\n<skills_instructions>\n## Skills\n";
    out += "A skill is a set of instructions provided through a `SKILL.md` source. Below is the "
           "list of skills that can be used. Each entry includes a name, description, and source "
           "locator. Every locator here is the `activate_skill` tool: call it with the skill's "
           "`name` to read its `SKILL.md`, and add `resource_path` to read a file the skill "
           "references.\n";
    out += "### Available skills\n";
    for (const auto& skill : skills) {
        out += "- " + skill.name + ": " + skill.description + " (tool: activate_skill";
        if (skill.from_additional_workspace && !skill.workspace_root.empty()) {
            out += ", workspace: " + skill.workspace_root.string();
        }
        out += ")\n";
    }
    out += "</skills_instructions>";
    return out;
}

std::shared_ptr<const PromptFraming> make_prompt_framing(CodexPrompt codex,
                                                         const ModelGuidance* guidance) {
    if (!codex.instructions.empty()) {
        return std::make_shared<const CodexPromptFraming>(std::move(codex));
    }
    return std::make_shared<const FiloPromptFraming>(guidance);
}

} // namespace core::context
