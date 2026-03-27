#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace core::tools {

/**
 * @brief Distinguishes how a skill is executed.
 *
 * - @c Tool   — A Python script invoked by the LLM as a callable tool.
 *               Requires a non-empty @c entry_point.
 * - @c Prompt — A markdown template injected as a user turn.
 *               No @c entry_point; has a @c body template with optional @c $ARGUMENTS.
 */
enum class SkillType {
    Tool,   ///< Python entry-point skill (LLM-callable)
    Prompt, ///< Markdown prompt template (slash-command)
};

/**
 * @brief Parsed contents of a @c SKILL.md frontmatter block.
 *
 * A skill lives in its own subdirectory inside a skills root
 * (@c ~/.config/filo/skills/ or @c ./.filo/skills/).  The subdirectory
 * must contain a @c SKILL.md file whose YAML frontmatter describes the skill.
 *
 * ## Tool skill (entry_point present)
 *
 * @code{.yaml}
 * ---
 * name: weather_lookup
 * description: Fetches current weather for a city using the Open-Meteo API.
 * entry_point: weather.py
 * enabled: true
 * ---
 *
 * ## Full documentation for the LLM (optional)
 * …
 * @endcode
 *
 * ## Prompt skill (no entry_point — slash command)
 *
 * @code{.yaml}
 * ---
 * name: review-pr
 * description: Security-focused code review for a GitHub pull request.
 * model: grok-3-mini
 * allowed-tools: shell, read_file
 * enabled: true
 * ---
 *
 * Please review the changes in pull request #$ARGUMENTS for security issues.
 * Focus on: SQL injection, XSS, auth bypass, and secrets in source.
 * @endcode
 *
 * The skill type is auto-detected:
 * - @c entry_point present → @c SkillType::Tool
 * - @c entry_point absent  → @c SkillType::Prompt
 * - @c type: prompt         → always @c SkillType::Prompt (explicit override)
 *
 * ### Common frontmatter fields
 *
 * | Field             | Required       | Default       | Notes                                               |
 * |-------------------|----------------|---------------|-----------------------------------------------------|
 * | @c name           | yes            | —             | Slash command verb or MCP tool identifier           |
 * | @c description    | yes            | —             | Prose description shown to the user and LLM         |
 * | @c entry_point    | Tool only      | @c ""         | Python script filename relative to @c skill_dir     |
 * | @c type           | no             | auto-detected | @c "tool" or @c "prompt"; overrides auto-detection  |
 * | @c model          | no             | @c ""         | Preferred model hint (advisory; not yet enforced)   |
 * | @c allowed-tools  | no             | @c ""         | Comma-separated tool whitelist (future enforcement) |
 * | @c enabled        | no             | @c true       | Set to @c false to skip this skill at load time     |
 *
 * @note @c allowed-tools and @c allowed_tools are both accepted.
 * @note Inline YAML comments (@c #) on value lines are not supported.
 * @note Boolean @c false variants: @c false, @c 0, @c no (case-sensitive).
 *
 * @see SkillLoader, core::commands::SkillCommand
 */
struct SkillManifest {
    // ── Required fields ──────────────────────────────────────────────────────

    /// Slash command verb (Prompt) or MCP tool identifier (Tool), e.g. @c "review-pr".
    std::string name;

    /// Prose description shown to the user in autocomplete and to the LLM for tool selection.
    std::string description;

    // ── Tool-only fields ─────────────────────────────────────────────────────

    /// Python script filename relative to @c skill_dir.  Empty for Prompt skills.
    std::string entry_point;

    // ── Common optional fields ───────────────────────────────────────────────

    /// Whether to register this skill; @c false skips it at load time.
    bool enabled = true;

    /// Resolved absolute path to the directory that contains @c SKILL.md.
    std::filesystem::path skill_dir;

    /// Execution type; auto-detected from presence of @c entry_point when not explicit.
    SkillType type = SkillType::Tool;

    // ── Prompt-skill fields ──────────────────────────────────────────────────

    /// Preferred model hint from the @c model frontmatter key (advisory; not yet enforced).
    std::string model_hint;

    /// Comma-separated tool whitelist from @c allowed-tools (empty = all tools allowed).
    /// Future: passed to the agent to restrict available tools for this turn.
    std::vector<std::string> allowed_tools;

    /// Markdown body of the skill file (everything after the closing @c ---).
    /// May contain @c $ARGUMENTS placeholder, which is substituted at invocation time.
    std::string body;
};

} // namespace core::tools
