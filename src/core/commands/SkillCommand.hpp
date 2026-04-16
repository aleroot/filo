#pragma once

#include "CommandExecutor.hpp"
#include "SkillTurnResolver.hpp"
#include "../tools/SkillManifest.hpp"
#include <string>
#include <string_view>

namespace core::commands {

namespace detail {

/**
 * @brief Expands every occurrence of @c $ARGUMENTS in @p body with @p args.
 *
 * If @p body contains no @c $ARGUMENTS placeholder the body is returned as-is,
 * allowing fixed prompts that ignore trailing arguments.
 *
 * @param body  Template string (the @c SKILL.md body section).
 * @param args  Caller-supplied arguments (everything after the command verb).
 * @return      Expanded string with all @c $ARGUMENTS replaced.
 */
inline std::string expand_skill_arguments(std::string_view body, std::string_view args) {
    constexpr std::string_view kPlaceholder = "$ARGUMENTS";
    std::string result(body);
    std::size_t pos = 0;
    while ((pos = result.find(kPlaceholder, pos)) != std::string::npos) {
        result.replace(pos, kPlaceholder.size(), args);
        pos += args.size();
    }
    return result;
}

/**
 * @brief Returns everything in @p text after the first whitespace-separated token.
 *
 * For input @c "/review-pr 123" this returns @c "123".
 * For input @c "/review-pr" (no trailing args) this returns @c "".
 *
 * @param text  Raw input text including the command verb.
 * @return      Trimmed argument string, empty if none.
 */
inline std::string extract_trailing_args(std::string_view text) {
    // Skip leading whitespace
    const auto tok_start = text.find_first_not_of(" \t");
    if (tok_start == std::string_view::npos) return {};
    text = text.substr(tok_start);

    // Find end of command verb
    const auto space = text.find_first_of(" \t");
    if (space == std::string_view::npos) return {};

    // Skip whitespace between verb and args
    const auto args_start = text.find_first_not_of(" \t", space);
    if (args_start == std::string_view::npos) return {};

    return std::string(text.substr(args_start));
}

} // namespace detail

/**
 * @brief A slash command synthesised from a @c SkillType::Prompt SKILL.md manifest.
 *
 * When invoked as @c /<name> [args] the skill's body template is expanded
 * (replacing @c $ARGUMENTS with trailing args) and submitted to the agent as
 * a normal user turn, using the full TUI callback path for rich feedback.
 *
 * If @c CommandContext::send_user_message_fn is not wired (e.g. in headless or
 * test contexts) the command falls back to a direct @c agent->send_message() call.
 *
 * ## Example SKILL.md
 *
 * @code{.yaml}
 * ---
 * name: review-pr
 * description: Security-focused code review for a pull request number.
 * ---
 *
 * Review PR #$ARGUMENTS for security vulnerabilities.
 * Focus on: SQL injection, XSS, auth bypass, secrets in source.
 * @endcode
 *
 * Invoked as: @c /review-pr 42
 * Sends to agent: @c "Review PR #42 for security vulnerabilities.…"
 *
 * @see core::commands::SkillCommandLoader, core::tools::SkillManifest
 */
class SkillCommand : public Command {
public:
    explicit SkillCommand(core::tools::SkillManifest manifest)
        : manifest_(std::move(manifest)) {}

    std::string get_name() const override {
        return "/" + manifest_.name;
    }

    std::string get_description() const override {
        return manifest_.description;
    }

    bool accepts_arguments() const override { return true; }

    void execute(const CommandContext& ctx) override {
        ctx.clear_input_fn();

        const std::string args   = detail::extract_trailing_args(ctx.text);
        const std::string prompt = detail::expand_skill_arguments(manifest_.body, args);

        if (prompt.empty()) {
            ctx.append_history_fn(
                "\n\xe2\x9c\x97  Skill '/" + manifest_.name + "' has an empty body.\n");
            return;
        }

        // Preferred path: inject through the full TUI turn (rich UI feedback).
        if (ctx.send_user_skill_message_fn) {
            ctx.send_user_skill_message_fn(prompt, manifest_.model_hint, manifest_.allowed_tools);
            return;
        }

        const bool needs_turn_resolution =
            !manifest_.model_hint.empty() || !manifest_.allowed_tools.empty();

        if (ctx.send_user_message_fn && !needs_turn_resolution) {
            ctx.send_user_message_fn(prompt);
            return;
        }

        // Fallback: direct agent call (headless, test, or daemon contexts).
        if (ctx.agent) {
            auto append_fn = ctx.append_history_fn;
            const auto resolution = detail::resolve_skill_turn(
                manifest_.model_hint,
                manifest_.allowed_tools);
            if (!resolution.warning.empty()) {
                append_fn("\n⚠  " + resolution.warning + "\n");
            }
            ctx.agent->send_message(
                prompt,
                [append_fn](const std::string& chunk) { append_fn(chunk); },
                [](const std::string&, const std::string&) {},
                []() {},
                resolution.callbacks
            );
            return;
        }

        if (ctx.send_user_message_fn) {
            if (needs_turn_resolution) {
                ctx.append_history_fn(
                    "\n⚠  Skill model/tool constraints are unavailable in this context; sending the prompt without overrides.\n");
            }
            ctx.send_user_message_fn(prompt);
            return;
        }

        ctx.append_history_fn("\n\xe2\x9c\x97  No agent available.\n");
    }

    /// Returns the underlying manifest (read-only).
    const core::tools::SkillManifest& manifest() const { return manifest_; }

private:
    core::tools::SkillManifest manifest_;
};

} // namespace core::commands
