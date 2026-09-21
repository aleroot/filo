#include "ReviewExecutor.hpp"

#include "../review/Engine.hpp"
#include "../review/Findings.hpp"
#include "../review/Git.hpp"
#include "../review/Prompt.hpp"
#include "../review/ReviewSteering.hpp"
#include "../context/SteeringLoader.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <cctype>
#include <format>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace core::commands {
namespace {

using core::utils::str::to_lower_ascii_copy;
using core::utils::str::trim_ascii_copy;
using core::utils::str::trim_ascii_view;

[[nodiscard]] bool token_equals(std::string_view token, std::string_view expected) {
    return to_lower_ascii_copy(token) == expected;
}

enum class ReviewTargetKind {
    UncommittedChanges,
    BaseBranch,
    Commit,
    Custom,
    StagedChanges,
};

struct ReviewTargetSelection {
    ReviewTargetKind kind = ReviewTargetKind::UncommittedChanges;
    std::string branch;
    std::string sha;
    std::string title;
    std::string instructions;
};

struct ParsedReviewCommand {
    bool show_help = false;
    std::string error;
    ReviewTargetSelection target;
};

std::string join(std::string_view separator, const std::vector<std::string>& values) {
    std::string out;
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i > 0) out += separator;
        out += values[i];
    }
    return out;
}

std::string review_usage_text() {
    return
        "\nℹ  Usage: /review [uncommitted | staged | base <branch> | commit <sha> [title <title>] | <instructions>]\n"
        "   /review                      Open the review menu.\n"
        "   /review uncommitted          Review current changes (staged, unstaged, and untracked).\n"
        "   /review staged               Review staged changes only.\n"
        "   /review base main            Review changes against a base branch.\n"
        "   /review commit <sha>         Review a specific commit.\n"
        "   /review commit <sha> title \"Subject\"\n"
        "                               Review a commit and show a custom title.\n"
        "   /review <instructions>       Run a custom review prompt.\n"
        "   Legacy aliases: --uncommitted, --staged, --base, --commit.\n";
}

std::optional<std::vector<std::string>> split_command_arguments(
    std::string_view input,
    std::string& error)
{
    std::vector<std::string> tokens;
    std::string current;
    bool in_single = false;
    bool in_double = false;
    bool escaping = false;

    for (char ch : input) {
        if (escaping) {
            current.push_back(ch);
            escaping = false;
            continue;
        }

        if (ch == '\\' && !in_single) {
            escaping = true;
            continue;
        }
        if (in_single) {
            if (ch == '\'') in_single = false;
            else current.push_back(ch);
            continue;
        }
        if (in_double) {
            if (ch == '"') in_double = false;
            else current.push_back(ch);
            continue;
        }
        if (ch == '\'') {
            in_single = true;
            continue;
        }
        if (ch == '"') {
            in_double = true;
            continue;
        }
        if (std::isspace(static_cast<unsigned char>(ch))) {
            if (!current.empty()) {
                tokens.push_back(std::move(current));
                current.clear();
            }
            continue;
        }
        current.push_back(ch);
    }

    if (escaping) {
        current.push_back('\\');
    }
    if (in_single || in_double) {
        error = "Unterminated quote in /review arguments.";
        return std::nullopt;
    }
    if (!current.empty()) {
        tokens.push_back(std::move(current));
    }
    return tokens;
}

std::string take_short_sha(std::string_view sha) {
    constexpr std::size_t kShortShaLen = 7;
    if (sha.size() <= kShortShaLen) {
        return std::string(sha);
    }
    return std::string(sha.substr(0, kShortShaLen));
}

ParsedReviewCommand parse_review_command(std::string_view raw_args) {
    ParsedReviewCommand parsed;
    parsed.target.kind = ReviewTargetKind::UncommittedChanges;

    const std::string trimmed_args = trim_ascii_copy(raw_args);
    if (trimmed_args.empty()) {
        return parsed;
    }

    std::string token_error;
    const auto tokens_opt = split_command_arguments(trimmed_args, token_error);
    if (!tokens_opt.has_value()) {
        parsed.error = token_error;
        return parsed;
    }

    const auto& tokens = *tokens_opt;
    if (tokens.empty()) {
        return parsed;
    }

    if (tokens.size() == 1 && (tokens[0] == "-h" || tokens[0] == "--help")) {
        parsed.show_help = true;
        return parsed;
    }

    bool staged = false;
    bool uncommitted = false;
    std::optional<std::string> base_branch;
    std::optional<std::string> commit_sha;
    std::optional<std::string> commit_title;
    std::vector<std::string> positional;

    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const auto& token = tokens[i];
        const auto take_option_value = [&](std::string_view option)
            -> std::optional<std::string> {
            if (token == option) {
                if (i + 1 >= tokens.size()) {
                    parsed.error = std::format("Missing value for {}.", option);
                    return std::nullopt;
                }
                return tokens[++i];
            }
            const std::string prefix = std::format("{}=", option);
            if (token.starts_with(prefix)) {
                const auto value = token.substr(prefix.size());
                if (value.empty()) {
                    parsed.error = std::format("Missing value for {}.", option);
                    return std::nullopt;
                }
                return std::string(value);
            }
            return std::string{};
        };

        if (token == "--uncommitted") {
            uncommitted = true;
        } else if (token == "--staged" || token == "--cached") {
            staged = true;
        } else if (token == "--base" || token.starts_with("--base=")) {
            const auto value = take_option_value("--base");
            if (!parsed.error.empty()) return parsed;
            base_branch = *value;
        } else if (token == "--commit" || token.starts_with("--commit=")) {
            const auto value = take_option_value("--commit");
            if (!parsed.error.empty()) return parsed;
            commit_sha = *value;
        } else if (token == "--title" || token.starts_with("--title=")) {
            const auto value = take_option_value("--title");
            if (!parsed.error.empty()) return parsed;
            commit_title = *value;
        } else if (token.starts_with('-')) {
            parsed.error = std::format(
                "Unknown /review option '{}'. Use: uncommitted, staged, base <branch>, commit <sha>.",
                token);
            return parsed;
        } else {
            positional.push_back(token);
        }
    }

    // First token is the target. `/review look at uncommitted changes` stays custom.
    if (!positional.empty()) {
        if (token_equals(positional.front(), "help")) {
            parsed.show_help = true;
            return parsed;
        }
        if (token_equals(positional.front(), "uncommitted")) {
            uncommitted = true;
            positional.erase(positional.begin());
        } else if (token_equals(positional.front(), "staged")) {
            staged = true;
            positional.erase(positional.begin());
        } else if (token_equals(positional.front(), "base")) {
            if (positional.size() < 2) {
                parsed.error = "Missing branch for base.";
                return parsed;
            }
            base_branch = positional[1];
            positional.erase(positional.begin(), positional.begin() + 2);
        } else if (token_equals(positional.front(), "commit")) {
            if (positional.size() < 2) {
                parsed.error = "Missing commit SHA.";
                return parsed;
            }
            commit_sha = positional[1];
            positional.erase(positional.begin(), positional.begin() + 2);
            if (!positional.empty() && token_equals(positional.front(), "title")) {
                if (positional.size() < 2) {
                    parsed.error = "Missing value for title.";
                    return parsed;
                }
                commit_title = join(
                    " ",
                    std::vector<std::string>(positional.begin() + 1, positional.end()));
                positional.clear();
            }
        }
    }

    if (commit_title.has_value() && !commit_sha.has_value()) {
        parsed.error = "title requires a commit target.";
        return parsed;
    }

    int mode_count = 0;
    mode_count += staged ? 1 : 0;
    mode_count += uncommitted ? 1 : 0;
    mode_count += base_branch.has_value() ? 1 : 0;
    mode_count += commit_sha.has_value() ? 1 : 0;

    if (mode_count > 1) {
        parsed.error = "Choose only one review target: uncommitted, staged, base, or commit.";
        return parsed;
    }
    if (mode_count > 0 && !positional.empty()) {
        parsed.error = "Cannot combine custom instructions with a review target.";
        return parsed;
    }

    if (mode_count == 0) {
        if (!positional.empty()) {
            parsed.target.kind = ReviewTargetKind::Custom;
            parsed.target.instructions = join(" ", positional);
        }
        return parsed;
    }

    if (staged) {
        parsed.target.kind = ReviewTargetKind::StagedChanges;
        return parsed;
    }
    if (uncommitted) {
        parsed.target.kind = ReviewTargetKind::UncommittedChanges;
        return parsed;
    }
    if (base_branch.has_value()) {
        parsed.target.kind = ReviewTargetKind::BaseBranch;
        parsed.target.branch = std::move(*base_branch);
        return parsed;
    }
    if (commit_sha.has_value()) {
        parsed.target.kind = ReviewTargetKind::Commit;
        parsed.target.sha = std::move(*commit_sha);
        parsed.target.title = commit_title.value_or("");
        return parsed;
    }

    parsed.error = "Unable to parse /review arguments.";
    return parsed;
}

std::string review_user_hint(const ReviewTargetSelection& target) {
    switch (target.kind) {
        case ReviewTargetKind::UncommittedChanges:
            return "current changes";
        case ReviewTargetKind::BaseBranch:
            return std::format("changes against '{}'", target.branch);
        case ReviewTargetKind::Commit:
            if (!trim_ascii_view(target.title).empty()) {
                return std::format("commit {}: {}", take_short_sha(target.sha), target.title);
            }
            return std::format("commit {}", take_short_sha(target.sha));
        case ReviewTargetKind::Custom:
            return trim_ascii_copy(target.instructions);
        case ReviewTargetKind::StagedChanges:
            return "staged changes";
    }
    return "current changes";
}

std::string build_custom_submission_prompt(std::string_view task) {
    std::string prompt;
    prompt += "You are running Filo's standalone /review task.\n";
    prompt += "Use available tools to inspect the relevant git changes and code.\n";
    prompt += "Minimize token usage while staying accurate:\n";
    prompt += "- Start with diff summaries (`git diff --stat`, changed files) before deep dives.\n";
    prompt += "- Read only targeted slices (`read` with offset_line/limit_lines) when possible.\n";
    prompt += "- Avoid redundant reads of the same large files unless needed to verify a finding.\n";
    prompt += "- Do not run broad repository scans or tests unless the diff makes them necessary.\n";
    prompt += "- Finish once you have inspected the relevant diff and any directly related code.\n";
    prompt += "Return only valid JSON that matches the schema below.\n";
    prompt += "Do not wrap JSON in markdown fences.\n\n";
    prompt += "[Review task]\n";
    prompt += task;
    if (!prompt.empty() && prompt.back() != '\n') {
        prompt.push_back('\n');
    }
    prompt += "\n";
    prompt += core::review::review_rubric();
    return prompt;
}

using AssistantOutputCallback = std::function<void(const std::string&)>;
using AssistantDisclosureOutputCallback =
    std::function<void(const std::string&, const std::string&, const std::string&)>;

void append_review_report(const std::function<void(const std::string&)>& append_history,
                          const AssistantOutputCallback& append_assistant,
                          const AssistantDisclosureOutputCallback& append_disclosure,
                          const core::review::Report& report,
                          std::string_view extra_text = {}) {
    std::string full_output = core::review::render_report(report);
    full_output += extra_text;

    const std::string low_notes =
        core::review::render_low_severity_findings(report);
    if (!low_notes.empty() && append_disclosure) {
        std::string display = core::review::render_report(report, false);
        display += extra_text;
        const auto low_count = std::count_if(
            report.findings.begin(), report.findings.end(), [](const auto& finding) {
                return core::review::effective_severity(finding)
                    == core::review::Severity::Low;
            });
        const std::string summary = low_count == 1
            ? "1 low-severity note omitted"
            : std::format("{} low-severity notes omitted", low_count);
        append_disclosure(
            display,
            summary,
            low_notes);
    } else if (append_assistant) {
        append_assistant(full_output);
    } else {
        append_history(std::format("\n{}\n", full_output));
    }
}

void run_agent_review_turn(const CommandContext& ctx,
                           std::string prompt,
                           bool allow_tools) {
    const auto set_review_activity_fn = ctx.set_review_activity_fn;
    const auto append_assistant_output_fn = ctx.append_assistant_output_fn;
    const auto append_assistant_disclosure_output_fn =
        ctx.append_assistant_disclosure_output_fn;
    auto append_fn = ctx.append_history_fn;
    auto collected_output = std::make_shared<std::string>();
    auto interrupted = std::make_shared<bool>(false);

    core::agent::Agent::TurnCallbacks review_callbacks{
        .min_context_utilization_for_rotation =
            core::review::kReviewRotationMinContextUtilization,
    };
    review_callbacks.effort_override = "off";
    review_callbacks.max_tokens_override = core::review::kReviewMaxOutputTokens;
    review_callbacks.response_format_override = core::llm::ResponseFormat{
        .type = core::llm::ResponseFormat::Type::JsonObject,
    };
    if (!allow_tools) {
        review_callbacks.allowed_tools = {"__filo_no_tools__"};
    }

    ctx.agent->send_message(
        std::move(prompt),
        [collected_output, interrupted](const std::string& chunk) {
            *collected_output += chunk;
            if (chunk.find(core::review::kStopMarker) != std::string::npos) {
                *interrupted = true;
            }
        },
        [](const std::string&, const std::string&) {},
        [append_fn,
         collected_output,
         interrupted,
         set_review_activity_fn,
         append_assistant_output_fn,
         append_assistant_disclosure_output_fn]() {
            const std::string raw = trim_ascii_copy(*collected_output);
            if (*interrupted || raw.find(core::review::kStopMarker) != std::string::npos) {
                append_fn(
                    "\nℹ  Review was interrupted. Re-run /review and wait for it to complete.\n");
                if (set_review_activity_fn) {
                    set_review_activity_fn(false, "");
                }
                return;
            }

            const auto output = core::review::parse_review_output(raw);
            append_review_report(append_fn,
                                 append_assistant_output_fn,
                                 append_assistant_disclosure_output_fn,
                                 output);
            if (set_review_activity_fn) {
                set_review_activity_fn(false, "");
            }
        },
        std::move(review_callbacks));
}

void persist_review_summary(const CommandContext& ctx,
                            std::string_view hint,
                            std::string_view rendered) {
    if (!ctx.agent) return;
    ctx.agent->append_history_message(core::llm::Message{
        .role = "user",
        .content = std::format("[/review] {}", hint),
        .synthetic = true,
    });
    ctx.agent->append_history_message(core::llm::Message{
        .role = "assistant",
        .content = std::string(rendered),
        .synthetic = true,
    });
}

/// Renders engine lifecycle events into the two surfaces the user watches:
/// the footer activity pill (what is running now) and the transcript (what
/// each group concluded). Progress is reported from the review worker thread;
/// both callbacks are the thread-safe UI sinks the command layer already owns.
[[nodiscard]] std::function<void(const core::review::Progress&)>
make_progress_renderer(const CommandContext& ctx, std::string hint) {
    const auto set_review_activity_fn = ctx.set_review_activity_fn;
    const auto review_progress_fn = ctx.review_progress_fn;
    // Hosts that render the structured card must not also receive the plain
    // text lines, or every group would be reported twice.
    const auto append_history_fn =
        review_progress_fn ? std::function<void(const std::string&)>{} : ctx.append_history_fn;

    return [set_review_activity_fn, review_progress_fn, append_history_fn, hint = std::move(hint)](
               const core::review::Progress& progress) {
        using core::review::ProgressPhase;

        if (review_progress_fn) {
            review_progress_fn(progress);
        }

        // The footer pill shows review (N/M). File names and elapsed time stay
        // on the card and the pill's popover so they do not crowd the bar.
        const auto keep_pill_alive = [&] {
            if (!set_review_activity_fn) return;
            set_review_activity_fn(true, hint);
        };

        switch (progress.phase) {
            case ProgressPhase::Planned: {
                if (append_history_fn) {
                    std::string line = std::format(
                        "   Planned {} review pass(es) over {} file(s), {} changed line(s)",
                        progress.group_total,
                        progress.files,
                        progress.changed_lines);
                    if (progress.risk_passes > 0) {
                        line += std::format("; {} risk pass(es)", progress.risk_passes);
                    }
                    if (progress.skipped_files > 0) {
                        line += std::format("; {} file(s) too large to review",
                                            progress.skipped_files);
                    }
                    line += ".\n";
                    append_history_fn(line);
                }
                keep_pill_alive();
                return;
            }
            case ProgressPhase::GroupStarted:
            case ProgressPhase::RiskPass:
            case ProgressPhase::Summarizing:
                keep_pill_alive();
                return;
            case ProgressPhase::GroupFailed: {
                keep_pill_alive();
                if (!append_history_fn) return;
                append_history_fn(std::format(
                    "   [{}/{}] {} — not reviewed: {}\n",
                    progress.group_index,
                    progress.group_total,
                    progress.label,
                    progress.failure));
                return;
            }
            case ProgressPhase::Finished:
                // The campaign outcome is reported by the caller (rendered
                // report, interruption notice, or error line).
                return;
            case ProgressPhase::GroupFinished: {
                if (!append_history_fn) return;
                std::string outcome;
                if (progress.findings == 0) {
                    outcome = "no findings";
                } else if (progress.blocking_findings > 0) {
                    outcome = std::format("{} finding(s), {} blocking",
                                          progress.findings,
                                          progress.blocking_findings);
                } else {
                    outcome = std::format("{} finding(s)", progress.findings);
                }
                append_history_fn(std::format(
                    "   [{}/{}] {} — {}\n",
                    progress.group_index,
                    progress.group_total,
                    progress.label,
                    outcome));
                return;
            }
        }
    };
}

void run_grouped_review(const CommandContext& ctx,
                        core::review::CampaignInput input) {
    const auto set_review_activity_fn = ctx.set_review_activity_fn;
    const std::string hint = input.user_facing_hint;
    auto agent = ctx.agent;
    auto review_agent = agent->make_isolated_agent(agent->get_provider());
    if (!review_agent) {
        ctx.append_history_fn("\n✗  Review requires an active provider.\n");
        if (set_review_activity_fn) {
            set_review_activity_fn(false, "");
        }
        return;
    }

    core::review::Engine engine({
        .runner = std::make_unique<core::review::AgentTurnRunner>(
            std::move(review_agent),
            [agent] {
                return agent && agent->is_stop_requested();
            }),
        .cancellation_requested = [agent] {
            return agent && agent->is_stop_requested();
        },
        .on_progress = make_progress_renderer(ctx, hint),
    });

    auto result = engine.run(input);

    const auto publish_report = [&](std::string extra_text, bool persist) {
        std::string rendered = core::review::render_report(result.report);
        rendered += extra_text;
        if (persist) {
            persist_review_summary(ctx, hint, rendered);
        }
        append_review_report(ctx.append_history_fn,
                             ctx.append_assistant_output_fn,
                             ctx.append_assistant_disclosure_output_fn,
                             result.report,
                             extra_text);
    };

    if (result.interrupted) {
        const bool has_partial = result.report.groups_reviewed > 0
            || !result.report.findings.empty();
        if (has_partial) {
            publish_report(
                "\n\nReview was interrupted. Findings above cover the units that finished.",
                true);
        } else {
            ctx.append_history_fn(
                "\nℹ  Review was interrupted. Re-run /review and wait for it to complete.\n");
        }
        if (set_review_activity_fn) {
            set_review_activity_fn(false, "");
        }
        return;
    }
    if (!result.error.empty()) {
        ctx.append_history_fn(std::format("\n✗  Review failed: {}\n", result.error));
        if (set_review_activity_fn) {
            set_review_activity_fn(false, "");
        }
        return;
    }

    publish_report({}, true);
    if (set_review_activity_fn) {
        set_review_activity_fn(false, "");
    }
}

} // namespace

void ReviewExecutor::execute(const CommandContext& ctx, std::string_view raw_args) {
    const ParsedReviewCommand parsed = parse_review_command(raw_args);
    if (parsed.show_help) {
        ctx.append_history_fn(review_usage_text());
        return;
    }
    if (!parsed.error.empty()) {
        ctx.append_history_fn(std::format(
            "\n✗  {}\n{}",
            parsed.error,
            review_usage_text()));
        return;
    }

    if (!ctx.agent) {
        ctx.append_history_fn("\n✗  Review requires an active agent session.\n");
        return;
    }

    const auto set_review_activity_fn = ctx.set_review_activity_fn;
    const std::string initial_review_hint = review_user_hint(parsed.target);
    if (set_review_activity_fn) {
        set_review_activity_fn(
            true,
            initial_review_hint.empty() ? std::string("current changes") : initial_review_hint);
    }
    auto stop_review_activity = [&]() {
        if (set_review_activity_fn) {
            set_review_activity_fn(false, "");
        }
    };

    if (parsed.target.kind != ReviewTargetKind::Custom) {
        std::string git_error;
        if (!core::review::GitClient::is_repository(&git_error)) {
            const std::string detail = trim_ascii_copy(git_error);
            stop_review_activity();
            ctx.append_history_fn(std::format(
                "\n✗  /review requires a git repository. {}\n",
                detail.empty() ? std::string() : std::format("({})", detail)));
            return;
        }
    }

    const std::string review_hint = initial_review_hint.empty()
        ? std::string("current changes")
        : initial_review_hint;
    ctx.append_history_fn(std::format("\n»  Code review started: {}…\n", review_hint));

    if (parsed.target.kind == ReviewTargetKind::Custom) {
        const std::string prompt = build_custom_submission_prompt(parsed.target.instructions);
        run_agent_review_turn(ctx, prompt, true);
        return;
    }

    core::review::CampaignInput input;
    input.user_facing_hint = review_hint;

    switch (parsed.target.kind) {
        case ReviewTargetKind::UncommittedChanges: {
            auto snapshot = core::review::GitClient::collect_uncommitted();
            if (!snapshot.has_value()) {
                stop_review_activity();
                ctx.append_history_fn(std::format(
                    "\n✗  Could not start review: {}\n",
                    snapshot.error()));
                return;
            }
            input.task = core::review::uncommitted_task();
            input.snapshot = std::move(*snapshot);
            break;
        }
        case ReviewTargetKind::StagedChanges: {
            auto snapshot = core::review::GitClient::collect_staged();
            if (!snapshot.has_value()) {
                stop_review_activity();
                ctx.append_history_fn(std::format(
                    "\n✗  Could not start review: {}\n",
                    snapshot.error()));
                return;
            }
            input.task = core::review::staged_task();
            input.snapshot = std::move(*snapshot);
            break;
        }
        case ReviewTargetKind::BaseBranch: {
            if (trim_ascii_view(parsed.target.branch).empty()) {
                stop_review_activity();
                ctx.append_history_fn("\n✗  Could not start review: Base branch name cannot be empty.\n");
                return;
            }
            const auto merge_base =
                core::review::GitClient::merge_base_with_head(parsed.target.branch);
            if (!merge_base.has_value()) {
                const std::string prompt = build_custom_submission_prompt(
                    core::review::base_branch_backup_task(parsed.target.branch));
                run_agent_review_turn(ctx, prompt, true);
                return;
            }
            auto snapshot = core::review::GitClient::collect_against_ref(*merge_base);
            if (!snapshot.has_value()) {
                stop_review_activity();
                ctx.append_history_fn(std::format(
                    "\n✗  Could not start review: {}\n",
                    snapshot.error()));
                return;
            }
            input.task = core::review::base_branch_task(parsed.target.branch, *merge_base);
            input.snapshot = std::move(*snapshot);
            break;
        }
        case ReviewTargetKind::Commit: {
            if (trim_ascii_view(parsed.target.sha).empty()) {
                stop_review_activity();
                ctx.append_history_fn("\n✗  Could not start review: Commit SHA cannot be empty.\n");
                return;
            }
            if (!core::review::GitClient::rev_parse(parsed.target.sha).has_value()) {
                stop_review_activity();
                ctx.append_history_fn(std::format(
                    "\n✗  Could not start review: Commit '{}' could not be resolved.\n",
                    parsed.target.sha));
                return;
            }
            auto snapshot = core::review::GitClient::collect_commit(parsed.target.sha);
            if (!snapshot.has_value()) {
                stop_review_activity();
                ctx.append_history_fn(std::format(
                    "\n✗  Could not start review: {}\n",
                    snapshot.error()));
                return;
            }
            input.task = core::review::commit_task(parsed.target.sha, parsed.target.title);
            input.snapshot = std::move(*snapshot);
            break;
        }
        case ReviewTargetKind::Custom:
            break;
    }

    const auto session = ctx.agent->session_context_snapshot();
    const auto workspace = ctx.agent->workspace_snapshot();
    const auto steering_roots = core::context::collect_steering_roots(
        workspace.primary(), workspace.additional());
    input.steering = core::review::load_review_steering_guidance(
        input.snapshot, steering_roots, session.steering_policy);
    run_grouped_review(ctx, std::move(input));
}

} // namespace core::commands
