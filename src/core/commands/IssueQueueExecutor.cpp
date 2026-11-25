#include "IssueQueueExecutor.hpp"

#include "../goal/IssueHost.hpp"
#include "../goal/IssueQueue.hpp"
#include "../review/RunReview.hpp"
#include "../utils/StringUtils.hpp"

#include <format>
#include <utility>

namespace core::commands {
namespace {

using core::utils::str::trim_ascii_copy;

[[nodiscard]] std::string fix_prompt(const core::goal::IssueRecord& issue,
                                     std::string_view retry) {
    const std::string body = issue.body.size() > 6000
        ? issue.body.substr(0, 6000)
        : issue.body;
    std::string prompt = std::format(
        "Fix GitHub issue #{}: {}\n\n{}\n\n"
        "Work only in this checkout. Do not commit, push, or open a pull request. "
        "Leave the fix as uncommitted changes and run the most relevant check you can.\n",
        issue.number,
        issue.title,
        body);
    if (!trim_ascii_copy(retry).empty()) {
        prompt += "\n";
        prompt += retry;
    }
    return prompt;
}

void execute(const CommandContext& ctx, int max_issues) {
    auto append = [&](std::string_view line) {
        if (ctx.append_history_fn) ctx.append_history_fn(std::string(line));
    };
    if (!ctx.agent) {
        append("\n✗  /goal issues requires an active agent.\n");
        return;
    }
    const auto repository = ctx.agent->workspace_snapshot().primary();
    core::goal::IssueHost host(repository);
    auto listed = host.list_open();
    if (!listed.has_value()) {
        append(std::format("\n✗  {}\n", listed.error()));
        return;
    }
    if (static_cast<int>(listed->size()) > max_issues) {
        listed->resize(static_cast<std::size_t>(max_issues));
    }
    ctx.agent->clear_stop_request();

    core::goal::IssueHooks hooks;
    hooks.prepare = [&](const core::goal::IssueRecord& issue) { return host.prepare(issue); };
    hooks.release = [&](const core::goal::PreparedWorkspace& workspace) { host.release(workspace); };
    hooks.has_changes = [&](const core::goal::PreparedWorkspace& workspace) {
        return host.has_changes(workspace);
    };
    hooks.existing_pr = [&](const core::goal::IssueRecord& issue) {
        return host.open_pull_request(issue);
    };
    hooks.fix = [&](const core::goal::IssueRecord& issue,
                    const core::goal::PreparedWorkspace& workspace,
                    std::string_view retry) {
        return ctx.agent->run_workspace_task(
            std::format("fix #{}", issue.number),
            fix_prompt(issue, retry),
            workspace.path,
            false);
    };
    hooks.review = [&](const core::goal::PreparedWorkspace& workspace)
        -> std::expected<core::goal::ReviewStep, std::string> {
        append("   reviewing…\n");
        const auto cancel = [&] { return ctx.agent->is_stop_requested(); };
        auto reviewed = core::review::review_workspace(
            ctx.agent, workspace.path, "uncommitted", {}, {}, cancel);
        if (reviewed.ok && reviewed.empty) {
            reviewed = core::review::review_workspace(
                ctx.agent, workspace.path, "against", workspace.base_ref, {}, cancel);
        }
        if (!reviewed.ok) return std::unexpected(reviewed.error);
        return core::goal::ReviewStep{
            .findings = std::move(reviewed.findings),
            .empty = reviewed.empty,
        };
    };
    hooks.file_follow_ups = [&](const core::goal::IssueRecord& issue,
                                const core::review::ClassifiedFindings& findings) {
        return host.file_follow_ups(issue, findings);
    };
    hooks.approve_publish = [&](const core::goal::IssueRecord& issue,
                                const core::goal::PreparedWorkspace& workspace) {
        const std::string args = std::format(
            "{{\"command\":\"git push -u origin {} && gh pr create for issue #{}\"}}",
            workspace.branch,
            issue.number);
        return ctx.agent->confirm_tool_call("run_terminal_command", args);
    };
    hooks.publish = [&](const core::goal::IssueRecord& issue,
                        const core::goal::PreparedWorkspace& workspace,
                        std::string_view summary) {
        return host.publish(issue, workspace, summary);
    };
    hooks.cancelled = [&] { return ctx.agent->is_stop_requested(); };
    hooks.on_event = [&](std::string_view line) {
        append(std::format("   {}\n", line));
    };

    append("\n»  Fixing open issues one at a time, each in its own worktree.\n");
    const auto result = core::goal::run_issues(*listed, 2, hooks);
    append(std::format("\n{}  {}\n", result.cancelled ? "ℹ" : "✓", result.summary));
}

} // namespace

void IssueQueueExecutor::run(const CommandContext& ctx, int max_issues) {
    execute(ctx, max_issues);
}

} // namespace core::commands
