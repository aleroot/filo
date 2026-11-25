#include "RunReview.hpp"

#include "Engine.hpp"
#include "Git.hpp"
#include "Prompt.hpp"
#include "TurnRunner.hpp"

#include "../agent/Agent.hpp"
#include "../utils/JsonWriter.hpp"
#include "../utils/StringUtils.hpp"

#include <utility>

namespace core::review {
namespace {

using core::utils::str::trim_ascii_copy;

[[nodiscard]] bool has_diff(const GitSnapshot& snapshot) {
    return !trim_ascii_copy(snapshot.patch).empty()
        || !trim_ascii_copy(snapshot.untracked_paths).empty()
        || !trim_ascii_copy(snapshot.stat).empty()
        || !trim_ascii_copy(snapshot.status).empty();
}

[[nodiscard]] std::string clamp_text(std::string_view text, std::size_t limit) {
    if (text.size() <= limit) return std::string(text);
    std::string out(text.substr(0, limit));
    out += "...";
    return out;
}

void write_finding(core::utils::JsonWriter& writer, const Finding& finding) {
    auto object = writer.object();
    writer.kv_str("title", finding.title).comma();
    writer.kv_str("severity", std::string(to_string(effective_severity(finding)))).comma();
    writer.kv_str("category", finding.category.has_value()
                                 ? std::string(to_string(*finding.category))
                                 : std::string("unset"))
        .comma();
    writer.kv_str("path", finding.absolute_file_path).comma();
    writer.kv_num("line_start", finding.line_start).comma();
    writer.kv_num("line_end", finding.line_end).comma();
    writer.kv_bool("blocks_merge", blocks_merge(finding)).comma();
    writer.kv_str("body", clamp_text(finding.body, 400));
}

void write_finding_array(core::utils::JsonWriter& writer,
                         const std::vector<Finding>& findings) {
    auto array = writer.array();
    for (std::size_t i = 0; i < findings.size(); ++i) {
        if (i != 0) writer.comma();
        write_finding(writer, findings[i]);
    }
}

} // namespace

WorkspaceReview review_workspace(
    const std::shared_ptr<core::agent::Agent>& session_agent,
    const std::filesystem::path& repository,
    std::string_view target,
    std::string_view against_ref,
    const std::function<void(const Progress&)>& on_progress,
    const std::function<bool()>& cancellation_requested) {
    WorkspaceReview review;
    if (!session_agent) {
        review.error = "review requires an active agent";
        return review;
    }
    if (repository.empty()) {
        review.error = "review requires a repository";
        return review;
    }
    if (cancellation_requested && cancellation_requested()) {
        review.error = "review was interrupted";
        return review;
    }

    const GitClient git(repository);
    std::string git_error;
    if (!git.is_repository(&git_error)) {
        review.error = git_error.empty() ? "not a git repository" : git_error;
        return review;
    }

    std::expected<GitSnapshot, std::string> snapshot;
    CampaignInput input;
    if (target == "uncommitted" || target.empty()) {
        snapshot = git.collect_uncommitted();
        input.task = uncommitted_task();
        input.user_facing_hint = "uncommitted changes";
    } else if (target == "staged") {
        snapshot = git.collect_staged();
        input.task = staged_task();
        input.user_facing_hint = "staged changes";
    } else if (target == "against") {
        if (trim_ascii_copy(against_ref).empty()) {
            review.error = "review target 'against' requires a ref";
            return review;
        }
        snapshot = git.collect_against_ref(against_ref);
        input.task = base_branch_backup_task(against_ref);
        input.user_facing_hint = std::string(against_ref);
    } else {
        review.error = "review target must be uncommitted, staged, or against";
        return review;
    }
    if (!snapshot.has_value()) {
        review.error = snapshot.error();
        return review;
    }
    if (!has_diff(*snapshot)) {
        review.ok = true;
        review.empty = true;
        return review;
    }

    auto review_agent = session_agent->make_isolated_agent(session_agent->get_provider());
    if (!review_agent) {
        review.error = "review requires an active provider";
        return review;
    }
    const std::filesystem::path reviewed = repository;
    review_agent->update_session_context([&](core::context::SessionContext& context) {
        (void)context.set_workspace_primary(reviewed);
    });

    input.snapshot = std::move(*snapshot);
    Engine engine({
        .runner = std::make_unique<AgentTurnRunner>(review_agent, cancellation_requested),
        .cancellation_requested = cancellation_requested,
        .on_progress = on_progress,
    });
    CampaignResult result = engine.run(std::move(input));
    review.findings = classify_findings(result.report);
    if (result.interrupted) {
        review.error = "review was interrupted";
        review.findings.incomplete = true;
        return review;
    }
    if (!result.error.empty()) {
        review.error = result.error;
        review.findings.incomplete = true;
        return review;
    }
    review.ok = true;
    return review;
}

std::string render_review_json(const WorkspaceReview& review) {
    core::utils::JsonWriter writer(1024);
    {
        auto root = writer.object();
        if (!review.ok) {
            writer.kv_bool("ok", false).comma();
            writer.kv_str("error", review.error);
        } else {
            writer.kv_bool("ok", true).comma();
            writer.kv_bool("empty", review.empty).comma();
            writer.kv_bool("incomplete", review.findings.incomplete).comma();
            writer.kv_num("blocking_count", static_cast<int>(review.findings.blocking.size())).comma();
            writer.kv_num("follow_up_count", static_cast<int>(review.findings.follow_ups.size())).comma();
            writer.key("blocking");
            write_finding_array(writer, review.findings.blocking);
            writer.comma().key("follow_ups");
            write_finding_array(writer, review.findings.follow_ups);
        }
    }
    return std::move(writer).take();
}

} // namespace core::review
