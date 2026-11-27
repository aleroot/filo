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

// A finding's body carries the suggested fix in its last sentences, so the
// limit is generous; past it the cut lands on a word boundary and never inside
// a UTF-8 sequence.
inline constexpr std::size_t kMaxBodyBytes = 1500;
inline constexpr std::size_t kMaxRationaleBytes = 800;

[[nodiscard]] std::string clamp_text(std::string_view text, std::size_t limit) {
    if (text.size() <= limit) return std::string(text);
    std::size_t cut = limit;
    while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0) == 0x80) --cut;
    if (const auto space = text.find_last_of(" \n\t", cut); space != std::string_view::npos
        && space > limit / 2) {
        cut = space;
    }
    std::string out(trim_ascii_copy(text.substr(0, cut)));
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
    writer.kv_str("body", clamp_text(finding.body, kMaxBodyBytes));
    if (!finding.rationale.empty()) {
        writer.comma().kv_str("rationale", clamp_text(finding.rationale, kMaxRationaleBytes));
    }
}

void write_finding_array(core::utils::JsonWriter& writer,
                         const std::vector<Finding>& findings) {
    auto array = writer.array();
    for (std::size_t i = 0; i < findings.size(); ++i) {
        if (i != 0) writer.comma();
        write_finding(writer, findings[i]);
    }
}

void write_review_body(core::utils::JsonWriter& writer, const WorkspaceReview& review) {
    if (!review.repository.empty()) {
        writer.kv_str("repository", review.repository).comma();
    }
    writer.kv_bool("empty", review.empty).comma();
    writer.kv_bool("incomplete", review.findings.incomplete).comma();
    writer.kv_num("blocking_count", static_cast<int>(review.findings.blocking.size())).comma();
    writer.kv_num("follow_up_count", static_cast<int>(review.findings.follow_ups.size())).comma();
    writer.key("blocking");
    write_finding_array(writer, review.findings.blocking);
    writer.comma().key("follow_ups");
    write_finding_array(writer, review.findings.follow_ups);
    if (review.failed_groups.empty()) return;
    writer.comma().key("unreviewed");
    auto array = writer.array();
    for (std::size_t i = 0; i < review.failed_groups.size(); ++i) {
        if (i != 0) writer.comma();
        auto object = writer.object();
        writer.kv_str("unit", review.failed_groups[i].label).comma();
        writer.kv_str("reason", clamp_text(review.failed_groups[i].reason, 300));
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
    review.repository = repository.string();
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
    review.failed_groups = result.report.failed_groups;
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
        const bool has_findings =
            !review.findings.blocking.empty() || !review.findings.follow_ups.empty();
        writer.kv_bool("ok", review.ok).comma();
        if (!review.ok) {
            writer.kv_str("error", review.error);
        }
        // An interrupted or failed campaign still reports what the finished
        // units found and which units it never reached; dropping either would
        // pass a partial review off as a bare error.
        if (review.ok || has_findings || !review.failed_groups.empty()) {
            if (!review.ok) writer.comma();
            write_review_body(writer, review);
        }
    }
    return std::move(writer).take();
}

} // namespace core::review
