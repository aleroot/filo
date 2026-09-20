#include "Engine.hpp"

#include "Findings.hpp"
#include "Prompt.hpp"
#include "ReviewSteering.hpp"
#include "SummaryPrompt.hpp"

#include "../tools/ToolNames.hpp"
#include "../utils/StringUtils.hpp"

#include <algorithm>
#include <atomic>
#include <format>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

namespace core::review {
namespace {

[[nodiscard]] bool cancelled(const std::function<bool()>& cancellation_requested) {
    return cancellation_requested && cancellation_requested();
}

[[nodiscard]] std::string group_label(const ReviewGroup& group) {
    auto names = join_group_paths(group);
    return names.empty() ? std::string("changes") : names;
}

[[nodiscard]] int count_blocking(const Report& report) noexcept {
    int blocking = 0;
    for (const auto& finding : report.findings) {
        const auto severity = effective_severity(finding);
        if (severity == Severity::Critical || severity == Severity::High) {
            ++blocking;
        }
    }
    return blocking;
}

void resolve_finding_paths(Report& report, std::string_view worktree_root) {
    for (auto& finding : report.findings) {
        if (finding.absolute_file_path.empty()) continue;
        finding.absolute_file_path =
            absolute_review_path(worktree_root, finding.absolute_file_path);
    }
}

struct GroupOutcome {
    Report report;
    bool reviewed = false;
    bool failed = false;
    bool interrupted = false;
    bool plan_passed = false;
    std::string label;
    std::string failure;
};

} // namespace

Engine::Engine(Options options) : options_(std::move(options)) {}

CampaignResult Engine::run(const CampaignInput& input) const {
    std::mutex progress_mutex;
    const auto report_progress = [this, &progress_mutex](const Progress& progress) {
        if (options_.on_progress) {
            // Group workers report concurrently. Keep adapters such as the
            // transcript writer serialized without forcing model turns back
            // onto one lane.
            std::lock_guard lock(progress_mutex);
            options_.on_progress(progress);
        }
    };
    // Exactly one terminal event per campaign, on every exit path, so a live
    // view can always settle into a final state instead of spinning forever.
    const auto finish = [&report_progress](CampaignResult result) {
        Progress done{.phase = ProgressPhase::Finished};
        done.findings = static_cast<int>(result.report.findings.size());
        done.blocking_findings = count_blocking(result.report);
        done.skipped_files = static_cast<int>(result.report.skipped_paths.size());
        done.skipped_paths = result.report.skipped_paths;
        done.failed_groups = static_cast<int>(result.report.failed_groups.size());
        done.interrupted = result.interrupted;
        done.failure = result.error;
        report_progress(done);
        return result;
    };

    CampaignResult result;
    if (!options_.runner) {
        result.error = "review engine is missing a turn runner";
        return finish(std::move(result));
    }

    const auto files = parse_unified_diff(input.snapshot.patch);
    const auto plan = plan_review(files, options_.grouper);

    if (cancelled(options_.cancellation_requested)) {
        result.interrupted = true;
        return finish(std::move(result));
    }

    if (plan.groups.empty()) {
        result.report = aggregate_reports({}, plan.skipped_paths, plan.warnings);
        return finish(std::move(result));
    }

    std::vector<Report> group_reports;
    group_reports.reserve(plan.groups.size());
    std::vector<FailedGroup> failed_groups;
    int plan_passes = 0;

    // Stop/Esc used to return an empty CampaignResult, throwing away every
    // group that had already finished. Seal whatever we have so the caller can
    // still render a partial summary.
    const auto seal = [&](bool interrupted) {
        CampaignResult out;
        out.interrupted = interrupted;

        auto warnings = plan.warnings;
        for (const auto& failure : failed_groups) {
            warnings.push_back(std::format(
                "Could not review {}: {}", failure.label, failure.reason));
        }

        if (group_reports.empty()) {
            out.report.skipped_paths.assign(
                plan.skipped_paths.begin(), plan.skipped_paths.end());
            out.report.warnings = std::move(warnings);
            out.report.failed_groups = failed_groups;
            out.report.plan_passes = plan_passes;
            out.report.overall_explanation = !failed_groups.empty()
                ? std::format("No file groups completed review; {} group(s) could not be reviewed.",
                              failed_groups.size())
                : "Review stopped before any file group completed.";
            if (!interrupted && !failed_groups.empty()) {
                out.error = failed_groups.front().reason;
            }
            return finish(std::move(out));
        }

        out.report = aggregate_reports(group_reports, plan.skipped_paths, warnings);
        out.report.plan_passes = plan_passes;
        if (!failed_groups.empty()) {
            out.report.overall_explanation += std::format(
                " {} file-group(s) could not be reviewed.", failed_groups.size());
        }
        if (interrupted
            && group_reports.size() + failed_groups.size() < plan.groups.size()) {
            out.report.overall_explanation += std::format(
                " Review stopped after {} of {} file-group(s).",
                group_reports.size() + failed_groups.size(),
                plan.groups.size());
        }
        out.report.failed_groups = failed_groups;

        // Findings and the verdict are already sealed. A synthesizer turn is
        // optional: skip it for a one-group review that already has a staff
        // summary, and never relabel a completed findings pass as interrupted
        // just because the synthesizer was cancelled or returned junk.
        if (!interrupted
            && campaign_needs_summary(out.report)
            && !cancelled(options_.cancellation_requested)) {
            Progress summarizing{.phase = ProgressPhase::Summarizing};
            summarizing.group_total = plan.groups.size();
            summarizing.findings = static_cast<int>(out.report.findings.size());
            summarizing.blocking_findings = count_blocking(out.report);
            summarizing.label = "campaign summary";
            report_progress(summarizing);

            const auto response = options_.runner->run(TurnRunner::Request{
                .prompt = build_summary_prompt(input, out.report),
                .max_tokens = 1024,
                .json_object = true,
            });
            bool summary_applied = false;
            if (!response.interrupted && response.error.empty()) {
                if (auto summary = parse_summary_response(response.text);
                    summary.has_value()) {
                    out.report.overall_explanation = std::move(*summary);
                    summary_applied = true;
                }
            }
            if (!summary_applied) {
                out.report.warnings.push_back(
                    "The campaign summary could not be generated; using the group summaries.");
            }
        }
        return finish(std::move(out));
    };

    if (options_.on_progress) {
        Progress planned{.phase = ProgressPhase::Planned};
        planned.group_total = plan.groups.size();
        planned.skipped_files = static_cast<int>(plan.skipped_paths.size());
        planned.skipped_paths = plan.skipped_paths;
        for (const auto& group : plan.groups) {
            planned.files += static_cast<int>(group.files.size());
            planned.changed_lines += group.changed_lines();
            planned.risk_passes += group.needs_plan ? 1 : 0;
        }
        report_progress(planned);
    }

    std::vector<GroupOutcome> outcomes(plan.groups.size());
    const auto run_group = [&](std::size_t i, TurnRunner& runner) {
        auto& outcome = outcomes[i];
        const auto& group = plan.groups[i];
        outcome.label = group_label(group);
        const auto base_progress = [&](ProgressPhase phase) {
            Progress progress{.phase = phase};
            progress.group_index = i + 1;
            progress.group_total = plan.groups.size();
            progress.label = outcome.label;
            progress.files = static_cast<int>(group.files.size());
            progress.changed_lines = group.changed_lines();
            return progress;
        };

        if (cancelled(options_.cancellation_requested)) {
            outcome.interrupted = true;
            return;
        }
        report_progress(base_progress(ProgressPhase::GroupStarted));

        std::vector<RiskItem> risks;
        if (group.needs_plan) {
            report_progress(base_progress(ProgressPhase::RiskPass));
            const auto plan_response = runner.run(TurnRunner::Request{
                .prompt = build_plan_prompt(input, group),
                .max_tokens = kPlanMaxOutputTokens,
            });
            if (plan_response.interrupted) {
                outcome.interrupted = true;
                return;
            }
            if (plan_response.error.empty()) {
                risks = parse_risk_output(plan_response.text);
                outcome.plan_passed = true;
            }
        }

        if (cancelled(options_.cancellation_requested)) {
            outcome.interrupted = true;
            return;
        }

        // Large units may use read-only tools. If that turn fails or answers
        // with something that never becomes JSON, the unit is retried once as
        // a plain JSON-mode turn: the patch alone is still a usable review, and
        // losing a whole file because one tool turn misbehaved is not.
        const auto run_review_turn = [&](bool with_tools) {
            return runner.run(TurnRunner::Request{
                .prompt = build_review_prompt(input, group, risks, with_tools),
                .max_tokens = kReviewMaxOutputTokens,
                .json_object = !with_tools,
                .allowed_tools = with_tools
                    ? std::vector<std::string>{
                          std::string(core::tools::names::kRead),
                          std::string(core::tools::names::kGrepSearch),
                          std::string(core::tools::names::kFileSearch),
                      }
                    : std::vector<std::string>{},
            });
        };

        const bool with_tools = group.needs_plan;
        auto review_response = run_review_turn(with_tools);
        if (review_response.interrupted) {
            outcome.interrupted = true;
            return;
        }

        std::optional<Report> parsed_json;
        if (review_response.error.empty()) {
            parsed_json = parse_review_json(review_response.text);
        }
        if (!parsed_json.has_value() && with_tools) {
            auto retry = run_review_turn(false);
            if (retry.interrupted) {
                outcome.interrupted = true;
                return;
            }
            if (retry.error.empty()) {
                parsed_json = parse_review_json(retry.text);
            }
            if (parsed_json.has_value() || !review_response.error.empty()) {
                review_response = std::move(retry);
            }
        }

        if (!parsed_json.has_value()) {
            outcome.failed = true;
            outcome.failure = review_response.error.empty()
                ? std::string("the model did not return a JSON review")
                : review_response.error;
            auto failure = base_progress(ProgressPhase::GroupFailed);
            failure.failure = outcome.failure;
            report_progress(failure);
            return;
        }

        auto parsed = std::move(*parsed_json);
        const auto dropped = localize_findings(
            parsed.findings, group.files, input.snapshot.worktree_root);
        if (dropped > 0) {
            parsed.warnings.push_back(std::format(
                "Dropped {} finding(s) whose location did not overlap the diff.",
                dropped));
        }
        resolve_finding_paths(parsed, input.snapshot.worktree_root);
        const auto rejected_steering_references = validate_steering_references(
            parsed,
            group,
            input.steering,
            input.snapshot.worktree_root);
        if (rejected_steering_references > 0) {
            parsed.warnings.push_back(std::format(
                "Dropped {} steering reference(s) without a matching applicable source quote.",
                rejected_steering_references));
        }
        parsed.groups_reviewed = 1;

        auto finished = base_progress(ProgressPhase::GroupFinished);
        finished.findings = static_cast<int>(parsed.findings.size());
        finished.blocking_findings = count_blocking(parsed);
        report_progress(finished);

        outcome.report = std::move(parsed);
        outcome.reviewed = true;
    };

    // Use provider-forked runners for independent groups. A custom runner (or
    // a provider that serializes requests) returns no fork and therefore keeps
    // the old deterministic one-at-a-time behavior.
    const auto requested_workers = std::min(
        options_.max_parallel_groups,
        plan.groups.size());
    std::vector<std::unique_ptr<TurnRunner>> parallel_runners;
    std::vector<TurnRunner*> runners;
    runners.reserve(requested_workers);
    parallel_runners.reserve(requested_workers > 0 ? requested_workers - 1 : 0);
    runners.push_back(options_.runner.get());
    while (runners.size() < requested_workers) {
        auto fork = options_.runner->fork_for_parallel_request();
        if (!fork) break;
        runners.push_back(fork.get());
        parallel_runners.push_back(std::move(fork));
    }

    std::atomic_size_t next_group{0};
    std::atomic_bool stop_dispatching{false};
    std::vector<std::jthread> workers;
    workers.reserve(runners.size());
    for (TurnRunner* runner : runners) {
        workers.emplace_back([&, runner](std::stop_token) {
            for (;;) {
                if (stop_dispatching.load(std::memory_order_acquire)
                    || cancelled(options_.cancellation_requested)) {
                    return;
                }
                const auto i = next_group.fetch_add(1, std::memory_order_relaxed);
                if (i >= plan.groups.size()) return;
                run_group(i, *runner);
                if (outcomes[i].interrupted) {
                    // A timed-out/cancelled runner may still have a late
                    // provider callback. Never reuse that runner for another
                    // group, and stop handing out new work to sibling lanes.
                    stop_dispatching.store(true, std::memory_order_release);
                    return;
                }
            }
        });
    }
    // jthread joins before outcomes are aggregated, so provider-forked agents
    // have released their turns before the campaign publishes its result.
    workers.clear();

    bool interrupted_group = false;
    for (const auto& outcome : outcomes) {
        if (outcome.plan_passed) ++plan_passes;
        interrupted_group = interrupted_group || outcome.interrupted;
        if (outcome.reviewed) {
            group_reports.push_back(outcome.report);
        } else if (outcome.failed) {
            failed_groups.push_back(FailedGroup{
                .label = outcome.label,
                .reason = outcome.failure,
            });
        }
    }

    if (interrupted_group || cancelled(options_.cancellation_requested)) {
        return seal(true);
    }

    return seal(false);
}

} // namespace core::review
