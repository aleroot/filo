#include "IssueQueue.hpp"

#include <algorithm>
#include <format>
#include <utility>

namespace core::goal {
namespace {

constexpr int kMinAttempts = 1;
constexpr int kMaxAttempts = 8;
constexpr std::size_t kMaxRetryChars = 4000;

[[nodiscard]] std::string one_line(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const char c : text) {
        out.push_back(c == '\n' || c == '\r' ? ' ' : c);
    }
    return out;
}

[[nodiscard]] std::string blocking_retry(const core::review::ClassifiedFindings& findings) {
    std::string out = "The review reported merge-blocking findings. Fix these and leave the changes uncommitted:\n";
    for (const auto& finding : findings.blocking) {
        out += "- [";
        out += core::review::to_string(core::review::effective_severity(finding));
        out += "] ";
        out += one_line(finding.title);
        if (!finding.absolute_file_path.empty()) {
            out += std::format(" ({}:{})", finding.absolute_file_path, finding.line_start);
        }
        out += "\n";
        if (!finding.body.empty()) {
            out += "  ";
            out += one_line(finding.body);
            out += "\n";
        }
        if (out.size() >= kMaxRetryChars) break;
    }
    if (out.size() > kMaxRetryChars) out.resize(kMaxRetryChars);
    return out;
}

void emit(const IssueHooks& hooks, std::string_view line) {
    if (hooks.on_event && !line.empty()) hooks.on_event(line);
}

[[nodiscard]] bool stopped(const IssueHooks& hooks) {
    return hooks.cancelled && hooks.cancelled();
}

struct WorktreeGuard {
    const IssueHooks* hooks = nullptr;
    const PreparedWorkspace* workspace = nullptr;
    bool keep = false;

    ~WorktreeGuard() {
        if (keep || hooks == nullptr || workspace == nullptr || !hooks->release) return;
        hooks->release(*workspace);
    }
};

} // namespace

RunResult run_issues(std::span<const IssueRecord> issues,
                     int max_attempts,
                     const IssueHooks& hooks) {
    RunResult result;
    max_attempts = std::clamp(max_attempts, kMinAttempts, kMaxAttempts);
    if (!hooks.prepare || !hooks.fix || !hooks.review || !hooks.has_changes
        || !hooks.approve_publish || !hooks.publish) {
        result.summary = "issue run is missing a required hook";
        return result;
    }

    for (const IssueRecord& issue : issues) {
        if (stopped(hooks)) {
            result.cancelled = true;
            emit(hooks, "Stopped.");
            break;
        }
        if (hooks.existing_pr) {
            const std::string url = hooks.existing_pr(issue);
            if (!url.empty()) {
                ++result.skipped;
                emit(hooks, std::format("▸ #{} already has {}", issue.number, url));
                continue;
            }
        }
        emit(hooks, std::format("▸ #{} {}", issue.number, one_line(issue.title)));

        auto prepared = hooks.prepare(issue);
        if (!prepared.has_value()) {
            ++result.blocked;
            emit(hooks, std::format("  blocked: {}", prepared.error()));
            continue;
        }
        WorktreeGuard guard{.hooks = &hooks, .workspace = &*prepared};
        std::string retry;
        int attempts = 0;
        bool finished = false;
        while (!finished) {
            if (stopped(hooks)) {
                result.cancelled = true;
                emit(hooks, "Stopped.");
                finished = true;
                break;
            }
            emit(hooks, std::format("  fix attempt {}/{}", attempts + 1, max_attempts));
            auto fixed = hooks.fix(issue, *prepared, retry);
            const auto fail = [&](std::string note, std::string next_retry) {
                ++attempts;
                if (attempts >= max_attempts) {
                    ++result.blocked;
                    emit(hooks, std::format("  blocked: {}", note));
                    finished = true;
                } else {
                    retry = std::move(next_retry);
                }
            };
            if (!fixed.has_value()) {
                fail(fixed.error(), "The previous fix attempt failed: " + fixed.error());
                continue;
            }
            if (!hooks.has_changes(*prepared)) {
                fail("the fix did not change the worktree",
                     "the fix did not change the worktree. Try a different fix.");
                continue;
            }
            auto reviewed = hooks.review(*prepared);
            if (!reviewed.has_value()) {
                fail(reviewed.error(), "Review did not finish: " + reviewed.error());
                continue;
            }
            if (reviewed->empty) {
                fail("the worktree has no diff to review", "the worktree has no diff to review");
                continue;
            }
            const auto& findings = reviewed->findings;
            emit(hooks, std::format(
                     "  review: {} blocking, {} follow-up{}",
                     findings.blocking.size(),
                     findings.follow_ups.size(),
                     findings.incomplete ? ", incomplete" : ""));
            if (!findings.blocking.empty()) {
                fail("blocking findings remain", blocking_retry(findings));
                continue;
            }
            if (findings.incomplete) {
                fail("the review did not cover every change",
                     "the review did not cover every change");
                continue;
            }
            if (!findings.follow_ups.empty() && hooks.file_follow_ups) {
                auto filed = hooks.file_follow_ups(issue, findings);
                if (!filed.has_value()) {
                    emit(hooks, std::format("  follow-ups were not filed: {}", filed.error()));
                } else if (!filed->empty()) {
                    emit(hooks, std::format("  filed {} follow-up issue(s)", filed->size()));
                }
            }
            if (!hooks.approve_publish(issue, *prepared)) {
                guard.keep = true;
                ++result.blocked;
                emit(hooks, std::format("  publish skipped; worktree kept at {}", prepared->path.string()));
                finished = true;
                continue;
            }
            auto pr = hooks.publish(issue, *prepared, fixed->empty() ? issue.title : *fixed);
            if (!pr.has_value()) {
                guard.keep = true;
                ++result.blocked;
                emit(hooks, std::format("  publish failed: {} ({})", pr.error(), prepared->path.string()));
                finished = true;
                continue;
            }
            ++result.opened;
            emit(hooks, std::format("  opened {}", *pr));
            finished = true;
        }
        if (result.cancelled) break;
    }

    result.summary = std::format(
        "{} opened, {} blocked, {} skipped", result.opened, result.blocked, result.skipped);
    if (result.cancelled) result.summary += ", stopped";
    emit(hooks, result.summary);
    return result;
}

} // namespace core::goal
