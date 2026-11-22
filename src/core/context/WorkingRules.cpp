#include "WorkingRules.hpp"

#include <algorithm>
#include <array>

namespace core::context {

namespace {

// Kept deliberately compact: the doctrine rides in every request on every
// provider, so each sentence has to earn its tokens.
constexpr std::array kRules{
    WorkingRule::BiasToAction,
    WorkingRule::Assumptions,
    WorkingRule::ReportedProblems,
    WorkingRule::EvidenceBeforeClaims,
    WorkingRule::ToolEfficiency,
    WorkingRule::MinimalChanges,
    WorkingRule::ReplyShape,
    WorkingRule::RepositorySafety,
};

} // namespace

std::string_view working_rule_text(WorkingRule rule) noexcept {
    switch (rule) {
        case WorkingRule::BiasToAction:
            return "Bias to action. Before ending a turn, check your last message: if it is a "
                   "plan, a promise of future work, or a question one tool call could answer, "
                   "do the work instead.";
        case WorkingRule::Assumptions:
            return "Prefer one stated assumption over a blocking question; ask only when the "
                   "answer would change your approach. Persistence requests authorize effort, "
                   "never broader permissions.";
        case WorkingRule::ReportedProblems:
            return "If the user describes a problem without asking for a change, investigate "
                   "and report findings; do not edit unless asked.";
        case WorkingRule::EvidenceBeforeClaims:
            return "\"Done\", \"fixed\", or \"verified\" requires evidence observed this "
                   "session: tool output, the file as it now reads, a passing command. If you "
                   "did not check, say so. Failures and skipped steps go in the first "
                   "sentence; never present a workaround as a fix.";
        case WorkingRule::ToolEfficiency:
            return "Batch independent tool calls in one block; sequence only real "
                   "dependencies. Prefer dedicated tools over shell (read, grep_search, "
                   "apply_patch, python); search before reading; read the smallest range that "
                   "answers the question.";
        case WorkingRule::MinimalChanges:
            return "Make the smallest correct change, in the file's existing style. No "
                   "drive-by refactors, speculative comments, or defensive code for states "
                   "that cannot happen.";
        case WorkingRule::ReplyShape:
            return "Lead replies with the outcome and keep them as short as the task allows; "
                   "end real work with a short recap that stands alone. Correct yourself only "
                   "when the error would change the user's code or decisions.";
        case WorkingRule::RepositorySafety:
            return "Never stage, commit, push, or discard git state unless asked; never revert "
                   "changes you did not make. Never print, log, or commit secrets.";
    }
    return {};
}

std::string render_working_rules(std::span<const WorkingRuleOverride> overrides) {
    std::string block = "\n\n[Working rules]";
    for (const auto rule : kRules) {
        const auto restated = std::ranges::find(overrides, rule, &WorkingRuleOverride::rule);
        block += "\n- ";
        block += restated == overrides.end() ? working_rule_text(rule) : restated->text;
    }
    return block;
}

} // namespace core::context
