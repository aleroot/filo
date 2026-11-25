#pragma once

#include "Types.hpp"

#include <vector>

namespace core::review {

// One rule for "this finding blocks the merge", shared by the review progress
// counter and the issue queue. Critical or high blocks when the category is a
// bug, security, or unset. Style, docs, and other nits stay follow-ups even
// when the model marks them high: the review rubric already calls those
// non-blocking, and a queue must not re-fix them.
[[nodiscard]] inline bool blocks_merge(const Finding& finding) noexcept {
    const Severity severity = effective_severity(finding);
    if (severity != Severity::Critical && severity != Severity::High) {
        return false;
    }
    if (!finding.category.has_value()) {
        return true;
    }
    return *finding.category == Category::Bug
        || *finding.category == Category::Security;
}

// Findings split for a caller that will either re-fix or file follow-ups.
// `incomplete` means at least one review unit failed, so an empty blocking
// list is not a clean bill.
struct ClassifiedFindings {
    std::vector<Finding> blocking;
    std::vector<Finding> follow_ups;
    bool incomplete = false;
};

[[nodiscard]] inline ClassifiedFindings classify_findings(const Report& report) {
    ClassifiedFindings out;
    out.incomplete = !report.failed_groups.empty();
    out.blocking.reserve(report.findings.size());
    out.follow_ups.reserve(report.findings.size());
    for (const Finding& finding : report.findings) {
        if (blocks_merge(finding)) {
            out.blocking.push_back(finding);
        } else {
            out.follow_ups.push_back(finding);
        }
    }
    return out;
}

} // namespace core::review
