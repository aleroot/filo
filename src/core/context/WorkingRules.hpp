#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace core::context {

/// One rule of Filo's shared working doctrine. Every provider receives the
/// doctrine; a model profile (ModelGuidance) may restate individual rules
/// for a model that reads the shared wording differently.
enum class WorkingRule : std::uint8_t {
    BiasToAction,
    Assumptions,
    ReportedProblems,
    EvidenceBeforeClaims,
    ToolEfficiency,
    MinimalChanges,
    ReplyShape,
    RepositorySafety,
};

/// One shared rule, restated in other words.
struct WorkingRuleOverride {
    WorkingRule rule;
    std::string_view text;
};

/// The shared wording of @p rule: one bullet's text, without "- " or newline.
[[nodiscard]] std::string_view working_rule_text(WorkingRule rule) noexcept;

/// The doctrine block: "\n\n[Working rules]" and one "- " bullet per rule, in
/// declaration order. A rule named in @p overrides is rendered in the
/// override's words, in the shared rule's place, so a model never reads two
/// versions of the same rule.
[[nodiscard]] std::string render_working_rules(
    std::span<const WorkingRuleOverride> overrides = {});

} // namespace core::context
