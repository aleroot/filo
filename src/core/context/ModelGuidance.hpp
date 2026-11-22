#pragma once

#include "WorkingRules.hpp"

#include <span>
#include <string_view>

namespace core::context {

/// What one model family needs on top of Filo's shared doctrine.
///
/// The doctrine is written once for every provider, but models do not all
/// read it the same way: a literal reader can turn a guard rail into a
/// refusal. A profile restates those rules in place and adds the
/// instructions the family's own vendor client gives it. ContextBuilder
/// renders a profile; the Agent picks it for the model a turn targets.
/// Supporting a new family means adding one profile to the registry in
/// ModelGuidance.cpp; nothing else changes.
struct ModelGuidance {
    /// Stable identifier, for diagnostics and tests.
    std::string_view name;
    /// Whether the profile applies to @p model_id.
    bool (*matches)(std::string_view model_id) noexcept;
    std::span<const WorkingRuleOverride> rule_overrides;
    /// Extra instructions rendered right after the working rules; may be empty.
    std::string_view instructions;
};

/// Every registered profile, in match order.
[[nodiscard]] std::span<const ModelGuidance> model_guidance_profiles() noexcept;

/// The first registered profile that matches @p model_id, or nullptr when
/// the shared doctrine stands as written.
[[nodiscard]] const ModelGuidance* find_model_guidance(std::string_view model_id) noexcept;

} // namespace core::context
