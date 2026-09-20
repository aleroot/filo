#pragma once

#include "Types.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace core::review {

[[nodiscard]] std::string build_summary_prompt(
    const CampaignInput& input,
    const Report& campaign_report);

/// True when a dedicated synthesizer turn is worth the extra latency:
/// several groups, incomplete coverage, or verified policy citations.
/// A one-file review already has a per-unit overall_explanation.
[[nodiscard]] bool campaign_needs_summary(const Report& campaign_report) noexcept;

[[nodiscard]] std::optional<std::string> parse_summary_response(
    std::string_view response);

} // namespace core::review
