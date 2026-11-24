#pragma once

#include "Types.hpp"

#include "../llm/DisplaySection.hpp"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace core::review {

/// Parses a review answer that is expected to be JSON. Returns nullopt when the
/// model answered with something else, which is what lets the engine retry the
/// unit instead of reporting free-form prose as a review.
[[nodiscard]] std::optional<Report> parse_review_json(std::string_view text);

/// parse_review_json with a prose fallback (the raw text becomes the overall
/// explanation). Used where no retry is possible.
[[nodiscard]] Report parse_review_output(std::string_view text);
[[nodiscard]] std::vector<RiskItem> parse_risk_output(std::string_view text);

[[nodiscard]] Report aggregate_reports(std::span<const Report> reports,
                                       std::span<const std::string> skipped_paths = {},
                                       std::span<const std::string> warnings = {});

/// Evidence is a claim that code outside the patch was read. A turn that ran
/// no tool cannot have read any, so its findings lose it whatever they say.
void drop_unchecked_evidence(Report& report, int tools_run);

/// Short stable handle for a finding ("f-3c91a7e2"), derived from where it is
/// and what it says, so a later prompt can point at it ("fix f-3c91a7e2").
[[nodiscard]] std::string finding_id(const Finding& finding);

/// The complete report as text: every finding with its reasons, low-severity
/// notes included. This is what history keeps, so the model can still act on
/// any of it in a later turn.
[[nodiscard]] std::string render_report(const Report& report);

/// The same report laid out for a transcript: a head section, one section per
/// comment with its reasons behind a collapsed "Why this was flagged", and a
/// tail holding review notes, @p trailer and the low-severity notes collapsed.
[[nodiscard]] std::vector<core::llm::DisplaySection> render_report_sections(
    const Report& report,
    std::string_view trailer = {});

} // namespace core::review
