#pragma once

#include "Types.hpp"

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

[[nodiscard]] std::string render_report(const Report& report);

} // namespace core::review
