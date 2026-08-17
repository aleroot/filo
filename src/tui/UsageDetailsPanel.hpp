#pragma once

#include "core/llm/Models.hpp"
#include "core/llm/protocols/ApiProtocol.hpp"
#include <ftxui/dom/elements.hpp>
#include <cstdint>
#include <string>
#include <string_view>

namespace tui {

/**
 * @brief Formats a unix timestamp into human-readable reset countdown and local time.
 * @param resets_at Unix epoch timestamp in seconds.
 * @param now_seconds Current unix epoch timestamp (0 = use system_clock::now()).
 * @return Formatted string like "18:30:00 (in 2h 15m)" or empty if resets_at == 0.
 */
[[nodiscard]] std::string format_quota_reset_display(int64_t resets_at, int64_t now_seconds = 0);

/**
 * @brief Maps a short window identifier (e.g. "5h", "7d") to a descriptive title.
 */
[[nodiscard]] std::string human_window_title(std::string_view label);

/**
 * @brief Renders a visual quota progress bar with color coding.
 */
[[nodiscard]] ftxui::Element render_quota_progress_bar(float utilization, int width = 20);

/**
 * @brief Renders the complete usage details popover panel for active provider quotas and rate limits.
 */
[[nodiscard]] ftxui::Element render_usage_details_panel(
    const core::llm::protocols::RateLimitInfo& rate_limit_info,
    std::string_view active_provider_name,
    std::string_view active_model_name,
    std::string_view session_effort_value,
    bool is_subscription,
    const core::llm::TokenUsage& session_tokens,
    double session_cost_usd,
    int context_pct);

} // namespace tui
