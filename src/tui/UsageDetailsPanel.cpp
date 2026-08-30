#include "UsageDetailsPanel.hpp"

#include "TuiTheme.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <format>
#include <utility>
#include <vector>

namespace tui {

namespace {

using namespace ftxui;

[[nodiscard]] std::string format_compact_number(int64_t count) {
    if (count < 1'000) return std::to_string(count);
    if (count < 1'000'000) {
        return std::format("{:.1f}k", static_cast<double>(count) / 1'000.0);
    }
    return std::format("{:.2f}M", static_cast<double>(count) / 1'000'000.0);
}

[[nodiscard]] ftxui::Color utilization_color(float utilization) {
    if (utilization >= 0.90f) return ftxui::Color::Red;
    if (utilization >= 0.75f) return ftxui::Color(ColorWarn);
    if (utilization >= 0.50f) return ftxui::Color::Yellow;
    return ftxui::Color::Green;
}

} // namespace

std::string format_quota_reset_display(int64_t resets_at, int64_t now_seconds) {
    if (resets_at <= 0) return {};
    if (now_seconds == 0) {
        now_seconds = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    const int64_t diff = resets_at - now_seconds;
    std::string rel_str;
    if (diff <= 0) {
        rel_str = "resets now";
    } else if (diff < 60) {
        rel_str = std::format("in {}s", diff);
    } else if (diff < 3600) {
        const auto m = diff / 60;
        const auto s = diff % 60;
        rel_str = s > 0 ? std::format("in {}m {}s", m, s) : std::format("in {}m", m);
    } else if (diff < 86400) {
        const auto h = diff / 3600;
        const auto m = (diff % 3600) / 60;
        rel_str = m > 0 ? std::format("in {}h {}m", h, m) : std::format("in {}h", h);
    } else {
        const auto d = diff / 86400;
        const auto h = (diff % 86400) / 3600;
        rel_str = h > 0 ? std::format("in {}d {}h", d, h) : std::format("in {}d", d);
    }

    const auto tt = static_cast<std::time_t>(resets_at);
    std::tm tm{};
#if defined(_WIN32)
    localtime_s(&tm, &tt);
#else
    localtime_r(&tt, &tm);
#endif

    std::string abs_str;
    if (diff < 86400 && diff >= -3600) {
        abs_str = std::format("{:02d}:{:02d}:{:02d}", tm.tm_hour, tm.tm_min, tm.tm_sec);
    } else {
        abs_str = std::format("{:02d}-{:02d} {:02d}:{:02d}",
                              tm.tm_mon + 1, tm.tm_mday, tm.tm_hour, tm.tm_min);
    }

    return std::format("{} ({})", abs_str, rel_str);
}

std::string human_window_title(std::string_view label) {
    if (label == "5h") return "5-Hour Rolling Window";
    if (label == "7d") return "7-Day Subscription Window";
    if (label == "30d") return "30-Day Monthly Allowance";
    if (label == "4h") return "4-Hour Rolling Window";
    if (label == "oauth-apps" || label == "seven_day_oauth_apps") return "OAuth Applications Allowance (7d)";
    if (label == "opus" || label == "seven_day_opus") return "Claude Opus Allocation (7d)";
    if (label == "sonnet" || label == "seven_day_sonnet") return "Claude Sonnet Allocation (7d)";
    if (label == "overage") return "Extra Usage / Overage Allowance";
    if (label == "web") return "Web Search / Realtime Quota";
    return std::format("{} Window", label);
}

Element render_quota_progress_bar(float utilization, int width) {
    const float clamped = std::clamp(utilization, 0.0f, 1.5f);
    const int filled = std::clamp(static_cast<int>(clamped * static_cast<float>(width)), 0, width);
    const int empty = std::max(0, width - filled);

    std::string bar_filled;
    for (int i = 0; i < filled; ++i) bar_filled += "█";
    std::string bar_empty;
    for (int i = 0; i < empty; ++i) bar_empty += "░";

    return hbox({
        text("[") | ftxui::color(ftxui::Color::GrayDark),
        text(std::move(bar_filled)) | ftxui::color(utilization_color(utilization)),
        text(std::move(bar_empty)) | ftxui::color(ftxui::Color::GrayDark),
        text("]") | ftxui::color(ftxui::Color::GrayDark),
    });
}

Element render_usage_details_panel(
    const core::llm::protocols::RateLimitInfo& rate_limit_info,
    std::string_view active_provider_name,
    std::string_view active_model_name,
    std::string_view session_effort_value,
    bool is_subscription,
    const core::llm::TokenUsage& session_tokens,
    double session_cost_usd,
    int context_pct) {
    Elements rows;

    // ── Header & Active Model Badge ──────────────────────────────────────────
    std::string tier_label = is_subscription ? "Subscription (OAuth / Plan)" : "API Key / Metered";
    if (!rate_limit_info.subscription.tier_label.empty()) {
        tier_label = rate_limit_info.subscription.tier_label;
    }
    if (!rate_limit_info.unified_status.empty()) {
        tier_label += " · " + rate_limit_info.unified_status;
    }

    std::string model_str = active_model_name.empty() ? "<default>" : std::string(active_model_name);
    if (!session_effort_value.empty() && session_effort_value != "auto") {
        model_str += std::format(" ({})", session_effort_value);
    }

    rows.push_back(hbox({
        text("  Provider: ") | ftxui::color(ftxui::Color::GrayLight),
        text(std::string(active_provider_name)) | ftxui::bold | ftxui::color(ColorYellowBright),
        text("   Model: ") | ftxui::color(ftxui::Color::GrayLight),
        text(std::move(model_str)) | ftxui::bold | ftxui::color(ftxui::Color::White),
        text("   Tier: ") | ftxui::color(ftxui::Color::GrayLight),
        text(std::move(tier_label)) | ftxui::color(ftxui::Color::GrayLight) | ftxui::dim,
    }));
    rows.push_back(text(""));

    // ── Subscription Usage Windows ───────────────────────────────────────────
    if (!rate_limit_info.usage_windows.empty()) {
        rows.push_back(text("  Subscription Utilization & Limits:") | ftxui::bold | ftxui::color(ColorYellowDark));

        for (const auto& w : rate_limit_info.usage_windows) {
            const float pct = w.utilization * 100.0f;
            const float remaining_pct = std::max(0.0f, 100.0f - pct);
            const ftxui::Color col = utilization_color(w.utilization);
            const std::string title = human_window_title(w.label);

            const bool is_governing = (!rate_limit_info.unified_representative_claim.empty()
                && (rate_limit_info.unified_representative_claim == w.label
                    || (w.label == "5h" && rate_limit_info.unified_representative_claim == "five_hour")
                    || (w.label == "7d" && rate_limit_info.unified_representative_claim == "seven_day")));

            std::string pct_label = std::format("{:.0f}% used", pct);
            if (pct <= 100.0f) {
                pct_label += std::format(" ({:.0f}% left)", remaining_pct);
            } else {
                pct_label += " (quota exceeded)";
            }

            Elements window_items;
            window_items.push_back(text("    • ") | ftxui::color(ftxui::Color::GrayDark));
            window_items.push_back(text(title + ": ") | ftxui::bold | ftxui::color(ftxui::Color::White));
            window_items.push_back(render_quota_progress_bar(w.utilization, 16));
            window_items.push_back(text(" " + pct_label + " ") | ftxui::bold | ftxui::color(col));

            if (is_governing) {
                window_items.push_back(text(" ★ Governing ") | ftxui::bgcolor(ColorYellowDark) | ftxui::color(ftxui::Color::Black));
            }

            rows.push_back(hbox(std::move(window_items)));

            // Reset time display
            if (w.resets_at > 0) {
                const std::string reset_str = format_quota_reset_display(w.resets_at);
                if (!reset_str.empty()) {
                    rows.push_back(hbox({
                        text("      ⏳ Resets at: ") | ftxui::color(ftxui::Color::GrayLight) | ftxui::dim,
                        text(reset_str) | ftxui::color(ColorYellowBright),
                    }));
                }
            } else {
                rows.push_back(hbox({
                    text("      ⏳ Rolling window: ") | ftxui::color(ftxui::Color::GrayLight) | ftxui::dim,
                    text("updates continuously with sliding usage") | ftxui::color(ftxui::Color::GrayLight) | ftxui::dim,
                }));
            }
        }
        rows.push_back(text(""));
    }

    // ── Subscription End Date (only when the provider API reports it) ───────
    if (rate_limit_info.subscription_ends_at > 0) {
        const std::string sub_end_str =
            format_quota_reset_display(rate_limit_info.subscription_ends_at);
        if (!sub_end_str.empty()) {
            rows.push_back(hbox({
                text("  🗓 Subscription period ends: ")
                    | ftxui::color(ftxui::Color::GrayLight),
                text(sub_end_str) | ftxui::bold | ftxui::color(ColorYellowBright),
            }));
            rows.push_back(text(""));
        }
    }

    if (rate_limit_info.subscription.supplemental_credits.has_value()) {
        const auto& credits = *rate_limit_info.subscription.supplemental_credits;
        std::string credits_text = "  Extra Credits: ";
        if (credits.unlimited) {
            credits_text += "unlimited";
        } else if (!credits.balance.empty()) {
            credits_text += credits.balance;
        } else {
            credits_text += credits.available ? "available" : "none available";
        }
        rows.push_back(text(std::move(credits_text))
                       | ftxui::color(credits.available
                           || credits.unlimited
                               ? ftxui::Color::Green
                               : ftxui::Color(ColorWarn)));
    }
    if (!rate_limit_info.subscription.notice.empty()) {
        rows.push_back(text("  " + rate_limit_info.subscription.notice)
                       | ftxui::color(ColorYellowBright));
    }
    if (rate_limit_info.subscription.supplemental_credits.has_value()
        || !rate_limit_info.subscription.notice.empty()) {
        rows.push_back(text(""));
    }

    // ── Request & Token Limits (RPM / TPM) ───────────────────────────────────
    if (rate_limit_info.requests_limit > 0 || rate_limit_info.tokens_limit > 0
        || rate_limit_info.requests_remaining > 0 || rate_limit_info.tokens_remaining > 0
        || rate_limit_info.is_rate_limited) {
        rows.push_back(text("  API Rate Limits & Thresholds:") | ftxui::bold | ftxui::color(ColorYellowDark));

        if (rate_limit_info.requests_limit > 0 || rate_limit_info.requests_remaining > 0) {
            std::string req_text = std::format("    • Requests: {} / {} remaining",
                                               format_compact_number(rate_limit_info.requests_remaining),
                                               format_compact_number(rate_limit_info.requests_limit));
            if (rate_limit_info.requests_reset > 0) {
                req_text += "   ⏳ Resets: " + format_quota_reset_display(rate_limit_info.requests_reset);
            }
            rows.push_back(text(std::move(req_text)) | ftxui::color(ftxui::Color::White));
        }

        if (rate_limit_info.tokens_limit > 0 || rate_limit_info.tokens_remaining > 0) {
            std::string tok_text = std::format("    • Tokens:   {} / {} remaining",
                                               format_compact_number(rate_limit_info.tokens_remaining),
                                               format_compact_number(rate_limit_info.tokens_limit));
            if (rate_limit_info.tokens_reset > 0) {
                tok_text += "   ⏳ Resets: " + format_quota_reset_display(rate_limit_info.tokens_reset);
            }
            rows.push_back(text(std::move(tok_text)) | ftxui::color(ftxui::Color::White));
        }

        if (rate_limit_info.is_rate_limited) {
            std::string block_msg = "    ⛔ Provider Rate Limited (429)";
            if (!rate_limit_info.subscription.limit_reached_reason.empty()) {
                block_msg += " · " + rate_limit_info.subscription.limit_reached_reason;
            }
            if (rate_limit_info.retry_after > 0) {
                block_msg += std::format(" · Retry allowed in {}s", rate_limit_info.retry_after);
            }
            rows.push_back(text(std::move(block_msg)) | ftxui::bold | ftxui::color(ftxui::Color::Red));
        }

        rows.push_back(text(""));
    }

    // ── Session Metrics & Context Window ─────────────────────────────────────
    rows.push_back(text("  Session Consumption:") | ftxui::bold | ftxui::color(ColorYellowDark));
    {
        Elements session_items;
        session_items.push_back(text("    • Total Tokens: ") | ftxui::color(ftxui::Color::GrayLight));
        session_items.push_back(text(format_compact_number(session_tokens.total_tokens)) | ftxui::bold | ftxui::color(ftxui::Color::White));
        session_items.push_back(text(std::format(" ({} prompt, {} completion)",
                                                 format_compact_number(session_tokens.prompt_tokens),
                                                 format_compact_number(session_tokens.completion_tokens)))
                                | ftxui::color(ftxui::Color::GrayLight) | ftxui::dim);

        if (session_tokens.cached_prompt_tokens > 0) {
            session_items.push_back(text(std::format(" · {} cached", format_compact_number(session_tokens.cached_prompt_tokens)))
                                    | ftxui::color(ftxui::Color::Green));
        }
        if (session_tokens.reasoning_tokens > 0) {
            session_items.push_back(text(std::format(" · {} reasoning", format_compact_number(session_tokens.reasoning_tokens)))
                                    | ftxui::color(ColorYellowDark));
        }
        rows.push_back(hbox(std::move(session_items)));
    }

    if (!is_subscription && session_cost_usd > 0.0) {
        rows.push_back(hbox({
            text("    • Estimated Cost: ") | ftxui::color(ftxui::Color::GrayLight),
            text(std::format("${:.4f} USD", session_cost_usd)) | ftxui::bold | ftxui::color(ftxui::Color::Green),
        }));
    }

    if (context_pct >= 0) {
        const ftxui::Color ctx_col = context_pct < 25
            ? ftxui::Color(ftxui::Color::Red)
            : (context_pct < 50 ? ftxui::Color(ColorWarn) : ftxui::Color(ftxui::Color::Green));
        rows.push_back(hbox({
            text("    • Context Window: ") | ftxui::color(ftxui::Color::GrayLight),
            text(std::format("{}% remaining", context_pct)) | ftxui::bold | ftxui::color(ctx_col),
        }));
    }

    // ── Overage Status if available ──────────────────────────────────────────
    if (!rate_limit_info.unified_overage_status.empty() && rate_limit_info.unified_overage_status != "allowed") {
        std::string overage_msg = std::format("  Overage State: {}", rate_limit_info.unified_overage_status);
        if (rate_limit_info.unified_overage_reset > 0) {
            overage_msg += " · Resets: " + format_quota_reset_display(rate_limit_info.unified_overage_reset);
        }
        rows.push_back(text(""));
        rows.push_back(text(std::move(overage_msg)) | ftxui::color(ftxui::Color(ColorWarn)));
    }

    // ── Footer Dismiss Hint ──────────────────────────────────────────────────
    rows.push_back(text(""));
    rows.push_back(
        text("  Esc, q, or click status bar to close")
        | ftxui::color(ftxui::Color::GrayDark)
        | ftxui::dim);

    return UiWindow(
        text(" 📊 Provider Usage & Rate Limits ") | ftxui::color(ColorYellowBright) | ftxui::bold,
        vbox(std::move(rows)) | xflex);
}

} // namespace tui
