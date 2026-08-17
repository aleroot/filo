#include "core/utils/TimeUtils.hpp"
#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/llm/protocols/KimiProtocol.hpp"
#include "core/llm/protocols/GrokBillingUsage.hpp"
#include "core/llm/protocols/ZaiProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "tui/UsageDetailsPanel.hpp"
#include <catch2/catch_test_macros.hpp>
#include <ftxui/dom/node.hpp>
#include <chrono>

TEST_CASE("TimeUtils: parse_timestamp_or_duration", "[time_utils]") {
    SECTION("Unix epoch seconds and milliseconds") {
        CHECK(core::utils::time::parse_timestamp_or_duration("1700000000") == 1700000000LL);
        CHECK(core::utils::time::parse_timestamp_or_duration("1700000000000") == 1700000000LL); // ms to sec
    }

    SECTION("Duration formats") {
        const auto now = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();

        const int64_t in_6s = core::utils::time::parse_timestamp_or_duration("6s");
        CHECK(in_6s >= now + 5);
        CHECK(in_6s <= now + 7);

        const int64_t in_1m30s = core::utils::time::parse_timestamp_or_duration("1m30s");
        CHECK(in_1m30s >= now + 88);
        CHECK(in_1m30s <= now + 92);

        const int64_t in_20ms = core::utils::time::parse_timestamp_or_duration("20ms");
        CHECK(in_20ms >= now);
        CHECK(in_20ms <= now + 1);
    }

    SECTION("ISO 8601 strings") {
        const int64_t t_utc = core::utils::time::parse_timestamp_or_duration("2026-08-17T12:00:00Z");
        CHECK(t_utc > 0);

        const int64_t t_ms = core::utils::time::parse_timestamp_or_duration("2026-08-17T12:00:00.123Z");
        CHECK(t_ms == t_utc);

        const int64_t t_offset = core::utils::time::parse_timestamp_or_duration("2026-08-17T20:00:00+08:00");
        CHECK(t_offset == t_utc);
    }

    SECTION("Invalid inputs return 0") {
        CHECK(core::utils::time::parse_timestamp_or_duration("") == 0);
        CHECK(core::utils::time::parse_timestamp_or_duration("not-a-date") == 0);
    }
}

TEST_CASE("UsageDetailsPanel: helpers", "[usage_details_panel]") {
    SECTION("human_window_title mapping") {
        CHECK(tui::human_window_title("5h") == "5-Hour Rolling Window");
        CHECK(tui::human_window_title("7d") == "7-Day Subscription Window");
        CHECK(tui::human_window_title("30d") == "30-Day Monthly Allowance");
        CHECK(tui::human_window_title("opus") == "Claude Opus Allocation (7d)");
        CHECK(tui::human_window_title("sonnet") == "Claude Sonnet Allocation (7d)");
        CHECK(tui::human_window_title("oauth-apps") == "OAuth Applications Allowance (7d)");
        CHECK(tui::human_window_title("overage") == "Extra Usage / Overage Allowance");
        CHECK(tui::human_window_title("custom") == "custom Window");
    }

    SECTION("format_quota_reset_display relative formats") {
        const int64_t now = 100000;
        CHECK(tui::format_quota_reset_display(0, now).empty());
        CHECK(tui::format_quota_reset_display(now - 5, now).find("resets now") != std::string::npos);
        CHECK(tui::format_quota_reset_display(now + 30, now).find("in 30s") != std::string::npos);
        CHECK(tui::format_quota_reset_display(now + 150, now).find("in 2m 30s") != std::string::npos);
        CHECK(tui::format_quota_reset_display(now + 7320, now).find("in 2h 2m") != std::string::npos);
        CHECK(tui::format_quota_reset_display(now + 100000, now).find("in 1d 3h") != std::string::npos);
    }

    SECTION("render_quota_progress_bar renders without throwing") {
        auto bar0 = tui::render_quota_progress_bar(0.0f, 16);
        CHECK(bar0 != nullptr);
        auto bar50 = tui::render_quota_progress_bar(0.5f, 16);
        CHECK(bar50 != nullptr);
        auto bar100 = tui::render_quota_progress_bar(1.0f, 16);
        CHECK(bar100 != nullptr);
    }

    SECTION("render_usage_details_panel popover renders successfully") {
        core::llm::protocols::RateLimitInfo info;
        info.requests_limit = 1000;
        info.requests_remaining = 950;
        info.tokens_limit = 50000;
        info.tokens_remaining = 42000;
        info.usage_windows.push_back(core::llm::protocols::UsageWindow{
            .label = "5h",
            .utilization = 0.35f,
            .resets_at = 1750000000LL,
        });
        info.usage_windows.push_back(core::llm::protocols::UsageWindow{
            .label = "7d",
            .utilization = 0.82f,
            .resets_at = 1750500000LL,
        });
        info.unified_representative_claim = "7d";

        core::llm::TokenUsage session_tokens{
            .prompt_tokens = 12000,
            .completion_tokens = 3500,
            .total_tokens = 15500,
            .cached_prompt_tokens = 4000,
            .reasoning_tokens = 1200,
        };

        auto element = tui::render_usage_details_panel(
            info,
            "Anthropic",
            "claude-3-7-sonnet",
            "medium",
            true,
            session_tokens,
            0.045,
            75);
        CHECK(element != nullptr);
    }
}

TEST_CASE("Protocol reset parsing: Anthropic, Kimi, Grok, Zai, OpenAI", "[protocol_resets]") {
    SECTION("Anthropic on_response parses resets_at from headers") {
        core::llm::protocols::AnthropicProtocol protocol;
        cpr::Header h;
        h["anthropic-ratelimit-unified-5h-utilization"] = "0.45";
        h["anthropic-ratelimit-unified-5h-resets-at"] = "2026-08-17T18:30:00Z";
        h["anthropic-ratelimit-unified-7d-utilization"] = "0.12";
        h["anthropic-ratelimit-unified-7d-resets-at"] = "1750000000";
        core::llm::protocols::HttpResponse res{
            .status_code = 200,
            .headers = h,
        };
        protocol.on_response(res);

        const auto info = protocol.last_rate_limit();
        REQUIRE(info.usage_windows.size() == 2);
        CHECK(info.usage_windows[0].label == "5h");
        CHECK(info.usage_windows[0].resets_at > 0);
        CHECK(info.usage_windows[1].label == "7d");
        CHECK(info.usage_windows[1].resets_at == 1750000000LL);
    }

    SECTION("Kimi on_response parses reset headers") {
        core::llm::protocols::KimiProtocol protocol;
        cpr::Header h;
        h["x-ratelimit-5h-utilization"] = "0.35";
        h["x-ratelimit-5h-resets-at"] = "2026-08-17T19:00:00Z";
        h["x-ratelimit-7d-utilization"] = "0.80";
        h["x-ratelimit-7d-reset"] = "1750500000";
        core::llm::protocols::HttpResponse res{
            .status_code = 200,
            .headers = h,
        };
        protocol.on_response(res);

        const auto info = protocol.last_rate_limit();
        REQUIRE(info.usage_windows.size() == 2);
        CHECK(info.usage_windows[0].resets_at > 0);
        CHECK(info.usage_windows[1].resets_at == 1750500000LL);
    }

    SECTION("Grok billing usage parses period endTime") {
        const std::string payload = R"({
            "config": {
                "creditUsagePercent": 42.0,
                "currentPeriod": {
                    "type": "WEEKLY",
                    "endTime": "2026-08-17T23:59:59Z"
                }
            }
        })";
        const auto windows = core::llm::protocols::parse_grok_billing_usage(payload);
        REQUIRE(!windows.empty());
        CHECK(windows[0].label == "7d");
        CHECK(windows[0].resets_at > 0);
    }

    SECTION("Zai on_response parses rate limit headers") {
        core::llm::protocols::ZaiProtocol protocol;
        cpr::Header h;
        h["x-ratelimit-unified-5h-utilization"] = "0.55";
        h["x-ratelimit-unified-5h-resets-at"] = "2026-08-17T19:00:00Z";
        core::llm::protocols::HttpResponse res{
            .status_code = 200,
            .headers = h,
        };
        protocol.on_response(res);

        const auto info = protocol.last_rate_limit();
        REQUIRE(!info.usage_windows.empty());
        CHECK(info.usage_windows[0].resets_at > 0);
    }

    SECTION("OpenAI on_response parses reset headers") {
        core::llm::protocols::OpenAIProtocol protocol;
        cpr::Header h;
        h["x-ratelimit-limit-requests"] = "5000";
        h["x-ratelimit-remaining-requests"] = "4900";
        h["x-ratelimit-reset-requests"] = "15s";
        h["x-ratelimit-limit-tokens"] = "100000";
        h["x-ratelimit-remaining-tokens"] = "85000";
        h["x-ratelimit-reset-tokens"] = "250ms";
        core::llm::protocols::HttpResponse res{
            .status_code = 200,
            .headers = h,
        };
        protocol.on_response(res);

        const auto info = protocol.last_rate_limit();
        CHECK(info.requests_limit == 5000);
        CHECK(info.requests_remaining == 4900);
        CHECK(info.requests_reset > 0);
        CHECK(info.tokens_limit == 100000);
        CHECK(info.tokens_remaining == 85000);
        CHECK(info.tokens_reset > 0);
    }
}
