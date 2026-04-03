#include "SessionReport.hpp"
#include <ftxui/screen/string.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <unistd.h>

namespace core::session {

namespace {

constexpr int kBoxWidth = 64; // inner width between │ characters

// ANSI codes — used only when color is supported.
constexpr std::string_view kReset         = "\033[0m";
constexpr std::string_view kBold          = "\033[1m";
constexpr std::string_view kDim           = "\033[2m";
constexpr std::string_view kCyan          = "\033[36m";
constexpr std::string_view kGreen         = "\033[32m";
constexpr std::string_view kYellow        = "\033[33m";
constexpr std::string_view kYellowBright  = "\033[93m";
constexpr std::string_view kGray          = "\033[90m";

// Box-drawing characters (UTF-8).
constexpr std::string_view kH   = "─";
constexpr std::string_view kV   = "│";
constexpr std::string_view kTL  = "╭";
constexpr std::string_view kTR  = "╮";
constexpr std::string_view kBL  = "╰";
constexpr std::string_view kBR  = "╯";
constexpr std::string_view kML  = "├";
constexpr std::string_view kMR  = "┤";
constexpr std::string_view kEllipsis = "...";

std::string rep(std::string_view s, int n) {
    std::string out;
    out.reserve(s.size() * static_cast<std::size_t>(std::max(n, 0)));
    for (int i = 0; i < n; ++i) out += s;
    return out;
}

int display_width(std::string_view s) {
    return ftxui::string_width(std::string{s});
}

std::size_t utf8_next_codepoint(std::string_view s, std::size_t pos) {
    if (pos >= s.size()) return s.size();
    ++pos;
    while (pos < s.size() &&
           (static_cast<unsigned char>(s[pos]) & 0xC0u) == 0x80u) {
        ++pos;
    }
    return pos;
}

std::size_t utf8_prev_codepoint(std::string_view s, std::size_t pos) {
    if (pos == 0) return 0;
    --pos;
    while (pos > 0 &&
           (static_cast<unsigned char>(s[pos]) & 0xC0u) == 0x80u) {
        --pos;
    }
    return pos;
}

std::string trim_right_to_width(std::string s, int max_width) {
    if (max_width <= 0) return {};
    while (!s.empty() && display_width(s) > max_width) {
        const auto start = utf8_prev_codepoint(s, s.size());
        s.erase(start);
    }
    return s;
}

std::string trim_left_to_width(std::string s, int max_width) {
    if (max_width <= 0) return {};
    while (!s.empty() && display_width(s) > max_width) {
        const auto next = utf8_next_codepoint(s, 0);
        s.erase(0, next);
    }
    return s;
}

std::string ellipsize_right(std::string s, int max_width) {
    if (max_width <= 0) return {};
    if (display_width(s) <= max_width) return s;
    const int ellipsis_width = display_width(kEllipsis);
    if (max_width <= ellipsis_width) {
        return trim_right_to_width(std::string{kEllipsis}, max_width);
    }
    s = trim_right_to_width(std::move(s), max_width - ellipsis_width);
    return s + std::string{kEllipsis};
}

std::string ellipsize_left(std::string s, int max_width) {
    if (max_width <= 0) return {};
    if (display_width(s) <= max_width) return s;
    const int ellipsis_width = display_width(kEllipsis);
    if (max_width <= ellipsis_width) {
        return trim_right_to_width(std::string{kEllipsis}, max_width);
    }
    s = trim_left_to_width(std::move(s), max_width - ellipsis_width);
    return std::string{kEllipsis} + s;
}

std::string pad_right(std::string s, int width) {
    if (width <= 0) return {};
    s = trim_right_to_width(std::move(s), width);
    const int gap = width - display_width(s);
    if (gap > 0) {
        s.append(static_cast<std::size_t>(gap), ' ');
    }
    return s;
}

} // namespace

bool SessionReport::supports_color() noexcept {
    if (const char* no_color = std::getenv("NO_COLOR"); no_color) return false;
    if (const char* term = std::getenv("TERM"); term && std::string_view{term} == "dumb") return false;
    return isatty(STDOUT_FILENO) != 0;
}

std::string SessionReport::fmt_tokens(int32_t n) {
    if (n >= 1'000'000) return std::format("{:.1f}M", n / 1'000'000.0);
    if (n >= 1'000)     return std::format("{:.1f}k", n / 1'000.0);
    return std::to_string(n);
}

std::string SessionReport::fmt_cost(double usd) {
    if (usd < 0.0001) return "$0";
    if (usd < 0.01)   return std::format("${:.4f}", usd);
    if (usd < 1.0)    return std::format("${:.3f}", usd);
    return std::format("${:.2f}", usd);
}

std::string SessionReport::fmt_duration(std::chrono::seconds s) {
    const int64_t total = s.count();
    if (total < 60)   return std::format("{}s", total);
    if (total < 3600) return std::format("{}m {}s", total / 60, total % 60);
    return std::format("{}h {}m", total / 3600, (total % 3600) / 60);
}

void SessionReport::print(const core::budget::BudgetTracker& budget,
                           const SessionStats::Snapshot& snap,
                           std::string_view session_id,
                           std::string_view session_file_path) {
    if (snap.turn_count == 0) return;

    const bool color = supports_color();
    auto C = [&](std::string_view code) -> std::string_view {
        return color ? code : std::string_view{};
    };

    const auto now   = std::chrono::system_clock::now();
    const auto wall  = std::chrono::duration_cast<std::chrono::seconds>(now - snap.started_at);
    const auto total = budget.session_total();
    const double cost = budget.session_cost_usd();

    // ── horizontal rule helper ──────────────────────────────────────────────
    auto hr = [&](std::string_view left, std::string_view right) {
        std::cout << C(kGray) << left << rep(kH, kBoxWidth) << right << C(kReset) << '\n';
    };

    // ── single-value row ────────────────────────────────────────────────────
    constexpr int kLabelWidth = 18;
    const int kValueWidth = kBoxWidth - kLabelWidth;
    auto row = [&](std::string_view label, std::string value,
                   std::string_view vcol = {}) {
        const std::string label_str = std::string("  ") + std::string(label);
        std::cout
            << C(kGray) << kV << C(kReset)
            << C(kDim)  << pad_right(label_str, kLabelWidth) << C(kReset)
            << C(vcol)  << pad_right(std::move(value), kValueWidth) << C(kReset)
            << C(kGray) << kV << C(kReset) << '\n';
    };

    // ── header ──────────────────────────────────────────────────────────────
    std::cout << '\n';
    {
        constexpr std::string_view kTitle = " Session Summary ";
        const int title_width = display_width(kTitle);
        const int pad = std::max(0, (kBoxWidth - title_width) / 2);
        const int right_pad = std::max(0, kBoxWidth - pad - title_width);
        std::cout
            << C(kGray) << kTL << rep(kH, pad) << C(kReset)
            << C(kBold) << C(kYellowBright) << kTitle << C(kReset)
            << C(kGray) << rep(kH, right_pad)
            << kTR << C(kReset) << '\n';
    }

    // ── Overview ────────────────────────────────────────────────────────────
    row("Duration", fmt_duration(wall), kReset);
    row("Turns", std::to_string(snap.turn_count), kReset);

    if (snap.api_calls_total > 0) {
        const int32_t api_failed = snap.api_calls_total - snap.api_calls_success;
        row("API calls",
            std::format("{} (\xe2\x9c\x93 {} \xe2\x9c\x97 {})",
                snap.api_calls_total, snap.api_calls_success, api_failed),
            (api_failed > 0) ? kYellow : kGreen);
    }

    if (snap.tool_calls_total > 0) {
        const int32_t failed = snap.tool_calls_total - snap.tool_calls_success;
        row("Tool calls",
            std::format("{} total  ({} \xe2\x9c\x93  {} \xe2\x9c\x97)",
                snap.tool_calls_total, snap.tool_calls_success, failed),
            (failed > 0) ? kYellow : kGreen);
    }

    // ── Per-model table ──────────────────────────────────────────────────────
    if (!snap.per_model.empty()) {
        hr(kML, kMR);

        constexpr int kMW = kLabelWidth; // model name column
        constexpr int kCW =  6; // calls
        constexpr int kIW =  9; // input tokens
        constexpr int kOW =  9; // output tokens
        const int kCostW = kBoxWidth - kMW - kCW - kIW - kOW;

        // Table header
        std::cout
            << C(kGray) << kV << C(kReset)
            << C(kBold) << C(kDim)
            << "  " << pad_right("Model",  kMW - 2)
            << pad_right("Calls", kCW)
            << pad_right("Input",  kIW)
            << pad_right("Output", kOW)
            << pad_right("Cost",   kCostW)
            << C(kReset)
            << C(kGray) << kV << C(kReset) << '\n';

        for (const auto& m : snap.per_model) {
            std::string mname = ellipsize_right(m.model, kMW - 2);
            const std::string cost_str = (m.cost_usd > 0.0) ? fmt_cost(m.cost_usd) : "-";

            std::cout
                << C(kGray) << kV << C(kReset)
                << "  "
                << pad_right(mname, kMW - 2)
                << C(kCyan)  << pad_right(std::to_string(m.call_count), kCW)   << C(kReset)
                << pad_right(fmt_tokens(m.prompt_tokens),     kIW)
                << pad_right(fmt_tokens(m.completion_tokens), kOW)
                << C(m.cost_usd > 0.0 ? kYellow : kDim)
                << pad_right(cost_str, kCostW) << C(kReset)
                << C(kGray) << kV << C(kReset) << '\n';
        }
    }

    // ── Totals ───────────────────────────────────────────────────────────────
    hr(kML, kMR);
    // ↑ (\xe2\x86\x91) = prompt tokens going UP to the LLM (input/sent)
    // ↓ (\xe2\x86\x93) = completion tokens coming DOWN from the LLM (output/received)
    row("Total tokens",
        std::format("\xe2\x86\x91{}  \xe2\x86\x93{}  total {}",
            fmt_tokens(total.prompt_tokens),
            fmt_tokens(total.completion_tokens),
            fmt_tokens(total.total_tokens)),
        kReset);
    if (cost > 0.0) {
        row("Total cost", fmt_cost(cost), kYellow);
    }

    // ── Session persistence ──────────────────────────────────────────────────
    if (!session_id.empty()) {
        hr(kML, kMR);
        row("Session ID", std::string(session_id), kReset);

        if (!session_file_path.empty()) {
            // Abbreviate home directory.
            std::string path_str{session_file_path};
            if (const char* home = std::getenv("HOME"); home && home[0] != '\0') {
                const std::string_view home_sv{home};
                if (std::string_view(path_str).starts_with(home_sv)) {
                    path_str = "~" + path_str.substr(home_sv.size());
                }
            }
            path_str = ellipsize_left(std::move(path_str), kValueWidth);
            row("Saved to", path_str, kReset);
            row("Resume with",
                std::format("filo --resume {}", session_id), kReset);
        }
    }

    // ── Footer ───────────────────────────────────────────────────────────────
    std::cout << C(kGray) << kBL << rep(kH, kBoxWidth) << kBR << C(kReset) << '\n';
    std::cout << '\n';
    std::cout.flush();
}

} // namespace core::session
