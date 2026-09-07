#include <catch2/catch_test_macros.hpp>

#include "tui/ActivityTimer.hpp"

#include <chrono>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

using namespace tui;

namespace {

/// One "<number><unit>" pair of a rendered span, e.g. {'m', 13}.
struct SpanUnit {
    char    unit;
    int64_t value;
};

/// Seconds in one of the units the formatter can emit. Zero marks an unknown
/// suffix so a test can fail loudly instead of silently dividing by it.
[[nodiscard]] int64_t unit_seconds(char unit) {
    switch (unit) {
        case 'd': return 24 * 3600;
        case 'h': return 3600;
        case 'm': return 60;
        case 's': return 1;
        default:  return 0;
    }
}

/// Value at which a unit must carry into the next coarser one. Days are the
/// coarsest unit rendered, so they alone may grow without bound.
[[nodiscard]] int64_t unit_carry_limit(char unit) {
    switch (unit) {
        case 'h': return 24;
        case 'm': return 60;
        case 's': return 60;
        default:  return std::numeric_limits<int64_t>::max();
    }
}

/// Parses "2d 1h" or "6m 16s" back into units, so a test can assert
/// properties over a whole range instead of a handful of literal strings.
[[nodiscard]] std::vector<SpanUnit> parse_span_units(std::string_view span) {
    std::vector<SpanUnit> units;
    for (std::size_t i = 0; i < span.size();) {
        while (i < span.size() && span[i] == ' ') ++i;
        int64_t value = 0;
        bool    seen_digit = false;
        while (i < span.size() && span[i] >= '0' && span[i] <= '9') {
            value = value * 10 + (span[i++] - '0');
            seen_digit = true;
        }
        if (!seen_digit || i >= span.size()) break;
        units.push_back({span[i++], value});
    }
    return units;
}

/// Reports why `span` is not a well-formed rendering of `elapsed`, or an
/// empty string when it is. Returning the reason instead of asserting keeps a
/// sweep to one assertion rather than one per sample.
[[nodiscard]] std::string span_defect(std::string_view span,
                                      std::chrono::seconds elapsed,
                                      bool humanized) {
    const std::vector<SpanUnit> units = parse_span_units(span);
    if (units.empty()) return "no unit parsed";
    if (humanized && units.size() > 2) return "more than two units";

    int64_t reconstructed = 0;
    for (std::size_t i = 0; i < units.size(); ++i) {
        const SpanUnit& unit = units[i];
        if (unit_seconds(unit.unit) == 0) return "unknown unit suffix";
        // Every unit must respect its own carry point, the leading one above
        // all: "200m 15s" satisfies every other rule here and is the bug.
        if (unit.value >= unit_carry_limit(unit.unit)) return "unit overflow";
        if (i > 0 && unit_seconds(units[i - 1].unit) <= unit_seconds(unit.unit)) {
            return "units not ordered coarse to fine";
        }
        reconstructed += unit.value * unit_seconds(unit.unit);
    }
    // Ages read at a glance drop seconds once an hour is on the clock.
    if (humanized && units.size() == 2
        && (units[0].unit == 'h' || units[0].unit == 'd')
        && units[1].unit == 's') {
        return "seconds trailing an hour or day";
    }
    // The span must still describe the duration it was given: truncating
    // toward the coarser unit is fine, inventing or losing time is not.
    if (reconstructed > elapsed.count()) return "span exceeds the duration";
    if (elapsed.count() - reconstructed >= unit_seconds(units.back().unit)) {
        return "span drops more than its finest unit";
    }
    return {};
}

} // namespace


TEST_CASE("ActivityTimerRegistry tracks elapsed time per operation id",
          "[tui][timer]") {
    ActivityTimerRegistry timer;

    const auto start = ActivityTimerRegistry::Clock::now();
    timer.start_at("assistant-turn-1", start);

    const auto elapsed = timer.elapsed(
        "assistant-turn-1",
        start + std::chrono::seconds(76));

    REQUIRE(elapsed.has_value());
    REQUIRE(elapsed->count() == 76);

    timer.stop("assistant-turn-1");
    REQUIRE_FALSE(timer.elapsed("assistant-turn-1").has_value());
}

TEST_CASE("format_elapsed_compact renders concise human-friendly durations",
          "[tui][timer]") {
    REQUIRE(format_elapsed_compact(std::chrono::seconds(0)) == "0s");
    REQUIRE(format_elapsed_compact(std::chrono::seconds(59)) == "59s");
    REQUIRE(format_elapsed_compact(std::chrono::seconds(76)) == "1m 16s");
    REQUIRE(format_elapsed_compact(std::chrono::seconds(3661)) == "1h 01m 01s");
    REQUIRE(format_elapsed_compact(std::chrono::hours(49))
            == "2d 01h 00m 00s");
}

TEST_CASE("format_elapsed_compact keeps seconds only below the hour",
          "[tui][timer]") {
    // Below a minute the second is the only unit there is.
    REQUIRE(format_elapsed_compact(
                std::chrono::seconds(45), ElapsedFormat::humanized)
            == "45s");
    // Minutes still read better with their seconds, but a "00s" tail is noise.
    REQUIRE(format_elapsed_compact(
                std::chrono::minutes(1), ElapsedFormat::humanized)
            == "1m");
    REQUIRE(format_elapsed_compact(
                std::chrono::seconds(376), ElapsedFormat::humanized)
            == "6m 16s");
    REQUIRE(format_elapsed_compact(
                std::chrono::seconds(3599), ElapsedFormat::humanized)
            == "59m 59s");
    // From the first hour on, the tail becomes minutes and never seconds:
    // this is the case that used to keep counting in minutes forever.
    REQUIRE(format_elapsed_compact(
                std::chrono::hours(1), ElapsedFormat::humanized)
            == "1h");
    REQUIRE(format_elapsed_compact(
                std::chrono::seconds(3601), ElapsedFormat::humanized)
            == "1h");
    REQUIRE(format_elapsed_compact(
                std::chrono::minutes(253) + std::chrono::seconds(49),
                ElapsedFormat::humanized)
            == "4h 13m");
    REQUIRE(format_elapsed_compact(
                std::chrono::hours(48), ElapsedFormat::humanized)
            == "2d");
    REQUIRE(format_elapsed_compact(
                std::chrono::hours(49) + std::chrono::seconds(30),
                ElapsedFormat::humanized)
            == "2d 1h");
}

TEST_CASE("format_elapsed_compact is well-formed for every duration",
          "[tui][timer]") {
    // A sweep, because the formatter this replaced was correct on every case
    // anyone had checked by hand and still rendered "200m 15s" in the field.
    // Defects are collected and asserted once: a CHECK per sample would add
    // millions of assertions to the suite for no extra signal.
    struct Defect {
        int64_t     elapsed;
        std::string span;
        std::string reason;
        bool        humanized;
    };
    std::vector<Defect> defects;

    const auto examine = [&](int64_t total_seconds) {
        for (const bool humanized : {false, true}) {
            const auto elapsed = std::chrono::seconds(total_seconds);
            const std::string span = format_elapsed_compact(
                elapsed,
                humanized ? ElapsedFormat::humanized : ElapsedFormat::precise);
            if (std::string reason = span_defect(span, elapsed, humanized);
                !reason.empty()) {
                defects.push_back({total_seconds, span, std::move(reason),
                                   humanized});
            }
        }
    };

    // Second by second through the first three hours, which covers every
    // minute and hour carry, then a coarse stride out to 90 days.
    for (int64_t s = 0; s <= 3 * 3600; ++s) examine(s);
    for (int64_t s = 3 * 3600; s <= 90 * 24 * 3600; s += 37) examine(s);
    for (const int64_t boundary : {59LL, 60LL, 61LL, 3599LL, 3600LL, 3601LL,
                                   86399LL, 86400LL, 86401LL,
                                   30LL * 86400, 365LL * 86400,
                                   // std::chrono::days holds 32 bits here, so
                                   // a 64-bit seconds count must be decomposed
                                   // without ever being cast into it.
                                   std::chrono::seconds::max().count()}) {
        examine(boundary);
    }

    if (!defects.empty()) {
        const Defect& first = defects.front();
        UNSCOPED_INFO("defects=" << defects.size()
                      << " first: elapsed=" << first.elapsed
                      << "s format=" << (first.humanized ? "humanized" : "precise")
                      << " span=\"" << first.span << "\" reason=" << first.reason);
    }
    CHECK(defects.empty());
}
