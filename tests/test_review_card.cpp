#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/ReviewCard.hpp"

#include <ftxui/dom/node.hpp>
#include <ftxui/screen/screen.hpp>

#include <string>

using Catch::Matchers::ContainsSubstring;
using tui::ReviewGroupRow;
using tui::ReviewProgressView;

namespace {

[[nodiscard]] std::string render(const ReviewProgressView& view,
                                 std::size_t tick = 0,
                                 bool show_spinner = false) {
    auto element = tui::render_review_card(view, tick, show_spinner);
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(100),
                                        ftxui::Dimension::Fit(element));
    ftxui::Render(screen, element);
    return screen.ToString();
}

[[nodiscard]] ReviewProgressView running_view() {
    ReviewProgressView view;
    view.hint = "current changes";
    view.planned = true;
    view.total_groups = 3;
    view.files = 5;
    view.changed_lines = 212;
    view.rows.push_back(ReviewGroupRow{
        .label = "src/core/review/Engine.cpp",
        .state = ReviewGroupRow::State::Done,
        .findings = 2,
        .blocking = 1,
        .elapsed = "3.2s",
    });
    view.rows.push_back(ReviewGroupRow{
        .label = "src/tui/MainApp.cpp",
        .state = ReviewGroupRow::State::Running,
    });
    return view;
}

} // namespace

TEST_CASE("review card shows the target, progress and per-file outcomes",
          "[tui][review_card]") {
    const auto output = render(running_view());

    CHECK_THAT(output, ContainsSubstring("review"));
    CHECK_THAT(output, ContainsSubstring("current changes"));
    CHECK_THAT(output, ContainsSubstring("src/core/review/Engine.cpp"));
    CHECK_THAT(output, ContainsSubstring("src/tui/MainApp.cpp"));
    CHECK_THAT(output, ContainsSubstring("2 finding(s)"));
    CHECK_THAT(output, ContainsSubstring("1 blocking"));
    CHECK_THAT(output, ContainsSubstring("3.2s"));
    // One of three units has finished.
    CHECK_THAT(output, ContainsSubstring("1/3"));
}

TEST_CASE("review card stays static while the footer owns activity animation",
          "[tui][review_card]") {
    const auto view = running_view();
    CHECK(render(view, 0, true) == render(view, 7, false));
}

TEST_CASE("review card marks clean units and risk passes", "[tui][review_card]") {
    ReviewProgressView view;
    view.total_groups = 2;
    view.rows.push_back(ReviewGroupRow{
        .label = "src/a.cpp",
        .state = ReviewGroupRow::State::Done,
    });
    view.rows.push_back(ReviewGroupRow{
        .label = "src/b.cpp",
        .state = ReviewGroupRow::State::RiskPass,
    });

    const auto output = render(view);
    CHECK_THAT(output, ContainsSubstring("clean"));
    CHECK_THAT(output, ContainsSubstring("risk pass"));
}

TEST_CASE("review card summarises a finished review", "[tui][review_card]") {
    auto view = running_view();
    view.rows.back().state = ReviewGroupRow::State::Done;
    view.rows.back().findings = 0;
    view.rows.back().elapsed = "1.1s";
    view.finished = true;
    view.skipped_files = 1;
    view.elapsed = "12.5s";

    const auto output = render(view);
    CHECK_THAT(output, ContainsSubstring("5 file(s)"));
    CHECK_THAT(output, ContainsSubstring("212 changed line(s)"));
    CHECK_THAT(output, ContainsSubstring("2 comment(s)"));
    CHECK_THAT(output, ContainsSubstring("1 skipped"));
    CHECK_THAT(output, ContainsSubstring("12.5s"));
}

TEST_CASE("review card reports interruption and failure", "[tui][review_card]") {
    SECTION("interrupted") {
        auto view = running_view();
        view.finished = true;
        view.interrupted = true;
        CHECK_THAT(render(view), ContainsSubstring("Interrupted"));
    }

    SECTION("failed") {
        auto view = running_view();
        view.finished = true;
        view.failure = "provider returned an error";
        CHECK_THAT(render(view), ContainsSubstring("provider returned an error"));
    }
}

TEST_CASE("review card lists skipped files as declined rows", "[tui][review_card]") {
    ReviewProgressView view;
    view.finished = true;
    view.files = 2;
    view.skipped_files = 1;
    view.total_groups = 1;
    view.rows.push_back(ReviewGroupRow{
        .label = "src/ok.cpp",
        .state = ReviewGroupRow::State::Done,
    });
    view.rows.push_back(ReviewGroupRow{
        .label = "src/huge.cpp",
        .state = ReviewGroupRow::State::Skipped,
        .note = "diff too large",
    });

    const auto output = render(view);
    CHECK_THAT(output, ContainsSubstring("src/huge.cpp"));
    CHECK_THAT(output, ContainsSubstring("skipped · diff too large"));
    CHECK_THAT(output, ContainsSubstring("1 skipped"));
}

TEST_CASE("review card marks a failed unit without stopping the review",
          "[tui][review_card]") {
    ReviewProgressView view;
    view.finished = true;
    view.files = 2;
    view.total_groups = 2;
    view.rows.push_back(ReviewGroupRow{
        .label = "src/ok.cpp",
        .state = ReviewGroupRow::State::Done,
        .findings = 1,
    });
    view.rows.push_back(ReviewGroupRow{
        .label = "src/bad.cpp",
        .state = ReviewGroupRow::State::Failed,
        .note = "provider returned an error",
    });

    const auto output = render(view);
    CHECK_THAT(output, ContainsSubstring("src/bad.cpp"));
    CHECK_THAT(output, ContainsSubstring("failed · provider returned an error"));
    CHECK_THAT(output, ContainsSubstring("1 failed"));
    CHECK(view.failed_rows() == 1);
    CHECK(view.completed_rows() == 1);
    CHECK(view.processed_rows() == 2);
    CHECK_THAT(output, ContainsSubstring("2/2"));
}

TEST_CASE("review details popover carries what the pill no longer shows",
          "[tui][review_card]") {
    ReviewProgressView view;
    view.hint = "staged changes";
    view.total_groups = 3;
    view.files = 3;
    view.changed_lines = 120;
    view.risk_passes = 1;
    view.elapsed = "12.0s";
    view.rows.push_back(ReviewGroupRow{
        .label = "src/ok.cpp",
        .state = ReviewGroupRow::State::Done,
        .findings = 2,
        .blocking = 1,
    });
    view.rows.push_back(ReviewGroupRow{
        .label = "src/live.cpp",
        .state = ReviewGroupRow::State::RiskPass,
    });

    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(90),
                                        ftxui::Dimension::Fixed(20));
    auto element = tui::render_review_details_panel(view, 0, false, 8);
    ftxui::Render(screen, element);
    const auto output = screen.ToString();

    CHECK_THAT(output, ContainsSubstring("staged changes"));
    CHECK_THAT(output, ContainsSubstring("1/3 unit(s) reviewed"));
    CHECK_THAT(output, ContainsSubstring("120 changed line(s)"));
    CHECK_THAT(output, ContainsSubstring("src/live.cpp"));
    CHECK_THAT(output, ContainsSubstring("risk pass"));
    CHECK_THAT(output, ContainsSubstring("12.0s"));
}

TEST_CASE("review summary helpers describe outcomes", "[tui][review_card]") {
    CHECK(tui::review_row_outcome(ReviewGroupRow{
              .state = ReviewGroupRow::State::Done,
          }) == "clean");
    CHECK(tui::review_row_outcome(ReviewGroupRow{
              .state = ReviewGroupRow::State::Done,
              .findings = 3,
          }) == "3 finding(s)");
    CHECK(tui::review_row_outcome(ReviewGroupRow{
              .state = ReviewGroupRow::State::Skipped,
              .note = "diff too large",
          }) == "skipped · diff too large");
    // A unit the review never reached must not claim its diff was oversized.
    CHECK(tui::review_row_outcome(ReviewGroupRow{
              .state = ReviewGroupRow::State::Skipped,
              .note = "not reviewed",
          }) == "skipped · not reviewed");
    CHECK(tui::review_row_outcome(ReviewGroupRow{
              .state = ReviewGroupRow::State::Failed,
              .note = "provider returned an error",
          }) == "failed · provider returned an error");

    ReviewProgressView view;
    view.files = 2;
    view.elapsed = "9.0s";
    view.rows.push_back(ReviewGroupRow{
        .state = ReviewGroupRow::State::Done,
        .findings = 1,
        .blocking = 1,
    });
    CHECK(tui::review_summary_line(view)
          == "2 file(s) · 1 comment(s) · 1 blocking · 9.0s");
}

TEST_CASE("review status pill shows live N/M progress", "[tui][review_card]") {
    ReviewProgressView view;
    CHECK(tui::review_status_pill_label(view) == "review");

    view.total_groups = 17;
    CHECK(tui::review_status_pill_label(view) == "review (0/17)");

    view.rows.push_back(ReviewGroupRow{.state = ReviewGroupRow::State::Done});
    view.rows.push_back(ReviewGroupRow{.state = ReviewGroupRow::State::Failed});
    view.rows.push_back(ReviewGroupRow{.state = ReviewGroupRow::State::RiskPass});
    CHECK(tui::review_status_pill_label(view) == "review (2/17)");
}
