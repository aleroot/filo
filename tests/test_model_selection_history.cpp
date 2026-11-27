#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/ModelPickerRows.hpp"
#include "tui/ModelSelectionHistory.hpp"
#include "tui/PromptComponents.hpp"

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>

#include <regex>
#include <string>
#include <vector>

using tui::ModelSelectionMode;
using tui::ModelSelectionSnapshot;

namespace {

ModelSelectionSnapshot manual(std::string provider, std::string model) {
    return {.mode = ModelSelectionMode::Manual,
            .manual_provider_name = std::move(provider),
            .manual_model_name = std::move(model)};
}

ModelSelectionSnapshot router(std::string policy) {
    return {.mode = ModelSelectionMode::Router, .router_policy = std::move(policy)};
}

bool exact(std::string_view lhs, std::string_view rhs) { return lhs == rhs; }

tui::ModelPickerRow row(std::string provider, std::string model) {
    return {.id = model, .selector = model, .source_provider = std::move(provider)};
}

std::string strip_ansi(std::string text) {
    static const std::regex ansi("\x1B\\[[0-9;?]*[ -/]*[@-~]");
    return std::regex_replace(text, ansi, "");
}

std::string render(const ftxui::Element& panel) {
    auto screen = ftxui::Screen::Create(ftxui::Dimension::Fixed(120),
                                        ftxui::Dimension::Fixed(16));
    ftxui::Render(screen, panel);
    return strip_ansi(screen.ToString());
}

} // namespace

TEST_CASE("Model switches build a most-recent-first history without the current model",
          "[tui][model][history]") {
    std::vector<ModelSelectionSnapshot> recent;
    const auto opus = manual("claude", "opus");
    const auto sonnet = manual("claude", "sonnet");
    const auto kimi = manual("kimi", "k2");

    tui::record_model_switch(recent, opus, sonnet);
    tui::record_model_switch(recent, sonnet, kimi);
    REQUIRE(recent == std::vector{sonnet, opus});

    // Going back to opus moves it out of history and puts kimi on top: the
    // list never holds the model in use and never holds one twice.
    tui::record_model_switch(recent, kimi, opus);
    CHECK(recent == std::vector{kimi, sonnet});

    // A no-op switch records nothing.
    tui::record_model_switch(recent, opus, opus);
    CHECK(recent == std::vector{kimi, sonnet});
}

TEST_CASE("Model history treats manual selections differing only in router policy as one",
          "[tui][model][history]") {
    auto opus_a = manual("claude", "opus");
    opus_a.router_policy = "a";
    auto opus_b = manual("claude", "opus");
    opus_b.router_policy = "b";
    CHECK(tui::same_selection_target(opus_a, opus_b));
    CHECK_FALSE(tui::same_selection_target(router("a"), router("b")));

    std::vector<ModelSelectionSnapshot> recent;
    tui::record_model_switch(recent, opus_a, manual("kimi", "k2"));
    tui::record_model_switch(recent, manual("kimi", "k2"), opus_b);
    CHECK(recent == std::vector{manual("kimi", "k2")});
}

TEST_CASE("Model history is capped", "[tui][model][history]") {
    std::vector<ModelSelectionSnapshot> recent;
    for (std::size_t i = 0; i < tui::kMaxRecentModelSelections + 4; ++i) {
        tui::record_model_switch(recent,
                                 manual("p", "m" + std::to_string(i)),
                                 manual("p", "m" + std::to_string(i + 1)));
    }
    REQUIRE(recent.size() == tui::kMaxRecentModelSelections);
    CHECK(recent.front().manual_model_name
          == "m" + std::to_string(tui::kMaxRecentModelSelections + 3));
}

TEST_CASE("The picker's Ctrl+P / Ctrl+N walk visits model selections only",
          "[tui][model][history]") {
    const std::vector recent{
        manual("claude", "sonnet"),  // 0
        router("work"),              // 1: no model row, skipped
        manual("kimi", "k2"),        // 2
    };
    CHECK(tui::older_model_choice(recent, std::nullopt) == 0u);
    CHECK(tui::older_model_choice(recent, 0u) == 2u);
    CHECK_FALSE(tui::older_model_choice(recent, 2u).has_value());

    CHECK(tui::newer_model_choice(recent, 2u) == 0u);
    CHECK_FALSE(tui::newer_model_choice(recent, 0u).has_value());  // back to current

    const std::vector only_router{router("work")};
    CHECK_FALSE(tui::older_model_choice(only_router, std::nullopt).has_value());
    CHECK_FALSE(tui::older_model_choice(std::vector<ModelSelectionSnapshot>{}, std::nullopt)
                    .has_value());
}

TEST_CASE("The picker lands on the selection's row, adding one the catalogue omits",
          "[tui][model][history]") {
    std::vector rows{row("claude", "opus"), row("claude", "sonnet")};

    CHECK(tui::focus_model_row(rows, manual("claude", "sonnet"), exact) == 1u);
    REQUIRE(rows.size() == 2);

    // Same id on another endpoint is a different target.
    const auto custom = manual("claude-sub", "sonnet");
    REQUIRE(tui::focus_model_row(rows, custom, exact) == 0u);
    REQUIRE(rows.size() == 3);
    CHECK(rows.front().id == "sonnet");
    CHECK(rows.front().source_provider == "claude-sub");
    CHECK(tui::row_selects(rows.front(), custom, exact));

    // An empty model name maps to the provider-default row.
    std::vector defaults{tui::ModelPickerRow{.id = "<provider default>", .source_provider = "x"}};
    CHECK(tui::focus_model_row(defaults, manual("x", ""), exact) == 0u);
    CHECK(defaults.size() == 1);
}

TEST_CASE("Model rows mark the active and the previous model",
          "[tui][model][history]") {
    std::vector rows{row("claude", "opus"), row("claude", "sonnet"), row("claude", "haiku")};
    // The newest entry is a router selection: "previous" is the newest
    // MODEL, which is where Ctrl+P lands.
    const std::vector recent{router("work"), manual("claude", "haiku"), manual("claude", "opus")};

    tui::mark_model_rows(rows, manual("claude", "sonnet"), recent, exact);
    CHECK_FALSE(rows[0].active);
    CHECK_FALSE(rows[0].previous);
    CHECK(rows[1].active);
    CHECK_FALSE(rows[1].previous);
    CHECK(rows[2].previous);

    // In router mode no row is active.
    tui::mark_model_rows(rows, router("work"), recent, exact);
    CHECK_FALSE(rows[1].active);
}

TEST_CASE("The model page footer advertises only the keys that act",
          "[tui][picker][model][history]") {
    std::vector rows{row("claude", "opus"), row("claude", "sonnet")};
    rows[0].active = true;
    rows[1].previous = true;

    const auto opened_directly = render(tui::render_provider_model_picker_panel(
        "claude", rows, 0,
        {.escape_returns_to_providers = false, .older_model = true, .newer_model = false}));
    CHECK_THAT(opened_directly, Catch::Matchers::ContainsSubstring("opus (active)"));
    CHECK_THAT(opened_directly, Catch::Matchers::ContainsSubstring("sonnet (previous)"));
    CHECK_THAT(opened_directly, Catch::Matchers::ContainsSubstring("Ctrl+P: previous model"));
    CHECK_THAT(opened_directly, Catch::Matchers::ContainsSubstring("Esc: close"));
    CHECK_THAT(opened_directly, !Catch::Matchers::ContainsSubstring("Ctrl+N"));

    const auto walked = render(tui::render_provider_model_picker_panel(
        "claude", rows, 1,
        {.escape_returns_to_providers = true, .older_model = false, .newer_model = true}));
    CHECK_THAT(walked, Catch::Matchers::ContainsSubstring("Ctrl+N: newer model"));
    CHECK_THAT(walked, Catch::Matchers::ContainsSubstring("Esc: providers"));
    CHECK_THAT(walked, !Catch::Matchers::ContainsSubstring("Ctrl+P"));

    const std::vector<tui::ModelProviderPickerRow> providers{{.name = "claude", .active = true}};
    CHECK_THAT(render(tui::render_model_provider_picker_panel(providers, 0, false, true)),
               Catch::Matchers::ContainsSubstring("Ctrl+P: previous model"));
    CHECK_THAT(render(tui::render_model_provider_picker_panel(providers, 0, false, false)),
               !Catch::Matchers::ContainsSubstring("Ctrl+P"));
}
