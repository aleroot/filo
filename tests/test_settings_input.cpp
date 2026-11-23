#include <catch2/catch_test_macros.hpp>

#include "tui/SettingsInput.hpp"

#include <ftxui/component/event.hpp>

using ftxui::Event;
using tui::SettingsTextEntry;

TEST_CASE("A settings text entry types, erases and submits", "[tui][settings]") {
    SettingsTextEntry entry;
    CHECK_FALSE(entry.active());
    entry.begin();
    REQUIRE(entry.active());

    for (const char ch : std::string_view("40x")) {
        CHECK(entry.handle(Event::Character(ch)) == SettingsTextEntry::Outcome::Editing);
    }
    entry.handle(Event::Backspace);
    entry.handle(Event::Character("%"));
    entry.handle(Event::Character("\u00e9"));  // two UTF-8 bytes
    entry.handle(Event::Backspace);             // removes the whole character
    CHECK(entry.text() == "40%");

    CHECK(entry.handle(Event::Return) == SettingsTextEntry::Outcome::Submitted);
    CHECK_FALSE(entry.active());
    CHECK(entry.text() == "40%");

    entry.begin();
    CHECK(entry.text().empty());
    entry.handle(Event::Character('9'));
    CHECK(entry.handle(Event::Escape) == SettingsTextEntry::Outcome::Cancelled);
    CHECK_FALSE(entry.active());
}

TEST_CASE("Auto-compaction values read as shares, token counts or the default",
          "[tui][settings][auto-compact]") {
    CHECK(tui::describe_auto_compact_setting("auto") == "Default (75% of window)");
    CHECK(tui::describe_auto_compact_setting("0") == "Disabled");
    CHECK(tui::describe_auto_compact_setting("40%") == "40% of window");
    CHECK(tui::describe_auto_compact_setting("400000") == "400k tokens");
    CHECK(tui::describe_auto_compact_setting("1000000") == "1M tokens");
    CHECK(tui::describe_auto_compact_setting("12345") == "12345 tokens");
    CHECK(tui::describe_auto_compact_setting("garbage") == "garbage");

    // Every preset is a value the typed form accepts unchanged.
    const auto input = tui::auto_compact_custom_input();
    for (const auto preset : tui::auto_compact_presets()) {
        CAPTURE(preset);
        CHECK(input.parse(preset) == std::string(preset));
    }
    CHECK(input.parse("400k") == "400000");
    CHECK_FALSE(input.parse("lots").has_value());
}
