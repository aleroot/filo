#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/Conversation.hpp"
#include "tui/Text.hpp"

TEST_CASE("Text compact and trim helpers normalize whitespace", "[tui][text]") {
    REQUIRE(tui::trim_ascii("  alpha \n") == "alpha");
    REQUIRE(tui::trim_ascii("   ") == "");
    REQUIRE(tui::to_lower_ascii("HeLLo") == "hello");
    REQUIRE(tui::compact_single_line("  hello\n\tworld  ", 64) == "hello world");
    REQUIRE(tui::compact_single_line("abcdefghijk", 6) == "abc...");
    REQUIRE(tui::compact_single_line("abc", 2) == ".."); // Ellipsis fits partially
}

TEST_CASE("Text whitespace and role helpers", "[tui][text]") {
    REQUIRE(tui::flatten_whitespace("  multiple   spaces  \n  newlines  ") == "multiple spaces newlines");
    REQUIRE(tui::flatten_whitespace("already_flat") == "already_flat");
    REQUIRE(tui::flatten_whitespace("   ") == "");

    REQUIRE(tui::search_role_label(tui::MessageType::User) == "User");
    REQUIRE(tui::search_role_label(tui::MessageType::Assistant) == "Assistant");
    REQUIRE(tui::search_role_label(tui::MessageType::ToolGroup) == "Tool");
    REQUIRE(tui::search_role_label(tui::MessageType::System) == "System");
}

TEST_CASE("Text visibility parser accepts user-friendly values", "[tui][text]") {
    REQUIRE(tui::visibility_setting_enabled("show"));
    REQUIRE(tui::visibility_setting_enabled("YES"));
    REQUIRE_FALSE(tui::visibility_setting_enabled("hide"));
    REQUIRE_FALSE(tui::visibility_setting_enabled("off"));
    REQUIRE(tui::visibility_setting_enabled("unknown", true));
    REQUIRE_FALSE(tui::visibility_setting_enabled("unknown", false));
}

TEST_CASE("Text newline and utf8 editing helpers are stable", "[tui][text]") {
    REQUIRE(tui::normalize_newlines("a\r\nb\rc") == "a\nb\nc");

    std::string utf8 = "A\xe2\x9c\x85"; // A + check mark
    REQUIRE(tui::erase_last_utf8_codepoint(utf8));
    REQUIRE(utf8 == "A");
    REQUIRE(tui::erase_last_utf8_codepoint(utf8));
    REQUIRE(utf8.empty());
    REQUIRE_FALSE(tui::erase_last_utf8_codepoint(utf8));
}

TEST_CASE("Text search text includes tool metadata", "[tui][text]") {
    tui::UiMessage msg = tui::make_assistant_message("Primary answer", "", false);
    msg.secondary_text = "Secondary note";
    auto tool = tui::make_tool_activity("tool-1", "read_file", "{}", "Open config");
    tool.result.summary = "line one\nline two";
    msg.tools.push_back(tool);

    const std::string searchable = tui::search_text_for_message(msg);
    REQUIRE_THAT(searchable, Catch::Matchers::ContainsSubstring("Primary answer"));
    REQUIRE_THAT(searchable, Catch::Matchers::ContainsSubstring("Secondary note"));
    REQUIRE_THAT(searchable, Catch::Matchers::ContainsSubstring("read_file Open config"));
    REQUIRE_THAT(searchable, Catch::Matchers::ContainsSubstring("line one line two"));
}

TEST_CASE("Text builds bounded search snippets", "[tui][text]") {
    const std::string corpus =
        "prefix text that keeps going and going so the preview has enough context "
        "needle marker and some more suffix content";
    const auto pos = corpus.find("needle");
    REQUIRE(pos != std::string::npos);

    const std::string snippet = tui::build_search_snippet(corpus, pos, 6);
    REQUIRE_THAT(snippet, Catch::Matchers::ContainsSubstring("needle"));
    REQUIRE(snippet.size() <= 206); // 200 chars + optional leading/trailing "..."
}
