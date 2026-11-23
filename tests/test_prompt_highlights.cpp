#include <catch2/catch_test_macros.hpp>

#include "tui/PromptHighlights.hpp"
#include "tui/PromptInput.hpp"
#include "tui/TuiTheme.hpp"

#include <ftxui/component/component_options.hpp>
#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>

#include <algorithm>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

using namespace ftxui;
using core::commands::InlineCommandCatalog;
using core::context::MentionTarget;
using tui::PromptTokenKind;

namespace {

InlineCommandCatalog make_catalog() {
    return InlineCommandCatalog({{
        .name = "/effort",
        .description = "Reasoning effort",
        .values = {{.value = "auto"}, {.value = "low"}, {.value = "high"}},
        .current_value = "auto",
    }});
}

// Stand-ins for the session: one file, one subagent, one saved conversation.
tui::PromptTokenSources make_sources(const InlineCommandCatalog& catalog) {
    return {
        .commands = &catalog,
        .classify_mention = [](const core::context::MentionToken& mention) {
            if (mention.raw_path == "src/a.cpp") return MentionTarget::File;
            if (mention.raw_path == "explore") return MentionTarget::Subagent;
            return mention.quoted ? MentionTarget::Missing : MentionTarget::Prose;
        },
        .conversation_title = [](const core::session::SessionReferenceToken& token)
            -> std::optional<std::string> {
            if (token.target == "a1b2c3d4") return "Parser rewrite";
            return std::nullopt;
        },
        .ultrathink = true,
    };
}

std::vector<PromptTokenKind> kinds(const std::vector<tui::PromptToken>& tokens) {
    std::vector<PromptTokenKind> out;
    std::ranges::transform(tokens, std::back_inserter(out), &tui::PromptToken::kind);
    return out;
}

// The characters on row @p y, ignoring styles.
std::string row_text(const Screen& screen, int y = 0) {
    std::string out;
    for (int x = 0; x < screen.dimx(); ++x) out += screen.at(x, y);
    return out;
}

} // namespace

TEST_CASE("prompt tokens report what a draft does on send, and only that",
          "[prompt-highlights]") {
    const auto catalog = make_catalog();
    const std::string prompt =
        R"(/effort high ultrathink and read @src/a.cpp with @explore and #a1b2c3d4, )"
        R"(not @here #123 or `@src/a.cpp`, and @"gone.txt")";
    const auto tokens = tui::scan_prompt_tokens(prompt, 0, make_sources(catalog));

    CHECK(kinds(tokens) == std::vector{
        PromptTokenKind::Command,
        PromptTokenKind::Ultrathink,
        PromptTokenKind::Attachment,
        PromptTokenKind::Subagent,
        PromptTokenKind::Conversation,
        PromptTokenKind::Unavailable,
    });
    CHECK(tokens[0].subject == "/effort");
    CHECK(tokens[0].detail == "high");
    CHECK(prompt.substr(tokens[2].begin, tokens[2].end - tokens[2].begin) == "@src/a.cpp");
    CHECK(tokens[4].subject == "Parser rewrite");
    CHECK(prompt.substr(tokens[4].begin, tokens[4].end - tokens[4].begin) == "#a1b2c3d4");
    CHECK(tokens[5].detail == "not found");
    CHECK(std::ranges::is_sorted(tokens, {}, &tui::PromptToken::begin));
}

TEST_CASE("an inline command is highlighted while typed, then left as prose",
          "[prompt-highlights]") {
    const auto catalog = make_catalog();
    const auto sources = make_sources(catalog);
    const std::string draft = "fix /effort";

    CHECK(kinds(tui::scan_prompt_tokens(draft, draft.size(), sources))
          == std::vector{PromptTokenKind::PendingCommand});
    CHECK(tui::scan_prompt_tokens(draft, 0, sources).empty());

    CHECK(kinds(tui::scan_prompt_tokens("/effort low then /effort high", 0, sources))
          == std::vector{PromptTokenKind::SupersededCommand, PromptTokenKind::Command});
}

TEST_CASE("ultrathink is coloured glyph by glyph", "[prompt-highlights]") {
    const auto catalog = make_catalog();
    const auto tokens = tui::scan_prompt_tokens("please ultrathink now", 0, make_sources(catalog));
    REQUIRE(kinds(tokens) == std::vector{PromptTokenKind::Ultrathink});

    const auto spans = tui::prompt_input_spans(tokens);
    REQUIRE(spans.size() == 10);
    CHECK(spans.front().begin == 7);
    CHECK(spans.back().end == 17);
}

TEST_CASE("PromptInput styles spans without moving the text or the cursor",
          "[prompt-highlights][prompt_input][render]") {
    std::string content = "fix /effort high now";
    int cursor = 6;
    InputOption option;
    option.cursor_position = &cursor;

    const auto render = [&](tui::PromptInputStyler styler) {
        auto input = tui::PromptInput(&content, "", option, std::move(styler));
        auto screen = Screen::Create(Dimension::Fixed(30), Dimension::Fixed(1));
        Render(screen, input->Render());
        return screen;
    };

    const auto plain = render({});
    const auto styled = render([](std::string_view, std::size_t) {
        return std::vector<tui::PromptInputSpan>{
            {.begin = 4, .end = 16, .style = bgcolor(Color::Red)},
        };
    });

    CHECK(row_text(styled) == row_text(plain));
    CHECK(styled.cursor().x == plain.cursor().x);
    CHECK(styled.cursor().y == plain.cursor().y);
    CHECK(styled.CellAt(3, 0).background_color != Color(Color::Red));
    for (int x = 4; x < 16; ++x) {
        INFO(x);
        CHECK(styled.CellAt(x, 0).background_color == Color(Color::Red));
    }
    CHECK(styled.CellAt(16, 0).background_color != Color(Color::Red));
}

TEST_CASE("PromptInput keeps styles across lines and ignores overlapping spans",
          "[prompt-highlights][prompt_input][render]") {
    std::string content = "one /x\ntwo @y";
    int cursor = 0;
    InputOption option;
    option.cursor_position = &cursor;
    option.multiline = true;

    auto input = tui::PromptInput(&content, "", option, [](std::string_view, std::size_t) {
        return std::vector<tui::PromptInputSpan>{
            {.begin = 11, .end = 13, .style = bgcolor(Color::Blue)},  // "@y", line 2
            {.begin = 4, .end = 6, .style = bgcolor(Color::Red)},     // "/x", line 1
            {.begin = 5, .end = 9, .style = bgcolor(Color::Green)},   // overlaps: dropped
        };
    });
    auto screen = Screen::Create(Dimension::Fixed(10), Dimension::Fixed(2));
    Render(screen, input->Render());

    CHECK(row_text(screen, 0).starts_with("one /x"));
    CHECK(row_text(screen, 1).starts_with("two @y"));
    CHECK(screen.CellAt(4, 0).background_color == Color(Color::Red));
    CHECK(screen.CellAt(5, 0).background_color == Color(Color::Red));
    CHECK(screen.CellAt(0, 1).background_color != Color(Color::Green));
    CHECK(screen.CellAt(4, 1).background_color == Color(Color::Blue));
    CHECK(screen.CellAt(5, 1).background_color == Color(Color::Blue));
}
