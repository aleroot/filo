#include <catch2/catch_test_macros.hpp>

#include "core/commands/InlineCommands.hpp"

#include <string>
#include <string_view>
#include <vector>

using namespace core::commands;

namespace {

InlineCommandCatalog make_catalog() {
    return InlineCommandCatalog({
        {
            .name = "/effort",
            .description = "Reasoning effort",
            .values = {{.value = "auto"}, {.value = "low"}, {.value = "medium"},
                       {.value = "high"}, {.value = "xhigh"}, {.value = "off"}},
            .current_value = "auto",
        },
        {
            .name = "/yolo",
            .description = "Approve tool calls",
            .values = {{.value = "on"}, {.value = "off"}},
            .current_value = "off",
        },
    });
}

std::vector<InlineCommandInvocation> invocations(std::string_view prompt) {
    return extract_inline_commands(prompt, make_catalog()).invocations;
}

std::string remainder(std::string_view prompt) {
    return extract_inline_commands(prompt, make_catalog()).prompt;
}

} // namespace

TEST_CASE("an inline command anywhere in a prompt is applied and taken out",
          "[inline-commands]") {
    CHECK(invocations("fix the parser /effort high")
          == std::vector<InlineCommandInvocation>{{.name = "/effort", .value = "high"}});
    CHECK(remainder("fix the parser /effort high") == "fix the parser");
    CHECK(remainder("/effort high fix the bug") == "fix the bug");
    CHECK(remainder("fix /effort high the bug") == "fix the bug");
    CHECK(remainder("line one /effort high\nline two") == "line one\nline two");
    CHECK(remainder("/effort high /yolo on do it") == "do it");
    CHECK(invocations("/effort high /yolo on do it").size() == 2);

    // A prompt that only sets things sends nothing.
    CHECK(remainder("/effort high") == "");
    CHECK(remainder("  /effort high  ") == "");
    CHECK(invocations("  /effort high").size() == 1);
}

TEST_CASE("inline command values are a closed set, matched ignoring case",
          "[inline-commands]") {
    CHECK(invocations("go /effort HIGH").front().value == "high");
    // The name folds case too, and a repeat still collapses onto the canonical one.
    CHECK(invocations("go /EFFORT high").front().name == "/effort");
    CHECK(remainder("/effort low then /EFFORT high") == "then");
    // A value it does not take makes the whole thing prose.
    CHECK(invocations("tune /effort please").empty());
    CHECK(remainder("tune /effort please") == "tune /effort please");
    // Sentence punctuation stays with the sentence.
    CHECK(invocations("use /effort high, then fix it").front().value == "high");
    CHECK(remainder("use /effort high, then fix it") == "use , then fix it");
}

TEST_CASE("slash words that are not inline commands reach the model untouched",
          "[inline-commands]") {
    for (const std::string_view prose : {
             "open /usr/bin/effort",       // a path
             "see /effortless high",       // a longer word
             "and/effort high",            // not at a word boundary
             "the /settings page",         // not an inline command
             "run `/effort high` here",    // inline code
             "```\n/effort high\n```",     // fenced code
             "!echo /effort high",         // a shell command
             "s/effort high/low/g",        // a substitution
         }) {
        INFO(prose);
        CHECK(scan_inline_commands(prose, make_catalog()).empty());
        CHECK(remainder(prose) == prose);
    }
}

TEST_CASE("an inline command being typed is reported but never applied",
          "[inline-commands]") {
    const auto catalog = make_catalog();

    auto tokens = scan_inline_commands("fix /effort", catalog);
    REQUIRE(tokens.size() == 1);
    CHECK(tokens[0].state == InlineCommandState::AwaitingValue);
    CHECK(tokens[0].begin == 4);
    CHECK(tokens[0].end == 11);

    tokens = scan_inline_commands("fix /effort hi", catalog);
    REQUIRE(tokens.size() == 1);
    CHECK(tokens[0].state == InlineCommandState::PartialValue);
    CHECK(tokens[0].end == 14);

    CHECK(invocations("fix /effort hi").empty());
    CHECK(remainder("fix /effort") == "fix /effort");
}

TEST_CASE("a repeated inline command keeps its last value", "[inline-commands]") {
    const auto tokens = scan_inline_commands("/effort low and /effort high", make_catalog());
    REQUIRE(tokens.size() == 2);
    CHECK(tokens[0].superseded);
    CHECK_FALSE(tokens[1].superseded);
    CHECK(invocations("/effort low and /effort high")
          == std::vector<InlineCommandInvocation>{{.name = "/effort", .value = "high"}});
    CHECK(remainder("/effort low and /effort high") == "and");
}

TEST_CASE("the inline picker offers names mid-prompt and values anywhere",
          "[inline-commands]") {
    const auto catalog = make_catalog();
    const auto active = [&](std::string_view prompt) {
        return find_active_inline_command(prompt, prompt.size(), catalog);
    };

    const auto name = active("fix this /ef");
    REQUIRE(name.has_value());
    CHECK(name->part == ActiveInlineCommand::Part::Name);
    CHECK(name->query == "/ef");
    CHECK(suggest_inline_commands(*name, catalog, 8).front().insertion == "/effort");

    // The start of the prompt belongs to the full command picker.
    CHECK_FALSE(active("/ef").has_value());
    // Paths are not command names.
    CHECK_FALSE(active("look in /usr/b").has_value());
    CHECK_FALSE(active("run `/ef").has_value());
    CHECK_FALSE(active("!ls /ef").has_value());

    const auto value = active("fix this /effort h");
    REQUIRE(value.has_value());
    CHECK(value->part == ActiveInlineCommand::Part::Value);
    CHECK(value->command == "/effort");
    CHECK(value->query == "h");
    CHECK(suggest_inline_commands(*value, catalog, 8).front().insertion == "high");

    // Values are offered at the start too: the full picker stops at the name.
    const auto all_values = active("/effort ");
    REQUIRE(all_values.has_value());
    const auto suggestions = suggest_inline_commands(*all_values, catalog, 8);
    REQUIRE(suggestions.size() == 6);
    CHECK(suggestions.front().insertion == "auto");
    CHECK(suggestions.front().current);
    CHECK_FALSE(suggestions[1].current);

    CHECK_FALSE(active("fix /effort high ").has_value());
}

TEST_CASE("accepting an inline suggestion leaves the cursor ready for what follows",
          "[inline-commands]") {
    const auto catalog = make_catalog();

    const std::string draft = "fix this /ef";
    const auto name = find_active_inline_command(draft, draft.size(), catalog);
    REQUIRE(name.has_value());
    const auto named = apply_inline_completion(draft, *name, "/effort");
    CHECK(named.text == "fix this /effort ");
    CHECK(named.cursor == named.text.size());

    const std::string middle = "fix /effort h and more";
    const auto value = find_active_inline_command(middle, 13, catalog);
    REQUIRE(value.has_value());
    const auto valued = apply_inline_completion(middle, *value, "high");
    CHECK(valued.text == "fix /effort high and more");
    CHECK(valued.cursor == 17);
}
