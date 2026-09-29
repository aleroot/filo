#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/Conversation.hpp"
#include "tui/SessionReplay.hpp"

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>

#include <algorithm>
#include <format>
#include <string>
#include <unordered_map>

using namespace tui;
using core::changes::FileChange;
using core::changes::FileChangeContent;
using core::changes::FileChangeKind;
using Catch::Matchers::ContainsSubstring;

namespace {

std::string strip_ansi(std::string_view input) {
    std::string out;
    out.reserve(input.size());
    for (std::size_t i = 0; i < input.size();) {
        if (input[i] == '\x1b' && i + 1 < input.size() && input[i + 1] == '[') {
            i += 2;
            while (i < input.size()) {
                const char ch = input[i++];
                if (ch >= '@' && ch <= '~') {
                    break;
                }
            }
            continue;
        }
        out.push_back(input[i]);
        ++i;
    }
    return out;
}

std::string render_text(const std::vector<UiMessage>& messages,
                        ConversationRenderOptions options = {}) {
    auto content = render_history_content(messages, 0, std::move(options));
    auto screen = ftxui::Screen::Create(
        ftxui::Dimension::Fixed(100),
        ftxui::Dimension::Fit(content, true));
    ftxui::Render(screen, content);
    return strip_ansi(screen.ToString());
}

std::vector<FileChange> sample_changes() {
    return {
        FileChange{
            .kind = FileChangeKind::Modified,
            .path = "src/app.cpp",
            .diff = "--- a/src/app.cpp\n+++ b/src/app.cpp\n@@ -1,2 +1,3 @@\n keep\n-old-app-line\n+new-app-line\n+second-app-line\n",
            .added = 2,
            .deleted = 1,
        },
        FileChange{
            .kind = FileChangeKind::Added,
            .path = "docs/guide.md",
            .diff = "--- a/docs/guide.md\n+++ b/docs/guide.md\n@@ -0,0 +1 @@\n+guide-body-line\n",
            .added = 1,
        },
        FileChange{
            .kind = FileChangeKind::Deleted,
            .content = FileChangeContent::Binary,
            .path = "assets/logo.png",
        },
    };
}

} // namespace

TEST_CASE("The turn's file changes follow the answer, one collapsed row per file",
          "[tui][turn_file_changes]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = sample_changes();

    const auto text = render_text({answer});
    CHECK_THAT(text, ContainsSubstring("3 files changed  +3 -1"));
    CHECK_THAT(text, ContainsSubstring("▶ src/app.cpp"));
    CHECK_THAT(text, ContainsSubstring("▶ docs/guide.md  new"));
    CHECK_THAT(text, ContainsSubstring("• assets/logo.png  deleted, binary"));
    CHECK(text.find("All done.") < text.find("3 files changed"));
    CHECK(text.find("new-app-line") == std::string::npos);
    CHECK(text.find("guide-body-line") == std::string::npos);
}

TEST_CASE("Opening one file shows only that file's diff", "[tui][turn_file_changes]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = sample_changes();

    std::unordered_map<std::string, bool> expanded{
        {turn_file_change_key(answer.id, "src/app.cpp"), true},
    };
    ConversationRenderOptions options;
    options.system_disclosure_expanded = &expanded;

    const auto text = render_text({answer}, options);
    CHECK_THAT(text, ContainsSubstring("▼ src/app.cpp"));
    CHECK_THAT(text, ContainsSubstring("old-app-line"));
    CHECK_THAT(text, ContainsSubstring("second-app-line"));
    CHECK(text.find("guide-body-line") == std::string::npos);
}

TEST_CASE("Messages without file changes draw no box", "[tui][turn_file_changes]") {
    const auto text = render_text({make_assistant_message("Just an answer.", "", false)});
    CHECK(text.find("changed") == std::string::npos);
}

TEST_CASE("A resumed session shows the recorded file changes on the same message",
          "[tui][turn_file_changes][session_replay]") {
    core::llm::Message user{.role = "user", .content = "edit the app"};
    core::llm::Message answer{.role = "assistant", .content = "All done."};
    answer.turn_changes.files = sample_changes();

    core::session::SessionData data;
    data.messages = {user, answer};
    const auto messages = build_resumed_ui_messages(data);
    const auto it = std::ranges::find_if(messages, [](const UiMessage& message) {
        return message.type == MessageType::Assistant;
    });
    REQUIRE(it != messages.end());
    CHECK(it->turn_changes == answer.turn_changes);
}

TEST_CASE("A summary that fell short of the truth says so", "[tui][turn_file_changes]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = sample_changes();
    answer.turn_changes.partial_enumeration = true;
    answer.turn_changes.unscoped_mutations = true;

    const auto text = render_text({answer});
    CHECK_THAT(text, ContainsSubstring("too many files to detail fully"));
    CHECK_THAT(text, ContainsSubstring(
        "files created by shell, scripts or MCP tools are not listed"));
}

TEST_CASE("A complete summary claims nothing it cannot back", "[tui][turn_file_changes]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = sample_changes();

    const auto text = render_text({answer});
    CHECK(text.find("too many files") == std::string::npos);
    CHECK(text.find("are not listed") == std::string::npos);
}

TEST_CASE("A long file list is counted past the rows the box draws",
          "[tui][turn_file_changes]") {
    auto answer = make_assistant_message("All done.", "", false);
    const std::size_t total = kMaxRenderedFileChanges + 5;
    for (std::size_t i = 0; i < total; ++i) {
        answer.turn_changes.files.push_back(FileChange{
            .kind = FileChangeKind::Modified,
            .content = FileChangeContent::BudgetSpent,
            .path = std::format("src/f{:03}.cpp", i),
        });
    }

    const auto text = render_text({answer});
    CHECK_THAT(text, ContainsSubstring(std::format("{} files changed", total)));
    CHECK_THAT(text, ContainsSubstring("5 more not listed"));
    CHECK_THAT(text, ContainsSubstring("src/f000.cpp"));
    CHECK(text.find("src/f029.cpp") == std::string::npos);
}

TEST_CASE("A file that cannot be diffed still shows how large the change was",
          "[tui][turn_file_changes]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = {
        FileChange{.kind = FileChangeKind::Deleted,
                   .content = FileChangeContent::BudgetSpent,
                   .path = "logs/big.log",
                   .deleted = 1234},
    };

    const auto text = render_text({answer});
    CHECK_THAT(text, ContainsSubstring("1 file changed"));
    CHECK_THAT(text, ContainsSubstring("-1234"));
    CHECK_THAT(text, ContainsSubstring("deleted, diff omitted"));
    // Not expandable: there is no diff to open, only a count.
    CHECK_THAT(text, ContainsSubstring("• logs/big.log"));
}

TEST_CASE("A change whose size cannot be known shows no counts at all",
          "[tui][turn_file_changes]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = {
        FileChange{.kind = FileChangeKind::Modified,
                   .content = FileChangeContent::Binary,
                   .path = "assets/logo.png"},
    };

    const auto text = render_text({answer});
    CHECK_THAT(text, ContainsSubstring("1 file changed"));
    CHECK_THAT(text, ContainsSubstring("binary"));
    // Guessing a size would be worse than showing none.
    CHECK(text.find("+0") == std::string::npos);
    CHECK(text.find("-0") == std::string::npos);
}

namespace {

[[nodiscard]] std::size_t count_of(std::string_view haystack, std::string_view needle) {
    std::size_t count = 0;
    for (auto at = haystack.find(needle); at != std::string_view::npos;
         at = haystack.find(needle, at + needle.size())) {
        ++count;
    }
    return count;
}

} // namespace

TEST_CASE("The comparer affordance marks only the files a comparer could show",
          "[tui][turn_file_changes][diff_comparer]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = sample_changes();

    std::unordered_map<std::string, ftxui::Box> hitboxes;
    ConversationRenderOptions options;
    options.show_diff_comparer_affordance = true;
    options.system_disclosure_hitboxes = &hitboxes;

    const auto text = render_text({answer}, options);
    // Two of the three changes carry a diff; the binary one has nothing to
    // compare, so it offers nothing to click.
    CHECK(count_of(text, "\u2197") == 2);
    CHECK(hitboxes.contains(turn_file_change_open_key(answer.id, "src/app.cpp")));
    CHECK(hitboxes.contains(turn_file_change_open_key(answer.id, "docs/guide.md")));
    CHECK_FALSE(hitboxes.contains(turn_file_change_open_key(answer.id, "assets/logo.png")));
    // The row's own disclosure box is still there, affordance or not.
    CHECK(hitboxes.contains(turn_file_change_key(answer.id, "src/app.cpp")));
}

TEST_CASE("No affordance is drawn while the transcript is the comparer",
          "[tui][turn_file_changes][diff_comparer]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = sample_changes();

    std::unordered_map<std::string, ftxui::Box> hitboxes;
    ConversationRenderOptions options;
    options.system_disclosure_hitboxes = &hitboxes;

    const auto text = render_text({answer}, options);
    CHECK(count_of(text, "\u2197") == 0);
    CHECK_FALSE(hitboxes.contains(turn_file_change_open_key(answer.id, "src/app.cpp")));
}

TEST_CASE("An affordance key names the turn and file it came from",
          "[tui][turn_file_changes][diff_comparer]") {
    const auto key = turn_file_change_open_key("0123456789abcdef", "src/a:b.cpp");
    CHECK(is_turn_file_change_open_key(key));
    CHECK_FALSE(is_turn_file_change_open_key(turn_file_change_key("0123456789abcdef", "src/a:b.cpp")));

    const auto ref = parse_turn_file_change_open_key(key);
    REQUIRE(ref.has_value());
    CHECK(ref->message_id == "0123456789abcdef");
    // A path may hold colons; a message id never does, so the first one ends it.
    CHECK(ref->path == "src/a:b.cpp");

    CHECK_FALSE(parse_turn_file_change_open_key("file-change:0123456789abcdef:src/app.cpp").has_value());
    CHECK_FALSE(parse_turn_file_change_open_key("file-change-open:0123456789abcdef").has_value());
}

TEST_CASE("A turn exports one patch for the comparer", "[tui][turn_file_changes][diff_comparer]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = sample_changes();

    const auto comparison = turn_comparison(answer);
    REQUIRE(comparison.has_value());
    CHECK_THAT(comparison->patch, ContainsSubstring("diff --git a/src/app.cpp b/src/app.cpp"));
    CHECK_THAT(comparison->patch, ContainsSubstring("diff --git a/docs/guide.md b/docs/guide.md"));
    CHECK_THAT(comparison->patch, ContainsSubstring("+new-app-line"));
    CHECK_THAT(comparison->patch, ContainsSubstring("+guide-body-line"));
    // The binary change has no diff to export.
    CHECK(comparison->patch.find("logo.png") == std::string::npos);

    REQUIRE(comparison->disclosure_keys.size() == 2);
    CHECK(comparison->disclosure_keys[0] == turn_file_change_key(answer.id, "src/app.cpp"));
    CHECK(comparison->disclosure_keys[1] == turn_file_change_key(answer.id, "docs/guide.md"));
}

TEST_CASE("One file exports on its own", "[tui][turn_file_changes][diff_comparer]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = sample_changes();

    const auto comparison = turn_comparison(answer, "docs/guide.md");
    REQUIRE(comparison.has_value());
    CHECK_THAT(comparison->patch, ContainsSubstring("diff --git a/docs/guide.md b/docs/guide.md"));
    CHECK(comparison->patch.find("app.cpp") == std::string::npos);
    REQUIRE(comparison->disclosure_keys.size() == 1);
    CHECK(comparison->disclosure_keys[0] == turn_file_change_key(answer.id, "docs/guide.md"));

    // Nothing to compare, so nothing to open.
    CHECK_FALSE(turn_comparison(answer, "assets/logo.png").has_value());
    CHECK_FALSE(turn_comparison(answer, "src/missing.cpp").has_value());
}

TEST_CASE("A turn with nothing diffable opens no comparer",
          "[tui][turn_file_changes][diff_comparer]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = {
        FileChange{.kind = FileChangeKind::Modified,
                   .content = FileChangeContent::Binary,
                   .path = "assets/logo.png"},
    };

    const std::vector<UiMessage> messages{answer};
    CHECK(latest_turn_with_changes(messages).message != nullptr);
    CHECK_FALSE(turn_comparison(answer).has_value());
}

TEST_CASE("The comparer opens on the most recent turn that changed files",
          "[tui][turn_file_changes][diff_comparer]") {
    auto first = make_assistant_message("First.", "", false);
    first.turn_changes.files = sample_changes();
    auto quiet = make_assistant_message("Nothing touched.", "", false);
    auto second = make_assistant_message("Second.", "", false);
    second.turn_changes.files = {
        FileChange{.kind = FileChangeKind::Added,
                   .path = "src/late.cpp",
                   .diff = "--- a/src/late.cpp\n+++ b/src/late.cpp\n@@ -0,0 +1 @@\n+late\n",
                   .added = 1},
    };

    const std::vector<UiMessage> messages{first, quiet, second};
    const auto latest = latest_turn_with_changes(messages);
    REQUIRE(latest.message != nullptr);
    CHECK(latest.message->id == second.id);
    CHECK(latest.index == 2);
    CHECK_FALSE(latest.superseded);

    const auto comparison = turn_comparison(*latest.message);
    REQUIRE(comparison.has_value());
    CHECK_THAT(comparison->patch, ContainsSubstring("src/late.cpp"));
    CHECK(comparison->patch.find("app.cpp") == std::string::npos);

    const std::vector<UiMessage> quiet_only{quiet};
    CHECK(latest_turn_with_changes(quiet_only).message == nullptr);
    const std::vector<UiMessage> none;
    CHECK(latest_turn_with_changes(none).message == nullptr);
}

TEST_CASE("A turn that changed files is flagged once a later turn started",
          "[tui][turn_file_changes][diff_comparer]") {
    auto changed = make_assistant_message("Edited.", "", false);
    changed.turn_changes.files = sample_changes();
    UiMessage later_prompt;
    later_prompt.type = MessageType::User;
    later_prompt.text = "Now explain it.";
    auto later_answer = make_assistant_message("It works like this.", "", false);

    SECTION("a later turn that changed nothing") {
        const std::vector<UiMessage> messages{changed, later_prompt, later_answer};
        const auto latest = latest_turn_with_changes(messages);
        REQUIRE(latest.message != nullptr);
        CHECK(latest.message->id == changed.id);
        CHECK(latest.index == 0);
        CHECK(latest.superseded);
    }
    SECTION("a later turn still running, with no answer yet") {
        const std::vector<UiMessage> messages{changed, later_prompt};
        const auto latest = latest_turn_with_changes(messages);
        REQUIRE(latest.message != nullptr);
        CHECK(latest.superseded);
    }
    SECTION("notices after the turn do not start a new one") {
        const std::vector<UiMessage> messages{changed, make_info_message("Model switched.")};
        const auto latest = latest_turn_with_changes(messages);
        REQUIRE(latest.message != nullptr);
        CHECK_FALSE(latest.superseded);
    }
}

TEST_CASE("A comparison names the changes its patch leaves out",
          "[tui][turn_file_changes][diff_comparer]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = sample_changes();

    SECTION("a binary file has no diff to carry") {
        const auto comparison = turn_comparison(answer);
        REQUIRE(comparison.has_value());
        CHECK(comparison->undiffed == 1);
        CHECK(comparison->caveats.empty());
        CHECK(comparison_omissions(*comparison)
              == "The comparison is not the whole turn: 1 changed file has no diff "
                 "(binary, too large or over the diff budget).");
    }
    SECTION("the turn's own caveats travel with it") {
        answer.turn_changes.unscoped_mutations = true;
        answer.turn_changes.partial_enumeration = true;
        answer.turn_changes.files.pop_back();  // Drop the binary change.
        const auto comparison = turn_comparison(answer);
        REQUIRE(comparison.has_value());
        CHECK(comparison->undiffed == 0);
        CHECK(comparison_omissions(*comparison)
              == "The comparison is not the whole turn: too many files to detail fully; "
                 "files created by shell, scripts or MCP tools are not listed.");
    }
    SECTION("a complete turn has nothing to report") {
        answer.turn_changes.files.pop_back();
        const auto comparison = turn_comparison(answer);
        REQUIRE(comparison.has_value());
        CHECK(comparison_omissions(*comparison).empty());
    }
    SECTION("one file is the whole of what was asked for") {
        answer.turn_changes.unscoped_mutations = true;
        const auto comparison = turn_comparison(answer, "src/app.cpp");
        REQUIRE(comparison.has_value());
        CHECK(comparison_omissions(*comparison).empty());
    }
}
