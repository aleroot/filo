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
