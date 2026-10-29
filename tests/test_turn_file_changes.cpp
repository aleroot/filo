#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/Conversation.hpp"
#include "tui/SessionReplay.hpp"
#include "core/changes/TurnChangeTracker.hpp"
#include "core/changes/TurnRecovery.hpp"
#include "core/changes/FileChangeJson.hpp"

#include <simdjson.h>

#include <ftxui/dom/elements.hpp>
#include <ftxui/screen/screen.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <format>
#include <random>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace tui;
using core::changes::FileChange;
using core::changes::FileChangeContent;
using core::changes::FileChangeKind;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::EndsWith;

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
                        ConversationRenderOptions options = {}, int width = 100) {
    auto content = render_history_content(messages, 0, std::move(options));
    auto screen = ftxui::Screen::Create(
        ftxui::Dimension::Fixed(width),
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

std::vector<std::string> split_lines(std::string_view text) {
    std::vector<std::string> lines;
    std::size_t start = 0;
    while (start <= text.size()) {
        const auto end = text.find('\n', start);
        auto line = text.substr(start, end == std::string_view::npos
                                           ? std::string_view::npos
                                           : end - start);
        // The screen ends its lines with a carriage return.
        if (!line.empty() && line.back() == '\r') line.remove_suffix(1);
        lines.emplace_back(line);
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return lines;
}

/// The line without the spaces that pad it to the width of the box.
std::string trimmed(std::string_view text) {
    const auto end = text.find_last_not_of(' ');
    if (end == std::string_view::npos) return {};
    return std::string(text.substr(0, end + 1));
}

/// The line of the box that carries the summary of the turn.
std::string summary_line(std::string_view text) {
    for (const auto& line : split_lines(text)) {
        if (line.find("changed") != std::string::npos) return line;
    }
    return {};
}

/// The content of a line of the box, without the border that frames it.
std::string box_content(std::string_view line) {
    constexpr std::string_view border = "\u2502";  // │
    std::string_view content = line;
    if (content.starts_with(border)) content.remove_prefix(border.size());
    if (content.ends_with(border)) content.remove_suffix(border.size());
    return trimmed(content);
}

/// The column of the sign of every count token of @p sign in @p line.
std::vector<std::size_t> count_sign_columns(std::string_view line, char sign) {
    std::vector<std::size_t> columns;
    for (std::size_t i = 0; i + 1 < line.size(); ++i) {
        if (line[i] != sign) continue;
        if (!std::isdigit(static_cast<unsigned char>(line[i + 1]))) continue;
        columns.push_back(i);
    }
    return columns;
}

/// The column of the kind letter of @p line, if it has one.
std::optional<std::size_t> kind_column(std::string_view line) {
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char letter = line[i];
        if (letter != 'M' && letter != 'A' && letter != 'D' && letter != 'R') {
            continue;
        }
        const bool blank_before = i == 0 || line[i - 1] == ' ';
        const bool blank_after = i + 1 == line.size() || line[i + 1] == ' ';
        if (blank_before && blank_after) return i;
    }
    return std::nullopt;
}

} // namespace

TEST_CASE("The turn's file changes follow the answer, one collapsed row per file",
          "[tui][turn_file_changes]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = sample_changes();

    const auto text = render_text({answer});
    CHECK_THAT(text, ContainsSubstring("3 files changed  1 added · 1 modified · 1 deleted"));
    // The totals close the summary line, in the columns the rows below use.
    CHECK_THAT(box_content(summary_line(text)), EndsWith("+3 -1"));
    CHECK_THAT(text, ContainsSubstring("▶ src/app.cpp"));
    CHECK_THAT(text, ContainsSubstring("M  +2 -1"));
    CHECK_THAT(text, ContainsSubstring("▶ docs/guide.md"));
    CHECK_THAT(text, ContainsSubstring("A  +1"));
    CHECK_THAT(text, ContainsSubstring("• assets/logo.png  binary"));
    // The row without counts keeps the column of the ones that have them.
    CHECK_THAT(text, ContainsSubstring(" D "));
    CHECK(text.find("All done.") < text.find("3 files changed"));
    CHECK(text.find("new-app-line") == std::string::npos);
    CHECK(text.find("guide-body-line") == std::string::npos);
}

TEST_CASE("The counts of every file change share one column",
          "[tui][turn_file_changes]") {
    auto answer = make_assistant_message("All done.", "", false);
    // The totals are wider than any single row's count, so the columns only
    // hold if the sign comes first in every one of them.
    answer.turn_changes.files = {
        FileChange{.kind = FileChangeKind::Modified, .path = "src/app.cpp",
                   .added = 90, .deleted = 1},
        FileChange{.kind = FileChangeKind::Added, .path = "docs/guide.md",
                   .added = 13},
        FileChange{.kind = FileChangeKind::Deleted,
                   .content = FileChangeContent::Binary,
                   .path = "assets/logo.png"},
        FileChange{.kind = FileChangeKind::Modified, .path = "src/other.cpp",
                   .added = 5, .deleted = 40},
    };

    const auto text = render_text({answer});
    std::vector<std::size_t> kinds;
    std::vector<std::size_t> added;
    std::vector<std::size_t> deleted;
    for (const auto& line : split_lines(text)) {
        if (const auto column = kind_column(line); column.has_value()) {
            kinds.push_back(*column);
        }
        for (const auto column : count_sign_columns(line, '+')) added.push_back(column);
        for (const auto column : count_sign_columns(line, '-')) deleted.push_back(column);
    }

    // The summary's totals and the four rows share the columns: the sign of
    // every count stands in the same place, whichever row it belongs to.
    REQUIRE(added.size() == 4);
    REQUIRE(deleted.size() == 3);
    REQUIRE(kinds.size() == 4);
    for (const auto column : added) CHECK(column == added.front());
    for (const auto column : deleted) CHECK(column == deleted.front());
    for (const auto column : kinds) CHECK(column == kinds.front());
}

TEST_CASE("File change counts omit zero directions in rows and summary",
          "[tui][turn_file_changes]") {
    for (const auto& [added, deleted, expected] :
         std::vector<std::tuple<std::size_t, std::size_t, std::string>>{
             {3, 0, "+3"}, {0, 4, "-4"}, {3, 4, "+3 -4"}}) {
        auto answer = make_assistant_message("Done.", "", false);
        answer.turn_changes.files = {
            FileChange{.kind = FileChangeKind::Modified, .path = "file.txt",
                       .added = added, .deleted = deleted},
        };
        const auto text = render_text({answer});
        CHECK_THAT(text, ContainsSubstring("1 file changed"));
        CHECK_THAT(box_content(summary_line(text)), EndsWith(expected));
        CHECK_THAT(text, ContainsSubstring("M  " + expected));
        CHECK(text.find("+0") == std::string::npos);
        CHECK(text.find("-0") == std::string::npos);
    }
}

TEST_CASE("A renamed file shows its kind letter and previous path",
          "[tui][turn_file_changes]") {
    auto answer = make_assistant_message("Renamed.", "", false);
    answer.turn_changes.files = {
        FileChange{
            .kind = FileChangeKind::Renamed,
            .path = "src/new_name.cpp",
            .previous_path = "src/old_name.cpp",
            .diff = "--- a/src/old_name.cpp\n+++ b/src/new_name.cpp\n@@ -1 +1 @@\n-old\n+new\n",
            .added = 1,
            .deleted = 1,
        },
    };

    const auto text = render_text({answer});
    CHECK_THAT(text, ContainsSubstring("1 file changed  1 renamed"));
    CHECK_THAT(box_content(summary_line(text)), EndsWith("+1 -1"));
    CHECK_THAT(text, ContainsSubstring("▶ src/new_name.cpp  from src/old_name.cpp"));
    CHECK_THAT(text, ContainsSubstring("R  +1 -1"));
}

TEST_CASE("Every change kind is visible even without line counts",
          "[tui][turn_file_changes]") {
    const std::vector<std::pair<FileChangeKind, std::string>> kinds{
        {FileChangeKind::Added, "A"},
        {FileChangeKind::Modified, "M"},
        {FileChangeKind::Deleted, "D"},
        {FileChangeKind::Renamed, "R"},
    };
    for (const auto& [kind, letter] : kinds) {
        CAPTURE(letter);
        auto answer = make_assistant_message("Done.", "", false);
        answer.turn_changes.files = {
            FileChange{.kind = kind, .path = "新しい.cpp", .previous_path = "古い.cpp"},
        };
        const auto text = render_text({answer}, {}, 40);
        CHECK_THAT(text, ContainsSubstring("• 新しい.cpp"));
        CHECK_THAT(text, ContainsSubstring(" " + letter + "│"));
        CHECK(text.find("+0") == std::string::npos);
        CHECK(text.find("-0") == std::string::npos);
        if (kind == FileChangeKind::Renamed) {
            CHECK_THAT(text, ContainsSubstring("from 古い.cpp"));
        }
    }
}

TEST_CASE("File status and comparer remain visible beside long paths",
          "[tui][turn_file_changes][diff_comparer]") {
    auto answer = make_assistant_message("Done.", "", false);
    auto change = sample_changes().front();
    change.kind = FileChangeKind::Renamed;
    change.path = std::string(120, 'x') + "/新しい.cpp";
    change.previous_path = std::string(120, 'y') + "/古い.cpp";
    answer.turn_changes.files = {change};
    answer.turn_changes.reverted = true;

    for (const int width : {40, 60, 100}) {
        CAPTURE(width);
        std::unordered_map<std::string, ftxui::Box> hitboxes;
        ConversationRenderOptions options;
        options.show_diff_comparer_affordance = true;
        options.system_disclosure_hitboxes = &hitboxes;
        const auto text = render_text({answer}, options, width);
        CHECK_THAT(text, ContainsSubstring("R  +2 -1  ↗"));
        CHECK_THAT(text, ContainsSubstring("↶ undone"));
        const auto& button = hitboxes.at(turn_file_change_open_key(answer.id, change.path));
        const auto& row = hitboxes.at(turn_file_change_key(answer.id, change.path));
        CHECK(button.x_min == button.x_max);
        CHECK(button.x_min >= row.x_min);
        CHECK(button.x_max <= row.x_max);
        CHECK(button.x_max < width - 1);
        CHECK(button.y_min == row.y_min);
    }
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
        const bool listed = i < kMaxRenderedFileChanges;
        answer.turn_changes.files.push_back(FileChange{
            .kind = FileChangeKind::Modified,
            .content = FileChangeContent::BudgetSpent,
            .path = std::format("src/f{:03}.cpp", i),
            // The files past the cap carry counts of their own, so the header
            // is the one place their numbers can still be seen.
            .added = listed ? std::size_t{1} : std::size_t{100},
            .deleted = listed ? std::size_t{1} : std::size_t{50},
        });
    }

    const auto text = render_text({answer});
    CHECK_THAT(text, ContainsSubstring(std::format("{} files changed", total)));
    CHECK_THAT(text, ContainsSubstring("5 more not listed"));
    CHECK_THAT(text, ContainsSubstring("src/f000.cpp"));
    CHECK(text.find("src/f029.cpp") == std::string::npos);
    // The total covers every file, not only the drawn rows: 25 at +1 -1 and 5
    // at +100 -50. A turn too large to list is still a turn whose size can be
    // read off the summary.
    CHECK_THAT(text, ContainsSubstring("+525"));
    CHECK_THAT(text, ContainsSubstring("-275"));
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
    CHECK_THAT(text, ContainsSubstring("D  -1234"));
    CHECK_THAT(text, ContainsSubstring("diff omitted"));
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

TEST_CASE("A modification too large to diff still shows its counts",
          "[tui][turn_file_changes]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = {
        FileChange{.kind = FileChangeKind::Modified,
                   .content = FileChangeContent::TooLarge,
                   .path = "src/tui/MainApp.cpp",
                   .added = 12,
                   .deleted = 3},
    };

    const auto text = render_text({answer});
    CHECK_THAT(text, ContainsSubstring("too large to diff"));
    CHECK_THAT(text, ContainsSubstring("M  +12 -3"));
    // Still not expandable: the counts are exact, the diff is not there.
    CHECK_THAT(text, ContainsSubstring("\u2022 src/tui/MainApp.cpp"));
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

TEST_CASE("A resumed multi-file review keeps every diff across long-running turns",
          "[changes][recovery][tui][turn_file_changes][diff_comparer][regression]") {
    namespace fs = std::filesystem;
    using core::changes::MutationScope;
    using core::changes::TurnChangeTracker;
    using core::changes::TurnRecovery;
    const auto temporary = fs::temp_directory_path()
        / std::format("filo_multifile_review_{}", std::random_device{}());
    struct Cleanup {
        fs::path path;
        ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); }
    } cleanup{temporary};
    const auto root = temporary / "workspace";
    const auto storage = temporary / "checkpoints";
    fs::create_directories(root);
    const auto put = [&](const std::string& name, const std::string& text) {
        std::ofstream(root / name) << text;
    };
    constexpr int changed_files = 7;
    for (int i = 0; i < changed_files; ++i) put(std::format("source{}.swift", i), "original\n");
    for (int i = 0; i < 40; ++i) put(std::format("untouched{}.swift", i), std::string(255, 'x') + "\n");
    TurnChangeTracker::Limits limits;
    limits.max_tracked_bytes = 1024; // Changes fit, the entire workspace does not.
    const auto run = [&](std::string_view text, bool wait_for_build) {
        TurnChangeTracker tracker(root, limits);
        tracker.set_recovery(std::make_shared<TurnRecovery>(
            core::changes::RecoveryStore::new_directory(storage), root));
        tracker.track(MutationScope{.unbounded = true, .tool = "run_terminal_command"}, [&] {
            for (int i = 0; i < changed_files; ++i) put(std::format("source{}.swift", i), std::string(text));
            if (wait_for_build) {
                std::this_thread::sleep_for(core::utils::FileStamp::kSettleTime + std::chrono::milliseconds(200));
            }
            return 0;
        });
        tracker.settle();
        tracker.finish_recovery();
        return tracker.changes();
    };
    core::llm::Message first{.role = "assistant", .content = "First build finished."};
    first.turn_changes = run("intermediate\n", true);
    core::llm::Message last{.role = "assistant", .content = "Second build finished."};
    last.turn_changes = run("final\n", false);
    last.turn_changes.id = "second-turn";
    REQUIRE(last.turn_changes.files.size() == changed_files);
    CHECK_FALSE(last.turn_changes.partial_enumeration);

    // Exercise the saved representation before rebuilding the transcript.
    std::string json;
    core::changes::append_turn_changes_json(json, last.turn_changes);
    simdjson::dom::parser parser;
    simdjson::dom::object object;
    REQUIRE(parser.parse(json).get(object) == simdjson::SUCCESS);
    last.turn_changes = core::changes::parse_turn_changes_json(object);
    core::session::SessionData data;
    data.messages = {first, last};
    const auto messages = build_resumed_ui_messages(data);
    const auto latest = latest_turn_with_changes(messages);
    REQUIRE(latest.message);
    const auto comparison = turn_comparison(*latest.message);
    REQUIRE(comparison);
    CHECK(comparison->turn_id == "second-turn");
    CHECK(comparison->undiffed == 0);
    CHECK(comparison->disclosure_keys.size() == changed_files);
    for (int i = 0; i < changed_files; ++i) {
        CHECK_THAT(comparison->patch, ContainsSubstring(std::format(
            "diff --git a/source{0}.swift b/source{0}.swift", i)));
    }
    CHECK_THAT(comparison->patch, ContainsSubstring("-intermediate\n+final\n"));
    CHECK(comparison->patch.find("untouched") == std::string::npos);
    const auto all = conversation_comparison(conversation_turns(messages), root);
    REQUIRE(all);
    CHECK(all->undiffed == 0);
    CHECK_THAT(all->patch, ContainsSubstring("-original\n+final\n"));
    CHECK(all->patch.find("untouched") == std::string::npos);
    CHECK(comparison_omissions(*all).find("outside the observed workspace") != std::string::npos);
}

TEST_CASE("A turn comparison carries its turn and only the shown files' comments",
          "[tui][turn_file_changes][diff_comparer]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = sample_changes();
    answer.turn_changes.id = "turn-1";
    const core::review::Annotation on_app{.id = "app", .path = "src/app.cpp", .comment = "App."};
    const core::review::Annotation on_guide{.id = "guide", .path = "docs/guide.md", .comment = "Guide."};
    answer.turn_changes.annotations = {on_app, on_guide};

    const auto turn = turn_comparison(answer);
    REQUIRE(turn.has_value());
    CHECK(turn->turn_id == "turn-1");
    CHECK(turn->annotations == answer.turn_changes.annotations);

    const auto file = turn_comparison(answer, "docs/guide.md");
    REQUIRE(file.has_value());
    CHECK(file->turn_id == "turn-1");
    CHECK(file->annotations == std::vector{on_guide});
}

TEST_CASE("A turn's id and review comments survive a resume", "[tui][turn_file_changes]") {
    core::changes::TurnChanges changes;
    changes.id = "turn-1";
    changes.files = sample_changes();
    changes.annotations = {{
        .id = "15319332-3c51-4bd8-a391-30de79a90551",
        .path = "src/app.cpp",
        .side = "new",
        .start_line = 2,
        .end_line = 2,
        .selected_text = "new-app-line",
        .comment = "Keep the old line.",
        .revision = "revision-1",
    }};
    std::string json;
    core::changes::append_turn_changes_json(json, changes);

    simdjson::dom::parser parser;
    simdjson::dom::object object;
    REQUIRE(parser.parse(json).get(object) == simdjson::SUCCESS);
    CHECK(core::changes::parse_turn_changes_json(object) == changes);
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

TEST_CASE("The whole conversation compares as one net diff per file",
          "[tui][turn_file_changes][diff_comparer]") {
    namespace fs = std::filesystem;
    const fs::path root = fs::temp_directory_path()
        / std::format("filo_conversation_comparison_{}", std::random_device{}());
    fs::create_directories(root);
    struct Cleanup {
        fs::path path;
        ~Cleanup() { std::error_code ec; fs::remove_all(path, ec); }
    } cleanup{root};
    const auto put = [&](std::string_view name, std::string_view content) {
        std::ofstream(root / name, std::ios::binary) << content;
    };
    const auto run_turn = [&](std::string_view name, std::string_view content) {
        core::changes::TurnChangeTracker tracker(root);
        tracker.track(core::changes::MutationScope{.paths = {root / name}}, [&] {
            put(name, content);
            return 0;
        });
        return tracker.changes();
    };

    put("app.cpp", "one\ntwo\n");
    UiMessage prompt;
    prompt.type = MessageType::User;
    prompt.text = "Start.";
    auto first = make_assistant_message("First.", "", false);
    first.turn_changes = run_turn("app.cpp", "one\nTWO\n");
    auto quiet = make_assistant_message("Nothing touched.", "", false);
    auto second = make_assistant_message("Second.", "", false);
    second.turn_changes = run_turn("app.cpp", "one\nTWO\nthree\n");

    const std::vector<UiMessage> messages{prompt, first, quiet, second};
    const auto source = conversation_turns(messages);
    CHECK(source.turns.size() == 2);
    CHECK(source.first_index == 1);
    CHECK(source.disclosure_keys
          == std::vector<std::string>{turn_file_change_key(first.id, "app.cpp"),
                                      turn_file_change_key(second.id, "app.cpp")});

    SECTION("one section per file, from the original to now") {
        const auto comparison = conversation_comparison(source, root);
        REQUIRE(comparison.has_value());
        CHECK(comparison->whole_conversation);
        CHECK(comparison->patch.find("diff --git") == comparison->patch.rfind("diff --git"));
        CHECK_THAT(comparison->patch, ContainsSubstring("-two\n"));
        CHECK_THAT(comparison->patch, ContainsSubstring("+TWO\n+three\n"));
        CHECK(comparison_omissions(*comparison).empty());
    }
    SECTION("a file edited by hand since is named, not folded in") {
        put("app.cpp", "one\nTWO\nthree, by hand\n");
        const auto comparison = conversation_comparison(source, root);
        REQUIRE_FALSE(comparison.has_value());
        CHECK(comparison.error()
              == "Nothing the conversation changed can be compared: "
                 "app.cpp changed outside the conversation.");
    }
    SECTION("a conversation without changes says so") {
        const std::vector<UiMessage> without{prompt, quiet};
        const auto comparison = conversation_comparison(conversation_turns(without), root);
        REQUIRE_FALSE(comparison.has_value());
        CHECK(comparison.error() == "No turn in this conversation has changed files yet.");
    }
}

TEST_CASE("A conversation comparison says what it leaves out",
          "[tui][turn_file_changes][diff_comparer]") {
    TurnComparison comparison{
        .patch = "diff",
        .caveats = {"b.txt changed outside the conversation"},
        .whole_conversation = true,
    };
    CHECK(comparison_omissions(comparison)
          == "The comparison is not the whole conversation: "
             "b.txt changed outside the conversation.");
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

TEST_CASE("A summary names the tools that made it incomplete", "[tui][turn_file_changes]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = sample_changes();
    answer.turn_changes.files.pop_back();  // The binary change, so only the caveat remains.
    answer.turn_changes.unscoped_mutations = true;

    SECTION("a build is named as a build") {
        answer.turn_changes.unscoped_tools = {"run_verification"};
        const auto comparison = turn_comparison(answer);
        REQUIRE(comparison.has_value());
        CHECK(comparison_omissions(*comparison)
              == "The comparison is not the whole turn: files created by run_verification "
                 "are not listed.");
    }
    SECTION("two are joined as a choice") {
        answer.turn_changes.unscoped_tools = {"python", "run_terminal_command"};
        const auto comparison = turn_comparison(answer);
        REQUIRE(comparison.has_value());
        CHECK(comparison_omissions(*comparison)
              == "The comparison is not the whole turn: files created by python or "
                 "run_terminal_command are not listed.");
    }
    SECTION("a long list is cut short rather than read out") {
        answer.turn_changes.unscoped_tools = {
            "python", "run_terminal_command", "run_verification", "mcp__x", "mcp__y"};
        const auto comparison = turn_comparison(answer);
        REQUIRE(comparison.has_value());
        CHECK(comparison_omissions(*comparison)
              == "The comparison is not the whole turn: files created by python, "
                 "run_terminal_command or run_verification and 2 more are not listed.");
    }
    SECTION("a summary recorded before tools were named still says it falls short") {
        const auto comparison = turn_comparison(answer);
        REQUIRE(comparison.has_value());
        CHECK(comparison_omissions(*comparison)
              == "The comparison is not the whole turn: files created by shell, scripts "
                 "or MCP tools are not listed.");
    }
    SECTION("the box draws the name it was given") {
        answer.turn_changes.unscoped_tools = {"run_verification"};
        CHECK_THAT(render_text({answer}),
                   ContainsSubstring("files created by run_verification are not listed"));
    }
}

TEST_CASE("A reverted turn says its summary no longer describes the workspace",
          "[tui][turn_file_changes]") {
    auto answer = make_assistant_message("All done.", "", false);
    answer.turn_changes.files = sample_changes();
    CHECK(render_text({answer}).find("undone") == std::string::npos);

    answer.turn_changes.reverted = true;
    const auto text = render_text({answer});
    CHECK_THAT(text, ContainsSubstring("\u21b6 undone"));
    // The summary is still what the turn did; only its standing changed.
    CHECK_THAT(text, ContainsSubstring("3 files changed"));
}

TEST_CASE("A revert says what went back, or what stopped it",
          "[tui][turn_file_changes][revert]") {
    core::changes::RevertResult result;

    SECTION("one file is named") {
        result.restored = {"src/app.cpp"};
        CHECK(describe_revert(result) == "Reverted src/app.cpp.");
    }
    SECTION("a few files are named") {
        result.restored = {"src/a.cpp", "include/b.hpp"};
        CHECK(describe_revert(result) == "Reverted src/a.cpp and include/b.hpp.");
    }
    SECTION("a large revert is counted instead") {
        result.restored = {"a", "b", "c", "d"};
        CHECK(describe_revert(result) == "Reverted 4 files.");
    }
    SECTION("a refusal names the file and the reason") {
        result.refused = {core::changes::RevertRefusal{
            .path = "src/app.cpp", .reason = "changed since that turn"}};
        CHECK(describe_revert(result)
              == "Nothing was reverted: src/app.cpp changed since that turn.");
    }
    SECTION("only the first refusals are named, and the rest counted") {
        result.refused = {
            core::changes::RevertRefusal{.path = "a.cpp", .reason = "changed since that turn"},
            core::changes::RevertRefusal{.path = "b.png", .reason = "is a binary file"},
            core::changes::RevertRefusal{.path = "c.log", .reason = "kept no diff"},
        };
        CHECK(describe_revert(result)
              == "Nothing was reverted: a.cpp changed since that turn; b.png is a binary "
                 "file. 1 more file refused it.");
    }
    SECTION("a revert that stopped halfway says where") {
        result.restored = {"a.cpp", "b.hpp"};
        result.error = "c.md could not be written: permission denied.";
        CHECK(describe_revert(result)
              == "Reverted a.cpp and b.hpp, then stopped: c.md could not be written: "
                 "permission denied.");
    }
}

TEST_CASE("A revert repeats what its summary could not cover",
          "[tui][turn_file_changes][revert]") {
    core::changes::TurnChanges changes;
    changes.files = sample_changes();
    CHECK(revert_omissions(changes).empty());

    changes.unscoped_mutations = true;
    changes.unscoped_tools = {"run_terminal_command"};
    CHECK(revert_omissions(changes)
          == "That turn's summary was incomplete, so this revert is too: files created "
             "by run_terminal_command are not listed.");

    changes.partial_enumeration = true;
    CHECK_THAT(revert_omissions(changes),
               ContainsSubstring("too many files to detail fully"));
}

