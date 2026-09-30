#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestSessionContext.hpp"
#include "core/changes/FileChangeJson.hpp"
#include "core/changes/MutationScope.hpp"
#include "core/changes/TurnChangeTracker.hpp"
#include "core/session/SessionStore.hpp"
#include "core/tools/ToolDiffUtils.hpp"
#include "core/tools/AskUserQuestionTool.hpp"
#include "core/tools/ListDirectoryTool.hpp"
#include "core/tools/MemoryTool.hpp"
#include "core/tools/ShellTool.hpp"
#include "core/tools/Tool.hpp"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <unistd.h>

using core::changes::FileChange;
using core::changes::FileChangeContent;
using core::changes::FileChangeKind;
using core::changes::MutationScope;
using core::changes::TurnChangeTracker;
using Catch::Matchers::ContainsSubstring;

namespace fs = std::filesystem;

namespace {

struct TempDir {
    fs::path path;
    TempDir() {
        static std::atomic<unsigned> counter{0};
        path = fs::temp_directory_path()
            / std::format("filo_turn_changes_{}_{}_{}", ::getpid(), std::random_device{}(),
                          counter.fetch_add(1));
        fs::create_directories(path);
        path = fs::canonical(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;
};

void put(const fs::path& path, std::string_view content) {
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary) << content;
}

MutationScope scope_of(std::initializer_list<fs::path> paths) {
    return MutationScope{.paths = paths};
}

/// The files a turn reports. The fidelity flags are asserted separately.
std::vector<FileChange> files_of(const TurnChangeTracker& tracker) {
    return tracker.changes().files;
}

// Mirrors how the agent drives the tracker around one tool call.
template <typename Mutation>
void edit(TurnChangeTracker& tracker, const MutationScope& scope, Mutation&& mutate) {
    tracker.track(scope, [&] {
        mutate();
        return 0;
    });
}

core::context::SessionContext context_for(const fs::path& root) {
    return test_support::make_session_context(core::workspace::WorkspaceSnapshot{
        .primary = root,
        .enforce = true,
        .version = 1,
    });
}

} // namespace

TEST_CASE("Repeated edits of one file collapse into its net diff", "[changes]") {
    TempDir dir;
    const auto file = dir.path / "src" / "main.cpp";
    put(file, "one\ntwo\nthree\n");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({file}), [&] { put(file, "one\nTWO\nthree\n"); });
    edit(tracker, scope_of({file}), [&] { put(file, "one\nTWO\nthree\nfour\n"); });

    const auto changes = files_of(tracker);
    REQUIRE(changes.size() == 1);
    CHECK(changes[0].kind == FileChangeKind::Modified);
    CHECK(changes[0].content == FileChangeContent::Text);
    CHECK(changes[0].path == "src/main.cpp");
    CHECK(changes[0].added == 2);
    CHECK(changes[0].deleted == 1);
    CHECK_THAT(changes[0].diff, ContainsSubstring("-two"));
    CHECK_THAT(changes[0].diff, ContainsSubstring("+TWO"));
    CHECK_THAT(changes[0].diff, ContainsSubstring("+four"));
}

TEST_CASE("Edits that restore the original content are not a change", "[changes]") {
    TempDir dir;
    const auto file = dir.path / "a.txt";
    put(file, "original\n");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({file}), [&] { put(file, "changed\n"); });
    edit(tracker, scope_of({file}), [&] { put(file, "original\n"); });

    CHECK(tracker.changes().empty());
}

TEST_CASE("Created, deleted and created-then-deleted files", "[changes]") {
    TempDir dir;
    const auto created = dir.path / "new.txt";
    const auto removed = dir.path / "old.txt";
    const auto transient = dir.path / "tmp.txt";
    put(removed, "bye\n");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({created}), [&] { put(created, "hello\nworld\n"); });
    edit(tracker, scope_of({removed}), [&] { fs::remove(removed); });
    edit(tracker, scope_of({transient}), [&] { put(transient, "x\n"); });
    edit(tracker, scope_of({transient}), [&] { fs::remove(transient); });

    const auto changes = files_of(tracker);
    REQUIRE(changes.size() == 2);
    CHECK(changes[0].path == "new.txt");
    CHECK(changes[0].kind == FileChangeKind::Added);
    CHECK(changes[0].added == 2);
    CHECK(changes[1].path == "old.txt");
    CHECK(changes[1].kind == FileChangeKind::Deleted);
    CHECK(changes[1].deleted == 1);
}

TEST_CASE("A move is a rename carrying any edit made along the way", "[changes]") {
    TempDir dir;
    const auto from = dir.path / "before.txt";
    const auto to = dir.path / "after.txt";
    const auto final_name = dir.path / "final.txt";
    put(from, "alpha\n");
    TurnChangeTracker tracker(dir.path);

    MutationScope move{.paths = {from, to}, .moves = {{from, to}}};
    edit(tracker, move, [&] { fs::rename(from, to); });
    edit(tracker, scope_of({to}), [&] { put(to, "alpha\nbeta\n"); });
    MutationScope second{.paths = {to, final_name}, .moves = {{to, final_name}}};
    edit(tracker, second, [&] { fs::rename(to, final_name); });

    const auto changes = files_of(tracker);
    REQUIRE(changes.size() == 1);
    CHECK(changes[0].kind == FileChangeKind::Renamed);
    CHECK(changes[0].previous_path == "before.txt");
    CHECK(changes[0].path == "final.txt");
    CHECK(changes[0].added == 1);
    CHECK(changes[0].deleted == 0);
}

TEST_CASE("Deleting and moving directories reports each file", "[changes]") {
    TempDir dir;
    put(dir.path / "gone" / "a.txt", "a\n");
    put(dir.path / "gone" / "nested" / "b.txt", "b\n");
    put(dir.path / "src_dir" / "c.txt", "c\n");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({dir.path / "gone"}), [&] { fs::remove_all(dir.path / "gone"); });
    MutationScope move{
        .paths = {dir.path / "src_dir", dir.path / "dst_dir"},
        .moves = {{dir.path / "src_dir", dir.path / "dst_dir"}},
    };
    edit(tracker, move, [&] { fs::rename(dir.path / "src_dir", dir.path / "dst_dir"); });

    const auto changes = files_of(tracker);
    REQUIRE(changes.size() == 3);
    CHECK(changes[0].path == "dst_dir/c.txt");
    CHECK(changes[0].kind == FileChangeKind::Renamed);
    CHECK(changes[0].previous_path == "src_dir/c.txt");
    CHECK(changes[1].path == "gone/a.txt");
    CHECK(changes[1].kind == FileChangeKind::Deleted);
    CHECK(changes[2].path == "gone/nested/b.txt");
    CHECK(changes[2].kind == FileChangeKind::Deleted);
}

TEST_CASE("Binary and oversized files are listed without a diff", "[changes]") {
    TempDir dir;
    const auto binary = dir.path / "image.bin";
    const auto large = dir.path / "large.txt";
    put(binary, std::string("\x00\x01\x02", 3));
    put(large, "small\n");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({binary}), [&] { put(binary, std::string("\x00\x03\x04", 3)); });
    edit(tracker, scope_of({large}), [&] { put(large, std::string(600 * 1024, 'x')); });

    const auto changes = files_of(tracker);
    REQUIRE(changes.size() == 2);
    CHECK(changes[0].path == "image.bin");
    CHECK(changes[0].content == FileChangeContent::Binary);
    CHECK(changes[0].diff.empty());
    CHECK(changes[1].path == "large.txt");
    CHECK(changes[1].content == FileChangeContent::TooLarge);
    CHECK(changes[1].diff.empty());
    CHECK(changes[1].added == 0);
}

TEST_CASE("A mutation that throws is still observed", "[changes]") {
    TempDir dir;
    const auto file = dir.path / "partial.txt";
    TurnChangeTracker tracker(dir.path);

    CHECK_THROWS_AS(tracker.track(scope_of({file}), [&]() -> int {
        put(file, "half-written\n");
        throw std::runtime_error("tool crashed");
    }), std::runtime_error);

    const auto changes = files_of(tracker);
    REQUIRE(changes.size() == 1);
    CHECK(changes[0].kind == FileChangeKind::Added);
}

TEST_CASE("Files outside the workspace keep their absolute path", "[changes]") {
    TempDir workspace;
    TempDir outside;
    const auto file = outside.path / "elsewhere.txt";
    TurnChangeTracker tracker(workspace.path);

    edit(tracker, scope_of({file}), [&] { put(file, "x\n"); });

    const auto changes = files_of(tracker);
    REQUIRE(changes.size() == 1);
    CHECK(changes[0].path == file.generic_string());
}

TEST_CASE("Built-in file tools scope to the exact paths they touch", "[changes]") {
    TempDir dir;
    const auto context = context_for(dir.path);
    using core::changes::mutation_scope;
    // A tool with a rule is scoped by that rule, so what it declares about
    // itself is never consulted.
    const core::tools::ToolAnnotations unstated{};
    const auto scope = [&](std::string_view tool, std::string_view args) {
        return mutation_scope(tool, args, context, unstated);
    };

    CHECK(scope("write_file", R"({"file_path":"a.txt","content":""})").paths
          == std::vector<fs::path>{dir.path / "a.txt"});
    CHECK(scope("replace", R"({"file_path":"b.txt"})").paths
          == std::vector<fs::path>{dir.path / "b.txt"});
    CHECK(scope("search_replace", R"({"file_path":"c.txt"})").paths
          == std::vector<fs::path>{dir.path / "c.txt"});
    CHECK(scope("delete_file", R"({"file_path":"d"})").paths
          == std::vector<fs::path>{dir.path / "d"});

    const auto move = scope("move_file", R"({"source":"x.txt","destination":"y/x.txt"})");
    CHECK(move.paths == std::vector<fs::path>{dir.path / "x.txt", dir.path / "y/x.txt"});
    REQUIRE(move.moves.size() == 1);
    CHECK(move.moves[0].second == dir.path / "y/x.txt");

    fs::create_directories(dir.path / "sub");
    const auto patch = scope(
        "patch",  // alias of apply_patch
        R"({"working_dir":"sub","patch":"--- /dev/null\n+++ b/new.txt\n@@ -0,0 +1 @@\n+x\n--- a/old.txt\n+++ b/old.txt\n"})");
    CHECK(patch.paths == std::vector<fs::path>{
        dir.path / "sub" / "new.txt", dir.path / "sub" / "old.txt", dir.path / "sub" / "old.txt"});

    // Understood, and unable to change the content of any file: nothing to
    // observe, and no reason to doubt the rest of the summary.
    CHECK(scope("create_directory", R"({"dir_path":"gen"})").empty());
    CHECK(scope("write_file", R"({"content":"no path"})").empty());
}

TEST_CASE("A tool with no rule is judged by the contract it declares", "[changes]") {
    TempDir dir;
    const auto context = context_for(dir.path);
    using core::changes::mutation_scope;

    // Read-only by its own declaration, so it cannot change a summary.
    const core::tools::ListDirectoryTool list;
    CHECK(mutation_scope("list_directory", R"({"path":"."})", context,
                         list.get_definition().annotations)
              .empty());

    // Prompting the user writes nothing either.
    const core::tools::AskUserQuestionTool question;
    CHECK(mutation_scope("ask_user_question", R"({"questions":[]})", context,
                         question.get_definition().annotations)
              .empty());

    // A shell command may write anywhere, and where cannot be enumerated up
    // front, so it is unbounded rather than silently ignored.
    const core::tools::ShellTool shell;
    const auto unbounded = mutation_scope(
        "run_terminal_command", R"({"command":"rm -rf x"})", context,
        shell.get_definition().annotations);
    CHECK(unbounded.paths.empty());
    CHECK(unbounded.unbounded);
    CHECK_FALSE(unbounded.empty());

    // A tool Filo cannot resolve at all is assumed able to write.
    const core::tools::ToolAnnotations unstated{};
    CHECK(mutation_scope("some_mcp_tool", "{}", context, unstated).unbounded);
}

TEST_CASE("File changes survive a session save and load", "[changes][session]") {
    TempDir dir;
    core::session::SessionStore store{dir.path};
    core::session::SessionData data;
    data.session_id = "filechg1";
    data.created_at = "2026-09-29T10:00:00Z";
    data.last_active_at = data.created_at;
    core::llm::Message user{.role = "user", .content = "edit"};
    core::llm::Message assistant{.role = "assistant", .content = "done"};
    assistant.turn_changes.files = {
        FileChange{
            .kind = FileChangeKind::Renamed,
            .path = "b.txt",
            .previous_path = "a.txt",
            .diff = "--- a/b.txt\n+++ b/b.txt\n@@ -1 +1 @@\n-x\n+\"y\"\n",
            .added = 1,
            .deleted = 1,
        },
        FileChange{.kind = FileChangeKind::Added, .content = FileChangeContent::Binary,
                   .path = "logo.png"},
    };
    assistant.turn_changes.unscoped_mutations = true;
    assistant.turn_changes.unscoped_tools = {"run_terminal_command", "a post-tool hook"};
    assistant.turn_changes.reverted = true;
    data.messages = {user, assistant};

    REQUIRE(store.save(data));
    const auto loaded = store.load_by_id(data.session_id);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->messages.size() == 2);
    CHECK(loaded->messages[0].turn_changes.empty());
    CHECK(loaded->messages[1].turn_changes == assistant.turn_changes);
}

TEST_CASE("An unscoped mutation cannot leave an earlier diff stale", "[changes]") {
    TempDir dir;
    const auto file = dir.path / "notes.txt";
    put(file, "one\n");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({file}), [&] { put(file, "one\ntwo\n"); });
    // A shell command then appends behind the tracker's back.
    const MutationScope unbounded{.unbounded = true};
    edit(tracker, unbounded, [&] { put(file, "one\ntwo\nthree\n"); });

    const auto summary = tracker.changes();
    CHECK(summary.unscoped_mutations);
    REQUIRE(summary.files.size() == 1);
    // Both appended lines: the summary describes the file as it now is, not as
    // it stood when the last scoped tool finished.
    CHECK(summary.files[0].added == 2);
    CHECK(summary.files[0].deleted == 0);
    CHECK_THAT(summary.files[0].diff, ContainsSubstring("+three"));
}

TEST_CASE("An unscoped mutation leaves an untouched file alone", "[changes]") {
    TempDir dir;
    const auto file = dir.path / "notes.txt";
    put(file, "one\n");
    TurnChangeTracker tracker(dir.path);
    edit(tracker, scope_of({file}), [&] { put(file, "one\ntwo\n"); });

    // Nothing on disk moves, so reconciling must not invent a change.
    const MutationScope unbounded{.unbounded = true};
    edit(tracker, unbounded, [] {});

    const auto summary = tracker.changes();
    REQUIRE(summary.files.size() == 1);
    CHECK(summary.files[0].added == 1);
    CHECK(summary.unscoped_mutations);
}

TEST_CASE("A file an unscoped tool creates is flagged, not silently missed", "[changes]") {
    TempDir dir;
    TurnChangeTracker tracker(dir.path);
    const MutationScope unbounded{.unbounded = true};
    edit(tracker, unbounded, [&] { put(dir.path / "generated.txt", "x\n"); });

    const auto summary = tracker.changes();
    CHECK(summary.files.empty());
    CHECK(summary.unscoped_mutations);
    CHECK_FALSE(summary.complete());
}

TEST_CASE("A turn of scoped edits calls itself complete", "[changes]") {
    TempDir dir;
    const auto file = dir.path / "a.txt";
    TurnChangeTracker tracker(dir.path);
    edit(tracker, scope_of({file}), [&] { put(file, "x\n"); });

    const auto summary = tracker.changes();
    REQUIRE(summary.files.size() == 1);
    CHECK(summary.complete());
}

TEST_CASE("A directory too large to enumerate says so instead of looking complete",
          "[changes]") {
    TempDir dir;
    constexpr std::size_t kCap = 3;
    for (std::size_t i = 0; i <= kCap; ++i) {
        put(dir.path / "tree" / std::format("f{}.txt", i), "x\n");
    }
    TurnChangeTracker tracker(dir.path, {.max_files_per_directory = kCap});
    edit(tracker, scope_of({dir.path / "tree"}),
         [&] { fs::remove_all(dir.path / "tree"); });

    const auto summary = tracker.changes();
    CHECK(summary.partial_enumeration);
    CHECK(summary.files.size() == kCap);  // Listed up to the cap, gap declared.
}

TEST_CASE("The diff budget bounds a summary without dropping files from it", "[changes]") {
    TempDir dir;
    const std::string body(64, 'x');
    constexpr std::size_t kFiles = 6;
    for (std::size_t i = 0; i < kFiles; ++i) {
        put(dir.path / "bulk" / std::format("f{}.txt", i), body + "\n");
    }
    TurnChangeTracker tracker(dir.path, {.max_diff_bytes = 300});
    edit(tracker, scope_of({dir.path / "bulk"}),
         [&] { fs::remove_all(dir.path / "bulk"); });

    const auto summary = tracker.changes();
    std::size_t diff_bytes = 0;
    std::size_t detailed = 0;
    for (const auto& change : summary.files) {
        diff_bytes += change.diff.size();
        detailed += change.content == FileChangeContent::Text ? 1 : 0;
    }
    CHECK(summary.files.size() == kFiles);  // Every file is still listed.
    CHECK(diff_bytes <= 300);
    CHECK(detailed > 0);
    CHECK(detailed < kFiles);
    CHECK(summary.partial_enumeration);
    CHECK(std::ranges::any_of(summary.files, [](const FileChange& change) {
        return change.content == FileChangeContent::BudgetSpent;
    }));
}

TEST_CASE("Content past the retention budget is still reported as a change", "[changes]") {
    TempDir dir;
    const std::string body(512, 'x');
    put(dir.path / "big1.txt", body);
    put(dir.path / "big2.txt", body);
    // Room for the first file's baseline and current state, not the second's.
    TurnChangeTracker tracker(dir.path, {.max_tracked_bytes = 1200});

    edit(tracker, scope_of({dir.path / "big1.txt"}),
         [&] { put(dir.path / "big1.txt", body + "y"); });
    edit(tracker, scope_of({dir.path / "big2.txt"}),
         [&] { put(dir.path / "big2.txt", body + "z"); });

    const auto summary = tracker.changes();
    REQUIRE(summary.files.size() == 2);
    // The first file kept its detail; the second is known to have changed by
    // its stamp alone, which is enough to list it but not to diff it.
    CHECK(summary.files[0].path == "big1.txt");
    CHECK(summary.files[0].content == FileChangeContent::Text);
    CHECK(summary.files[0].added == 1);
    CHECK(summary.files[1].path == "big2.txt");
    CHECK(summary.files[1].content == FileChangeContent::BudgetSpent);
    CHECK(summary.files[1].diff.empty());
    CHECK(summary.partial_enumeration);
}

TEST_CASE("The fidelity flags survive a session save and load", "[changes][session]") {
    TempDir dir;
    core::session::SessionStore store{dir.path};
    core::session::SessionData data;
    data.session_id = "filechg2";
    data.created_at = "2026-09-29T10:00:00Z";
    data.last_active_at = data.created_at;
    core::llm::Message assistant{.role = "assistant", .content = "done"};
    assistant.turn_changes.files = {
        FileChange{.kind = FileChangeKind::Modified,
                   .content = FileChangeContent::BudgetSpent,
                   .path = "a.txt"},
    };
    assistant.turn_changes.partial_enumeration = true;
    assistant.turn_changes.unscoped_mutations = true;
    data.messages = {assistant};

    REQUIRE(store.save(data));
    const auto loaded = store.load_by_id(data.session_id);
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->messages.size() == 1);
    CHECK(loaded->messages[0].turn_changes == assistant.turn_changes);
}

TEST_CASE("A file whose content was not retained is still counted", "[changes]") {
    TempDir dir;
    put(dir.path / "gone.txt", "one\ntwo\nthree\n");
    // Nothing may be retained, so no diff is possible. The size of the change
    // still is: counting lines does not require keeping the bytes.
    TurnChangeTracker tracker(dir.path, {.max_tracked_bytes = 0});

    edit(tracker, scope_of({dir.path / "gone.txt"}),
         [&] { fs::remove(dir.path / "gone.txt"); });
    edit(tracker, scope_of({dir.path / "new.txt"}),
         [&] { put(dir.path / "new.txt", "a\nb\n"); });

    const auto summary = tracker.changes();
    REQUIRE(summary.files.size() == 2);
    CHECK(summary.partial_enumeration);

    const auto& gone = summary.files[0];
    CHECK(gone.path == "gone.txt");
    CHECK(gone.kind == FileChangeKind::Deleted);
    CHECK(gone.content == FileChangeContent::BudgetSpent);
    CHECK(gone.diff.empty());
    CHECK(gone.deleted == 3);
    CHECK(gone.added == 0);

    const auto& created = summary.files[1];
    CHECK(created.path == "new.txt");
    CHECK(created.kind == FileChangeKind::Added);
    CHECK(created.added == 2);
}

TEST_CASE("A modification without a diff reports no counts rather than guessing",
          "[changes]") {
    TempDir dir;
    const auto file = dir.path / "a.txt";
    put(file, "one\ntwo\n");
    TurnChangeTracker tracker(dir.path, {.max_tracked_bytes = 0});

    edit(tracker, scope_of({file}), [&] { put(file, "one\nTWO\nthree\n"); });

    const auto summary = tracker.changes();
    REQUIRE(summary.files.size() == 1);
    CHECK(summary.files[0].kind == FileChangeKind::Modified);
    // Additions and deletions cannot be separated without diffing the two.
    CHECK(summary.files[0].added == 0);
    CHECK(summary.files[0].deleted == 0);
}

TEST_CASE("An oversized file is counted even though it cannot be diffed", "[changes]") {
    TempDir dir;
    const auto big = dir.path / "big.log";
    std::string content;
    const std::string line(64, 'x');
    while (content.size() <= core::tools::detail::kMaxToolDiffInputBytes) {
        content += line + "\n";
    }
    const auto expected_lines =
        static_cast<std::size_t>(std::ranges::count(content, '\n'));

    TurnChangeTracker tracker(dir.path);
    edit(tracker, scope_of({big}), [&] { put(big, content); });

    const auto summary = tracker.changes();
    REQUIRE(summary.files.size() == 1);
    CHECK(summary.files[0].content == FileChangeContent::TooLarge);
    CHECK(summary.files[0].kind == FileChangeKind::Added);
    CHECK(summary.files[0].added == expected_lines);
    // One file too big to diff describes itself; it is not a gap in the summary.
    CHECK_FALSE(summary.partial_enumeration);
}

TEST_CASE("A binary file too large to read is named binary, not a budget casualty",
          "[changes]") {
    TempDir dir;
    const auto blob = dir.path / "blob.bin";
    std::string content(600 * 1024, '\0');
    TurnChangeTracker tracker(dir.path);
    edit(tracker, scope_of({blob}), [&] { put(blob, content); });

    const auto summary = tracker.changes();
    REQUIRE(summary.files.size() == 1);
    CHECK(summary.files[0].content == FileChangeContent::Binary);
    CHECK(summary.files[0].added == 0);
    CHECK_FALSE(summary.partial_enumeration);
}

TEST_CASE("A directory moved onto an existing directory reports renames, not deletions",
          "[changes]") {
    // The files a move brings into an existing directory are paths the tracker
    // has never seen. Seen for the first time after the mutation, they cannot be
    // assumed to have existed before it, or the move reads as a deletion of the
    // source and the destination silently swallows what arrived.
    for (const bool occupied : {false, true}) {
        DYNAMIC_SECTION("the destination "
                        << (occupied ? "already held files" : "was empty")) {
            TempDir dir;
            put(dir.path / "src_dir" / "c.txt", "c\n");
            put(dir.path / "src_dir" / "nested" / "d.txt", "d\n");
            fs::create_directories(dir.path / "dst_dir");
            if (occupied) {
                put(dir.path / "dst_dir" / "existing.txt", "e\n");
            }
            TurnChangeTracker tracker(dir.path);

            MutationScope move{
                .paths = {dir.path / "src_dir", dir.path / "dst_dir"},
                .moves = {{dir.path / "src_dir", dir.path / "dst_dir"}},
            };
            edit(tracker, move, [&] {
                std::error_code ec;
                fs::rename(dir.path / "src_dir", dir.path / "dst_dir", ec);
                if (!ec) {
                    return;
                }
                // What the move tool does when a rename cannot replace the
                // destination: merge the tree in, then remove the source.
                fs::copy(dir.path / "src_dir", dir.path / "dst_dir",
                         fs::copy_options::recursive
                             | fs::copy_options::overwrite_existing, ec);
                if (!ec) {
                    fs::remove_all(dir.path / "src_dir", ec);
                }
            });

            const auto changes = tracker.changes();
            REQUIRE(changes.files.size() == 2);
            CHECK(changes.files[0].kind == FileChangeKind::Renamed);
            CHECK(changes.files[0].path == "dst_dir/c.txt");
            CHECK(changes.files[0].previous_path == "src_dir/c.txt");
            CHECK(changes.files[1].kind == FileChangeKind::Renamed);
            CHECK(changes.files[1].path == "dst_dir/nested/d.txt");
            CHECK(changes.files[1].previous_path == "src_dir/nested/d.txt");
            CHECK_FALSE(changes.partial_enumeration);
            // What already lived in the destination is not this turn's doing.
            CHECK(std::ranges::none_of(changes.files, [](const FileChange& change) {
                return change.path == "dst_dir/existing.txt";
            }));
        }
    }
}

TEST_CASE("A file discovered under a scope after the mutation is reported, not swallowed",
          "[changes]") {
    TempDir dir;
    fs::create_directories(dir.path / "out");
    TurnChangeTracker tracker(dir.path);

    // A tool whose scope is a directory, and whose mutation adds a file to it:
    // the tracker meets that path for the first time after the write.
    const auto scope = scope_of({dir.path / "out"});
    edit(tracker, scope, [&] { put(dir.path / "out" / "generated.txt", "made\n"); });

    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    CHECK(changes.files[0].kind == FileChangeKind::Added);
    CHECK(changes.files[0].path == "out/generated.txt");
    CHECK(changes.files[0].added == 1);
}

TEST_CASE("The memory tool cannot make a summary incomplete", "[changes]") {
    using core::changes::mutation_scope;
    TempDir dir;
    const auto context = context_for(dir.path);
    const core::tools::MemoryTool memory;
    const auto scope = mutation_scope("memory", R"({"action":"remember"})", context,
                                      memory.get_definition().annotations);
    CHECK(scope.empty());
    CHECK_FALSE(scope.unbounded);

    // Its store is Filo's own config file, outside every workspace, so a turn
    // that remembered something still knows every file it changed.
    const auto file = dir.path / "a.txt";
    put(file, "one\n");
    TurnChangeTracker tracker(dir.path);
    edit(tracker, scope_of({file}), [&] { put(file, "two\n"); });
    edit(tracker, scope, [] {});

    const auto changes = tracker.changes();
    CHECK(changes.complete());
    CHECK(changes.unscoped_tools.empty());
    REQUIRE(changes.files.size() == 1);
}

TEST_CASE("An unscoped summary names the tools behind it, once each", "[changes]") {
    using core::changes::mutation_scope;
    TempDir dir;
    const auto context = context_for(dir.path);
    const core::tools::ShellTool shell;
    const auto annotations = shell.get_definition().annotations;
    const auto scope = mutation_scope("run_terminal_command", R"({"command":"make"})",
                                      context, annotations);
    CHECK(scope.unbounded);
    CHECK(scope.tool == "run_terminal_command");
    // An alias is named as the tool it stands for.
    CHECK(mutation_scope("shell", R"({"command":"make"})", context, annotations).tool
          == "run_terminal_command");

    TurnChangeTracker tracker(dir.path);
    edit(tracker, scope, [] {});
    edit(tracker, scope, [] {});
    tracker.note_unscoped_mutation("a post-tool hook");

    const auto changes = tracker.changes();
    CHECK_FALSE(changes.complete());
    CHECK(changes.unscoped_tools == std::vector<std::string>{"run_terminal_command",
                                                            "a post-tool hook"});
}

TEST_CASE("A rewrite that changes nothing is not a change, retained or not", "[changes]") {
    TempDir dir;
    const auto file = dir.path / "a.txt";
    const std::string content(64 * 1024, 'x');
    put(file, content);
    // Nothing is retained, so the file is known only by what a bounded read
    // says about it: its lines, and the identity of the bytes they came from.
    TurnChangeTracker tracker(dir.path, {.max_tracked_bytes = 0});

    edit(tracker, scope_of({file}), [] {});
    CHECK(tracker.changes().files.empty());

    // Only mtime moves. Reporting this would report a formatter that changed
    // nothing, or a build that rewrote a file byte for byte.
    edit(tracker, scope_of({file}), [&] { put(file, content); });
    CHECK(tracker.changes().files.empty());

    // A real edit is still seen, and still counts as one.
    edit(tracker, scope_of({file}), [&] { put(file, content + "tail\n"); });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    CHECK(changes.files[0].kind == FileChangeKind::Modified);
}

TEST_CASE("Settling takes one more look only for a turn something wrote behind its back",
          "[changes]") {
    TempDir dir;
    const auto file = dir.path / "a.txt";
    put(file, "one\n");
    TurnChangeTracker tracker(dir.path);
    edit(tracker, scope_of({file}), [&] { put(file, "two\n"); });

    // Somebody edits the file by hand after the last observation. A turn that
    // saw nothing unscoped must not claim that edit as its own.
    put(file, "by hand\n");
    tracker.settle();
    auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    CHECK_THAT(changes.files[0].diff, ContainsSubstring("+two"));
    CHECK_THAT(changes.files[0].diff, !ContainsSubstring("by hand"));

    // A detached hook, though, is the turn's own doing: settling after it
    // reports what the file holds when the turn is reported.
    tracker.note_unscoped_mutation("a post-tool hook");
    put(file, "formatted\n");
    tracker.settle();
    changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    CHECK_THAT(changes.files[0].diff, ContainsSubstring("+formatted"));
    CHECK(changes.unscoped_tools == std::vector<std::string>{"a post-tool hook"});
}
