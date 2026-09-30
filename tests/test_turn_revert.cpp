#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/changes/TurnChangeTracker.hpp"
#include "core/changes/TurnRevert.hpp"

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

using core::changes::FileChange;
using core::changes::FileChangeContent;
using core::changes::FileChangeKind;
using core::changes::MutationScope;
using core::changes::RevertResult;
using core::changes::TurnChangeTracker;
using core::changes::TurnChanges;
using core::changes::revert_turn;
using Catch::Matchers::ContainsSubstring;

namespace fs = std::filesystem;

namespace {

struct TempDir {
    fs::path path;
    TempDir() {
        static std::atomic<unsigned> counter{0};
        path = fs::temp_directory_path()
            / std::format("filo_turn_revert_{}_{}_{}", ::getpid(), std::random_device{}(),
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

std::string read(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

MutationScope scope_of(std::initializer_list<fs::path> paths) {
    return MutationScope{.paths = paths};
}

// Mirrors how the agent drives the tracker around one tool call.
template <typename Mutation>
void edit(TurnChangeTracker& tracker, const MutationScope& scope, Mutation&& mutate) {
    tracker.track(scope, [&] {
        mutate();
        return 0;
    });
}

/// Every regular file under `root`, as path and content, so a revert can be
/// checked against the whole tree rather than the files a test thought of.
std::vector<std::pair<std::string, std::string>> snapshot_tree(const fs::path& root) {
    std::vector<std::pair<std::string, std::string>> files;
    for (auto it = fs::recursive_directory_iterator(root);
         it != fs::recursive_directory_iterator(); ++it) {
        if (!it->is_regular_file()) {
            continue;
        }
        files.emplace_back(fs::relative(it->path(), root).generic_string(), read(it->path()));
    }
    std::ranges::sort(files);
    return files;
}

/// One refusal, worded the way a transcript would word it.
std::string refusal_of(const RevertResult& result, std::string_view path) {
    for (const auto& refusal : result.refused) {
        if (refusal.path == path) {
            return refusal.reason;
        }
    }
    return {};
}

} // namespace

TEST_CASE("A revert puts an edited file back the way the turn found it", "[changes][revert]") {
    TempDir dir;
    const auto file = dir.path / "src" / "app.cpp";
    put(file, "one\ntwo\nthree\n");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({file}), [&] { put(file, "one\nTWO\nthree\nfour\n"); });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);

    const auto result = revert_turn(changes, dir.path);
    CHECK(result.reverted());
    CHECK(result.restored == std::vector<std::string>{"src/app.cpp"});
    CHECK(read(file) == "one\ntwo\nthree\n");
}

TEST_CASE("A revert removes what a turn created and restores what it deleted",
          "[changes][revert]") {
    TempDir dir;
    put(dir.path / "kept.txt", "kept\n");
    put(dir.path / "gone" / "doomed.txt", "doomed line\nsecond\n");
    const auto before = snapshot_tree(dir.path);
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({dir.path / "new.txt"}),
         [&] { put(dir.path / "new.txt", "created by the turn\n"); });
    edit(tracker, scope_of({dir.path / "gone" / "doomed.txt"}),
         [&] { fs::remove(dir.path / "gone" / "doomed.txt"); });

    const auto result = revert_turn(tracker.changes(), dir.path);
    CHECK(result.reverted());
    CHECK_FALSE(fs::exists(dir.path / "new.txt"));
    CHECK(read(dir.path / "gone" / "doomed.txt") == "doomed line\nsecond\n");
    CHECK(snapshot_tree(dir.path) == before);
}

TEST_CASE("A revert moves a renamed file back, carrying its edits with it",
          "[changes][revert]") {
    TempDir dir;
    put(dir.path / "old" / "a.txt", "one\n");
    TurnChangeTracker tracker(dir.path);

    MutationScope move{
        .paths = {dir.path / "old" / "a.txt", dir.path / "new" / "b.txt"},
        .moves = {{dir.path / "old" / "a.txt", dir.path / "new" / "b.txt"}},
    };
    edit(tracker, move, [&] {
        fs::create_directories(dir.path / "new");
        fs::rename(dir.path / "old" / "a.txt", dir.path / "new" / "b.txt");
    });
    edit(tracker, scope_of({dir.path / "new" / "b.txt"}),
         [&] { put(dir.path / "new" / "b.txt", "one\ntwo\n"); });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    REQUIRE(changes.files[0].kind == FileChangeKind::Renamed);

    const auto result = revert_turn(changes, dir.path);
    CHECK(result.reverted());
    CHECK_FALSE(fs::exists(dir.path / "new" / "b.txt"));
    CHECK(read(dir.path / "old" / "a.txt") == "one\n");
}

TEST_CASE("A pure rename goes back without a diff to read", "[changes][revert]") {
    TempDir dir;
    put(dir.path / "a.txt", "content\n");
    TurnChangeTracker tracker(dir.path);

    MutationScope move{
        .paths = {dir.path / "a.txt", dir.path / "b.txt"},
        .moves = {{dir.path / "a.txt", dir.path / "b.txt"}},
    };
    edit(tracker, move, [&] { fs::rename(dir.path / "a.txt", dir.path / "b.txt"); });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    CHECK(changes.files[0].diff.empty());

    CHECK(revert_turn(changes, dir.path).reverted());
    CHECK(read(dir.path / "a.txt") == "content\n");
    CHECK_FALSE(fs::exists(dir.path / "b.txt"));
}

TEST_CASE("An empty file a turn created or deleted needs no diff to go back",
          "[changes][revert]") {
    TempDir dir;
    put(dir.path / "empty.txt", "");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({dir.path / "empty.txt"}), [&] { fs::remove(dir.path / "empty.txt"); });
    edit(tracker, scope_of({dir.path / "made.txt"}), [&] { put(dir.path / "made.txt", ""); });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 2);
    // Neither side has content, so neither kept a diff.
    CHECK(std::ranges::all_of(changes.files, [](const FileChange& change) {
        return change.diff.empty() && change.content == FileChangeContent::Text;
    }));

    CHECK(revert_turn(changes, dir.path).reverted());
    CHECK(fs::exists(dir.path / "empty.txt"));
    CHECK(read(dir.path / "empty.txt").empty());
    CHECK_FALSE(fs::exists(dir.path / "made.txt"));
}

TEST_CASE("A revert refuses a file that moved on, and writes nothing at all",
          "[changes][revert]") {
    TempDir dir;
    put(dir.path / "one.txt", "one\n");
    put(dir.path / "two.txt", "two\n");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({dir.path / "one.txt"}), [&] { put(dir.path / "one.txt", "ONE\n"); });
    edit(tracker, scope_of({dir.path / "two.txt"}), [&] { put(dir.path / "two.txt", "TWO\n"); });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 2);

    // Somebody edited one of them after the turn finished.
    put(dir.path / "one.txt", "edited by hand\n");

    const auto result = revert_turn(changes, dir.path);
    CHECK_FALSE(result.reverted());
    CHECK(result.restored.empty());
    CHECK(result.refused.size() == 1);
    CHECK(refusal_of(result, "one.txt") == "changed since that turn");
    // All or nothing: the file that could have been reverted was left alone.
    CHECK(read(dir.path / "one.txt") == "edited by hand\n");
    CHECK(read(dir.path / "two.txt") == "TWO\n");
}

TEST_CASE("A revert refuses what a summary never kept, and says why",
          "[changes][revert]") {
    TempDir dir;
    const auto binary = dir.path / "image.bin";
    put(binary, std::string("\x00\x01\x02", 3));
    put(dir.path / "text.txt", "before\n");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({binary}), [&] { put(binary, std::string("\x00\x03\x04", 3)); });
    edit(tracker, scope_of({dir.path / "text.txt"}),
         [&] { put(dir.path / "text.txt", "after\n"); });
    const auto changes = tracker.changes();

    const auto result = revert_turn(changes, dir.path);
    CHECK_FALSE(result.reverted());
    CHECK_THAT(refusal_of(result, "image.bin"), ContainsSubstring("binary"));
    // The refusal of one file protects the other: nothing was written.
    CHECK(read(dir.path / "text.txt") == "after\n");
}

TEST_CASE("A revert reaches a change outside the workspace by its absolute path",
          "[changes][revert]") {
    TempDir workspace;
    TempDir outside;
    const auto file = outside.path / "notes.md";
    put(file, "original\n");
    TurnChangeTracker tracker(workspace.path);

    edit(tracker, scope_of({file}), [&] { put(file, "rewritten\n"); });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    REQUIRE(fs::path(changes.files[0].path).is_absolute());

    CHECK(revert_turn(changes, workspace.path).reverted());
    CHECK(read(file) == "original\n");
}

TEST_CASE("A revert restores a final line that carries no newline", "[changes][revert]") {
    TempDir dir;
    const auto file = dir.path / "no-eol.txt";
    put(file, "first\nsecond");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({file}), [&] { put(file, "first\nSECOND\nthird"); });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    CHECK_THAT(changes.files[0].diff,
               ContainsSubstring("\\ No newline at end of file"));

    CHECK(revert_turn(changes, dir.path).reverted());
    CHECK(read(file) == "first\nsecond");
}

TEST_CASE("A revert undoes every hunk of a long file, not just the first",
          "[changes][revert]") {
    TempDir dir;
    const auto file = dir.path / "long.txt";
    std::string original;
    for (int line = 0; line < 200; ++line) {
        original += std::format("line {}\n", line);
    }
    put(file, original);
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({file}), [&] {
        std::string edited = original;
        // Two edits far enough apart to stay separate hunks.
        const auto first = edited.find("line 3\n");
        edited.replace(first, std::string("line 3\n").size(), "early edit\n");
        const auto last = edited.find("line 190\n");
        edited.replace(last, std::string("line 190\n").size(), "late edit\n");
        put(file, edited);
    });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    CHECK(std::ranges::count(changes.files[0].diff, '@') > 2);  // more than one hunk

    CHECK(revert_turn(changes, dir.path).reverted());
    CHECK(read(file) == original);
}

TEST_CASE("A whole turn of mixed edits goes back to the tree it started from",
          "[changes][revert]") {
    TempDir dir;
    put(dir.path / "src" / "kept.cpp", "int main() {}\n");
    put(dir.path / "src" / "edited.cpp", "one\ntwo\nthree\n");
    put(dir.path / "docs" / "doomed.md", "gone soon\n");
    put(dir.path / "src" / "moving.hpp", "#pragma once\n");
    const auto before = snapshot_tree(dir.path);
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({dir.path / "src" / "edited.cpp"}),
         [&] { put(dir.path / "src" / "edited.cpp", "one\nTWO\nthree\nfour\n"); });
    edit(tracker, scope_of({dir.path / "docs" / "doomed.md"}),
         [&] { fs::remove(dir.path / "docs" / "doomed.md"); });
    edit(tracker, scope_of({dir.path / "fresh.txt"}),
         [&] { put(dir.path / "fresh.txt", "brand new\n"); });
    MutationScope move{
        .paths = {dir.path / "src" / "moving.hpp", dir.path / "include" / "moved.hpp"},
        .moves = {{dir.path / "src" / "moving.hpp", dir.path / "include" / "moved.hpp"}},
    };
    edit(tracker, move, [&] {
        fs::create_directories(dir.path / "include");
        fs::rename(dir.path / "src" / "moving.hpp", dir.path / "include" / "moved.hpp");
    });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 4);
    CHECK(snapshot_tree(dir.path) != before);

    const auto result = revert_turn(changes, dir.path);
    CHECK(result.reverted());
    CHECK(result.restored.size() == 4);
    CHECK(snapshot_tree(dir.path) == before);
}

TEST_CASE("A second revert of the same turn is refused, not repeated",
          "[changes][revert]") {
    TempDir dir;
    const auto file = dir.path / "a.txt";
    put(file, "before\n");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({file}), [&] { put(file, "after\n"); });
    const auto changes = tracker.changes();
    REQUIRE(revert_turn(changes, dir.path).reverted());

    // The file now holds the pre-turn content, so the recorded "after" side no
    // longer matches it: a strict revert stops instead of guessing.
    const auto again = revert_turn(changes, dir.path);
    CHECK_FALSE(again.reverted());
    CHECK(again.restored.empty());
    CHECK(refusal_of(again, "a.txt") == "changed since that turn");
    CHECK(read(file) == "before\n");
}

TEST_CASE("A revert reports a move it could not finish", "[changes][revert]") {
    TempDir dir;
    put(dir.path / "sub" / "file.txt", "content\n");
    TurnChangeTracker tracker(dir.path);

    MutationScope move{
        .paths = {dir.path / "sub" / "file.txt", dir.path / "other" / "file.txt"},
        .moves = {{dir.path / "sub" / "file.txt", dir.path / "other" / "file.txt"}},
    };
    edit(tracker, move, [&] {
        fs::create_directories(dir.path / "other");
        fs::rename(dir.path / "sub" / "file.txt", dir.path / "other" / "file.txt");
    });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);

    // The plan is sound, so the revert starts; the directory the file has to
    // move back into is now blocked by a regular file of the same name.
    fs::remove_all(dir.path / "sub");
    put(dir.path / "sub", "in the way\n");

    const auto result = revert_turn(changes, dir.path);
    CHECK_FALSE(result.reverted());
    REQUIRE(result.error.has_value());
    // The failure names the file the summary named, which is the one the reader
    // is looking for, and says what stopped it.
    CHECK_THAT(*result.error, ContainsSubstring("other/file.txt"));
    CHECK_THAT(*result.error, ContainsSubstring("could not be moved back"));
    // The file that could not be moved back is still where the turn left it.
    CHECK(read(dir.path / "other" / "file.txt") == "content\n");
}

TEST_CASE("A turn with nothing to revert reports nothing", "[changes][revert]") {
    TempDir dir;
    const auto result = revert_turn(TurnChanges{}, dir.path);
    CHECK(result.empty());
    CHECK_FALSE(result.reverted());
}
