#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/changes/TurnChangeTracker.hpp"
#include "core/changes/TurnRevert.hpp"
#include "core/changes/WorkspaceMutationGate.hpp"
#include "core/changes/TurnRecovery.hpp"
#include "core/changes/WorkspaceInventory.hpp"
#include "core/changes/RecoveryStore.hpp"
#include "core/changes/FileChangeJson.hpp"
#include "core/changes/StoredDiff.hpp"
#include "core/utils/FileStamp.hpp"
#include "core/utils/InterprocessFile.hpp"
#include "core/utils/ProcessUtils.hpp"
#include "core/tools/ShellTool.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "core/tools/ListDirectoryTool.hpp"
#include "tui/DirectShellState.hpp"
#include "TestSessionContext.hpp"
#include <sys/wait.h>
#include <signal.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <map>
#include <set>
#include <random>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#include <simdjson.h>

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

struct ReleaseCommand {
    fs::path path;
    ~ReleaseCommand() { std::ofstream(path) << "release"; }
};

bool wait_for_file(const fs::path& path) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (!fs::exists(path) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    return fs::exists(path);
}

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

/// A clock far enough ahead that every file looks settled: what a turn sees
/// of files written long before it, without sleeping through kSettleTime.
std::chrono::system_clock::time_point settled_clock() {
    return std::chrono::system_clock::now() + core::utils::FileStamp::kSettleTime + std::chrono::seconds(1);
}

} // namespace

TEST_CASE("A revert puts an edited file back the way the turn found it", "[changes][revert]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir;
    const auto file = dir.path / "src" / "app.cpp";
    put(file, "one\ntwo\nthree\n");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({file}), [&] { put(file, "one\nTWO\nthree\nfour\n"); });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);

    const auto result = revert_turn(changes, dir.path, mutations);
    CHECK(result.reverted());
    CHECK(result.restored == std::vector<std::string>{"src/app.cpp"});
    CHECK(read(file) == "one\ntwo\nthree\n");
}

TEST_CASE("A revert removes what a turn created and restores what it deleted",
          "[changes][revert]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir;
    put(dir.path / "kept.txt", "kept\n");
    put(dir.path / "gone" / "doomed.txt", "doomed line\nsecond\n");
    const auto before = snapshot_tree(dir.path);
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({dir.path / "new.txt"}),
         [&] { put(dir.path / "new.txt", "created by the turn\n"); });
    edit(tracker, scope_of({dir.path / "gone" / "doomed.txt"}),
         [&] { fs::remove(dir.path / "gone" / "doomed.txt"); });

    const auto result = revert_turn(tracker.changes(), dir.path, mutations);
    CHECK(result.reverted());
    CHECK_FALSE(fs::exists(dir.path / "new.txt"));
    CHECK(read(dir.path / "gone" / "doomed.txt") == "doomed line\nsecond\n");
    CHECK(snapshot_tree(dir.path) == before);
}

TEST_CASE("A revert moves a renamed file back, carrying its edits with it",
          "[changes][revert]") {
    core::changes::WorkspaceMutationGate mutations;
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

    const auto result = revert_turn(changes, dir.path, mutations);
    CHECK(result.reverted());
    CHECK_FALSE(fs::exists(dir.path / "new" / "b.txt"));
    CHECK(read(dir.path / "old" / "a.txt") == "one\n");
}

TEST_CASE("A pure rename goes back without a diff to read", "[changes][revert]") {
    core::changes::WorkspaceMutationGate mutations;
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

    CHECK(revert_turn(changes, dir.path, mutations).reverted());
    CHECK(read(dir.path / "a.txt") == "content\n");
    CHECK_FALSE(fs::exists(dir.path / "b.txt"));
}

TEST_CASE("An empty file a turn created or deleted needs no diff to go back",
          "[changes][revert]") {
    core::changes::WorkspaceMutationGate mutations;
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

    CHECK(revert_turn(changes, dir.path, mutations).reverted());
    CHECK(fs::exists(dir.path / "empty.txt"));
    CHECK(read(dir.path / "empty.txt").empty());
    CHECK_FALSE(fs::exists(dir.path / "made.txt"));
}

TEST_CASE("A revert refuses a file that moved on, and writes nothing at all",
          "[changes][revert]") {
    core::changes::WorkspaceMutationGate mutations;
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

    const auto result = revert_turn(changes, dir.path, mutations);
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
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir;
    const auto binary = dir.path / "image.bin";
    put(binary, std::string("\x00\x01\x02", 3));
    put(dir.path / "text.txt", "before\n");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({binary}), [&] { put(binary, std::string("\x00\x03\x04", 3)); });
    edit(tracker, scope_of({dir.path / "text.txt"}),
         [&] { put(dir.path / "text.txt", "after\n"); });
    const auto changes = tracker.changes();

    const auto result = revert_turn(changes, dir.path, mutations);
    CHECK_FALSE(result.reverted());
    CHECK_THAT(refusal_of(result, "image.bin"), ContainsSubstring("binary"));
    // The refusal of one file protects the other: nothing was written.
    CHECK(read(dir.path / "text.txt") == "after\n");
}

TEST_CASE("A revert reaches a change outside the workspace by its absolute path",
          "[changes][revert]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir workspace;
    TempDir outside;
    const auto file = outside.path / "notes.md";
    put(file, "original\n");
    TurnChangeTracker tracker(workspace.path);

    edit(tracker, scope_of({file}), [&] { put(file, "rewritten\n"); });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    REQUIRE(fs::path(changes.files[0].path).is_absolute());

    CHECK(revert_turn(changes, workspace.path, mutations).reverted());
    CHECK(read(file) == "original\n");
}

TEST_CASE("A revert restores a final line that carries no newline", "[changes][revert]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir;
    const auto file = dir.path / "no-eol.txt";
    put(file, "first\nsecond");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({file}), [&] { put(file, "first\nSECOND\nthird"); });
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    CHECK_THAT(changes.files[0].diff,
               ContainsSubstring("\\ No newline at end of file"));

    CHECK(revert_turn(changes, dir.path, mutations).reverted());
    CHECK(read(file) == "first\nsecond");
}

TEST_CASE("A revert undoes every hunk of a long file, not just the first",
          "[changes][revert]") {
    core::changes::WorkspaceMutationGate mutations;
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

    CHECK(revert_turn(changes, dir.path, mutations).reverted());
    CHECK(read(file) == original);
}

TEST_CASE("A whole turn of mixed edits goes back to the tree it started from",
          "[changes][revert]") {
    core::changes::WorkspaceMutationGate mutations;
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

    const auto result = revert_turn(changes, dir.path, mutations);
    CHECK(result.reverted());
    CHECK(result.restored.size() == 4);
    CHECK(snapshot_tree(dir.path) == before);
}

TEST_CASE("A second revert of the same turn is refused, not repeated",
          "[changes][revert]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir;
    const auto file = dir.path / "a.txt";
    put(file, "before\n");
    TurnChangeTracker tracker(dir.path);

    edit(tracker, scope_of({file}), [&] { put(file, "after\n"); });
    const auto changes = tracker.changes();
    REQUIRE(revert_turn(changes, dir.path, mutations).reverted());

    // The file now holds the pre-turn content, so the recorded "after" side no
    // longer matches it: a strict revert stops instead of guessing.
    const auto again = revert_turn(changes, dir.path, mutations);
    CHECK_FALSE(again.reverted());
    CHECK(again.restored.empty());
    CHECK(refusal_of(again, "a.txt") == "changed since that turn");
    CHECK(read(file) == "before\n");
}

TEST_CASE("A revert reports a move it could not finish", "[changes][revert]") {
    core::changes::WorkspaceMutationGate mutations;
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

    const auto result = revert_turn(changes, dir.path, mutations);
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
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir;
    const auto result = revert_turn(TurnChanges{}, dir.path, mutations);
    CHECK(result.empty());
    CHECK_FALSE(result.reverted());
}

TEST_CASE("A revert refuses another thread's workspace writer before touching files",
          "[changes][revert][regression]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir;
    const auto file = dir.path / "app.cpp";
    put(file, "before\n");
    TurnChangeTracker tracker(dir.path);
    edit(tracker, scope_of({file}), [&] { put(file, "after\n"); });
    const std::vector<fs::path> roots{dir.path};
    {
        auto writer = mutations.acquire(roots);
        REQUIRE(writer);
        const auto result = revert_turn(tracker.changes(), dir.path, mutations);
        REQUIRE_FALSE(result.reverted());
        REQUIRE(result.refused.size() == 1);
        CHECK(read(file) == "after\n");
    }
    CHECK(revert_turn(tracker.changes(), dir.path, mutations).reverted());
    CHECK(read(file) == "before\n");
}

TEST_CASE("Reverting one file does not reserve other files in the same workspace",
          "[changes][revert][concurrency][regression]") {
    TempDir dir;
    const auto changed = dir.path / "changed.txt", unrelated = dir.path / "unrelated.txt";
    put(changed, "before");
    put(unrelated, "other work");
    TurnChangeTracker tracker(dir.path);
    edit(tracker, scope_of({changed}), [&] { put(changed, "after"); });
    core::changes::WorkspaceMutationGate mutations;
    const std::vector<fs::path> other_target{unrelated};
    auto writer = mutations.acquire(other_target);
    REQUIRE(writer);
    CHECK(revert_turn(tracker.changes(), dir.path, mutations).reverted());
    CHECK(read(changed) == "before");
    CHECK(read(unrelated) == "other work");
}

TEST_CASE("Durable restoration reserves its actual changed files even when the summary omits one",
          "[changes][revert][recovery][concurrency][regression]") {
    TempDir dir, storage;
    const auto a = dir.path / "a", b = dir.path / "b", c = dir.path / "c";
    for (const auto& file : {a, b, c}) put(file, "before");
    const auto checkpoint = storage.path / "turn";
    core::changes::TurnRecovery recovery(checkpoint, dir.path);
    const std::vector<fs::path> captured{a, b, c};
    recovery.before(captured);
    put(a, "after");
    put(b, "after");
    recovery.after(captured);
    recovery.finish();
    TurnChanges summary;
    summary.recovery_path = checkpoint.string();
    summary.files.push_back({.path = "a"});
    core::changes::WorkspaceMutationGate mutations;
    const std::vector<fs::path> omitted_target{b}, unrelated_target{c};
    {
        auto writer = mutations.acquire(omitted_target);
        REQUIRE(writer);
        CHECK_FALSE(revert_turn(summary, dir.path, mutations).reverted());
        CHECK(read(a) == "after");
        CHECK(read(b) == "after");
    }
    auto unrelated = mutations.acquire(unrelated_target);
    REQUIRE(unrelated);
    put(c, "independent work");
    CHECK(revert_turn(summary, dir.path, mutations).reverted());
    CHECK(read(a) == "before");
    CHECK(read(b) == "before");
    CHECK(read(c) == "independent work");
}

TEST_CASE("Workspace mutation tokens can be transferred and cancel waiting writers",
          "[changes][revert][concurrency]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir;
    const std::vector<fs::path> roots{dir.path};
    auto restore = mutations.acquire(roots);
    REQUIRE(restore);
    const auto blocked = mutations.acquire(roots,
        std::chrono::seconds(1), [] { return true; });
    CHECK_FALSE(blocked);
    std::thread release([token = std::move(*restore)] {});
    release.join();
    CHECK(mutations.acquire(roots, std::chrono::milliseconds(0)));
}

TEST_CASE("Durable recovery restores binary bytes and executable modes without a display diff",
          "[changes][recovery][revert]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, storage;
    const auto file = dir.path / "program";
    const std::string original("\0\1\2", 3);
    put(file, original);
    fs::permissions(file, fs::perms::owner_all);
    auto recovery = std::make_shared<core::changes::TurnRecovery>(storage.path / "turn", dir.path);
    TurnChangeTracker tracker(dir.path);
    tracker.set_recovery(recovery);
    edit(tracker, scope_of({file}), [&] { put(file, std::string("\0\3\4", 3)); });
    tracker.finish_recovery();
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    CHECK(changes.files[0].diff.empty());
    const auto result = revert_turn(changes, dir.path, mutations);
    REQUIRE(result.reverted());
    CHECK(read(file) == original);
    CHECK((fs::status(file).permissions() & fs::perms::owner_all) == fs::perms::owner_all);
}

TEST_CASE("A forced process exit after a write retains an actionable original and restoration scope",
          "[changes][recovery][crash]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, storage;
    const auto file = dir.path / "app.cpp";
    put(file, "pre-existing dirty contents\n");
    const auto checkpoint = storage.path / "turn";
    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        core::changes::TurnRecovery recovery(checkpoint, dir.path);
        const std::vector<fs::path> paths{file};
        recovery.before(paths);
        put(file, "partially completed task\n");
        ::_exit(0); // No after-record, finalizer, journal summary, or destructor.
    }
    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
    auto recovered = core::changes::TurnRecovery::reconcile(checkpoint, dir.path, mutations);
    REQUIRE(recovered);
    REQUIRE(recovered->files.size() == 1);
    CHECK(revert_turn(*recovered, dir.path, mutations).reverted());
    CHECK(read(file) == "pre-existing dirty contents\n");
}

TEST_CASE("A failed tool's partial writes remain revertible and later edits are protected",
          "[changes][recovery][regression]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, storage;
    const auto file = dir.path / "app.cpp";
    put(file, "original\n");
    TurnChangeTracker tracker(dir.path);
    tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(storage.path / "turn", dir.path));
    CHECK_THROWS(edit(tracker, scope_of({file}), [&] {
        put(file, "partial write\n");
        throw std::runtime_error("tool failed after its write");
    }));
    tracker.finish_recovery();
    auto changes = tracker.changes();
    put(file, "user's later edit\n");
    CHECK_FALSE(revert_turn(changes, dir.path, mutations).reverted());
    CHECK(read(file) == "user's later edit\n");
    put(file, "partial write\n");
    CHECK(revert_turn(changes, dir.path, mutations).reverted());
    CHECK(read(file) == "original\n");
}

TEST_CASE("Recovery storage failure prevents the mutation from starting",
          "[changes][recovery][regression]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, storage;
    put(storage.path / "blocked", "not a directory");
    const auto file = dir.path / "app.cpp";
    put(file, "original\n");
    TurnChangeTracker tracker(dir.path);
    tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(storage.path / "blocked" / "turn", dir.path));
    bool ran = false;
    CHECK_THROWS(edit(tracker, scope_of({file}), [&] { ran = true; put(file, "lost\n"); }));
    CHECK_FALSE(ran);
    CHECK(read(file) == "original\n");
}

TEST_CASE("Workspace recovery covers a generator's source output and stale checksum after interruption",
          "[changes][recovery][crash][inventory]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, storage;
    put(dir.path / "source.cpp", "old source\n");
    put(dir.path / "catalogue.json", "old output\n");
    put(dir.path / "manifest.json", "old checksum\n");
    const auto baseline = snapshot_tree(dir.path);
    const auto checkpoint = storage.path / "turn";
    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        TurnChangeTracker tracker(dir.path);
        tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(checkpoint, dir.path));
        edit(tracker, scope_of({dir.path / "source.cpp"}), [&] { put(dir.path / "source.cpp", "fixed source\n"); });
        tracker.track(MutationScope{.unbounded = true, .tool = "generator"}, [&]() -> int {
            put(dir.path / "catalogue.json", "new output\n");
            put(dir.path / "new generated.txt", "new file\n");
            ::_exit(0); // Abrupt exit while the generator is still unfinished.
        });
    }
    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
    auto changes = core::changes::TurnRecovery::reconcile(checkpoint, dir.path, mutations);
    REQUIRE(changes);
    CHECK(changes->files.size() == 3);
    CHECK(revert_turn(*changes, dir.path, mutations).reverted());
    CHECK(snapshot_tree(dir.path) == baseline);
}

TEST_CASE("Recovery refuses a live recorded process group and a corrupted original",
          "[changes][recovery][integrity]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, storage;
    const auto file = dir.path / "app.cpp";
    put(file, "original\n");
    const auto checkpoint = storage.path / "turn";
    core::changes::TurnRecovery recovery(checkpoint, dir.path);
    const std::vector<fs::path> paths{file};
    recovery.before(paths);
    put(file, "changed\n");
    // A group ID is never used to kill a process after restart: even PID reuse
    // must cause a conservative refusal instead of signalling an unrelated job.
    recovery.process_group(::getpgrp());
    CHECK_FALSE(core::changes::TurnRecovery::reconcile(checkpoint, dir.path, mutations));
    recovery.after(paths);
    recovery.finish();
    for (const auto& blob : fs::directory_iterator(checkpoint.parent_path() / "objects")) put(blob.path(), "corrupted");
    TurnChanges changes;
    changes.files.push_back({.path = "app.cpp"});
    changes.recovery_path = checkpoint.string();
    CHECK_FALSE(revert_turn(changes, dir.path, mutations).reverted());
    CHECK(read(file) == "changed\n");
}

TEST_CASE("A process group of unreaped zombies is not a live writer",
          "[changes][recovery][process][regression]") {
    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        ::setpgid(0, 0);
        ::_exit(0);
    }
    ::setpgid(child, child);
    siginfo_t info{};
    REQUIRE(::waitid(P_PID, static_cast<id_t>(child), &info, WEXITED | WNOWAIT) == 0);
    CHECK_FALSE(core::utils::process::group_may_exist(child));
    REQUIRE(::waitpid(child, nullptr, 0) == child);
    CHECK(core::utils::process::group_may_exist(::getpgrp()));
}

TEST_CASE("Native inventory respects exclusions symlinks and resource bounds without source control",
          "[changes][inventory]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, outside;
    put(dir.path / ".ignore", "cache/\n*.generated\n");
    put(dir.path / ".agentignore", "private/\n");
    put(dir.path / "src" / "a.cpp", "source\n");
    put(dir.path / "cache" / "build.log", "ignored\n");
    put(dir.path / "private" / "secret", "ignored\n");
    put(dir.path / "out.generated", "ignored\n");
    put(outside.path / "other.cpp", "outside\n");
    fs::create_directory_symlink(outside.path, dir.path / "link");
    const std::vector<fs::path> roots{dir.path};
    auto inventory = core::changes::workspace_inventory(roots, 100);
    REQUIRE(inventory.complete);
    REQUIRE(inventory.paths.size() == 3); // Two exclusion files and a.cpp.
    CHECK(std::ranges::any_of(inventory.paths, [](const auto& p) { return p.filename() == "a.cpp"; }));
    auto limited = core::changes::workspace_inventory(roots, 1);
    CHECK_FALSE(limited.complete);
    CHECK(limited.paths.size() == 1);
}

TEST_CASE("Writer exclusion follows overlapping native roots and leaves unrelated folders available",
          "[changes][revert][concurrency]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, unrelated;
    fs::create_directories(dir.path / "nested");
    const std::vector<fs::path> nested{dir.path / "nested"}, parent{dir.path}, other{unrelated.path};
    auto writer = mutations.acquire(nested);
    REQUIRE(writer);
    CHECK_FALSE(mutations.acquire(parent, std::chrono::milliseconds(0)));
    CHECK(mutations.acquire(other, std::chrono::milliseconds(0)));
}

TEST_CASE("File writer reservations do not expand to sibling attachments",
          "[changes][concurrency][regression][attachments]") {
    TempDir desktop;
    const auto first = desktop.path / "first.png", second = desktop.path / "second.png";
    put(first, "first");
    put(second, "second");
    using Gate = core::changes::WorkspaceMutationGate;
    Gate mutations;
    const std::vector<fs::path> a{first}, b{second}, parent{desktop.path};
    auto writer = mutations.acquire(a);
    REQUIRE(writer);
    // Deletion must not widen an existing or newly requested file scope.
    REQUIRE(fs::remove(first));
    REQUIRE(fs::remove(second));
    CHECK(mutations.acquire(b, std::chrono::milliseconds(0)));
    CHECK_FALSE(mutations.acquire(a, std::chrono::milliseconds(0)));
    CHECK_FALSE(mutations.acquire(parent, std::chrono::milliseconds(0)));
    writer->reset();
    auto directory_writer = mutations.acquire(parent);
    REQUIRE(directory_writer);
    CHECK_FALSE(mutations.acquire(b, std::chrono::milliseconds(0)));
}

TEST_CASE("A new project tab does not contend on screenshots inherited from its source tab",
          "[changes][concurrency][regression][attachments]") {
    TempDir first, second, desktop;
    const auto screenshot = desktop.path / "screenshot.png";
    put(screenshot, "image");
    core::workspace::SessionWorkspace original{core::workspace::WorkspaceSnapshot{
        .primary = first.path, .enforce = true}};
    REQUIRE(original.add_additional_paths({screenshot}) == 1);
    auto fresh = original.without_attachments();
    REQUIRE(fresh.set_primary(second.path));

    using Gate = core::changes::WorkspaceMutationGate;
    Gate mutations;
    auto running_tests = mutations.acquire(fresh.ordered_roots());
    REQUIRE(running_tests);
    CHECK(mutations.acquire(original.ordered_roots(),
                            std::chrono::milliseconds(0)));
}

TEST_CASE("Tool dispatch reserves targets rather than unrelated access grants",
          "[changes][concurrency][regression][tools]") {
    TempDir first, second, desktop;
    const auto screenshot = desktop.path / "screenshot.png";
    put(screenshot, "image");
    auto mutations = std::make_shared<core::changes::WorkspaceMutationGate>();
    core::tools::ToolManager tools(mutations);
    tools.register_tool(std::make_shared<core::tools::WriteFileTool>());
    tools.register_tool(std::make_shared<core::tools::ListDirectoryTool>());
    const auto context = test_support::make_session_context({
        .primary = second.path, .additional = {first.path, screenshot}, .enforce = true},
        core::context::SessionTransport::cli, "second-tab");
    const std::vector<fs::path> occupied{first.path, screenshot};
    auto writer = mutations->acquire(occupied);
    REQUIRE(writer);
    const auto invoke = [&](const std::string& name, const std::string& args) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
        return tools.execute_tool(name, args, core::tools::ToolInvocationContext{
            .session_context = context,
            .cancellation_requested = [deadline] { return std::chrono::steady_clock::now() >= deadline; },
        });
    };
    CHECK_THAT(invoke("write_file", R"({"file_path":"result.txt","content":"independent"})"),
               !ContainsSubstring("error"));
    CHECK(read(second.path / "result.txt") == "independent");
    CHECK_THAT(invoke("list_directory", std::format(R"({{"path":"{}"}})", first.path.string())),
               !ContainsSubstring("error"));
    const auto blocked = invoke("write_file", std::format(
        R"({{"file_path":"{}","content":"must not change"}})", screenshot.string()));
    CHECK_THAT(blocked, ContainsSubstring("cancelled"));
    CHECK_THAT(blocked, !ContainsSubstring("active writer"));
    CHECK(read(screenshot) == "image");
}

TEST_CASE("An active shell in one tab does not block another project's shell",
          "[changes][concurrency][regression][tools][shell]") {
    TempDir first, second, desktop;
    const auto first_shot = desktop.path / "first.png", second_shot = desktop.path / "second.png";
    put(first_shot, "first");
    put(second_shot, "second");
    auto first_context = test_support::make_session_context({
        .primary = first.path, .additional = {first_shot}, .enforce = true},
        core::context::SessionTransport::cli, "first-tab");
    auto second_workspace = first_context.workspace_view().without_attachments();
    REQUIRE(second_workspace.set_primary(second.path));
    SECTION("a new conversation inherits no file attachments") {}
    SECTION("separate conversations attach different files in the same folder") {
        REQUIRE(second_workspace.add_additional_paths({second_shot}) == 1);
    }
    SECTION("two conversations work on different files in the same project") {
        REQUIRE(second_workspace.set_primary(first.path));
    }
    const auto second_context = test_support::make_session_context(second_workspace.snapshot(),
        core::context::SessionTransport::cli, "second-tab");
    core::tools::ToolManager tools;
    tools.register_tool(std::make_shared<core::tools::ShellTool>(tools.shell_sessions()));
    tools.register_tool(std::make_shared<core::tools::WriteFileTool>());
    auto first_tracker = std::make_shared<TurnChangeTracker>(first.path, tools.mutation_observations());
    auto second_tracker = std::make_shared<TurnChangeTracker>(second_context.workspace_view().primary(),
                                                           tools.mutation_observations());
    auto running = std::async(std::launch::async, [&] {
        return tools.execute_tool("run_terminal_command",
            R"({"command":"export FILO_TAB_SECRET=first; : > running; while [ ! -e release ]; do sleep 0.01; done; printf first_done","timeout_seconds":5})",
            core::tools::ToolInvocationContext{.session_context = first_context, .change_tracker = first_tracker});
    });
    ReleaseCommand release{first.path / "release"};
    REQUIRE(wait_for_file(first.path / "running"));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(200);
    CHECK_THAT(tools.execute_tool("write_file", R"({"file_path":"unrelated.txt","content":"independent"})",
        core::tools::ToolInvocationContext{
            .session_context = first_context,
            .cancellation_requested = [deadline] { return std::chrono::steady_clock::now() >= deadline; },
        }), !ContainsSubstring("error"));
    auto independent = std::async(std::launch::async, [&] {
        return tools.execute_tool("run_terminal_command",
            R"({"command":"printf second > second.out; printf 'second_done[%s]' \"${FILO_TAB_SECRET-}\"","timeout_seconds":5})",
            core::tools::ToolInvocationContext{.session_context = second_context, .change_tracker = second_tracker});
    });
    CHECK(independent.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    const auto result = independent.get();
    auto settled = std::async(std::launch::async, [&] { second_tracker->settle(); });
    CHECK(settled.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    put(release.path, "release");
    settled.get();
    CHECK_THAT(running.get(), ContainsSubstring("first_done"));
    CHECK_THAT(result, ContainsSubstring("second_done[]"));
    CHECK_THAT(result, ContainsSubstring(R"("exit_code":0)"));
    CHECK_THAT(tools.execute_tool("run_terminal_command",
        R"({"command":"printf '%s' \"$FILO_TAB_SECRET\""})", first_context), ContainsSubstring("first"));
    CHECK_THAT(tools.execute_tool("run_terminal_command",
        R"({"command":"exit 9"})", first_context), ContainsSubstring(R"("exit_code":9)"));
    CHECK_THAT(tools.execute_tool("run_terminal_command",
        R"({"command":"printf recovered"})", first_context), ContainsSubstring("recovered"));
}

TEST_CASE("Concurrent snapshot ownership cannot authorize reverting another conversation's edits",
          "[changes][recovery][concurrency][regression]") {
    TempDir dir, storage;
    const auto own = dir.path / "own.txt", foreign = dir.path / "foreign.txt";
    put(own, "before");
    put(foreign, "before");
    core::tools::ToolManager tools;
    tools.register_tool(std::make_shared<core::tools::WriteFileTool>());
    TurnChangeTracker tracker(dir.path, tools.mutation_observations());
    const auto checkpoint = storage.path / "turn";
    tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(checkpoint, dir.path));
    edit(tracker, MutationScope{.unbounded = true, .tool = "shell"}, [&] { put(own, "own edit"); });
    const auto context = test_support::make_session_context({.primary = dir.path, .enforce = true},
        core::context::SessionTransport::cli, "other-conversation");
    CHECK_THAT(tools.execute_tool("write_file", R"({"file_path":"foreign.txt","content":"foreign edit"})", context),
               !ContainsSubstring("error"));
    tracker.settle();
    tracker.finish_recovery();
    auto changes = tracker.changes();
    REQUIRE(changes.ambiguous_ownership);
    CHECK_FALSE(revert_turn(changes, dir.path, *tools.mutation_gate()).reverted());
    CHECK(read(own) == "own edit");
    CHECK(read(foreign) == "foreign edit");

    std::string json;
    core::changes::append_turn_changes_json(json, changes);
    simdjson::dom::parser parser;
    auto saved = core::changes::parse_turn_changes_json(parser.parse(json).get_object());
    CHECK(saved.ambiguous_ownership);
    const auto recovered = core::changes::TurnRecovery::reconcile(checkpoint, dir.path, *tools.mutation_gate());
    REQUIRE(recovered);
    CHECK(recovered->ambiguous_ownership);
    // Even an older or altered summary cannot bypass the durable marker.
    changes.ambiguous_ownership = false;
    CHECK_FALSE(revert_turn(changes, dir.path, *tools.mutation_gate()).reverted());
    CHECK(read(foreign) == "foreign edit");
}

TEST_CASE("Scoped ownership and unrelated work remain independently revertible",
          "[changes][revert][concurrency][regression]") {
    TempDir dir, other;
    auto observations = std::make_shared<core::changes::MutationObservations>();
    TurnChangeTracker first(dir.path, observations), second(dir.path, observations), third(other.path, observations);
    put(dir.path / "a", "before a");
    put(dir.path / "b", "before b");
    put(other.path / "c", "before c");
    edit(first, scope_of({dir.path / "a"}), [&] { put(dir.path / "a", "after a"); });
    edit(second, scope_of({dir.path / "b"}), [&] { put(dir.path / "b", "after b"); });
    edit(third, MutationScope{.unbounded = true}, [&] { put(other.path / "c", "after c"); });
    first.finish_recovery();
    second.finish_recovery();
    third.finish_recovery();
    CHECK_FALSE(first.changes().ambiguous_ownership);
    CHECK_FALSE(second.changes().ambiguous_ownership);
    CHECK_FALSE(third.changes().ambiguous_ownership);
    core::changes::WorkspaceMutationGate mutations;
    CHECK(revert_turn(first.changes(), dir.path, mutations).reverted());
    CHECK(read(dir.path / "b") == "after b");
    CHECK(revert_turn(second.changes(), dir.path, mutations).reverted());
    CHECK(revert_turn(third.changes(), other.path, mutations).reverted());
}

TEST_CASE("A turn's delegated mutations do not make its own snapshots ambiguous",
          "[changes][concurrency][regression]") {
    TempDir dir;
    TurnChangeTracker tracker(dir.path);
    put(dir.path / "a", "before");
    edit(tracker, MutationScope{.unbounded = true}, [&] { put(dir.path / "a", "one"); });
    edit(tracker, scope_of({dir.path / "a"}), [&] { put(dir.path / "a", "two"); });
    tracker.finish_recovery();
    CHECK_FALSE(tracker.changes().ambiguous_ownership);
}

TEST_CASE("Unbounded overlap invalidates both conversations' snapshot ownership",
          "[changes][concurrency][regression]") {
    TempDir dir;
    put(dir.path / "a", "before");
    auto observations = std::make_shared<core::changes::MutationObservations>();
    TurnChangeTracker scoped(dir.path, observations), unbounded(dir.path, observations);
    const auto scoped_edit = [&] {
        edit(scoped, scope_of({dir.path / "a"}), [&] { put(dir.path / "a", "scoped"); });
    };
    const auto unbounded_edit = [&] {
        edit(unbounded, MutationScope{.unbounded = true}, [&] { put(dir.path / "a", "unbounded"); });
    };
    SECTION("scoped mutation starts first") { scoped_edit(); unbounded_edit(); }
    SECTION("unbounded mutation starts first") { unbounded_edit(); scoped_edit(); }
    scoped.finish_recovery();
    unbounded.finish_recovery();
    CHECK(scoped.changes().ambiguous_ownership);
    CHECK(unbounded.changes().ambiguous_ownership);
}

TEST_CASE("Slow attribution persistence does not hold the shared observation registry",
          "[changes][concurrency][regression]") {
    TempDir dir, unrelated;
    core::changes::MutationObservations observations;
    std::promise<void> entered, release;
    auto started = entered.get_future();
    auto released = release.get_future().share();
    auto owner = std::make_shared<core::changes::MutationObservations::Owner>([&] {
        entered.set_value();
        if (released.wait_for(std::chrono::seconds(3)) != std::future_status::ready)
            throw std::runtime_error("Test attribution persistence was not released");
    });
    const std::vector<fs::path> roots{dir.path}, other{unrelated.path};
    const auto window = observations.observe(owner, roots, true);
    auto overlapping = std::async(std::launch::async, [&] { return observations.observe({}, roots); });
    CHECK(started.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    auto independent = std::async(std::launch::async, [&] { return observations.observe({}, other); });
    CHECK(independent.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    auto closing = std::async(std::launch::async, [&] { observations.close(owner); });
    CHECK(closing.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout);
    release.set_value();
    CHECK(owner->ambiguous());
    CHECK(overlapping.get());
    CHECK(independent.get());
    closing.get();
}

TEST_CASE("Failed attribution persistence cannot be published as a safe snapshot",
          "[changes][recovery][concurrency][regression]") {
    TempDir dir;
    core::changes::MutationObservations observations;
    auto owner = std::make_shared<core::changes::MutationObservations::Owner>([] {
        throw std::runtime_error("checkpoint unavailable");
    });
    const std::vector<fs::path> roots{dir.path};
    const auto window = observations.observe(owner, roots, true);
    CHECK_THROWS_AS(observations.observe({}, roots), std::runtime_error);
    CHECK(owner->ambiguous());
    CHECK_THROWS_AS(observations.close(owner), std::runtime_error);
}

TEST_CASE("Direct shell ownership and cleanup are isolated by session and resource",
          "[changes][concurrency][regression][tui][shell]") {
    TempDir tree;
    const auto first = tree.path / "first", second = tree.path / "second";
    fs::create_directories(first);
    fs::create_directories(second);
    tui::DirectShellState shells;
    auto running = std::async(std::launch::async, [&] {
        return shells.run("export FILO_DIRECT_SECRET=first; : > running; while [ ! -e release ]; do sleep 0.01; done",
                          first.string(), "first-tab", std::chrono::seconds(5));
    });
    ReleaseCommand release{first / "release"};
    REQUIRE(wait_for_file(first / "running"));
    CHECK(shells.has_active_for("first-tab"));
    CHECK_FALSE(shells.has_active_for("second-tab"));
    CHECK(shells.quiesce_session("second-tab"));
    CHECK_FALSE(shells.quiesce_session("first-tab"));
    auto independent = std::async(std::launch::async, [&] {
        return shells.run("printf 'second[%s]' \"${FILO_DIRECT_SECRET-}\"", second.string(),
                          "second-tab", std::chrono::seconds(5));
    });
    CHECK(independent.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    const auto result = independent.get();
    shells.clear_session_state("second-tab");
    CHECK(shells.has_active_for("first-tab"));
    put(release.path, "release");
    CHECK(running.get().exit_code == 0);
    CHECK(result.exit_code == 0);
    CHECK(result.output == "second[]");
    CHECK(shells.run("printf recovered", first.string(), "first-tab").output == "recovered");
    CHECK_FALSE(shells.has_active_for("first-tab"));
    CHECK(shells.run("true", first.string(), "").exit_code == -1);
}

TEST_CASE("Session identity changes preserve shell ownership without replacing another session",
          "[changes][concurrency][regression][tui][shell]") {
    TempDir dir;
    tui::DirectShellState shells;
    auto running = std::async(std::launch::async, [&] {
        return shells.run("printf '%s' \"$$\" > shell-pid; : > running; while [ ! -e release ]; do sleep 0.01; done",
                          dir.path.string(), "original", std::chrono::seconds(5));
    });
    ReleaseCommand release{dir.path / "release"};
    REQUIRE(wait_for_file(dir.path / "running"));
    const auto original_process = read(dir.path / "shell-pid");
    REQUIRE_FALSE(original_process.empty());
    shells.move_session_state("original", "rotated");
    CHECK_FALSE(shells.has_active_for("original"));
    CHECK(shells.has_active_for("rotated"));
    shells.clear_session_state("original");
    CHECK(shells.has_active_for("rotated"));
    const auto other_process = shells.run("printf '%s' \"$$\"", dir.path.string(), "other").output;
    CHECK_FALSE(other_process.empty());
    CHECK(other_process != original_process);
    CHECK_THROWS_AS(shells.move_session_state("rotated", "other"), std::logic_error);
    put(release.path, "release");
    CHECK(running.get().exit_code == 0);
    CHECK_FALSE(shells.has_active_for("rotated"));
    CHECK(shells.run("printf '%s' \"$$\"", dir.path.string(), "rotated").output == original_process);
    CHECK(shells.run("printf '%s' \"$$\"", dir.path.string(), "other").output == other_process);

    core::tools::ToolManager tools;
    tools.register_tool(std::make_shared<core::tools::ShellTool>(tools.shell_sessions()));
    const auto context = [&](std::string id) {
        return test_support::make_session_context({.primary = dir.path, .enforce = true},
            core::context::SessionTransport::cli, std::move(id));
    };
    CHECK_THAT(tools.execute_tool("run_terminal_command", R"({"command":"export FILO_AGENT_ROTATED=retained"})",
                                context("agent-original")), ContainsSubstring(R"("exit_code":0)"));
    tools.move_session_state("agent-original", "agent-rotated");
    tools.clear_session_state("agent-original");
    CHECK_THAT(tools.execute_tool("run_terminal_command", R"({"command":"printf '%s' \"$FILO_AGENT_ROTATED\""})",
                                context("agent-rotated")), ContainsSubstring("retained"));
}

TEST_CASE("Cancelled and timed out reservations leave no stale workspace ownership",
          "[changes][concurrency][regression]") {
    using Gate = core::changes::WorkspaceMutationGate;
    TempDir first, second;
    Gate mutations;
    const std::vector<fs::path> roots{first.path}, other{second.path};
    const auto cancelled = mutations.acquire(roots,
        std::chrono::milliseconds(0), [] { return true; });
    REQUIRE_FALSE(cancelled);
    CHECK(cancelled.error() == Gate::Failure::Cancelled);
    auto writer = mutations.acquire(roots);
    REQUIRE(writer);
    const auto busy = mutations.acquire(roots, std::chrono::milliseconds(0));
    REQUIRE_FALSE(busy);
    CHECK(busy.error() == Gate::Failure::Busy);
    std::promise<void> waiting;
    auto started = waiting.get_future();
    std::promise<decltype(writer)> outcome;
    auto result = outcome.get_future();
    std::jthread waiter([&](std::stop_token stop) {
        bool announced = false;
        outcome.set_value(mutations.acquire(roots, std::chrono::seconds(2), [&] {
            if (!announced) { waiting.set_value(); announced = true; }
            return stop.stop_requested();
        }));
    });
    REQUIRE(started.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    CHECK(mutations.acquire(other, std::chrono::milliseconds(0)));
    waiter.request_stop();
    REQUIRE(result.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    const auto stopped = result.get();
    REQUIRE_FALSE(stopped);
    CHECK(stopped.error() == Gate::Failure::Cancelled);
    writer->reset();
    CHECK(mutations.acquire(roots, std::chrono::milliseconds(0)));
}

TEST_CASE("Incomplete native inventories never invent originals for files outside the capture budget",
          "[changes][recovery][inventory]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, storage;
    const auto file = dir.path / "large.bin";
    put(file, std::string(core::changes::TurnRecovery::kMaxFileBytes + 1, 'x'));
    const auto checkpoint = storage.path / "turn";
    auto recovery = std::make_shared<core::changes::TurnRecovery>(checkpoint, dir.path);
    TurnChangeTracker tracker(dir.path);
    tracker.set_recovery(recovery);
    tracker.track(MutationScope{.unbounded = true}, [&] { put(dir.path / "new.txt", "new\n"); return 0; });
    auto changes = core::changes::TurnRecovery::reconcile(checkpoint, dir.path, mutations);
    REQUIRE(changes);
    CHECK(changes->partial_enumeration);
    REQUIRE(changes->files.size() == 1);
    CHECK(changes->files[0].path == "new.txt");
    CHECK(revert_turn(*changes, dir.path, mutations).reverted());
    CHECK(fs::file_size(file) == core::changes::TurnRecovery::kMaxFileBytes + 1);
}

TEST_CASE("Durable recovery records permission-only changes and refuses an unquiesced writer",
          "[changes][recovery][revert]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, storage;
    const auto file = dir.path / "script";
    put(file, "unchanged bytes\n");
    fs::permissions(file, fs::perms::owner_read | fs::perms::owner_write);
    TurnChangeTracker tracker(dir.path);
    tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(storage.path / "turn", dir.path));
    edit(tracker, scope_of({file}), [&] { fs::permissions(file, fs::perms::owner_all); });
    tracker.finish_recovery();
    auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    CHECK_FALSE(revert_turn(changes, dir.path, mutations, [](auto) { return false; }).reverted());
    CHECK((fs::status(file).permissions() & fs::perms::owner_exec) != fs::perms::none);
    CHECK(revert_turn(changes, dir.path, mutations).reverted());
    CHECK((fs::status(file).permissions() & fs::perms::owner_exec) == fs::perms::none);
    CHECK(read(file) == "unchanged bytes\n");
}

TEST_CASE("A revert stops an idle managed shell's background writer before restoring bytes",
          "[changes][recovery][revert][shell]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, storage;
    const auto file = dir.path / "source.cpp";
    put(file, "original\n");
    auto tracker = std::make_shared<TurnChangeTracker>(dir.path);
    tracker->set_recovery(std::make_shared<core::changes::TurnRecovery>(storage.path / "turn", dir.path));
    edit(*tracker, scope_of({file}), [&] { put(file, "agent edit\n"); });
    std::string id;
    SECTION("identified session") { id = std::format("recovery-background-{}", ::getpid()); }
    SECTION("anonymous session") {}
    auto context = test_support::make_session_context({.primary = dir.path, .enforce = true},
        core::context::SessionTransport::cli, id);
    core::tools::ShellTool shell;
    const core::tools::ToolInvocationContext invocation{.session_context = context, .change_tracker = tracker};
    tracker->track(MutationScope{.unbounded = true, .tool = "run_terminal_command"}, [&] {
        return shell.execute(R"({"command":"(sleep 0.5; printf 'late write\\n' >> source.cpp) >/dev/null 2>&1 &"})", invocation);
    });
    tracker->finish_recovery();
    const auto result = revert_turn(tracker->changes(), dir.path, mutations, [&](auto) {
        return shell.sessions().quiesce_session(id);
    });
    INFO(result.error.value_or(""));
    std::string refusals;
    for (const auto& refusal : result.refused) refusals += refusal.path + ": " + refusal.reason + "\n";
    INFO(refusals);
    REQUIRE(result.reverted());
    std::this_thread::sleep_for(std::chrono::milliseconds(650));
    CHECK(read(file) == "original\n");
    shell.sessions().clear(id);
}

TEST_CASE("Native recovery never captures its own checkpoint inside the workspace",
          "[changes][recovery][inventory]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir;
    const auto file = dir.path / "source.cpp";
    const auto checkpoint = dir.path / ".cache" / "checkpoint";
    put(file, "before\n");
    TurnChangeTracker tracker(dir.path);
    tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(checkpoint, dir.path));
    tracker.track(MutationScope{.unbounded = true}, [&] { put(file, "after\n"); return 0; });
    tracker.finish_recovery();
    REQUIRE(tracker.changes().files.size() == 1);
    auto recovered = core::changes::TurnRecovery::reconcile(checkpoint, dir.path, mutations);
    REQUIRE(recovered);
    REQUIRE(recovered->files.size() == 1);
    CHECK(recovered->files.front().path == "source.cpp");
    CHECK(revert_turn(*recovered, dir.path, mutations).reverted());
    CHECK(read(file) == "before\n");
    CHECK(fs::exists(checkpoint / "manifest.json"));
}

TEST_CASE("Completed recovery storage retains only the configured number of turns",
          "[changes][recovery][limits]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, storage;
    const auto file = dir.path / "source.cpp";
    put(file, "initial\n");
    const std::vector<fs::path> paths{file};
    for (std::size_t i = 0; i < core::changes::TurnRecovery::kRetainedTurns + 3; ++i) {
        core::changes::TurnRecovery checkpoint(
            core::changes::RecoveryStore::new_directory(storage.path), dir.path);
        checkpoint.before(paths);
        put(file, std::format("turn {}\n", i));
        checkpoint.after(paths);
        checkpoint.finish();
    }
    const auto turns = core::changes::RecoveryStore::turn_directories(storage.path);
    REQUIRE(turns);
    CHECK(turns->size() == core::changes::TurnRecovery::kRetainedTurns);
    // The shared store keeps exactly what the retained turns refer to: each
    // turn's distinct original.
    CHECK(static_cast<std::size_t>(std::distance(fs::directory_iterator(storage.path / "objects"), fs::directory_iterator{}))
        == core::changes::TurnRecovery::kRetainedTurns);
}

TEST_CASE("Independent execution runtimes do not share hidden writer state", "[changes][concurrency]") {
    TempDir dir;
    const std::vector<fs::path> roots{dir.path};
    core::changes::WorkspaceMutationGate first, second;
    auto writer = first.acquire(roots);
    REQUIRE(writer);
    CHECK_FALSE(first.acquire(roots,
                              std::chrono::milliseconds(0)));
    CHECK(second.acquire(roots,
                         std::chrono::milliseconds(0)));
}

TEST_CASE("Detached writer reservations safely outlive their runtime owner", "[changes][concurrency]") {
    TempDir dir;
    const std::vector<fs::path> roots{dir.path};
    auto runtime = std::make_unique<core::changes::WorkspaceMutationGate>();
    auto reservation = runtime->acquire(roots);
    REQUIRE(reservation);
    runtime.reset();
    std::thread worker([lease = std::move(*reservation)] {});
    worker.join();
}

TEST_CASE("Recovery storage rejects unknown lifecycle phases before reconciliation", "[changes][recovery]") {
    TempDir dir, storage;
    core::changes::RecoveryStore store(storage.path / "checkpoint");
    core::changes::RecoveryManifest manifest{.workspace = dir.path};
    manifest.phase = core::changes::RecoveryPhase::Complete;
    store.save(manifest);
    auto loaded = store.load(dir.path);
    REQUIRE(loaded);
    CHECK(loaded->phase == core::changes::RecoveryPhase::Complete);
    CHECK_FALSE(store.load(storage.path));
    const auto path = store.directory() / "manifest.json";
    auto contents = read(path);
    const auto phase = contents.find("complete");
    REQUIRE(phase != std::string::npos);
    contents.replace(phase, 8, "unknown_phase");
    put(path, contents);
    CHECK_FALSE(store.load(dir.path));
}

TEST_CASE("Tool dispatch and revert use the injected writer coordinator", "[changes][recovery][tools]") {
    TempDir dir, storage;
    const auto file = dir.path / "app.txt";
    put(file, "before");
    auto mutations = std::make_shared<core::changes::WorkspaceMutationGate>();
    core::tools::ToolManager tools(mutations);
    tools.register_tool(std::make_shared<core::tools::WriteFileTool>());
    auto tracker = std::make_shared<TurnChangeTracker>(dir.path);
    tracker->set_recovery(std::make_shared<core::changes::TurnRecovery>(storage.path / "checkpoint", dir.path));
    auto context = test_support::make_session_context({dir.path, {}});
    core::tools::ToolInvocationContext invocation{
        .session_context = context,
        .cancellation_requested = [] { return true; },
        .change_tracker = tracker,
    };
    const std::string arguments = R"({"file_path":"app.txt","content":"after"})";
    const std::vector<fs::path> roots{dir.path};
    {
        auto writer = mutations->acquire(roots);
        REQUIRE(writer);
        const auto blocked = tools.execute_tool("write_file", arguments, invocation);
        CHECK(blocked.find("error") != std::string::npos);
        CHECK(read(file) == "before");
        CHECK_FALSE(fs::exists(storage.path / "checkpoint" / "manifest.json"));
    }
    invocation.cancellation_requested = {};
    const auto written = tools.execute_tool("write_file", arguments, invocation);
    CHECK(written.find("error") == std::string::npos);
    REQUIRE(read(file) == "after");
    tracker->finish_recovery();
    CHECK(revert_turn(tracker->changes(), dir.path, *mutations).reverted());
    CHECK(read(file) == "before");
}

TEST_CASE("A failed post-observation is reported once without retrying the checkpoint", "[changes][recovery]") {
    TempDir dir, storage;
    const auto file = dir.path / "source";
    put(file, "before");
    const auto checkpoint = storage.path / "checkpoint";
    TurnChangeTracker tracker(dir.path);
    tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(checkpoint, dir.path));
    bool mutated = false;
    int post_observations = 0;
    tracker.set_observation_filter([&](const auto&, bool) {
        if (mutated) ++post_observations;
        return true;
    });
    CHECK_THROWS(tracker.track(scope_of({file}), [&] {
        put(file, "after");
        mutated = true;
        fs::remove(checkpoint / "manifest.json");
        fs::create_directory(checkpoint / "manifest.json"); // Make the durable commit fail.
        return 0;
    }));
    CHECK(post_observations == 1);
    CHECK(read(file) == "after");
}

TEST_CASE("Private recovery state does not consume the workspace file budget", "[changes][recovery][inventory]") {
    TempDir dir;
    const auto state_root = dir.path / ".state";
    put(state_root / "journals" / "old.jsonl", "private journal");
    put(state_root / "recovery" / "old-session" / "objects" / "old", "private original");
    const auto checkpoint = state_root / "recovery" / "current-session" / "turn";
    const auto file = dir.path / "source";
    put(file, "before");
    TurnChangeTracker::Limits limits;
    limits.max_files_per_directory = 1;
    TurnChangeTracker tracker(dir.path, limits);
    tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(checkpoint, dir.path));
    edit(tracker, MutationScope{.unbounded = true, .tool = "script"}, [&] { put(file, "after"); });
    tracker.settle();
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    CHECK(changes.files[0].path == "source");
    CHECK_FALSE(changes.partial_enumeration);
}

TEST_CASE("Interruption reconciliation fences every recorded workspace", "[changes][recovery][concurrency]") {
    TempDir dir, other, storage;
    const auto file = other.path / "source";
    put(file, "before");
    core::changes::TurnRecovery recovery(storage.path / "turn", dir.path);
    const std::vector<fs::path> paths{file};
    recovery.before(paths);
    put(file, "after");
    recovery.after(paths);
    recovery.finish();
    core::changes::WorkspaceMutationGate mutations;
    const std::vector<fs::path> roots{other.path};
    auto writer = mutations.acquire(roots);
    REQUIRE(writer);
    CHECK_FALSE(core::changes::TurnRecovery::reconcile(recovery.directory(), dir.path, mutations));
    CHECK(read(file) == "after");
}

TEST_CASE("A newly created directory is removed from the durable file manifest", "[changes][recovery]") {
    TempDir dir, storage;
    core::changes::TurnRecovery recovery(storage.path / "turn", dir.path);
    const std::vector<fs::path> paths{dir.path / "created-directory"};
    recovery.before(paths);
    fs::create_directory(paths.front());
    recovery.after(paths);
    auto manifest = core::changes::RecoveryStore(recovery.directory()).load(dir.path);
    REQUIRE(manifest);
    CHECK(manifest->entries.empty());
}

TEST_CASE("A shell edit keeps its diff in a workspace larger than the retention budget",
          "[changes][recovery][inventory][regression]") {
    TempDir dir, storage;
    for (int i = 0; i < 8; ++i) {
        put(dir.path / "Lampo" / std::format("Source{}.swift", i), std::string(63, 'x') + "\n");
    }
    const auto project = dir.path / "Lampo.xcodeproj" / "project.pbxproj";
    put(project, "settings\n");
    const auto removed = dir.path / "Lampo" / "Source0.swift";
    TurnChangeTracker::Limits limits;
    limits.max_tracked_bytes = 256;  // The edited files fit; the workspace does not.
    TurnChangeTracker tracker(dir.path, limits);
    tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(storage.path / "turn", dir.path));
    edit(tracker, MutationScope{.unbounded = true, .tool = "run_terminal_command"}, [&] {
        put(project, "settings\nbuild phase\n");
        fs::remove(removed);
    });
    const auto edited = dir.path / "Lampo" / "Source1.swift";
    edit(tracker, scope_of({edited}), [&] { put(edited, "edited\n"); });
    tracker.settle();
    const auto changes = tracker.changes();
    CHECK_FALSE(changes.partial_enumeration);
    REQUIRE(changes.files.size() == 3);
    CHECK(changes.files[0].path == "Lampo.xcodeproj/project.pbxproj");
    CHECK(changes.files[0].content == FileChangeContent::Text);
    CHECK_THAT(changes.files[0].diff, ContainsSubstring("+build phase"));
    CHECK(changes.files[1].path == "Lampo/Source0.swift");
    CHECK(changes.files[1].kind == FileChangeKind::Deleted);
    CHECK(changes.files[1].deleted == 1);
    CHECK_THAT(changes.files[2].diff, ContainsSubstring("+edited"));
}

TEST_CASE("A later turn shares the originals of files it did not change",
          "[changes][recovery][store]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, storage;
    const auto kept = dir.path / "kept.cpp";
    const auto edited = dir.path / "edited.cpp";
    put(kept, "kept\n");
    put(edited, "first\n");
    const std::vector<fs::path> paths{kept, edited};
    const auto objects = [&] {
        return static_cast<std::size_t>(std::distance(
            fs::directory_iterator(storage.path / "objects"), fs::directory_iterator{}));
    };
    const auto entries = [&](const fs::path& turn) {
        auto manifest = core::changes::RecoveryStore(turn).load(dir.path);
        REQUIRE(manifest);
        return manifest->entries;
    };
    // A stamp is trusted only once its file has settled.
    std::this_thread::sleep_for(core::utils::FileStamp::kSettleTime + std::chrono::milliseconds(200));

    const auto first = core::changes::RecoveryStore::new_directory(storage.path);
    {
        core::changes::TurnRecovery turn(first, dir.path);
        turn.before(paths);
        put(edited, "second\n");
        turn.after(paths);
        turn.finish();
    }
    CHECK(objects() == 2);
    CHECK(entries(first).at(kept).before_stamp != 0);

    const auto second = core::changes::RecoveryStore::new_directory(storage.path);
    {
        core::changes::TurnRecovery turn(second, dir.path);
        turn.before(paths);
        put(kept, "changed in the second turn\n");
        turn.after(paths);
        turn.finish();
    }
    // The unchanged file shares the first turn's object; the one the first turn
    // edited was written too recently to trust its stamp, so its new content is
    // stored, once.
    CHECK(objects() == 3);
    CHECK(entries(second).at(kept).before == entries(first).at(kept).before);
    CHECK(entries(second).at(edited).before == entries(first).at(edited).after);
    CHECK(fs::directory_iterator(second) != fs::directory_iterator{});
    CHECK_FALSE(fs::exists(second / "objects"));

    // Reverting the second turn restores from the object it shared.
    TurnChanges changes;
    changes.files.push_back({.path = "kept.cpp"});
    changes.recovery_path = second.string();
    REQUIRE(revert_turn(changes, dir.path, mutations).reverted());
    CHECK(read(kept) == "kept\n");
}

TEST_CASE("A later turn stores the content an earlier turn's edit left behind",
          "[changes][recovery][store][regression]") {
    // The earlier turn only hashed its after image. Adopting that hash as the
    // next baseline once the file settled left a baseline with no object, so
    // the turn's summary showed "diff omitted" and its revert failed.
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, storage;
    const auto edited = dir.path / "edited.md";
    put(edited, "first\n");
    const std::vector<fs::path> paths{edited};
    const auto first = core::changes::RecoveryStore::new_directory(storage.path);
    {
        // Seen long after it was written, so its after stamp is trusted.
        core::changes::TurnRecovery turn(first, dir.path, settled_clock);
        turn.before(paths);
        put(edited, "second\n");
        turn.after(paths);
        turn.finish();
    }

    const auto second = core::changes::RecoveryStore::new_directory(storage.path);
    {
        core::changes::TurnRecovery turn(second, dir.path, settled_clock);
        turn.before(paths);
        const auto image = turn.captured(edited);
        REQUIRE(image);
        CHECK(core::changes::RecoveryStore(second).original(*image).value_or("") == "second\n");
        put(edited, "third\n");
        turn.after(paths);
        turn.finish();
    }
    TurnChanges changes;
    changes.files.push_back({.path = "edited.md"});
    changes.recovery_path = second.string();
    REQUIRE(revert_turn(changes, dir.path, mutations).reverted());
    CHECK(read(edited) == "second\n");
}

TEST_CASE("A settled post-image is captured before a later turn reuses it",
          "[changes][recovery][store][regression]") {
    TempDir dir, storage;
    const auto file = dir.path / "source.swift";
    const std::vector<fs::path> paths{file};
    put(file, "original\n");
    const auto first = core::changes::RecoveryStore::new_directory(storage.path);
    {
        core::changes::TurnRecovery turn(first, dir.path);
        turn.before(paths);
        put(file, "second\n");
        // A long-running build lets the edited file settle before the final
        // observation. Its post-image has a trusted stamp, but only a hash:
        // after() does not store the bytes.
        std::this_thread::sleep_for(core::utils::FileStamp::kSettleTime + std::chrono::milliseconds(200));
        turn.after(paths);
        turn.finish();
    }
    const auto previous = core::changes::RecoveryStore(first).load(dir.path);
    REQUIRE(previous);
    REQUIRE(previous->entries.at(file).after_stamp != 0);
    REQUIRE_FALSE(core::changes::RecoveryStore(first).original(previous->entries.at(file).after));

    // An intervening read-only turn must capture the missing bytes instead
    // of forwarding a reference to an object that was never stored.
    const auto second = core::changes::RecoveryStore::new_directory(storage.path);
    {
        TurnChangeTracker tracker(dir.path);
        tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(second, dir.path));
        edit(tracker, MutationScope{.unbounded = true, .tool = "run_terminal_command"}, [] {});
        CHECK(tracker.changes().empty());
        tracker.finish_recovery();
    }
    const auto captured = core::changes::RecoveryStore(second).load(dir.path);
    REQUIRE(captured);
    const auto bytes = core::changes::RecoveryStore(second).original(captured->entries.at(file).before);
    REQUIRE(bytes);
    CHECK(*bytes == "second\n");

    const auto third = core::changes::RecoveryStore::new_directory(storage.path);
    TurnChangeTracker tracker(dir.path);
    tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(third, dir.path));
    MutationScope scope;
    SECTION("shell mutation") { scope = {.unbounded = true, .tool = "run_terminal_command"}; }
    SECTION("file tool mutation") { scope = scope_of({file}); }
    edit(tracker, scope, [&] { put(file, "third\n"); });
    tracker.finish_recovery();
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    CHECK_FALSE(changes.partial_enumeration);
    CHECK(changes.files[0].content == FileChangeContent::Text);
    CHECK_THAT(changes.files[0].diff, ContainsSubstring("-second"));
    CHECK_THAT(changes.files[0].diff, ContainsSubstring("+third"));
    core::changes::WorkspaceMutationGate mutations;
    REQUIRE(revert_turn(changes, dir.path, mutations).reverted());
    CHECK(read(file) == "second\n");
}

TEST_CASE("Missing or replaced shared objects cannot become a later turn's baseline",
          "[changes][recovery][store][regression]") {
    TempDir dir, storage;
    const auto file = dir.path / "source.cpp";
    const auto unrelated = storage.path / "unrelated.txt";
    put(file, "before\n");
    put(unrelated, "leave alone\n");
    const std::vector<fs::path> paths{file};
    std::this_thread::sleep_for(core::utils::FileStamp::kSettleTime + std::chrono::milliseconds(200));
    const auto first = core::changes::RecoveryStore::new_directory(storage.path);
    {
        core::changes::TurnRecovery turn(first, dir.path);
        turn.before(paths);
        turn.after(paths);
        turn.finish();
    }
    const core::changes::RecoveryStore previous(first);
    const auto manifest = previous.load(dir.path);
    REQUIRE(manifest);
    REQUIRE(manifest->entries.at(file).latest_stamp() != 0);
    const auto image = manifest->entries.at(file).before;
    const auto object = storage.path / "objects" / image.hash;
    REQUIRE(fs::remove(object));
    bool blocked = false;
    SECTION("missing object is captured again") {}
    SECTION("a symlink is replaced without following it") { fs::create_symlink(unrelated, object); }
    SECTION("a directory blocks mutation before the tool runs") {
        fs::create_directory(object);
        blocked = true;
    }
    CHECK_FALSE(previous.has_object(image));
    const auto second = core::changes::RecoveryStore::new_directory(storage.path);
    TurnChangeTracker tracker(dir.path);
    tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(second, dir.path));
    bool ran = false;
    const auto mutate = [&] {
        edit(tracker, scope_of({file}), [&] { ran = true; put(file, "after\n"); });
    };
    if (blocked) {
        REQUIRE_THROWS_WITH(mutate(), ContainsSubstring("Cannot persist recovery contents"));
        CHECK_FALSE(ran);
        CHECK(read(file) == "before\n");
    } else {
        mutate();
        tracker.finish_recovery();
        CHECK(ran);
        const auto changes = tracker.changes();
        REQUIRE(changes.files.size() == 1);
        CHECK(changes.files[0].content == FileChangeContent::Text);
        CHECK_THAT(changes.files[0].diff, ContainsSubstring("-before\n+after\n"));
        CHECK(previous.has_object(image));
        core::changes::WorkspaceMutationGate mutations;
        REQUIRE(revert_turn(changes, dir.path, mutations).reverted());
        CHECK(read(file) == "before\n");
    }
    CHECK(read(unrelated) == "leave alone\n");
}

TEST_CASE("Shared baselines survive pruning their original checkpoint",
          "[changes][recovery][store][limits][regression]") {
    TempDir dir, storage;
    const auto file = dir.path / "shared.swift";
    put(file, "original\n");
    std::this_thread::sleep_for(core::utils::FileStamp::kSettleTime + std::chrono::milliseconds(200));
    fs::path first;
    for (std::size_t i = 0; i <= core::changes::TurnRecovery::kRetainedTurns; ++i) {
        const auto directory = core::changes::RecoveryStore::new_directory(storage.path);
        if (first.empty()) first = directory;
        TurnChangeTracker tracker(dir.path);
        tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(directory, dir.path));
        edit(tracker, MutationScope{.unbounded = true, .tool = "run_terminal_command"}, [] {});
        tracker.finish_recovery();
        CHECK(tracker.changes().empty());
        CHECK_FALSE(tracker.changes().partial_enumeration);
    }
    CHECK_FALSE(fs::exists(first));
    CHECK(std::distance(fs::directory_iterator(storage.path / "objects"), fs::directory_iterator{}) == 1);
    TurnChangeTracker tracker(dir.path);
    tracker.set_recovery(std::make_shared<core::changes::TurnRecovery>(
        core::changes::RecoveryStore::new_directory(storage.path), dir.path));
    edit(tracker, scope_of({file}), [&] { put(file, "edited\n"); });
    tracker.finish_recovery();
    const auto changes = tracker.changes();
    REQUIRE(changes.files.size() == 1);
    CHECK(changes.files[0].content == FileChangeContent::Text);
    CHECK_THAT(changes.files[0].diff, ContainsSubstring("-original\n+edited\n"));
    core::changes::WorkspaceMutationGate mutations;
    REQUIRE(revert_turn(changes, dir.path, mutations).reverted());
    CHECK(read(file) == "original\n");
}

TEST_CASE("A turn recorded with its own object store still reverts",
          "[changes][recovery][store]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir, storage;
    const auto file = dir.path / "legacy.cpp";
    put(file, "original\n");
    const std::vector<fs::path> paths{file};
    const auto turn_directory = storage.path / "turn";
    {
        core::changes::TurnRecovery turn(turn_directory, dir.path);
        turn.before(paths);
        put(file, "changed\n");
        turn.after(paths);
        turn.finish();
    }
    // The layout before the session-wide store: objects inside the turn.
    fs::rename(storage.path / "objects", turn_directory / "objects");
    CHECK_FALSE(core::changes::RecoveryStore(turn_directory).shares_objects());
    TurnChanges changes;
    changes.files.push_back({.path = "legacy.cpp"});
    changes.recovery_path = turn_directory.string();
    REQUIRE(revert_turn(changes, dir.path, mutations).reverted());
    CHECK(read(file) == "original\n");
}

TEST_CASE("Only paths an inventory found are offered to the filter as canonical",
          "[changes][inventory][regression]") {
    TempDir dir, outside;
    const auto file = dir.path / "inside.txt";
    put(file, "inside\n");
    put(outside.path / "secret.txt", "secret\n");
    // Scope expansion lists this link by its in-workspace name: the filter
    // must resolve it, or it would read the file it points to.
    fs::create_symlink(outside.path / "secret.txt", dir.path / "link.txt");
    TurnChangeTracker tracker(dir.path);
    std::vector<std::pair<std::string, bool>> offered;
    tracker.set_observation_filter([&](const fs::path& path, bool canonical) {
        offered.emplace_back(path.filename().string(), canonical);
        return true;
    });
    edit(tracker, MutationScope{.unbounded = true, .tool = "run_terminal_command"}, [] {});
    CHECK(std::ranges::contains(offered, std::pair<std::string, bool>{"inside.txt", true}));
    CHECK_FALSE(std::ranges::contains(offered, std::pair<std::string, bool>{"link.txt", true}));
    offered.clear();
    edit(tracker, scope_of({dir.path}), [] {});
    CHECK(std::ranges::contains(offered, std::pair<std::string, bool>{"link.txt", false}));
    CHECK(std::ranges::none_of(offered, [](const auto& entry) { return entry.second; }));
}

TEST_CASE("A turn still recording keeps the shared store from being swept",
          "[changes][recovery][store]") {
    TempDir dir, storage;
    const auto file = dir.path / "source.cpp";
    put(file, "original\n");
    const std::vector<fs::path> paths{file};
    // An object stored by a turn whose manifest does not list it yet.
    const auto in_flight = storage.path / "objects" / std::string(64, 'a');
    core::changes::TurnRecovery running(core::changes::RecoveryStore::new_directory(storage.path), dir.path);
    running.before(paths);
    put(in_flight, "not yet listed");
    {
        core::changes::TurnRecovery other(core::changes::RecoveryStore::new_directory(storage.path), dir.path);
        other.before(paths);
        other.after(paths);
        other.finish();
    }
    CHECK(fs::exists(in_flight));
    running.after(paths);
    running.finish();
    CHECK_FALSE(fs::exists(in_flight));
}

namespace {

/// What a workspace holds, by display path: the model a turn sequence is
/// checked against. Nothing in it comes from the code under test.
using Tree = std::map<std::string, std::string>;

std::size_t lines_in(std::string_view text) {
    return static_cast<std::size_t>(std::ranges::count(text, '\n'))
        + (!text.empty() && text.back() != '\n' ? 1 : 0);
}

/// Drives the real tracker, recovery and revert through many turns of random
/// edits and checks every turn against `Tree`. Each property is one the user
/// relies on, stated without reference to how it is achieved, so a new way of
/// reusing or skipping work cannot pass by breaking one quietly:
///  1. the summary lists exactly the files that differ from the turn's start;
///  2. every one of them carries a diff, and the diff unwinds to that start;
///  3. every original the turn recorded is readable and is what the file held;
///  4. a revert puts the whole tree back the way the turn found it.
class TurnSequence {
public:
    explicit TurnSequence(std::uint32_t seed) : rng_(seed) {
        for (const auto& name : names_) {
            if (chance(2)) write(name, text(name));
        }
    }

    void run(int turns) {
        for (int turn = 0; turn < turns; ++turn) {
            INFO("turn " << turn);
            if (chance(4)) edit_by_hand();
            run_turn();
        }
    }

private:
    bool chance(unsigned in) { return std::uniform_int_distribution<unsigned>(0, in - 1)(rng_) == 0; }
    const std::string& pick() { return names_[std::uniform_int_distribution<std::size_t>(0, names_.size() - 1)(rng_)]; }

    std::string text(const std::string& name) {
        std::string body;
        const auto count = std::uniform_int_distribution<int>(1, 6)(rng_);
        for (int i = 0; i < count; ++i) body += std::format("{} line {} v{}\n", name, i, version_++);
        return body;
    }

    void write(const std::string& name, const std::string& content) {
        put(dir_.path / name, content);
        tree_[name] = content;
    }
    void remove(const std::string& name) {
        fs::remove(dir_.path / name);
        tree_.erase(name);
    }

    /// One random mutation of `name`, the kinds a turn or a user makes.
    void mutate(const std::string& name) {
        const auto found = tree_.find(name);
        if (found == tree_.end()) return write(name, text(name));  // Created.
        switch (std::uniform_int_distribution<int>(0, 3)(rng_)) {
            case 0: return write(name, found->second + std::format("appended v{}\n", version_++));
            case 1: return write(name, text(name));                 // Rewritten.
            case 2: return write(name, std::string(found->second)); // Same bytes, new stamp.
            default: return remove(name);
        }
    }

    /// An edit between turns, made outside Filo: the next turn must start
    /// from it, not from what the previous turn left.
    void edit_by_hand() { mutate(pick()); }

    void run_turn() {
        // Half the turns see files long after they were written, so stamps
        // are trusted and earlier turns' records are reused; half see them
        // fresh, so everything is read again.
        const bool settled = chance(2);
        INFO("settled clock: " << settled);
        const auto turn_directory = core::changes::RecoveryStore::new_directory(storage_.path);
        const auto recovery = std::make_shared<core::changes::TurnRecovery>(
            turn_directory, dir_.path, [settled] {
                return settled ? settled_clock() : std::chrono::system_clock::now();
            });
        TurnChangeTracker tracker(dir_.path);
        tracker.set_recovery(recovery);
        const Tree start = tree_;

        const auto calls = std::uniform_int_distribution<int>(1, 3)(rng_);
        for (int call = 0; call < calls; ++call) {
            std::vector<std::string> touched;
            const auto count = std::uniform_int_distribution<int>(1, 3)(rng_);
            for (int i = 0; i < count; ++i) touched.push_back(pick());
            if (chance(2)) {
                // A shell command: only an inventory of the workspace sees it.
                edit(tracker, MutationScope{.unbounded = true, .tool = "run_terminal_command"},
                     [&] { for (const auto& name : touched) mutate(name); });
            } else {
                MutationScope scope;
                for (const auto& name : touched) scope.paths.push_back(dir_.path / name);
                edit(tracker, scope, [&] { for (const auto& name : touched) mutate(name); });
            }
        }
        tracker.settle();
        tracker.finish_recovery();
        const auto changes = tracker.changes();

        check_summary(changes, start);
        check_recorded_originals(turn_directory, start);
        if (!changes.files.empty() && chance(3)) {
            core::changes::WorkspaceMutationGate mutations;
            const auto result = revert_turn(changes, dir_.path, mutations);
            REQUIRE(result.reverted());
            tree_ = start;
            CHECK(snapshot_tree(dir_.path) == std::vector<std::pair<std::string, std::string>>(start.begin(), start.end()));
        }
    }

    void check_summary(const TurnChanges& changes, const Tree& start) const {
        std::vector<std::string> expected;
        for (const auto& name : std::set<std::string>(names_.begin(), names_.end())) {
            const auto before = start.find(name);
            const auto after = tree_.find(name);
            const bool existed = before != start.end(), exists = after != tree_.end();
            if (existed != exists || (existed && before->second != after->second)) expected.push_back(name);
        }
        std::vector<std::string> listed;
        for (const auto& file : changes.files) listed.push_back(file.path);
        REQUIRE(listed == expected);
        CHECK_FALSE(changes.partial_enumeration);

        for (const auto& file : changes.files) {
            INFO("file " << file.path << " (" << core::changes::to_string(file.content) << ")");
            REQUIRE(file.content == FileChangeContent::Text);
            REQUIRE_FALSE(file.diff.empty());
            const auto before = start.find(file.path);
            const auto after = tree_.find(file.path);
            if (before == start.end()) {
                CHECK(file.kind == FileChangeKind::Added);
                CHECK(file.added == lines_in(after->second));
            } else if (after == tree_.end()) {
                CHECK(file.kind == FileChangeKind::Deleted);
                CHECK(file.deleted == lines_in(before->second));
            } else {
                CHECK(file.kind == FileChangeKind::Modified);
                const auto hunks = core::changes::detail::parse_hunks(file.diff);
                REQUIRE(hunks);
                CHECK(core::changes::detail::reverse_apply(after->second, *hunks) == before->second);
            }
        }
    }

    void check_recorded_originals(const fs::path& turn_directory, const Tree& start) const {
        const core::changes::RecoveryStore store(turn_directory);
        auto manifest = store.load(dir_.path);
        if (!manifest) return;  // A turn that saw no file records nothing.
        for (const auto& [path, entry] : manifest->entries) {
            const auto name = fs::relative(path, dir_.path).generic_string();
            INFO("recorded original of " << name);
            const auto held = start.find(name);
            if (entry.before.hash.empty()) {
                CHECK(held == start.end());
                continue;
            }
            REQUIRE(held != start.end());
            const auto bytes = store.original(entry.before);
            REQUIRE(bytes.has_value());
            CHECK(*bytes == held->second);
        }
    }

    std::mt19937 rng_;
    TempDir dir_, storage_;
    const std::vector<std::string> names_{"README.md", "src/app.cpp", "src/app.hpp",
                                          "docs/notes.md", "docs/deep/plan.md", "build.txt"};
    Tree tree_;
    int version_ = 0;
};

} // namespace

TEST_CASE("Every turn of a session summarizes, records and reverts what it changed",
          "[changes][recovery][store][model]") {
    // Fixed seeds: a failure names the one that reproduces it.
    const auto seed = GENERATE(range(1u, 41u));
    INFO("seed " << seed);
    TurnSequence(seed).run(10);
}
