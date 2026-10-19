#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/changes/TurnChangeTracker.hpp"
#include "core/changes/TurnRevert.hpp"
#include "core/changes/WorkspaceMutationGate.hpp"
#include "core/changes/TurnRecovery.hpp"
#include "core/changes/WorkspaceInventory.hpp"
#include "core/utils/InterprocessFile.hpp"
#include "core/tools/ShellTool.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "TestSessionContext.hpp"
#include <sys/wait.h>
#include <signal.h>

#include <algorithm>
#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <random>
#include <string>
#include <thread>
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
    using Gate = core::changes::WorkspaceMutationGate;
    const std::vector<fs::path> roots{dir.path};
    {
        auto writer = mutations.acquire(roots, Gate::Access::ScopedWrite);
        REQUIRE(writer);
        const auto result = revert_turn(tracker.changes(), dir.path, mutations);
        REQUIRE_FALSE(result.reverted());
        REQUIRE(result.refused.size() == 1);
        CHECK(read(file) == "after\n");
    }
    CHECK(revert_turn(tracker.changes(), dir.path, mutations).reverted());
    CHECK(read(file) == "before\n");
}

TEST_CASE("Workspace mutation tokens can be transferred and cancel waiting writers",
          "[changes][revert][concurrency]") {
    core::changes::WorkspaceMutationGate mutations;
    TempDir dir;
    using Gate = core::changes::WorkspaceMutationGate;
    const std::vector<fs::path> roots{dir.path};
    auto restore = mutations.acquire(roots, Gate::Access::Exclusive);
    REQUIRE(restore);
    const auto blocked = mutations.acquire(roots, Gate::Access::ScopedWrite,
        std::chrono::seconds(1), [] { return true; });
    CHECK_FALSE(blocked);
    std::thread release([token = std::move(*restore)] {});
    release.join();
    CHECK(mutations.acquire(roots, Gate::Access::Exclusive, std::chrono::milliseconds(0)));
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
    for (const auto& blob : fs::directory_iterator(checkpoint / "objects")) put(blob.path(), "corrupted");
    TurnChanges changes;
    changes.files.push_back({.path = "app.cpp"});
    changes.recovery_path = checkpoint.string();
    CHECK_FALSE(revert_turn(changes, dir.path, mutations).reverted());
    CHECK(read(file) == "changed\n");
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
    using Gate = core::changes::WorkspaceMutationGate;
    const std::vector<fs::path> nested{dir.path / "nested"}, parent{dir.path}, other{unrelated.path};
    auto writer = mutations.acquire(nested, Gate::Access::ScopedWrite);
    REQUIRE(writer);
    CHECK_FALSE(mutations.acquire(parent, Gate::Access::Exclusive, std::chrono::milliseconds(0)));
    CHECK(mutations.acquire(other, Gate::Access::Exclusive, std::chrono::milliseconds(0)));
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
    const auto result = revert_turn(tracker->changes(), dir.path, mutations, [&shell](auto roots) {
        return std::ranges::all_of(roots, [&shell](const auto& root) {
            return shell.sessions().quiesce_workspace(root);
        });
    });
    INFO(result.error.value_or(""));
    for (const auto& refusal : result.refused) INFO(refusal.reason);
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
    CHECK(static_cast<std::size_t>(std::distance(fs::directory_iterator(storage.path), fs::directory_iterator{}))
        == core::changes::TurnRecovery::kRetainedTurns);
}

TEST_CASE("Independent execution runtimes do not share hidden writer state", "[changes][concurrency]") {
    TempDir dir;
    const std::vector<fs::path> roots{dir.path};
    core::changes::WorkspaceMutationGate first, second;
    auto writer = first.acquire(roots, core::changes::WorkspaceMutationGate::Access::Exclusive);
    REQUIRE(writer);
    CHECK_FALSE(first.acquire(roots, core::changes::WorkspaceMutationGate::Access::ScopedWrite,
                              std::chrono::milliseconds(0)));
    CHECK(second.acquire(roots, core::changes::WorkspaceMutationGate::Access::Exclusive,
                         std::chrono::milliseconds(0)));
}

TEST_CASE("Detached writer reservations safely outlive their runtime owner", "[changes][concurrency]") {
    TempDir dir;
    const std::vector<fs::path> roots{dir.path};
    auto runtime = std::make_unique<core::changes::WorkspaceMutationGate>();
    auto reservation = runtime->acquire(roots, core::changes::WorkspaceMutationGate::Access::Exclusive);
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
    auto tracker = std::make_shared<TurnChangeTracker>(dir.path, mutations);
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
        auto writer = mutations->acquire(roots, core::changes::WorkspaceMutationGate::Access::Exclusive);
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
    tracker.set_observation_filter([&](const auto&) {
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
    auto writer = mutations.acquire(roots, core::changes::WorkspaceMutationGate::Access::ScopedWrite);
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
