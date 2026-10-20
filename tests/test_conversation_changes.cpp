#include <catch2/catch_test_macros.hpp>

#include "core/changes/ConversationChanges.hpp"
#include "core/changes/TurnChangeTracker.hpp"

#include <atomic>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

using core::changes::ConversationChanges;
using core::changes::FileChange;
using core::changes::FileChangeKind;
using core::changes::MutationScope;
using core::changes::TurnChangeTracker;
using core::changes::TurnChanges;
using core::changes::conversation_changes;

namespace fs = std::filesystem;

namespace {

struct TempDir {
    fs::path path;
    TempDir() {
        static std::atomic<unsigned> counter{0};
        path = fs::temp_directory_path()
            / std::format("filo_conversation_changes_{}_{}_{}", ::getpid(),
                          std::random_device{}(), counter.fetch_add(1));
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

/// One agent turn: a fresh tracker around one tool call, as the agent drives it.
TurnChanges turn(const fs::path& root, const MutationScope& scope,
                 const std::function<void()>& mutate) {
    TurnChangeTracker tracker(root);
    tracker.track(scope, [&] {
        mutate();
        return 0;
    });
    return tracker.changes();
}

MutationScope scope_of(std::initializer_list<fs::path> paths) {
    return MutationScope{.paths = paths};
}

const FileChange* change_of(const ConversationChanges& net, std::string_view path) {
    for (const auto& change : net.changes.files) {
        if (change.path == path) {
            return &change;
        }
    }
    return nullptr;
}

} // namespace

TEST_CASE("Turns that edit one file net into a single diff against the original",
          "[changes][conversation]") {
    TempDir dir;
    const auto file = dir.path / "app.cpp";
    put(file, "one\ntwo\nthree\n");

    std::vector<TurnChanges> turns;
    turns.push_back(turn(dir.path, scope_of({file}), [&] { put(file, "one\nTWO\nthree\n"); }));
    turns.push_back(turn(dir.path, scope_of({file}),
                         [&] { put(file, "one\nTWO\nthree\nfour\n"); }));

    const auto net = conversation_changes(turns, dir.path);
    CHECK(net.gaps.empty());
    REQUIRE(net.changes.files.size() == 1);
    const FileChange& change = net.changes.files.front();
    CHECK(change.kind == FileChangeKind::Modified);
    CHECK(change.added == 2);
    CHECK(change.deleted == 1);
    CHECK(change.diff.find("-two\n") != std::string::npos);
    CHECK(change.diff.find("+TWO\n") != std::string::npos);
    CHECK(change.diff.find("+four\n") != std::string::npos);
}

TEST_CASE("Edits that cancel out across turns leave nothing to compare",
          "[changes][conversation]") {
    TempDir dir;
    const auto file = dir.path / "app.cpp";
    put(file, "one\n");

    std::vector<TurnChanges> turns;
    turns.push_back(turn(dir.path, scope_of({file}), [&] { put(file, "one\ntwo\n"); }));
    turns.push_back(turn(dir.path, scope_of({file}), [&] { put(file, "one\n"); }));

    const auto net = conversation_changes(turns, dir.path);
    CHECK(net.gaps.empty());
    CHECK(net.changes.files.empty());
}

TEST_CASE("A file created and then edited compares as one new file",
          "[changes][conversation]") {
    TempDir dir;
    const auto file = dir.path / "src" / "new.cpp";

    std::vector<TurnChanges> turns;
    turns.push_back(turn(dir.path, scope_of({file}), [&] { put(file, "a\n"); }));
    turns.push_back(turn(dir.path, scope_of({file}), [&] { put(file, "a\nb\n"); }));

    const auto net = conversation_changes(turns, dir.path);
    CHECK(net.gaps.empty());
    const auto* change = change_of(net, "src/new.cpp");
    REQUIRE(change != nullptr);
    CHECK(change->kind == FileChangeKind::Added);
    CHECK(change->added == 2);
    CHECK(change->deleted == 0);
}

TEST_CASE("A file created and deleted again is not part of the comparison",
          "[changes][conversation]") {
    TempDir dir;
    const auto scratch = dir.path / "scratch.txt";

    std::vector<TurnChanges> turns;
    turns.push_back(turn(dir.path, scope_of({scratch}), [&] { put(scratch, "tmp\n"); }));
    turns.push_back(turn(dir.path, scope_of({scratch}), [&] { fs::remove(scratch); }));

    const auto net = conversation_changes(turns, dir.path);
    CHECK(net.gaps.empty());
    CHECK(net.changes.files.empty());
}

TEST_CASE("A file deleted after an edit compares as deleted from its original",
          "[changes][conversation]") {
    TempDir dir;
    const auto file = dir.path / "old.txt";
    put(file, "original\n");

    std::vector<TurnChanges> turns;
    turns.push_back(turn(dir.path, scope_of({file}), [&] { put(file, "edited\n"); }));
    turns.push_back(turn(dir.path, scope_of({file}), [&] { fs::remove(file); }));

    const auto net = conversation_changes(turns, dir.path);
    CHECK(net.gaps.empty());
    const auto* change = change_of(net, "old.txt");
    REQUIRE(change != nullptr);
    CHECK(change->kind == FileChangeKind::Deleted);
    CHECK(change->diff.find("-original\n") != std::string::npos);
    CHECK(change->diff.find("edited") == std::string::npos);
}

TEST_CASE("A rename is followed back to the file's original path",
          "[changes][conversation]") {
    TempDir dir;
    const auto from = dir.path / "a.txt";
    const auto to = dir.path / "b.txt";
    put(from, "one\n");

    std::vector<TurnChanges> turns;
    turns.push_back(turn(dir.path, scope_of({from}), [&] { put(from, "one\ntwo\n"); }));
    turns.push_back(turn(dir.path, MutationScope{.paths = {from, to}, .moves = {{from, to}}},
                         [&] { fs::rename(from, to); }));
    turns.push_back(turn(dir.path, scope_of({to}), [&] { put(to, "one\ntwo\nthree\n"); }));

    const auto net = conversation_changes(turns, dir.path);
    CHECK(net.gaps.empty());
    REQUIRE(net.changes.files.size() == 1);
    const FileChange& change = net.changes.files.front();
    CHECK(change.kind == FileChangeKind::Renamed);
    CHECK(change.path == "b.txt");
    CHECK(change.previous_path == "a.txt");
    CHECK(change.added == 2);
    CHECK(change.deleted == 0);
}

TEST_CASE("A file edited outside the conversation becomes a gap, not a guess",
          "[changes][conversation]") {
    TempDir dir;
    const auto touched = dir.path / "touched.txt";
    const auto clean = dir.path / "clean.txt";
    put(touched, "one\n");
    put(clean, "one\n");

    std::vector<TurnChanges> turns;
    turns.push_back(turn(dir.path, scope_of({touched, clean}), [&] {
        put(touched, "one\ntwo\n");
        put(clean, "one\ntwo\n");
    }));
    put(touched, "one\ntwo, by hand\n");  // Between turns, over what the first one wrote.
    turns.push_back(turn(dir.path, scope_of({touched}),
                         [&] { put(touched, "zero\none\ntwo, by hand\n"); }));

    const auto net = conversation_changes(turns, dir.path);
    REQUIRE(net.gaps.size() == 1);
    CHECK(net.gaps.front().path == "touched.txt");
    CHECK(net.gaps.front().reason == "changed outside the conversation");
    REQUIRE(net.changes.files.size() == 1);
    CHECK(net.changes.files.front().path == "clean.txt");
}

TEST_CASE("An edit by hand away from the turns' lines is on both sides, not in the diff",
          "[changes][conversation]") {
    TempDir dir;
    const auto file = dir.path / "app.cpp";
    put(file, "1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n");

    std::vector<TurnChanges> turns;
    turns.push_back(turn(dir.path, scope_of({file}),
                         [&] { put(file, "1\nTWO\n3\n4\n5\n6\n7\n8\n9\n10\n"); }));
    put(file, "1\nTWO\n3\n4\n5\n6\n7\n8\n9\nTEN\n");  // By hand, far from line 2.

    const auto net = conversation_changes(turns, dir.path);
    CHECK(net.gaps.empty());
    REQUIRE(net.changes.files.size() == 1);
    CHECK(net.changes.files.front().added == 1);
    CHECK(net.changes.files.front().deleted == 1);
    CHECK(net.changes.files.front().diff.find("TEN") == std::string::npos);
}

TEST_CASE("A reverted turn is left out, as the workspace no longer holds it",
          "[changes][conversation]") {
    TempDir dir;
    const auto file = dir.path / "app.cpp";
    put(file, "one\n");

    std::vector<TurnChanges> turns;
    turns.push_back(turn(dir.path, scope_of({file}), [&] { put(file, "one\ntwo\n"); }));
    turns.push_back(turn(dir.path, scope_of({file}), [&] { put(file, "one\ntwo\nbad\n"); }));
    put(file, "one\ntwo\n");  // What /revert put back.
    turns.back().reverted = true;

    const auto net = conversation_changes(turns, dir.path);
    CHECK(net.gaps.empty());
    REQUIRE(net.changes.files.size() == 1);
    CHECK(net.changes.files.front().added == 1);
    CHECK(net.changes.files.front().diff.find("bad") == std::string::npos);
}

TEST_CASE("The conversation keeps every caveat any of its turns declared",
          "[changes][conversation]") {
    TempDir dir;
    std::vector<TurnChanges> turns(2);
    turns[0].unscoped_mutations = true;
    turns[0].unscoped_tools = {"run_terminal_command"};
    turns[1].unscoped_mutations = true;
    turns[1].unscoped_tools = {"python", "run_terminal_command"};
    turns[1].partial_enumeration = true;

    const auto net = conversation_changes(turns, dir.path);
    CHECK(net.changes.partial_enumeration);
    CHECK(net.changes.unscoped_mutations);
    CHECK(net.changes.unscoped_tools
          == std::vector<std::string>{"run_terminal_command", "python"});
}

TEST_CASE("Conversation coverage requires an inventory for every unscoped turn",
          "[changes][conversation][regression]") {
    TempDir dir;
    std::vector<TurnChanges> turns(3);
    turns[0].unscoped_mutations = true;
    turns[0].workspace_observed = true;
    turns[2].unscoped_mutations = true;
    turns[2].workspace_observed = true;
    bool covered = true;
    SECTION("all shell turns observed, with a scoped turn in between") {}
    SECTION("an earlier turn lacked an inventory") { turns[0].workspace_observed = false; covered = false; }
    SECTION("a later turn lacked an inventory") { turns[2].workspace_observed = false; covered = false; }
    SECTION("a reverted turn does not weaken coverage") {
        turns[1].unscoped_mutations = true;
        turns[1].reverted = true;
    }
    const auto net = conversation_changes(turns, dir.path);
    CHECK(net.changes.unscoped_mutations);
    CHECK(net.changes.workspace_observed == covered);
}
