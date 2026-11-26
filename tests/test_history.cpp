#include <catch2/catch_test_macros.hpp>
#include "core/history/PromptHistoryStore.hpp"
#include "core/persistence/AmendedFile.hpp"
#include "core/utils/InterprocessFile.hpp"
#include <filesystem>
#include <format>
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>

using namespace core::history;

TEST_CASE("PromptHistoryStore default path respects XDG_DATA_HOME", "[history]") {
    const char* original_xdg = std::getenv("XDG_DATA_HOME");
    
    // Test with XDG_DATA_HOME set
    setenv("XDG_DATA_HOME", "/tmp/xdg_test", 1);
    auto path1 = PromptHistoryStore::default_history_path();
    CHECK(path1 == std::filesystem::path("/tmp/xdg_test/filo/history.json"));
    
    // Restore environment
    if (original_xdg) {
        setenv("XDG_DATA_HOME", original_xdg, 1);
    } else {
        unsetenv("XDG_DATA_HOME");
    }
}

TEST_CASE("PromptHistoryStore load/save roundtrip", "[history]") {
    auto temp_path = std::filesystem::temp_directory_path() / "filo_test_history.json";
    
    // Cleanup before test
    std::filesystem::remove(temp_path);
    
    {
        PromptHistoryStore store(temp_path);
        CHECK(store.load() == true);  // Empty/non-existent is OK
        CHECK(store.empty() == true);
        
        store.add("first prompt");
        store.add("second prompt");
        store.add("third prompt");
        
        CHECK(store.size() == 3);
        CHECK(store.save() == true);
    }
    
    {
        PromptHistoryStore store(temp_path);
        CHECK(store.load() == true);
        CHECK(store.size() == 3);
        
        const auto& entries = store.entries();
        CHECK(entries[0] == "first prompt");
        CHECK(entries[1] == "second prompt");
        CHECK(entries[2] == "third prompt");
    }
    
    // Cleanup
    std::filesystem::remove(temp_path);
}

TEST_CASE("PromptHistoryStore deduplicates consecutive entries", "[history]") {
    auto temp_path = std::filesystem::temp_directory_path() / "filo_test_history_dup.json";
    std::filesystem::remove(temp_path);
    
    PromptHistoryStore store(temp_path);
    REQUIRE(store.load());
    
    store.add("hello");
    store.add("hello");  // Duplicate
    store.add("world");
    store.add("world");  // Duplicate
    store.add("hello");  // Not consecutive, should be added
    
    CHECK(store.size() == 3);
    
    const auto& entries = store.entries();
    CHECK(entries[0] == "hello");
    CHECK(entries[1] == "world");
    CHECK(entries[2] == "hello");
    
    std::filesystem::remove(temp_path);
}

TEST_CASE("PromptHistoryStore ignores empty entries", "[history]") {
    auto temp_path = std::filesystem::temp_directory_path() / "filo_test_history_empty.json";
    std::filesystem::remove(temp_path);
    
    PromptHistoryStore store(temp_path);
    REQUIRE(store.load());
    
    store.add("valid");
    store.add("");  // Empty - should be ignored
    store.add("also valid");
    
    CHECK(store.size() == 2);
    
    std::filesystem::remove(temp_path);
}

TEST_CASE("PromptHistoryStore size limit enforcement", "[history]") {
    auto temp_path = std::filesystem::temp_directory_path() / "filo_test_history_limit.json";
    std::filesystem::remove(temp_path);
    
    PromptHistoryStore store(temp_path);
    REQUIRE(store.load());
    store.set_max_entries(5);
    
    for (int i = 0; i < 10; ++i) {
        store.add(std::format("entry {}", i));
    }
    
    CHECK(store.size() == 5);
    
    const auto& entries = store.entries();
    // Should keep the most recent 5 entries
    CHECK(entries[0] == "entry 5");
    CHECK(entries[4] == "entry 9");
    
    std::filesystem::remove(temp_path);
}

TEST_CASE("PromptHistoryStore clear removes all entries", "[history]") {
    auto temp_path = std::filesystem::temp_directory_path() / "filo_test_history_clear.json";
    std::filesystem::remove(temp_path);
    
    PromptHistoryStore store(temp_path);
    REQUIRE(store.load());
    
    store.add("entry 1");
    store.add("entry 2");
    CHECK(store.size() == 2);
    
    store.clear();
    CHECK(store.size() == 0);
    CHECK(store.empty() == true);
    
    std::filesystem::remove(temp_path);
}

TEST_CASE("PromptHistoryStore entries_newest_first returns reversed order", "[history]") {
    PromptHistoryStore store(std::filesystem::path("/dev/null"));
    
    store.add("oldest");
    store.add("middle");
    store.add("newest");
    
    auto newest_first = store.entries_newest_first();
    REQUIRE(newest_first.size() == 3);
    CHECK(newest_first[0] == "newest");
    CHECK(newest_first[1] == "middle");
    CHECK(newest_first[2] == "oldest");
}

TEST_CASE("PromptHistoryStore remove_at_and_save removes by index", "[history]") {
    auto temp_path = std::filesystem::temp_directory_path() / "filo_test_history_remove.json";
    std::filesystem::remove(temp_path);

    {
        PromptHistoryStore store(temp_path);
        REQUIRE(store.load());
        store.add("alpha");
        store.add("beta");
        store.add("gamma");
        REQUIRE(store.save());
    }

    {
        PromptHistoryStore store(temp_path);
        REQUIRE(store.load());
        REQUIRE(store.size() == 3);

        // Remove index 1 ("beta") — oldest-first ordering.
        REQUIRE(store.remove_at_and_save(1));

        REQUIRE(store.load());
        REQUIRE(store.size() == 2);
        const auto& entries = store.entries();
        CHECK(entries[0] == "alpha");
        CHECK(entries[1] == "gamma");
    }

    std::filesystem::remove(temp_path);
}

TEST_CASE("PromptHistoryStore remove_at_and_save rejects out-of-range index", "[history]") {
    auto temp_path = std::filesystem::temp_directory_path() / "filo_test_history_remove_oor.json";
    std::filesystem::remove(temp_path);

    PromptHistoryStore store(temp_path);
    REQUIRE(store.load());
    store.add("only");

    std::string error;
    REQUIRE_FALSE(store.remove_at_and_save(99, &error));
    REQUIRE_FALSE(error.empty());
    REQUIRE(store.size() == 1);

    std::filesystem::remove(temp_path);
}

TEST_CASE("PersistentPromptHistory navigation", "[history]") {
    auto temp_path = std::filesystem::temp_directory_path() / "filo_test_history_nav.json";
    std::filesystem::remove(temp_path);
    
    auto store = std::make_shared<PromptHistoryStore>(temp_path);
    REQUIRE(store->load());
    store->add("first prompt");
    store->add("second prompt");
    store->add("third prompt");
    
    PersistentPromptHistory history(store);
    
    std::string input = "current typing";
    int cursor = 12;
    
    // Navigate previous (should show third prompt)
    CHECK(history.navigate_prev({}, input, cursor) == true);
    CHECK(input == "third prompt");
    CHECK(cursor == 12);  // Cursor at end
    
    // Navigate previous again (should show second prompt)
    CHECK(history.navigate_prev({}, input, cursor) == true);
    CHECK(input == "second prompt");
    
    // Navigate previous again (should show first prompt)
    CHECK(history.navigate_prev({}, input, cursor) == true);
    CHECK(input == "first prompt");
    
    // At oldest entry - should stay at first
    CHECK(history.navigate_prev({}, input, cursor) == true);
    CHECK(input == "first prompt");
    
    // Navigate next (should show second prompt)
    CHECK(history.navigate_next({}, input, cursor) == true);
    CHECK(input == "second prompt");
    
    // Navigate next (should show third prompt)
    CHECK(history.navigate_next({}, input, cursor) == true);
    CHECK(input == "third prompt");
    
    // Navigate next (should restore original input)
    CHECK(history.navigate_next({}, input, cursor) == true);
    CHECK(input == "current typing");
    
    // At current input - navigate_next should return false
    CHECK(history.navigate_next({}, input, cursor) == false);
    
    std::filesystem::remove(temp_path);
}

namespace {

struct TempHistoryFile {
    std::filesystem::path path;

    explicit TempHistoryFile(std::string_view name)
        : path(std::filesystem::temp_directory_path()
               / std::format("{}_{}.json", name, ::getpid())) {
        cleanup();
    }
    ~TempHistoryFile() { cleanup(); }

    void cleanup() const {
        std::error_code ec;
        std::filesystem::remove(path, ec);
        std::filesystem::remove(core::utils::lock_path_for(path), ec);
        std::filesystem::remove(log_path(), ec);
    }
    /// Prompts appended since the history was last saved whole.
    [[nodiscard]] std::filesystem::path log_path() const {
        return core::persistence::AmendedFile::log_path(path);
    }
};

const PromptHistoryScope kThreadA{.session_id = "session-a", .workspace = "/work/app"};
const PromptHistoryScope kThreadB{.session_id = "session-b", .workspace = "/work/app"};
const PromptHistoryScope kOtherProject{.session_id = "session-c", .workspace = "/work/lib"};

} // namespace

TEST_CASE("PersistentPromptHistory prefers the composer's own thread",
          "[history][threads]") {
    TempHistoryFile file("filo_test_history_scoped");
    auto store = std::make_shared<PromptHistoryStore>(file.path);
    PersistentPromptHistory history(store);

    history.save(kThreadA, "a1");
    history.save(kOtherProject, "lib1");
    history.save(kThreadB, "b1");
    history.save(kThreadA, "a2");
    history.save(kThreadB, "b2");

    std::string input;
    int cursor = 0;
    std::vector<std::string> seen;
    while (history.navigate_prev(kThreadA, input, cursor)
           && (seen.empty() || seen.back() != input)) {
        seen.push_back(input);
    }
    // Own thread newest first, then the same workspace, then everything else.
    CHECK(seen == std::vector<std::string>{"a2", "a1", "b2", "b1", "lib1"});

    input.clear();
    REQUIRE(history.navigate_prev(kThreadB, input, cursor));
    CHECK(input == "b2");
}

TEST_CASE("PersistentPromptHistory deduplicates across tiers", "[history][threads]") {
    TempHistoryFile file("filo_test_history_dedupe");
    auto store = std::make_shared<PromptHistoryStore>(file.path);
    PersistentPromptHistory history(store);

    history.save(kThreadA, "run the tests");
    history.save(kThreadB, "run the tests");
    history.save(kThreadB, "other");

    std::string input;
    int cursor = 0;
    REQUIRE(history.navigate_prev(kThreadA, input, cursor));
    CHECK(input == "run the tests");
    REQUIRE(history.navigate_prev(kThreadA, input, cursor));
    CHECK(input == "other");
    REQUIRE(history.navigate_prev(kThreadA, input, cursor));
    CHECK(input == "other");  // oldest reached; no duplicate shown
}

TEST_CASE("PersistentPromptHistory browses each thread independently",
          "[history][threads]") {
    TempHistoryFile file("filo_test_history_independent");
    auto store = std::make_shared<PromptHistoryStore>(file.path);
    PersistentPromptHistory history(store);
    history.save(kThreadA, "a1");
    history.save(kThreadB, "b1");

    // Thread A starts browsing with a draft, then the user switches tabs.
    std::string a_input = "A draft";
    int a_cursor = 0;
    REQUIRE(history.navigate_prev(kThreadA, a_input, a_cursor));
    CHECK(a_input == "a1");

    // Thread B's composer is untouched by A's browse.
    std::string b_input = "B draft";
    int b_cursor = 0;
    CHECK_FALSE(history.navigate_next(kThreadB, b_input, b_cursor));
    CHECK(b_input == "B draft");
    REQUIRE(history.navigate_prev(kThreadB, b_input, b_cursor));
    CHECK(b_input == "b1");
    REQUIRE(history.navigate_next(kThreadB, b_input, b_cursor));
    CHECK(b_input == "B draft");

    // Back on A: Down still restores A's own draft.
    REQUIRE(history.navigate_next(kThreadA, a_input, a_cursor));
    CHECK(a_input == "A draft");
    CHECK(a_cursor == static_cast<int>(a_input.size()));
}

TEST_CASE("PersistentPromptHistory forgets a closed thread's browse",
          "[history][threads]") {
    TempHistoryFile file("filo_test_history_forget");
    auto store = std::make_shared<PromptHistoryStore>(file.path);
    PersistentPromptHistory history(store);
    history.save(kThreadA, "a1");

    std::string input = "draft";
    int cursor = 0;
    REQUIRE(history.navigate_prev(kThreadA, input, cursor));
    CHECK(input == "a1");

    history.forget_session(kThreadA.session_id);
    CHECK_FALSE(history.navigate_next(kThreadA, input, cursor));
    CHECK(input == "a1");
    CHECK(history.size() == 1);  // Entries outlive the thread.
}

TEST_CASE("PersistentPromptHistory never overwrites an edited recall",
          "[history][threads]") {
    TempHistoryFile file("filo_test_history_edit");
    auto store = std::make_shared<PromptHistoryStore>(file.path);
    PersistentPromptHistory history(store);
    history.save(kThreadA, "first");
    history.save(kThreadA, "second");

    std::string input = "draft";
    int cursor = 0;
    REQUIRE(history.navigate_prev(kThreadA, input, cursor));
    CHECK(input == "second");
    input += " edited";

    // Down must not discard the edit.
    CHECK_FALSE(history.navigate_next(kThreadA, input, cursor));
    CHECK(input == "second edited");

    // Up starts over and keeps the edit as the draft to come back to.
    REQUIRE(history.navigate_prev(kThreadA, input, cursor));
    CHECK(input == "second");
    REQUIRE(history.navigate_next(kThreadA, input, cursor));
    CHECK(input == "second edited");
}

TEST_CASE("PersistentPromptHistory skips an entry equal to the draft",
          "[history][threads]") {
    TempHistoryFile file("filo_test_history_same_draft");
    auto store = std::make_shared<PromptHistoryStore>(file.path);
    PersistentPromptHistory history(store);
    history.save(kThreadA, "older");
    history.save(kThreadA, "stashed draft");

    std::string input = "stashed draft";
    int cursor = 0;
    REQUIRE(history.navigate_prev(kThreadA, input, cursor));
    CHECK(input == "older");
}

TEST_CASE("PersistentPromptHistory follows a thread across session id changes",
          "[history][threads]") {
    TempHistoryFile file("filo_test_history_rename");
    auto store = std::make_shared<PromptHistoryStore>(file.path);
    PersistentPromptHistory history(store);
    history.save(kThreadA, "before rotation");
    history.save(kThreadB, "from b");

    const PromptHistoryScope rotated{.session_id = "session-a2", .workspace = "/work/app"};
    history.rename_session(kThreadA.session_id, rotated.session_id);

    std::string input;
    int cursor = 0;
    REQUIRE(history.navigate_prev(rotated, input, cursor));
    CHECK(input == "before rotation");

    // The relabel is persisted for the next process.
    PromptHistoryStore reloaded(file.path);
    REQUIRE(reloaded.load());
    REQUIRE(reloaded.records().size() == 2);
    CHECK(reloaded.records()[0].session_id == "session-a2");
    CHECK(reloaded.records()[1].session_id == "session-b");
}

TEST_CASE("PromptHistoryStore persists entry scope and reads legacy files",
          "[history][threads]") {
    TempHistoryFile file("filo_test_history_scope_roundtrip");
    {
        PromptHistoryStore store(file.path);
        REQUIRE(store.append_and_save(PromptHistoryEntry{
            .text = "scoped \"quoted\"",
            .session_id = "s1",
            .workspace = "/w",
        }));
        REQUIRE(store.append_and_save("unscoped"));
    }
    {
        PromptHistoryStore store(file.path);
        REQUIRE(store.load());
        REQUIRE(store.records().size() == 2);
        CHECK(store.records()[0] == PromptHistoryEntry{
            .text = "scoped \"quoted\"", .session_id = "s1", .workspace = "/w"});
        CHECK(store.records()[1] == PromptHistoryEntry{.text = "unscoped"});
    }

    // Version 1 (and an older build rewriting version 2 without `meta`).
    file.cleanup();
    {
        std::ofstream out(file.path);
        out << R"({"version":1,"entries":["old one","old two"]})";
    }
    {
        PromptHistoryStore store(file.path);
        REQUIRE(store.load());
        CHECK(store.entries() == std::vector<std::string>{"old one", "old two"});
        CHECK(store.records()[0].session_id.empty());
    }

    // A `meta` array that no longer lines up with `entries` is ignored.
    {
        std::ofstream out(file.path);
        out << R"({"version":2,"entries":["x","y"],"meta":[{"session":"s"}]})";
    }
    {
        PromptHistoryStore store(file.path);
        REQUIRE(store.load());
        REQUIRE(store.records().size() == 2);
        CHECK(store.records()[0].session_id.empty());
    }
}

TEST_CASE("PromptHistoryStore keeps a repeat sent from another thread",
          "[history][threads]") {
    PromptHistoryStore store(std::filesystem::path("/dev/null"));
    store.add(PromptHistoryEntry{.text = "yes", .session_id = "a"});
    store.add(PromptHistoryEntry{.text = "yes", .session_id = "a"});
    store.add(PromptHistoryEntry{.text = "yes", .session_id = "b"});
    CHECK(store.size() == 2);
}

TEST_CASE("PersistentPromptHistory empty history navigation", "[history]") {
    auto temp_path = std::filesystem::temp_directory_path() / "filo_test_history_empty_nav.json";
    std::filesystem::remove(temp_path);
    
    auto store = std::make_shared<PromptHistoryStore>(temp_path);
    PersistentPromptHistory history(store);
    
    std::string input = "test";
    int cursor = 0;
    
    // Navigation on empty history should return false
    CHECK(history.navigate_prev({}, input, cursor) == false);
    CHECK(history.navigate_next({}, input, cursor) == false);
    CHECK(input == "test");  // Input unchanged
    
    std::filesystem::remove(temp_path);
}

TEST_CASE("PersistentPromptHistory save persists to disk", "[history]") {
    TempHistoryFile file("filo_test_history_persist");
    const auto& temp_path = file.path;
    
    {
        auto store = std::make_shared<PromptHistoryStore>(temp_path);
        PersistentPromptHistory history(store);
        
        history.save({}, "persisted prompt");
    }
    
    {
        PromptHistoryStore store(temp_path);
        CHECK(store.load() == true);
        CHECK(store.size() == 1);
        CHECK(store.entries()[0] == "persisted prompt");
    }
}

TEST_CASE("PersistentPromptHistory merges entries written by concurrent instances", "[history][concurrency]") {
    TempHistoryFile file("filo_test_history_concurrent");
    const auto& temp_path = file.path;

    auto first_store = std::make_shared<PromptHistoryStore>(temp_path);
    auto second_store = std::make_shared<PromptHistoryStore>(temp_path);
    PersistentPromptHistory first(first_store);
    PersistentPromptHistory second(second_store);

    first.save({}, "from first instance");
    second.save({}, "from second instance");

    PromptHistoryStore loaded(temp_path);
    REQUIRE(loaded.load());
    REQUIRE(loaded.size() == 2);
    CHECK(loaded.entries()[0] == "from first instance");
    CHECK(loaded.entries()[1] == "from second instance");
}

TEST_CASE("Prompt history remains valid under concurrent processes", "[history][concurrency]") {
    TempHistoryFile file("filo_test_history_processes");
    const auto& temp_path = file.path;

    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        PromptHistoryStore store(temp_path);
        for (int i = 0; i < 25; ++i) {
            if (!store.append_and_save(std::format("child-{}", i))) _exit(10);
        }
        _exit(0);
    }

    PromptHistoryStore parent_store(temp_path);
    for (int i = 0; i < 25; ++i) {
        REQUIRE(parent_store.append_and_save(std::format("parent-{}", i)));
    }

    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);

    PromptHistoryStore loaded(temp_path);
    REQUIRE(loaded.load());
    CHECK(loaded.size() == 50);
}

TEST_CASE("Sending a prompt records it instead of rewriting the history", "[history]") {
    TempHistoryFile file("filo_test_history_append");
    PromptHistoryStore store(file.path);
    store.set_max_entries(3);
    REQUIRE(store.append_and_save("first"));
    REQUIRE(store.append_and_save("second"));
    REQUIRE(store.save());  // Whole: folds the appended prompts in.
    CHECK_FALSE(std::filesystem::exists(file.log_path()));
    const auto saved = core::utils::read_file(file.path).value();

    REQUIRE(store.append_and_save("third"));
    REQUIRE(store.append_and_save("third"));  // A repeat adds nothing.
    REQUIRE(store.append_and_save("fourth"));
    CHECK(core::utils::read_file(file.path) == saved);
    REQUIRE(std::filesystem::exists(file.log_path()));

    // Replayed in order, the recorded prompts drop the oldest as saving each
    // would have; a record torn by a crash is skipped.
    std::ofstream(file.log_path(), std::ios::app) << "\x40\0";
    PromptHistoryStore loaded(file.path);
    loaded.set_max_entries(3);
    REQUIRE(loaded.load());
    CHECK(loaded.entries() == std::vector<std::string>{"second", "third", "fourth"});

    REQUIRE(loaded.remove_at_and_save(0));
    CHECK_FALSE(std::filesystem::exists(file.log_path()));
    PromptHistoryStore after(file.path);
    REQUIRE(after.load());
    CHECK(after.entries() == std::vector<std::string>{"third", "fourth"});
}

TEST_CASE("Prompt history reads the file again whenever another writer changed it",
          "[history][concurrency]") {
    TempHistoryFile file("filo_test_history_refresh");
    PromptHistoryStore first(file.path);
    PromptHistoryStore second(file.path);

    // A recorded prompt grows the log.
    REQUIRE(first.append_and_save("first-1"));
    REQUIRE(second.append_and_save("second-1"));
    REQUIRE(first.append_and_save("first-2"));
    CHECK(first.entries() == std::vector<std::string>{"first-1", "second-1", "first-2"});

    // A rewrite replaces the snapshot and drops the log.
    REQUIRE(second.load());
    REQUIRE(second.remove_at_and_save(0));
    REQUIRE(first.append_and_save("first-3"));
    CHECK(first.entries() == std::vector<std::string>{"second-1", "first-2", "first-3"});

    // A whole save by the const path, and a clear.
    REQUIRE(second.load());
    second.add("unsaved");
    REQUIRE(second.save());
    REQUIRE(first.load());
    CHECK(first.entries().back() == "unsaved");
    REQUIRE(second.clear_and_save());
    REQUIRE(first.load());
    CHECK(first.empty());

    // A file rewritten in place by another program (a different size: an
    // in-place rewrite of the same size within one clock tick is invisible).
    std::ofstream(file.path, std::ios::trunc) << R"({"version":2,"entries":["edited"]})";
    REQUIRE(first.load());
    CHECK(first.entries() == std::vector<std::string>{"edited"});
}

TEST_CASE("Prompt history never keeps what memory holds and the file does not",
          "[history]") {
    TempHistoryFile file("filo_test_history_memory_only");
    {
        PromptHistoryStore store(file.path);
        REQUIRE(store.append_and_save("saved"));
        store.add("only in memory");
        REQUIRE(store.append_and_save("sent"));
        CHECK(store.entries() == std::vector<std::string>{"saved", "sent"});
    }

    // Entries trimmed by a lower limit come back when the limit is raised:
    // the file still holds them.
    {
        PromptHistoryStore store(file.path);
        store.set_max_entries(1);
        REQUIRE(store.append_and_save("third"));
        CHECK(store.entries() == std::vector<std::string>{"third"});
        store.set_max_entries(10);
        REQUIRE(store.load());
        CHECK(store.entries() == std::vector<std::string>{"saved", "sent", "third"});
    }
}

TEST_CASE("Prompt history keeps in memory what reading the file gives back",
          "[history]") {
    TempHistoryFile file("filo_test_history_utf8");
    PromptHistoryStore store(file.path);
    REQUIRE(store.append_and_save(PromptHistoryEntry{
        .text = "bad \xff byte", .session_id = "s", .workspace = "/w\xfe"}));
    PromptHistoryStore reader(file.path);
    REQUIRE(reader.load());
    CHECK(store.records() == reader.records());
    CHECK(store.records().back().text != "bad \xff byte");
}

TEST_CASE("PromptHistoryStore reads every format the history was ever written in",
          "[history]") {
    TempHistoryFile file("filo_test_history_formats");
    const auto load = [&](std::string_view json) -> std::optional<std::vector<PromptHistoryEntry>> {
        file.cleanup();
        std::ofstream(file.path) << json;
        PromptHistoryStore store(file.path);
        if (!store.load()) return std::nullopt;
        return store.records();
    };
    using Entries = std::vector<PromptHistoryEntry>;

    // Keys in any order; non-string entries are skipped but keep `meta` aligned.
    CHECK(load(R"({"meta":[{"session":"a"},{},{"workspace":"/w","session":"c"}],)"
               R"("entries":["one",7,"three"],"version":2})")
          == Entries{{.text = "one", .session_id = "a"},
                     {.text = "three", .session_id = "c", .workspace = "/w"}});
    // Only the first of repeated keys counts, as a lookup would.
    CHECK(load(R"({"entries":["first"],"entries":["second"],)"
               R"("meta":[{"session":"x","session":"y"}],"meta":[{"session":"z"}]})")
          == Entries{{.text = "first", .session_id = "x"}});
    // Escaped keys and texts.
    CHECK(load(R"({"\u0065ntries":["a\n\"b\u00e9"],"meta":[{"s\u0065ssion":"\u0041"}]})")
          == Entries{{.text = "a\n\"b\xc3\xa9", .session_id = "A"}});
    // The legacy plain array; a document without entries.
    CHECK(load(R"(["old", {"x":1}, "older"])") == Entries{{.text = "old"}, {.text = "older"}});
    CHECK(load(R"({"version":2})") == Entries{});
    CHECK(load(R"({"entries":5,"meta":[]})") == Entries{});
    CHECK(load("5") == Entries{});
    CHECK(load(" null ") == Entries{});

    // A file that is not JSON is refused whole, even where nothing is read
    // from it, so it is never salvaged and saved over.
    CHECK_FALSE(load(R"({"version":tru,"entries":["a"]})"));
    CHECK_FALSE(load(R"({"entries":["a",nul]})"));
    CHECK_FALSE(load(R"({"entries":["a"],"meta":[{"x":[1 2]}]})"));
    CHECK_FALSE(load(R"({"other":["a" "b"],"entries":["a"]})"));
    CHECK_FALSE(load(R"({"entries":["a"]} trailing)"));
    CHECK_FALSE(load(R"({"entries":["a\x"]})"));
    CHECK_FALSE(load("falsy"));
    CHECK_FALSE(load(std::string(1100, '[') + std::string(1100, ']')));
}

TEST_CASE("PromptHistoryStore handles corrupted JSON gracefully", "[history]") {
    auto temp_path = std::filesystem::temp_directory_path() / "filo_test_history_bad.json";
    
    // Write invalid JSON
    {
        std::ofstream out(temp_path);
        out << "this is not valid json {[";
    }
    
    PromptHistoryStore store(temp_path);
    std::string error;
    CHECK(store.load(&error) == false);  // Should fail
    CHECK(error.empty() == false);        // Should provide error message
    
    std::filesystem::remove(temp_path);
}

TEST_CASE("now_iso8601 returns valid format", "[history]") {
    auto timestamp = PromptHistoryStore::now_iso8601();
    
    // Should be like: 2026-03-25T09:19:26Z
    CHECK(timestamp.size() == 20);  // YYYY-MM-DDTHH:MM:SSZ
    CHECK(timestamp[4] == '-');
    CHECK(timestamp[7] == '-');
    CHECK(timestamp[10] == 'T');
    CHECK(timestamp[13] == ':');
    CHECK(timestamp[16] == ':');
    CHECK(timestamp[19] == 'Z');
}
