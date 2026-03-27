#include <catch2/catch_test_macros.hpp>
#include "core/history/PromptHistoryStore.hpp"
#include <filesystem>
#include <fstream>

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
    CHECK(history.navigate_prev(input, cursor) == true);
    CHECK(input == "third prompt");
    CHECK(cursor == 12);  // Cursor at end
    
    // Navigate previous again (should show second prompt)
    CHECK(history.navigate_prev(input, cursor) == true);
    CHECK(input == "second prompt");
    
    // Navigate previous again (should show first prompt)
    CHECK(history.navigate_prev(input, cursor) == true);
    CHECK(input == "first prompt");
    
    // At oldest entry - should stay at first
    CHECK(history.navigate_prev(input, cursor) == true);
    CHECK(input == "first prompt");
    
    // Navigate next (should show second prompt)
    CHECK(history.navigate_next(input, cursor) == true);
    CHECK(input == "second prompt");
    
    // Navigate next (should show third prompt)
    CHECK(history.navigate_next(input, cursor) == true);
    CHECK(input == "third prompt");
    
    // Navigate next (should restore original input)
    CHECK(history.navigate_next(input, cursor) == true);
    CHECK(input == "current typing");
    
    // At current input - navigate_next should return false
    CHECK(history.navigate_next(input, cursor) == false);
    
    std::filesystem::remove(temp_path);
}

TEST_CASE("PersistentPromptHistory empty history navigation", "[history]") {
    auto temp_path = std::filesystem::temp_directory_path() / "filo_test_history_empty_nav.json";
    std::filesystem::remove(temp_path);
    
    auto store = std::make_shared<PromptHistoryStore>(temp_path);
    PersistentPromptHistory history(store);
    
    std::string input = "test";
    int cursor = 0;
    
    // Navigation on empty history should return false
    CHECK(history.navigate_prev(input, cursor) == false);
    CHECK(history.navigate_next(input, cursor) == false);
    CHECK(input == "test");  // Input unchanged
    
    std::filesystem::remove(temp_path);
}

TEST_CASE("PersistentPromptHistory save persists to disk", "[history]") {
    auto temp_path = std::filesystem::temp_directory_path() / "filo_test_history_persist.json";
    std::filesystem::remove(temp_path);
    
    {
        auto store = std::make_shared<PromptHistoryStore>(temp_path);
        PersistentPromptHistory history(store);
        
        history.save("persisted prompt");
    }
    
    {
        PromptHistoryStore store(temp_path);
        CHECK(store.load() == true);
        CHECK(store.size() == 1);
        CHECK(store.entries()[0] == "persisted prompt");
    }
    
    std::filesystem::remove(temp_path);
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
