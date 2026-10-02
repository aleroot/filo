#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/memory/MemoryRelevance.hpp"
#include "core/memory/MemoryStore.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace {

using Catch::Matchers::ContainsSubstring;
using core::memory::MemoryEntry;
using core::memory::MemoryState;

struct TempDir {
    std::filesystem::path path;
    explicit TempDir(std::string name)
        : path(std::filesystem::temp_directory_path() / std::move(name)) {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

[[nodiscard]] MemoryEntry make_entry(std::string id,
                                     std::string content,
                                     std::string last_used,
                                     std::string created) {
    MemoryEntry entry;
    entry.id = std::move(id);
    entry.content = std::move(content);
    entry.scope = "project";
    entry.last_used_at = std::move(last_used);
    entry.created_at = std::move(created);
    return entry;
}

[[nodiscard]] std::vector<std::string> ids_of(
    const std::vector<MemoryEntry>& entries,
    const std::vector<std::size_t>& order) {
    std::vector<std::string> ids;
    ids.reserve(order.size());
    for (const std::size_t index : order) ids.push_back(entries[index].id);
    return ids;
}

/// Newest first, so a test that expects a different order proves ranking did it.
[[nodiscard]] std::vector<MemoryEntry> sample_entries() {
    return {
        make_entry("m1", "TurnRevert replays the hunks of a turn backwards.",
                   "2026-03-01T00:00:00Z", "2026-03-01T00:00:00Z"),
        make_entry("m2", "The comparer opens patches in Lampo.",
                   "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        make_entry("m3", "PromptEditor launches the external editor.",
                   "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
    };
}

[[nodiscard]] std::vector<MemoryEntry> two_entries(std::string_view first_content,
                                                  std::string_view second_content) {
    return {
        // The SECOND is the more recently used one, so every test below asserts
        // on "first": a win for it can only come from scoring, never from the
        // recency tie-break. A test that expects "second" proves nothing.
        make_entry("first", std::string(first_content),
                   "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        make_entry("second", std::string(second_content),
                   "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
    };
}

[[nodiscard]] std::string top_id(const std::vector<MemoryEntry>& entries,
                                std::string_view query) {
    const auto order = core::memory::rank_memories(entries, query);
    return order.empty() ? std::string{} : entries[order.front()].id;
}

} // namespace

TEST_CASE("Memory recall without a usable query stays in recency order",
          "[memory][relevance]") {
    const auto entries = sample_entries();

    SECTION("empty query") {
        CHECK(ids_of(entries, core::memory::rank_memories(entries, ""))
              == std::vector<std::string>{"m1", "m2", "m3"});
    }
    SECTION("query whose terms appear in no memory") {
        // Nothing to score, so the order must fall back rather than be invented.
        CHECK(ids_of(entries, core::memory::rank_memories(entries, "xylophone zebra qq"))
              == std::vector<std::string>{"m1", "m2", "m3"});
    }
    SECTION("punctuation-only query") {
        CHECK(ids_of(entries, core::memory::rank_memories(entries, "… — !!! ??"))
              == std::vector<std::string>{"m1", "m2", "m3"});
    }
    SECTION("recency compares last use, then creation, then id") {
        const auto older = make_entry("m9", "x", "2026-01-01T00:00:00Z", "2026-05-01T00:00:00Z");
        const auto newer = make_entry("m8", "y", "2026-02-01T00:00:00Z", "2026-01-01T00:00:00Z");
        CHECK(core::memory::prompt_precedes(newer, older));
        CHECK_FALSE(core::memory::prompt_precedes(older, newer));

        const auto same_time_a = make_entry("ma", "x", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z");
        const auto same_time_b = make_entry("mb", "y", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z");
        CHECK(core::memory::prompt_precedes(same_time_a, same_time_b));

        const auto created_later = make_entry("mc", "x", "2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z");
        CHECK(core::memory::prompt_precedes(created_later, same_time_a));
    }
}

TEST_CASE("A memory about what the user asked outranks a more recent one",
          "[memory][relevance]") {
    const auto entries = sample_entries();
    const auto order = core::memory::rank_memories(entries, "PromptEditor never launches");
    REQUIRE(order.size() == 3);
    CHECK(entries[order.front()].id == "m3");
    // The two unmatched memories keep recency order behind the match.
    CHECK(entries[order[1]].id == "m1");
    CHECK(entries[order[2]].id == "m2");
}

TEST_CASE("A term every memory contains cannot decide the order",
          "[memory][relevance]") {
    const auto entries = two_entries(
        "filo counts TurnChangeTracker hunks per turn",
        "filo renders the transcript in the terminal");
    // "filo" is in both, so only the rare term may reorder them.
    CHECK(top_id(entries, "filo TurnChangeTracker") == "first");
    CHECK(top_id(entries, "filo transcript terminal") == "second");
}

TEST_CASE("BM25 does not reward a memory for being long",
          "[memory][relevance]") {
    const std::string filler(600, 'x');  // a term no query will ever match
    const auto entries = two_entries(
        "TurnRevert restores files.",
        "TurnRevert is mentioned once inside a very long entry " + filler);
    CHECK(top_id(entries, "TurnRevert") == "first");
}

TEST_CASE("A term repeated many times saturates instead of dominating",
          "[memory][relevance]") {
    // The repetitive memory is the more recent one; term-frequency saturation
    // keeps the entry that also matches the second term ahead of it.
    const auto entries = two_entries(
        "PromptEditor launches the external editor and the editor settings",
        "editor editor editor editor editor editor editor editor editor editor");
    CHECK(top_id(entries, "PromptEditor editor") == "first");
}

TEST_CASE("The analyzer is Unicode-aware, not ASCII pattern matching",
          "[memory][relevance]") {
    SECTION("case folding") {
        const auto entries = two_entries("PromptEditor launches the editor",
                                         "unrelated note about the build");
        CHECK(top_id(entries, "prompteditor") == "first");
        CHECK(top_id(entries, "PROMPTEDITOR") == "first");
    }
    SECTION("an accent typed in either Unicode form still matches") {
        // "perché" precomposed (U+00E9) against the same word written with a
        // combining acute (U+0301): NFC must make them one term.
        const auto entries = two_entries("Il comparatore apre le patch perché Lampo le legge",
                                         "unrelated note about the build");
        CHECK(top_id(entries, "perch\xC3\xA9") == "first");
        CHECK(top_id(entries, "perche\xCC\x81") == "first");
        CHECK(top_id(entries, "PERCHÉ") == "first");
    }
    SECTION("a script without spaces still produces terms") {
        const auto entries = two_entries("内存块有一个字符预算",
                                         "unrelated note about the build");
        CHECK(top_id(entries, "内存") == "first");
    }
    SECTION("an identifier keeps its underscores as one term") {
        const auto entries = two_entries("load_for_prompt selects the entries",
                                         "unrelated note about the build");
        CHECK(top_id(entries, "load_for_prompt") == "first");
        // Documented behaviour of UAX #29 word boundaries, not a gap to fill
        // with substring matching: a fragment of an identifier is not the term,
        // so nothing scores and recency decides.
        CHECK(top_id(entries, "prompt") == "second");
    }
    SECTION("ill-formed UTF-8 cannot make ranking throw or hang") {
        const auto entries = two_entries("PromptEditor launches the editor \xC3\x28 broken bytes",
                                         "unrelated note about the build");
        CHECK(top_id(entries, "PromptEditor") == "first");
    }
}

TEST_CASE("Memory selection respects the entry count and the character budget",
          "[memory][relevance]") {
    const std::vector<MemoryEntry> entries = {
        make_entry("top", std::string(30, 'a'), "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        make_entry("big", std::string(100, 'b'), "2026-02-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        make_entry("tail", std::string(20, 'c'), "2026-03-01T00:00:00Z", "2026-03-01T00:00:00Z"),
    };

    SECTION("an entry that does not fit is skipped so a later one still fits") {
        const auto chosen = core::memory::select_memories(entries, "", 24, 60);
        // Recency order is tail, big, top; the budget takes tail (20), skips big
        // (would reach 120) and still takes top (50 of 60 used).
        CHECK(ids_of(entries, chosen) == std::vector<std::string>{"tail", "top"});
    }
    SECTION("the best entry is recalled even when it alone exceeds the budget") {
        const auto chosen = core::memory::select_memories(entries, "", 1, 10);
        CHECK(ids_of(entries, chosen) == std::vector<std::string>{"tail"});
    }
    SECTION("the entry cap still applies") {
        const auto chosen = core::memory::select_memories(entries, "", 2, 0);
        CHECK(chosen.size() == 2);
    }
    SECTION("a zero character budget disables only the budget") {
        const auto chosen = core::memory::select_memories(entries, "", 24, 0);
        CHECK(chosen.size() == 3);
    }
    SECTION("a zero entry cap selects nothing") {
        CHECK(core::memory::select_memories(entries, "", 0, 0).empty());
    }
    SECTION("the query decides the order the budget is spent in") {
        const auto with_query = two_entries(
            std::string(60, 'a') + " PromptEditor",
            std::string(60, 'c'));
        const auto chosen = core::memory::select_memories(with_query, "PromptEditor", 1, 0);
        CHECK(ids_of(with_query, chosen) == std::vector<std::string>{"first"});
    }
}

TEST_CASE("The relevance query is the prompt that opened the conversation",
          "[memory][relevance]") {
    const auto system = core::llm::Message{.role = "system", .content = "runtime instructions"};
    const auto first = core::llm::Message{.role = "user", .content = "Fix PromptEditor"};
    const auto synthetic = core::llm::Message{
        .role = "user", .content = "Repository snapshot", .synthetic = true};
    const auto reply = core::llm::Message{.role = "assistant", .content = "Done with PromptEditor"};
    const auto second = core::llm::Message{.role = "user", .content = "Now the comparer"};

    const std::vector<core::llm::Message> conversation = {
        system, synthetic, first, reply, second};

    SECTION("the opening prompt, not the latest one") {
        // The block lives in the cached prompt prefix, so re-ranking on every
        // turn would rewrite that prefix and cost the provider's prompt cache.
        CHECK(core::memory::conversation_relevance_query(conversation) == "Fix PromptEditor");
    }
    SECTION("internal context records are not the user's words") {
        const std::vector<core::llm::Message> only_synthetic = {system, synthetic, reply};
        CHECK(core::memory::conversation_relevance_query(only_synthetic).empty());
    }
    SECTION("an empty conversation has no query") {
        CHECK(core::memory::conversation_relevance_query({}).empty());
    }
    SECTION("input_text is the fallback when content is empty") {
        const core::llm::Message typed{.role = "user", .content = "", .input_text = "Fix the comparer"};
        const std::vector<core::llm::Message> one = {typed};
        CHECK(core::memory::conversation_relevance_query(one) == "Fix the comparer");
    }
    SECTION("the query is bounded without splitting a character") {
        std::string long_prompt(20000, 'a');
        long_prompt += "é";  // a two-byte character past the bound
        const core::llm::Message message{.role = "user", .content = long_prompt};
        const std::vector<core::llm::Message> one = {message};
        const auto query = core::memory::conversation_relevance_query(one);
        CHECK(query.size() < long_prompt.size());
        // No dangling continuation byte: the cut landed on a character boundary.
        CHECK((static_cast<unsigned char>(query.back()) & 0xC0) != 0x80);
    }
}

TEST_CASE("The prompt projection ranks by query and renders best match first",
          "[memory][relevance][store]") {
    TempDir dir{"filo_memory_relevance_store"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    MemoryState state;
    state.settings.enabled = true;
    state.settings.auto_capture = true;
    for (auto& entry : sample_entries()) state.entries.push_back(std::move(entry));
    REQUIRE(store.save(state));

    SECTION("no query keeps the recency block") {
        const auto recalled = store.load_for_prompt();
        REQUIRE(recalled.entries.size() == 3);
        CHECK(recalled.entries[0].id == "m1");
        CHECK(recalled.entries[2].id == "m3");
    }
    SECTION("a query reorders the same entries") {
        core::memory::PromptProjection projection;
        projection.relevance_query = "The comparer window never opens";
        const auto recalled = store.load_for_prompt(projection);
        REQUIRE(recalled.entries.size() == 3);
        CHECK(recalled.entries[0].id == "m2");
    }
    SECTION("the rendered block follows the selection order") {
        core::memory::PromptProjection projection;
        projection.relevance_query = "PromptEditor never launches";
        const auto recalled = store.load_for_prompt(projection);
        const auto block = core::memory::build_memory_prompt_block(recalled);
        const auto editor = block.find("PromptEditor launches");
        const auto revert = block.find("TurnRevert replays");
        REQUIRE(editor != std::string::npos);
        REQUIRE(revert != std::string::npos);
        // m3 is the oldest entry, so leading with it can only be the ranking.
        CHECK(editor < revert);
    }
    SECTION("the character budget bounds what one prompt carries") {
        core::memory::PromptProjection projection;
        // m1 is 49 characters and m2 is 36: 85 fit, the 42-character m3 does not.
        projection.max_block_chars = 90;
        const auto recalled = store.load_for_prompt(projection);
        std::size_t chars = 0;
        for (const auto& entry : recalled.entries) chars += entry.content.size();
        CHECK(chars <= 90);
        CHECK(recalled.entries.size() == 2);
    }
    SECTION("archived and empty entries are never projected") {
        auto with_noise = store.load();
        for (auto& entry : with_noise.entries) {
            if (entry.id == "m2") entry.archived = true;
        }
        with_noise.entries.push_back(
            make_entry("m4", "", "2026-04-01T00:00:00Z", "2026-04-01T00:00:00Z"));
        REQUIRE(store.save(with_noise));
        const auto recalled = store.load_for_prompt();
        CHECK(recalled.entries.size() == 2);
    }
}

TEST_CASE("The capture instructions ask for calibrated, single-fact memories",
          "[memory][relevance]") {
    MemoryState state;
    state.settings.enabled = true;
    state.settings.auto_capture = true;
    state.entries.push_back(
        make_entry("m1", "Prefer direct answers.", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"));
    const auto block = core::memory::build_memory_prompt_block(state);
    CHECK_THAT(block, ContainsSubstring("[Memory Capture]"));
    // Selection is per entry, so a mixed entry cannot be partially recalled.
    CHECK_THAT(block, ContainsSubstring("one durable fact per entry"));
    CHECK_THAT(block, ContainsSubstring("at least two independent cases agree"));
    CHECK_THAT(block, ContainsSubstring("missing evidence as missing data"));
}
