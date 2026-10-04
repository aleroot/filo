#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/RepositoryContextMessage.hpp"
#include "core/memory/MemoryBackgroundService.hpp"
#include "core/memory/MemoryStore.hpp"
#include "core/tools/MemoryTool.hpp"
#include "core/tools/ToolNames.hpp"
#include "core/tools/ToolSchema.hpp"
#include "core/utils/JsonUtils.hpp"
#include "core/utils/JsonWriter.hpp"
#include "core/llm/protocols/AnthropicProtocol.hpp"

#include <simdjson.h>

#include <atomic>
#include <filesystem>
#include <fstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

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

} // namespace

namespace {

std::string revision_args(std::string_view action,
                          const std::vector<core::memory::MemoryRevisionTarget>& targets,
                          std::string_view content) {
    core::utils::JsonWriter writer;
    {
        auto object = writer.object();
        writer.kv_str("action", action).comma().kv_str("content", content);
        if (action == "update") {
            writer.comma().kv_str("id", targets.front().id).comma()
                  .kv_str("expected_content", targets.front().expected_content);
        } else {
            writer.comma().key("entries");
            auto array = writer.array();
            for (std::size_t i = 0; i < targets.size(); ++i) {
                if (i > 0) writer.comma();
                auto item = writer.object();
                writer.kv_str("id", targets[i].id).comma()
                      .kv_str("expected_content", targets[i].expected_content);
            }
        }
    }
    return std::move(writer).take();
}

} // namespace

TEST_CASE("Memory revision frees capacity atomically and retains original facts", "[memory][maintenance]") {
    TempDir dir{"filo_memory_revision_capacity"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    auto settings = store.settings();
    settings.max_active_entries = 2;
    REQUIRE(store.save_settings(settings));
    const auto context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path});
    auto scoped = store.for_context(context);
    const auto first = scoped.remember("Use the project virtual environment.", "project", {"python"});
    const auto second = scoped.remember("Run pytest in the project virtual environment.", "project", {"tests"});
    REQUIRE(first.ok);
    REQUIRE(second.ok);
    const auto rejected = scoped.remember("Prefer concise summaries.");
    REQUIRE_FALSE(rejected.ok);
    CHECK(rejected.code == "memory_capacity");
    REQUIRE(rejected.capacity);
    CHECK(rejected.capacity->active_entries == 2);
    CHECK(rejected.capacity->scope == "project");

    const auto merged = scoped.revise({{first.entry->id, first.entry->content},
                                       {second.entry->id, second.entry->content}},
                                      "Use the project virtual environment, including for pytest.");
    REQUIRE(merged.ok);
    REQUIRE(merged.entry);
    CHECK(merged.entry->supersedes == std::vector<std::string>{first.entry->id, second.entry->id});
    CHECK(merged.entry->tags == std::vector<std::string>{"python", "tests"});
    REQUIRE(scoped.list().size() == 1);
    const auto history = core::memory::MemoryStore{store.path()}.for_context(context).load();
    REQUIRE(history.entries.size() == 3);
    CHECK(history.entries[0].archived);
    CHECK(history.entries[0].content == first.entry->content);
    CHECK(history.entries[1].archived);
    CHECK(history.entries[1].content == second.entry->content);
    CHECK(history.entries[2].supersedes == merged.entry->supersedes);
    REQUIRE(scoped.remember("Prefer concise summaries.").ok);
    CHECK(scoped.list().size() == 2);
}

TEST_CASE("Memory update shortens legacy entries at capacity without truncating history", "[memory][maintenance]") {
    TempDir dir{"filo_memory_revision_legacy"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    const auto original = store.remember(std::string(6000, 'x'));
    REQUIRE(original.ok);
    REQUIRE(store.remember("Independent user preference.").ok);
    auto initial = store.load();
    for (auto& entry : initial.entries) entry.last_used_at = "2026-01-01T00:00:00Z";
    REQUIRE(store.save(initial));
    auto settings = store.settings();
    // Lowering the cap below current usage must not prevent maintenance.
    settings.max_active_entries = 1;
    REQUIRE(store.save_settings(settings));
    const auto updated = store.revise({{original.entry->id, original.entry->content}},
                                     "Build instructions live in docs/build.md.");
    REQUIRE(updated.ok);
    CHECK(store.list().size() == 2);
    const auto loaded = store.load();
    REQUIRE(loaded.entries.size() == 3);
    CHECK(loaded.entries[0].content.size() == 6000);
    CHECK(loaded.entries[0].archived);
    CHECK(loaded.entries[1].content == "Independent user preference.");
    CHECK_FALSE(loaded.entries[1].archived);
    CHECK(loaded.entries[2].supersedes == std::vector<std::string>{original.entry->id});
    CHECK(store.load_for_prompt({.max_entries = 1}).entries.front().id == updated.entry->id);
}

TEST_CASE("Memory revisions reject invalid or unavailable targets without partial writes", "[memory][maintenance][scope]") {
    TempDir dir{"filo_memory_revision_validation"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    auto context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path},
        core::context::SessionTransport::cli, "session-a");
    const auto scoped = store.for_context(context);
    const auto first = scoped.remember("First fact.");
    const auto second = scoped.remember("Second fact.");
    REQUIRE(first.ok);
    REQUIRE(second.ok);
    std::vector<core::memory::MemoryRevisionTarget> targets{
        {first.entry->id, first.entry->content}, {second.entry->id, second.entry->content}};
    auto view = scoped;
    SECTION("stale content") { targets[1].expected_content = "An obsolete value."; }
    SECTION("missing target") { targets[1].id = "m999"; }
    SECTION("duplicate target") { targets[1] = targets[0]; }
    SECTION("empty precondition") { targets[1].expected_content.clear(); }
    SECTION("another project") {
        context.workspace = core::workspace::SessionWorkspace{
            core::workspace::WorkspaceSnapshot{.primary = dir.path / "other"}};
        view = store.for_context(context);
    }
    SECTION("archived target") { REQUIRE(scoped.forget(second.entry->id).ok); }
    SECTION("mixed project and session scope") {
        const auto session = scoped.remember("Session fact.", "session");
        REQUIRE(session.ok);
        targets[1] = {session.entry->id, session.entry->content};
    }
    SECTION("other session") {
        const auto session = scoped.remember("Session fact.", "session");
        REQUIRE(session.ok);
        targets[1] = {session.entry->id, session.entry->content};
        context.session_id = "session-b";
        view = store.for_context(context);
    }
    SECTION("oversized revision") {
        const auto result = scoped.revise(targets, std::string(core::memory::kMaxAutoMemoryBytes + 1, 'x'));
        CHECK(result.code == "memory_content_too_long");
        CHECK(scoped.load().entries.size() == 2);
        return;
    }
    const auto bytes_before = [&] {
        std::ifstream file{store.path()};
        return std::string(std::istreambuf_iterator<char>(file), {});
    }();
    CHECK_FALSE(view.revise(targets, "Consolidated fact.").ok);
    std::ifstream file{store.path()};
    const std::string bytes_after(std::istreambuf_iterator<char>(file), {});
    CHECK(bytes_after == bytes_before);
}

TEST_CASE("Concurrent revisions of one memory commit only one replacement", "[memory][maintenance][concurrency]") {
    TempDir dir{"filo_memory_revision_concurrency"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    const auto original = store.remember("Original fact.");
    REQUIRE(original.ok);
    const std::vector<core::memory::MemoryRevisionTarget> targets{{original.entry->id, original.entry->content}};
    std::vector<core::memory::MemoryMutationResult> results(2);
    std::thread first([&] { results[0] = store.revise(targets, "First correction."); });
    std::thread second([&] { results[1] = store.revise(targets, "Second correction."); });
    first.join();
    second.join();
    CHECK(static_cast<int>(results[0].ok) + static_cast<int>(results[1].ok) == 1);
    CHECK((results[0].ok ? results[1] : results[0]).code == "memory_conflict");
    CHECK(store.list().size() == 1);
    CHECK(store.load().entries.size() == 2);
}

TEST_CASE("Revisions reuse an existing fact and no-op updates do not grow history", "[memory][maintenance]") {
    TempDir dir{"filo_memory_revision_deduplication"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    const auto first = store.remember("Use pytest.");
    const auto second = store.remember("pytest is the test runner.");
    REQUIRE(first.ok);
    REQUIRE(second.ok);
    CHECK(store.revise({{first.entry->id, first.entry->content}}, first.entry->content).ok);
    CHECK(store.load().entries.size() == 2);
    const auto merged = store.revise({{first.entry->id, first.entry->content},
                                      {second.entry->id, second.entry->content}}, "Use pytest.");
    REQUIRE(merged.ok);
    CHECK(merged.entry->id == first.entry->id);
    CHECK(merged.entry->supersedes == std::vector<std::string>{second.entry->id});
    CHECK(store.list().size() == 1);
    CHECK(store.load().entries.size() == 2);
}

TEST_CASE("Memory revisions preserve case-sensitive identifier corrections", "[memory][maintenance]") {
    TempDir dir{"filo_memory_revision_identifier_case"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    const auto original = store.remember("The table name is Foo.");
    REQUIRE(original.ok);
    const auto corrected = store.revise({{original.entry->id, original.entry->content}}, "The table name is foo.");
    REQUIRE(corrected.ok);
    CHECK(corrected.entry->content == "The table name is foo.");
    CHECK(corrected.entry->id != original.entry->id);
    CHECK(store.load().entries.front().archived);
    CHECK(store.list().front().content == "The table name is foo.");
}

TEST_CASE("Automatic capture limits bytes while preserving manual and existing long memories", "[memory][maintenance]") {
    TempDir dir{"filo_memory_capture_size"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    const auto at_limit = std::string(core::memory::kMaxAutoMemoryBytes, 'x');
    REQUIRE(store.remember(at_limit, {}, {}, "agent").ok);
    CHECK(store.remember(at_limit + "x", {}, {}, "agent").code == "memory_content_too_long");
    CHECK(store.remember(at_limit + "x", {}, {}, "background_review").code == "memory_content_too_long");
    const auto manual = store.remember(at_limit + "x");
    REQUIRE(manual.ok);
    CHECK(store.remember(manual.entry->content, {}, {}, "agent").ok);
    CHECK(store.list().size() == 2);
    std::string unicode;
    for (int i = 0; i < 401; ++i) unicode += "界";
    CHECK(store.remember(unicode, {}, {}, "agent").code == "memory_content_too_long");
    CHECK(store.list().size() == 2);
}

TEST_CASE("Capacity warnings count all scoped entries before recall selection", "[memory][maintenance]") {
    TempDir dir{"filo_memory_capacity_projection"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    auto settings = store.settings();
    settings.max_active_entries = 10;
    REQUIRE(store.save_settings(settings));
    auto context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path},
        core::context::SessionTransport::cli, "session-a");
    const auto scoped = store.for_context(context);
    for (int i = 0; i < 7; ++i) REQUIRE(scoped.remember("Fact " + std::to_string(i)).ok);
    CHECK(core::memory::build_memory_prompt_block(scoped.load_for_prompt()).find("Memory capacity:") == std::string::npos);
    REQUIRE(scoped.remember("Fact seven.").ok);
    REQUIRE(scoped.remember("Session-only fact.", "session").ok);
    const auto projected = scoped.load_for_prompt({.max_entries = 1});
    REQUIRE(projected.entries.size() == 1);
    REQUIRE(projected.scope_usage.size() == 2);
    CHECK(projected.scope_usage[0].active_entries == 8);
    CHECK(projected.scope_usage[1].active_entries == 1);
    CHECK_THAT(core::memory::build_memory_prompt_block(projected),
               Catch::Matchers::ContainsSubstring("project scope has 8/10"));
    const auto empty_projection = scoped.load_for_prompt({.max_entries = 0});
    CHECK(empty_projection.entries.empty());
    CHECK(empty_projection.scope_usage[0].active_entries == 8);
    context.session_id = "session-b";
    CHECK(store.for_context(context).load().scope_usage[1].active_entries == 0);
}

TEST_CASE("Memory tool recovers a full scope through merge then capture", "[memory][maintenance][tool]") {
    TempDir dir{"filo_memory_tool_recovery"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    auto settings = store.settings();
    settings.max_active_entries = 2;
    REQUIRE(store.save_settings(settings));
    const auto context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path});
    core::tools::MemoryTool tool{store};
    REQUIRE(core::utils::json::bool_field(tool.execute(R"({"action":"remember","content":"Use the project virtual environment."})", context), "ok"));
    REQUIRE(core::utils::json::bool_field(tool.execute(R"({"action":"remember","content":"Run pytest in the project virtual environment."})", context), "ok"));
    const auto blocked = tool.execute(R"({"action":"remember","content":"Prefer concise summaries."})", context);
    CHECK(core::utils::json::string_field(blocked, "code") == "memory_capacity");
    const auto status = tool.execute(R"({"action":"status"})", context);
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(status).get(doc) == simdjson::SUCCESS);
    CHECK(doc["scopes"].at(0)["active_entries"].get_uint64().value() == 2);
    CHECK(doc["scopes"].at(0)["remaining"].get_uint64().value() == 0);
    CHECK(doc["scopes"].at(0)["near_capacity"].get_bool().value());
    auto entries = store.for_context(context).load().entries;
    const auto args = revision_args("merge", {{entries[0].id, entries[0].content}, {entries[1].id, entries[1].content}},
                                    "Use the project virtual environment, including for pytest.");
    const auto validated = core::tools::schema::validate_arguments(tool.get_definition(), args);
    REQUIRE(validated.has_value());
    const auto merged = tool.execute(*validated, context);
    REQUIRE(core::tools::MemoryTool::committed_mutation(core::tools::names::kMemory, args, merged));
    REQUIRE(core::utils::json::bool_field(tool.execute(R"({"action":"remember","content":"Prefer concise summaries."})", context), "ok"));
    CHECK(store.for_context(context).list().size() == 2);
    const auto history_json = tool.execute(R"({"action":"list","include_archived":true})", context);
    REQUIRE(parser.parse(history_json).get(doc) == simdjson::SUCCESS);
    CHECK(doc["total"].get_uint64().value() == 4);
    CHECK(doc["entries"].at(0)["supersedes"].is_array());
}

TEST_CASE("Memory tool updates a fact at capacity and rejects stale retries", "[memory][maintenance][tool]") {
    TempDir dir{"filo_memory_tool_update"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    auto settings = store.settings();
    settings.max_active_entries = 1;
    REQUIRE(store.save_settings(settings));
    const auto context = core::context::make_session_context(core::workspace::WorkspaceSnapshot{.primary = dir.path});
    const auto scoped = store.for_context(context);
    const auto original = scoped.remember("Tests run with the old command.");
    REQUIRE(original.ok);
    core::tools::MemoryTool tool{store};
    const auto args = revision_args("update", {{original.entry->id, original.entry->content}}, "Tests run with pytest.");
    const auto validated = core::tools::schema::validate_arguments(tool.get_definition(), args);
    REQUIRE(validated.has_value());
    REQUIRE(core::tools::MemoryTool::committed_mutation(core::tools::names::kMemory, args, tool.execute(*validated, context)));
    CHECK(core::utils::json::string_field(tool.execute(args, context), "code") == "memory_conflict");
    REQUIRE(scoped.list().size() == 1);
    CHECK(scoped.list().front().content == "Tests run with pytest.");
    CHECK(scoped.load().entries.size() == 2);
    CHECK_FALSE(core::utils::json::bool_field(tool.execute(R"({"action":"update","id":"m2","content":"Missing precondition."})", context), "ok"));
    CHECK_FALSE(core::utils::json::bool_field(tool.execute(R"({"action":"merge","entries":["m2","m2"],"content":"Malformed."})", context), "ok"));
    CHECK_FALSE(core::utils::json::bool_field(tool.execute(R"({"action":"merge","entries":[],"content":"Malformed."})", context), "ok"));
    CHECK(scoped.load().entries.size() == 2);
}

TEST_CASE("Memory background cleanup runs before capture and reports unsaved candidates", "[memory][maintenance][background]") {
    TempDir dir{"filo_memory_background_capacity"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    auto settings = store.settings();
    settings.max_active_entries = 2;
    settings.background_review = true;
    settings.consolidation = true;
    settings.min_rate_limit_remaining_percent = 0;
    REQUIRE(store.save_settings(settings));
    const auto context = core::context::make_session_context(core::workspace::WorkspaceSnapshot{.primary = dir.path});
    const auto scoped = store.for_context(context);
    REQUIRE(scoped.remember("Existing fact.").ok);
    auto state = store.load();
    auto duplicate = state.entries.front();
    duplicate.id = "m2";
    state.entries.push_back(duplicate);
    REQUIRE(store.save(state));
    core::memory::MemoryBackgroundService service{store};
    core::memory::MemoryReviewInput input{
        .history = {core::llm::Message{.role = "user", .content = "Remember that pytest runs the tests."}},
        .session_context = context, .thread_policy = {},
    };
    const auto result = service.review(input);
    REQUIRE(result.ran);
    CHECK(result.memories_cleaned == 1);
    CHECK(result.memories_stored == 1);
    CHECK(result.memories_not_saved == 0);
    CHECK(scoped.list().size() == 2);
    input.history = {core::llm::Message{.role = "user", .content = "Remember that I prefer concise summaries."}};
    const auto full = service.review(input);
    CHECK(full.memories_stored == 0);
    CHECK(full.memories_not_saved == 1);
    CHECK_THAT(full.message, Catch::Matchers::ContainsSubstring("were not saved"));
    CHECK(scoped.list().size() == 2);
}

TEST_CASE("Background history replay cannot undo a revision or manual archival", "[memory][maintenance][background]") {
    TempDir dir{"filo_memory_background_revision_replay"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    auto settings = store.settings();
    settings.background_review = true;
    settings.min_rate_limit_remaining_percent = 0;
    REQUIRE(store.save_settings(settings));
    const auto context = core::context::make_session_context(core::workspace::WorkspaceSnapshot{.primary = dir.path});
    const auto scoped = store.for_context(context);
    core::memory::MemoryBackgroundService service{store};
    core::memory::MemoryReviewInput input{
        .history = {core::llm::Message{.role = "user", .content = "Remember that tests run with the old command."}},
        .session_context = context, .thread_policy = {},
    };
    REQUIRE(service.review(input).memories_stored == 1);
    const auto original = scoped.list().front();
    SECTION("revised fact") {
        REQUIRE(scoped.revise({{original.id, original.content}}, "Tests run with pytest.").ok);
    }
    SECTION("manually archived fact") { REQUIRE(scoped.forget(original.id).ok); }
    SECTION("already active fact") {}
    const auto before = scoped.load();
    const auto replayed = service.review(input);
    CHECK(replayed.memories_stored == 0);
    CHECK(replayed.memories_not_saved == 0);
    const auto after = scoped.load();
    REQUIRE(after.entries.size() == before.entries.size());
    for (std::size_t i = 0; i < after.entries.size(); ++i) {
        CHECK(after.entries[i].archived == before.entries[i].archived);
        CHECK(after.entries[i].use_count == before.entries[i].use_count);
        CHECK(after.entries[i].updated_at == before.entries[i].updated_at);
    }
}

TEST_CASE("Automatic capture cannot reactivate archived facts but explicit restoration can", "[memory][maintenance]") {
    TempDir dir{"filo_memory_capture_archive"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    const auto first = store.remember("Prefer concise summaries.");
    REQUIRE(first.ok);
    REQUIRE(store.forget(first.entry->id).ok);
    CHECK(store.remember(first.entry->content, {}, {}, "agent").code == "memory_archived");
    CHECK(store.remember(first.entry->content, {}, {}, "background_review").code == "memory_archived");
    CHECK(store.list().empty());
    REQUIRE(store.remember(first.entry->content).ok);
    CHECK(store.list().size() == 1);

    auto state = store.load();
    auto earlier = state.entries.front();
    earlier.id = "m2";
    earlier.archived = true;
    state.entries.insert(state.entries.begin(), earlier);
    REQUIRE(store.save(state));
    const auto remembered = store.remember(first.entry->content, {}, {}, "agent");
    REQUIRE(remembered.ok);
    CHECK(remembered.entry->id == first.entry->id);
    CHECK(store.list().size() == 1);
}

TEST_CASE("Memory tool update and merge respect capture and thread policies", "[memory][maintenance][tool]") {
    TempDir dir{"filo_memory_revision_policies"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    auto context = core::context::make_session_context(core::workspace::WorkspaceSnapshot{.primary = dir.path});
    const auto scoped = store.for_context(context);
    const auto first = scoped.remember("First fact.");
    const auto second = scoped.remember("Second fact.");
    REQUIRE(first.ok);
    REQUIRE(second.ok);
    core::tools::MemoryTool tool{store};
    const std::vector<core::memory::MemoryRevisionTarget> targets{{first.entry->id, first.entry->content},
                                                               {second.entry->id, second.entry->content}};
    SECTION("memory disabled") {
        auto settings = store.settings(); settings.enabled = false;
        REQUIRE(store.save_settings(settings));
    }
    SECTION("automatic capture disabled") {
        auto settings = store.settings(); settings.auto_capture = false;
        REQUIRE(store.save_settings(settings));
    }
    SECTION("thread generation disabled") { context.memory_policy.generate_memories = false; }
    for (const auto& action : {"update", "merge"}) {
        const auto args = revision_args(action, targets, "Revised fact.");
        CHECK_FALSE(core::tools::MemoryTool::committed_mutation(core::tools::names::kMemory, args, tool.execute(args, context)));
    }
    CHECK(scoped.load().entries.size() == 2);
}

TEST_CASE("Memory list searches and paginates without losing full revision content", "[memory][maintenance][tool]") {
    TempDir dir{"filo_memory_list_pages"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    const auto context = core::context::make_session_context(core::workspace::WorkspaceSnapshot{.primary = dir.path});
    const auto scoped = store.for_context(context);
    for (int i = 0; i < 26; ++i) REQUIRE(scoped.remember("Unrelated fact " + std::to_string(i)).ok);
    const auto relevant = scoped.remember("PromptEditor launches the external editor.");
    REQUIRE(relevant.ok);
    core::tools::MemoryTool tool{store};
    simdjson::dom::parser parser;
    simdjson::dom::element doc;
    auto result = tool.execute(R"({"action":"list","query":"PromptEditor","limit":1})", context);
    REQUIRE(parser.parse(result).get(doc) == simdjson::SUCCESS);
    CHECK(doc["total"].get_uint64().value() == 27);
    CHECK(doc["has_more"].get_bool().value());
    CHECK(doc["entries"].at(0)["id"].get_string().value() == relevant.entry->id);
    CHECK(doc["entries"].at(0)["content"].get_string().value() == relevant.entry->content);
    result = tool.execute(R"({"action":"list","offset":24})", context);
    REQUIRE(parser.parse(result).get(doc) == simdjson::SUCCESS);
    CHECK(doc["entries"].get_array().value().size() == 3);
    CHECK_FALSE(doc["has_more"].get_bool().value());
    result = tool.execute(R"({"action":"list","scope":"session"})", context);
    REQUIRE(parser.parse(result).get(doc) == simdjson::SUCCESS);
    CHECK(doc["total"].get_uint64().value() == 0);
}

TEST_CASE("MemoryStore stores and reloads settings plus entries", "[memory]") {
    TempDir dir{"filo_memory_store_reload"};
    core::memory::MemoryStore store{dir.path / "memory.json"};

    core::memory::MemorySettings settings;
    settings.enabled = true;
    settings.auto_capture = true;
    REQUIRE(store.save_settings(settings));

    auto result = store.remember("Prefer concise engineering summaries.", "global", {"style"}, "manual");
    REQUIRE(result.ok);
    REQUIRE(result.entry.has_value());
    REQUIRE(result.entry->id == "m1");

    const auto loaded = store.load();
    REQUIRE(loaded.settings.enabled);
    REQUIRE(loaded.settings.auto_capture);
    REQUIRE(loaded.entries.size() == 1);
    CHECK(loaded.entries[0].content == "Prefer concise engineering summaries.");
    CHECK(loaded.entries[0].tags.size() == 1);
}

TEST_CASE("MemoryStore deduplicates identical remembered content", "[memory]") {
    TempDir dir{"filo_memory_store_dedupe"};
    core::memory::MemoryStore store{dir.path / "memory.json"};

    REQUIRE(store.remember("Use rg before slower search tools.").ok);
    REQUIRE(store.remember("  use   RG before slower search tools. ").ok);

    const auto entries = store.list();
    REQUIRE(entries.size() == 1);
    CHECK(entries[0].id == "m1");
    CHECK(entries[0].use_count == 2);
}

TEST_CASE("MemoryStore recall updates only entries included in the prompt",
          "[memory][recall]") {
    TempDir dir{"filo_memory_store_recall_usage"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    const auto context_a = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path / "project-a"});
    const auto context_b = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path / "project-b"});
    const auto a = store.for_context(context_a);
    const auto b = store.for_context(context_b);

    REQUIRE(a.remember("Older project memory.").ok);
    REQUIRE(a.remember("Recently used project memory.").ok);
    REQUIRE(b.remember("Other project's memory.").ok);

    auto initial = store.load();
    for (auto& entry : initial.entries) {
        if (entry.content == "Older project memory.") {
            entry.last_used_at = "2026-01-01T00:00:00Z";
            entry.use_count = 3;
        } else if (entry.content == "Recently used project memory.") {
            entry.last_used_at = "2026-02-01T00:00:00Z";
            entry.use_count = 7;
        } else if (entry.content == "Other project's memory.") {
            entry.last_used_at = "2026-03-01T00:00:00Z";
            entry.use_count = 11;
        }
    }
    REQUIRE(store.save(initial));

    const auto recalled = a.load_for_prompt({.max_entries = 1});
    REQUIRE(recalled.entries.size() == 1);
    CHECK(recalled.entries.front().content == "Recently used project memory.");
    CHECK(recalled.entries.front().use_count == 7);
    CHECK(recalled.entries.front().last_used_at == "2026-02-01T00:00:00Z");
    REQUIRE(a.record_prompt_recall(
        std::vector<std::string>{recalled.entries.front().id}));

    const auto persisted = store.load();
    REQUIRE(persisted.entries.size() == 3);
    for (const auto& entry : persisted.entries) {
        if (entry.content == "Older project memory.") {
            CHECK(entry.use_count == 3);
            CHECK(entry.last_used_at == "2026-01-01T00:00:00Z");
        } else if (entry.content == "Recently used project memory.") {
            CHECK(entry.use_count == 8);
            CHECK(entry.last_used_at != "2026-02-01T00:00:00Z");
        } else if (entry.content == "Other project's memory.") {
            CHECK(entry.use_count == 11);
            CHECK(entry.last_used_at == "2026-03-01T00:00:00Z");
        }
    }
}

TEST_CASE("MemoryStore enforces max active entries without evicting memories",
          "[memory][limit]") {
    TempDir dir{"filo_memory_store_active_limit"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    auto settings = store.settings();
    settings.max_active_entries = 2;
    REQUIRE(store.save_settings(settings));

    REQUIRE(store.remember("First active memory.").ok);
    REQUIRE(store.remember("Second active memory.").ok);
    const auto rejected = store.remember("Third active memory.");
    CHECK_FALSE(rejected.ok);
    CHECK_THAT(rejected.message, Catch::Matchers::ContainsSubstring("Memory limit reached"));
    REQUIRE(store.list().size() == 2);

    // Updating an existing active entry remains allowed at the limit.
    CHECK(store.remember("First active memory.").ok);
    REQUIRE(store.forget("m1").ok);
    REQUIRE(store.remember("Third active memory.").ok);

    // Restoring an archived entry cannot push the scope over its configured cap.
    const auto restore_rejected = store.remember("First active memory.");
    CHECK_FALSE(restore_rejected.ok);
    CHECK_THAT(restore_rejected.message,
               Catch::Matchers::ContainsSubstring("Memory limit reached"));

    auto oversized = store.load();
    oversized.entries.push_back(core::memory::MemoryEntry{
        .id = "m99",
        .content = "Bypass through whole-store save.",
        .scope = "global",
    });
    std::string error;
    CHECK_FALSE(store.save(oversized, &error));
    CHECK_THAT(error, Catch::Matchers::ContainsSubstring("max_active_entries"));
    CHECK(store.list().size() == 2);
}

TEST_CASE("MemoryStore serializes concurrent remembers for the same file", "[memory][concurrency]") {
    TempDir dir{"filo_memory_store_concurrent"};
    const auto memory_path = dir.path / "memory.json";

    constexpr std::size_t kCount = 24;
    std::vector<core::memory::MemoryMutationResult> results(kCount);
    std::vector<std::thread> threads;
    threads.reserve(kCount);

    for (std::size_t i = 0; i < kCount; ++i) {
        threads.emplace_back([&, i, memory_path]() {
            core::memory::MemoryStore store{memory_path};
            results[i] = store.remember("Concurrent memory " + std::to_string(i));
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }

    for (const auto& result : results) {
        CHECK(result.ok);
    }
    CHECK(core::memory::MemoryStore{memory_path}.list().size() == kCount);
}

TEST_CASE("MemoryStore serializes remembers across processes", "[memory][concurrency]") {
    TempDir dir{"filo_memory_store_process_concurrency"};
    const auto memory_path = dir.path / "memory.json";

    const pid_t child = ::fork();
    REQUIRE(child >= 0);
    if (child == 0) {
        core::memory::MemoryStore store{memory_path};
        for (int i = 0; i < 15; ++i) {
            if (!store.remember("Child memory " + std::to_string(i)).ok) _exit(10);
        }
        _exit(0);
    }

    core::memory::MemoryStore store{memory_path};
    for (int i = 0; i < 15; ++i) {
        REQUIRE(store.remember("Parent memory " + std::to_string(i)).ok);
    }

    int status = 0;
    REQUIRE(::waitpid(child, &status, 0) == child);
    REQUIRE(WIFEXITED(status));
    REQUIRE(WEXITSTATUS(status) == 0);
    CHECK(store.list().size() == 30);
}

TEST_CASE("MemoryStore archives and cleans without deleting history", "[memory]") {
    TempDir dir{"filo_memory_store_archive"};
    core::memory::MemoryStore store{dir.path / "memory.json"};

    REQUIRE(store.remember("Keep patches small.").ok);
    REQUIRE(store.forget("m1").ok);

    CHECK(store.list(false).empty());
    auto all = store.list(true);
    REQUIRE(all.size() == 1);
    CHECK(all[0].archived);

    auto clear = store.clear();
    REQUIRE(clear.ok);
}

TEST_CASE("MemoryStore clean ignores archived entries when deduplicating active memories",
          "[memory]") {
    TempDir dir{"filo_memory_store_clean_archived"};
    core::memory::MemoryStore store{dir.path / "memory.json"};

    core::memory::MemoryState state;
    state.entries.push_back(core::memory::MemoryEntry{
        .id = "m1",
        .content = "Use rg before slower search tools.",
        .created_at = "2026-01-01T00:00:00Z",
        .updated_at = "2026-01-01T00:00:00Z",
        .last_used_at = "2026-01-01T00:00:00Z",
        .archived = true,
    });
    state.entries.push_back(core::memory::MemoryEntry{
        .id = "m2",
        .content = "  use   RG before slower search tools. ",
        .created_at = "2026-01-02T00:00:00Z",
        .updated_at = "2026-01-02T00:00:00Z",
        .last_used_at = "2026-01-02T00:00:00Z",
    });
    REQUIRE(store.save(state));

    auto clean = store.clean();
    REQUIRE(clean.ok);

    const auto active = store.list(false);
    REQUIRE(active.size() == 1);
    CHECK(active[0].id == "m2");
    CHECK_FALSE(active[0].archived);
}

TEST_CASE("Memory prompt block respects disabled memory and includes auto capture guidance", "[memory]") {
    core::memory::MemoryState state;
    state.settings.enabled = false;
    state.entries.push_back(core::memory::MemoryEntry{
        .id = "m1",
        .content = "Prefer direct answers.",
        .created_at = "2026-01-01T00:00:00Z",
    });
    CHECK(core::memory::build_memory_prompt_block(state).empty());

    state.settings.enabled = true;
    state.settings.auto_capture = true;
    const auto block = core::memory::build_memory_prompt_block(state);
    CHECK(block.find("[Memory]") != std::string::npos);
    CHECK(block.find("Prefer direct answers.") != std::string::npos);
    CHECK(block.find("[Memory Capture]") != std::string::npos);

    const auto no_capture_block = core::memory::build_memory_prompt_block(state, 24, false);
    CHECK(no_capture_block.find("[Memory]") != std::string::npos);
    CHECK(no_capture_block.find("[Memory Capture]") == std::string::npos);
}

TEST_CASE("MemoryStore defaults missing settings to on and preserves explicit opt-outs", "[memory]") {
    TempDir dir{"filo_memory_store_defaults"};
    const auto path = dir.path / "memory.json";
    core::memory::MemoryStore store{path};

    SECTION("missing file") {}
    SECTION("empty settings") {
        std::ofstream{path} << R"({"settings":{},"entries":[]})";
    }
    SECTION("legacy file without settings") {
        std::ofstream{path} << R"({"entries":[]})";
    }
    SECTION("explicit settings") {
        std::ofstream{path} << R"({"settings":{"enabled":false,"auto_capture":false}})";
        CHECK_FALSE(store.settings().enabled);
        CHECK_FALSE(store.settings().auto_capture);
        return;
    }
    SECTION("legacy explicit settings") {
        std::ofstream{path} << R"({"enabled":false,"auto_capture":false})";
        CHECK_FALSE(store.settings().enabled);
        CHECK_FALSE(store.settings().auto_capture);
        return;
    }
    SECTION("recall only") {
        std::ofstream{path} << R"({"settings":{"enabled":true,"auto_capture":false}})";
        CHECK(store.settings().enabled);
        CHECK_FALSE(store.settings().auto_capture);
        return;
    }

    const auto state = store.load();
    CHECK(state.settings.enabled);
    CHECK(state.settings.auto_capture);
    CHECK_FALSE(state.settings.background_review);
    CHECK_FALSE(state.settings.consolidation);
    CHECK_FALSE(state.settings.skill_curation);
    CHECK_THAT(core::memory::build_memory_prompt_block(state),
               Catch::Matchers::ContainsSubstring("[Memory Capture]"));
}

TEST_CASE("MemoryTool saves to disk by default and respects disabled settings", "[memory]") {
    TempDir dir{"filo_memory_tool_enable_and_save"};
    const auto path = dir.path / "memory.json";
    core::memory::MemoryStore store{path};
    core::tools::MemoryTool tool{store};
    const auto context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path});
    const std::string args =
        R"({"action":"remember","content":"Use the project virtual environment for formatting tools.","scope":"project"})";

    SECTION("fresh store") {}
    SECTION("disabled then re-enabled") {
        auto settings = store.settings();
        settings.enabled = false;
        settings.auto_capture = false;
        REQUIRE(store.save_settings(settings));
        const auto disabled = tool.execute(args, context);
        CHECK_THAT(disabled, Catch::Matchers::ContainsSubstring("/memory auto on"));
        CHECK_FALSE(core::tools::MemoryTool::committed_mutation(
            core::tools::names::kMemory, args, disabled));
        CHECK(store.list().empty());

        settings.enabled = true;
        REQUIRE(store.save_settings(settings));
        const auto recall_only = tool.execute(args, context);
        CHECK_THAT(recall_only, Catch::Matchers::ContainsSubstring("Automatic memory capture is disabled"));
        CHECK(store.list().empty());

        settings.auto_capture = true;
        REQUIRE(store.save_settings(settings));
    }
    const auto saved = tool.execute(args, context);
    REQUIRE(core::tools::MemoryTool::committed_mutation(
        core::tools::names::kMemory, args, saved));
    const auto reloaded = core::memory::MemoryStore{path}.load();
    REQUIRE(reloaded.entries.size() == 1);
    CHECK(reloaded.entries.front().content == "Use the project virtual environment for formatting tools.");
    CHECK(reloaded.entries.front().scope == "project");
    CHECK(reloaded.entries.front().source == "agent");
}

TEST_CASE("MemoryTool reports corrupt stores without overwriting them", "[memory]") {
    TempDir dir{"filo_memory_tool_corrupt"};
    const auto path = dir.path / "memory.json";
    std::ofstream{path} << "invalid JSON";
    core::tools::MemoryTool tool{core::memory::MemoryStore{path}};
    const auto context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path});
    for (const auto& args : {R"({"action":"status"})",
                             R"({"action":"remember","content":"Use rg."})"}) {
        CHECK_THAT(tool.execute(args, context),
                   Catch::Matchers::ContainsSubstring("contains invalid JSON"));
    }
    std::ifstream file{path};
    const std::string contents((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());
    CHECK(contents == "invalid JSON");
}

TEST_CASE("Memory views isolate project and session reads and mutations", "[memory][scope]") {
    TempDir dir{"filo_memory_scope"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    const auto context_a = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path / "a"},
        core::context::SessionTransport::cli, "session-a");
    const auto context_b = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path / "b"},
        core::context::SessionTransport::cli, "session-b");
    auto other_session = context_a;
    other_session.session_id = "other-session";
    const auto a = store.for_context(context_a);
    const auto b = store.for_context(context_b);

    REQUIRE(store.remember("Legacy memory without project identity.").ok);
    const auto first = a.remember("Use the local virtual environment.");
    const auto second = b.remember("Use the local virtual environment.");
    REQUIRE(first.ok);
    REQUIRE(second.ok);
    REQUIRE(first.entry.has_value());
    REQUIRE(second.entry.has_value());
    CHECK(first.entry->id != second.entry->id);
    REQUIRE(a.remember("Session-specific workflow.", "session").ok);
    CHECK(a.list().size() == 2);
    CHECK(a.load().entries.size() == 2);
    CHECK(b.list().size() == 1);
    CHECK(store.for_context(other_session).list().size() == 1);

    CHECK_FALSE(a.forget(second.entry->id).ok);
    REQUIRE(a.clean().ok);
    REQUIRE(store.clean().ok);
    CHECK(b.list().size() == 1);
    CHECK(a.list().size() == 2);
    CHECK_FALSE(a.remember("Should not become global.", "global").ok);
    CHECK_FALSE(a.save(a.load()));
    REQUIRE(a.clear().ok);
    CHECK(a.list().empty());
    CHECK(b.list().size() == 1);
    CHECK(store.list().size() == 2); // B and the untouched legacy entry.

    const auto restarted_b = core::memory::MemoryStore{store.path()}.for_context(context_b);
    REQUIRE(restarted_b.list().size() == 1);
    CHECK(restarted_b.list().front().project_root == context_b.workspace_view().primary().string());
}

TEST_CASE("Memory project identity groups subdirectories and isolates worktrees", "[memory][scope]") {
    TempDir dir{"filo_memory_project_identity"};
    const auto repo = dir.path / "repo";
    const auto worktree = dir.path / "worktree";
    std::filesystem::create_directories(repo / ".git");
    std::filesystem::create_directories(repo / "src");
    std::filesystem::create_directories(worktree);
    std::ofstream{worktree / ".git"} << "gitdir: ../repo/.git/worktrees/worktree\n";
    std::filesystem::create_directory_symlink(repo, dir.path / "alias");
    core::memory::MemoryStore store{dir.path / "memory.json"};
    const auto view = [&](const std::filesystem::path& root) {
        return store.for_context(core::context::make_session_context(
            core::workspace::WorkspaceSnapshot{.primary = root}));
    };
    REQUIRE(view(repo).remember("Checkout-specific build command.").ok);
    CHECK(view(repo / "src").list().size() == 1);
    CHECK(view(dir.path / "alias").list().size() == 1);
    CHECK(view(worktree).list().empty());
    CHECK(view({}).list().empty());
    CHECK_FALSE(view({}).remember("No workspace must not mean global.").ok);
    CHECK_FALSE(view(repo).remember("Missing session identity.", "session").ok);
}

TEST_CASE("MemoryTool binds each call to the caller project", "[memory][scope]") {
    TempDir dir{"filo_memory_tool_scope"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    core::tools::MemoryTool tool{store};
    auto context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path / "a"});
    const std::string args = R"({"action":"remember","content":"Project A uses Swift."})";
    REQUIRE(core::tools::MemoryTool::committed_mutation(
        core::tools::names::kMemory, args, tool.execute(args, context)));
    CHECK_THAT(tool.execute(R"({"action":"list"})", context),
               Catch::Matchers::ContainsSubstring("Project A uses Swift."));
    const auto entry = store.for_context(context).list().front();
    context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path / "b"});
    CHECK(tool.execute(R"({"action":"list"})", context).find("Project A") == std::string::npos);
    const auto forget = tool.execute("{\"action\":\"forget\",\"id\":\"" + entry.id + "\"}", context);
    CHECK_THAT(forget, Catch::Matchers::ContainsSubstring("Memory not found"));
    CHECK(store.list().size() == 1);
}

TEST_CASE("Claude memory schema and streamed tool calls round trip through persistence",
          "[memory][claude][integration]") {
    TempDir dir{"filo_claude_memory_roundtrip"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    core::tools::MemoryTool tool{store};
    const auto context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path / "project"});
    core::llm::ChatRequest request;
    request.model = "claude-opus-5";
    request.tools.push_back({.type = "function", .function = tool.get_definition()});
    request.messages.push_back({.role = "user", .content = "Remember this project's formatter."});
    const auto payload = core::llm::protocols::AnthropicSerializer::serialize(request);
    simdjson::dom::parser parser;
    const auto doc = parser.parse(payload);
    CHECK(doc["tools"].at(0)["name"].get_string().value() == "memory");
    CHECK(doc["tools"].at(0)["input_schema"]["required"].at(0).get_string().value() == "action");
    CHECK(doc["tools"].at(0)["input_schema"]["properties"]["scope"]["enum"].at(0).get_string().value() == "project");

    core::llm::protocols::AnthropicSSEParser stream;
    stream.process_event("content_block_start",
        R"({"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_memory","name":"memory","input":{}}})");
    stream.process_event("content_block_delta",
        R"({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"action\":\"remember\","}})");
    stream.process_event("content_block_delta",
        R"({"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"\"content\":\"Use the local Swift formatter.\"}"}})");
    const auto completed = stream.process_event("content_block_stop",
        R"({"type":"content_block_stop","index":0})");
    REQUIRE(completed.completed_tools.size() == 1);
    const auto& call = completed.completed_tools.front();
    const auto args = core::tools::schema::normalize_arguments(tool.get_definition(), call.function.arguments);
    REQUIRE(args.has_value());
    const auto result = tool.execute(*args, context);
    REQUIRE(core::tools::MemoryTool::committed_mutation(call.function.name, *args, result));
    const auto reloaded = core::memory::MemoryStore{store.path()}.for_context(context).list();
    REQUIRE(reloaded.size() == 1);
    CHECK(reloaded.front().content == "Use the local Swift formatter.");
    CHECK(reloaded.front().scope == "project");

    request.messages.push_back({.role = "assistant", .tool_calls = {call}});
    request.messages.push_back({.role = "tool", .content = result, .tool_call_id = call.id});
    const auto reply = core::llm::protocols::AnthropicSerializer::serialize(request);
    const auto reply_doc = parser.parse(reply);
    CHECK(reply_doc["messages"].at(2)["content"].at(0)["type"].get_string().value() == "tool_result");
    CHECK(reply_doc["messages"].at(2)["content"].at(0)["tool_use_id"].get_string().value() == call.id);
    CHECK(reply_doc["messages"].at(2)["content"].at(0)["content"].get_string().value() == result);
}

TEST_CASE("Memory markdown transfers only explicitly selected project entries", "[memory][scope]") {
    TempDir dir{"filo_memory_scoped_markdown"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    const auto view = [&](std::string_view project) {
        return store.for_context(core::context::make_session_context(
            core::workspace::WorkspaceSnapshot{.primary = dir.path / project}));
    };
    REQUIRE(view("a").remember("Project A build workflow.").ok);
    REQUIRE(view("b").remember("Project B private workflow.").ok);
    const auto exported = dir.path / "reviewed.md";
    REQUIRE(view("a").save_markdown(exported).count == 1);
    REQUIRE(view("c").load_markdown(exported).count == 1);
    REQUIRE(view("c").list().size() == 1);
    CHECK(view("c").list().front().content == "Project A build workflow.");
    CHECK(view("c").list().front().project_root ==
          core::workspace::SessionWorkspace::normalize_path(dir.path / "c").string());
}

TEST_CASE("MemoryTool respects thread generation policy", "[memory]") {
    TempDir dir{"filo_memory_tool_thread_policy"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    core::memory::MemorySettings settings;
    settings.enabled = true;
    settings.auto_capture = true;
    REQUIRE(store.save_settings(settings));

    core::tools::MemoryTool tool{store};
    auto context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path});
    context.memory_policy.generate_memories = false;

    const auto result = tool.execute(
        R"({"action":"remember","content":"Prefer precise status updates."})",
        context);

    CHECK_THAT(result, Catch::Matchers::ContainsSubstring("Thread memory generation is disabled"));
    CHECK(store.list().empty());
}

TEST_CASE("MemoryTool is treated as write-destructive for model mode filtering", "[memory]") {
    CHECK(core::tools::names::is_write_destructive_tool(core::tools::names::kMemory));
}

TEST_CASE("MemoryTool classifies committed mutations from tool IO", "[memory]") {
    CHECK(core::tools::MemoryTool::committed_mutation(
        core::tools::names::kMemory,
        R"({"action":"remember"})",
        R"({"ok":true})"));
    CHECK_FALSE(core::tools::MemoryTool::committed_mutation(
        core::tools::names::kMemory,
        R"({"action":"list"})",
        R"({"ok":true})"));
    CHECK_FALSE(core::tools::MemoryTool::committed_mutation(
        core::tools::names::kMemory,
        R"({"action":"remember"})",
        R"({"error":"disabled"})"));
}

TEST_CASE("MemoryStore saves and loads portable markdown", "[memory]") {
    TempDir dir{"filo_memory_store_markdown_roundtrip"};
    core::memory::MemoryStore source{dir.path / "source.json"};
    core::memory::MemoryStore target{dir.path / "target.json"};
    const auto markdown = dir.path / "memory.md";

    REQUIRE(source.remember("Prefer terse status updates.").ok);
    REQUIRE(source.remember("Use rg before slower search tools.", "project").ok);

    auto saved = source.save_markdown(markdown);
    REQUIRE(saved.ok);
    CHECK(saved.count == 2);

    auto loaded = target.load_markdown(markdown);
    REQUIRE(loaded.ok);
    CHECK(loaded.count == 2);

    const auto entries = target.list();
    REQUIRE(entries.size() == 2);
    bool found_project_scope = false;
    for (const auto& entry : entries) {
        CHECK(entry.source == "markdown");
        if (entry.content == "Use rg before slower search tools.") {
            found_project_scope = entry.scope == "project";
        }
    }
    CHECK(found_project_scope);
}

TEST_CASE("MemoryStore imports simple markdown lists", "[memory]") {
    TempDir dir{"filo_memory_store_markdown_import"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    const auto markdown = dir.path / "memory.md";

    {
        std::ofstream file(markdown);
        file << "# Portable Memory\n\n";
        file << "- Prefer direct engineering prose.\n";
        file << "1. Run focused tests before full suites.\n";
        file << "* [ ] Avoid storing secrets.\n";
    }

    auto loaded = store.load_markdown(markdown);
    REQUIRE(loaded.ok);
    CHECK(loaded.count == 3);
    CHECK(store.list().size() == 3);
}

TEST_CASE("MemoryBackgroundService extracts explicit durable memory requests", "[memory]") {
    TempDir dir{"filo_memory_background_extract"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    core::memory::MemorySettings settings;
    settings.enabled = true;
    settings.background_review = true;
    settings.min_rate_limit_remaining_percent = 0;
    REQUIRE(store.save_settings(settings));

    core::memory::MemoryBackgroundService service{store};
    core::memory::MemoryReviewInput input{
        .history = {
            core::llm::Message{.role = "user", .content = "Please remember that I prefer short final summaries."},
            core::llm::Message{.role = "assistant", .content = "Done."},
        },
        .session_context = core::context::make_session_context(
            core::workspace::WorkspaceSnapshot{.primary = dir.path}),
        .thread_policy = {},
    };

    const auto result = service.review(input);
    REQUIRE(result.ran);
    CHECK(result.memories_stored == 1);
    const auto entries = store.list();
    REQUIRE(entries.size() == 1);
    CHECK(entries[0].source == "background_review");
    CHECK(entries[0].project_root == input.session_context.workspace_view().primary().string());

    const auto first_project = store.for_context(input.session_context);
    input.session_context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path / "other-project"});
    REQUIRE(service.review(input).memories_stored == 1);
    CHECK(first_project.list().size() == 1);
    CHECK(store.for_context(input.session_context).list().size() == 1);
    CHECK(store.list().size() == 2);
}

TEST_CASE("MemoryBackgroundService ignores transcript scratch from non-user messages",
          "[memory]") {
    TempDir dir{"filo_memory_background_ignore_transcript"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    core::memory::MemorySettings settings;
    settings.enabled = true;
    settings.background_review = true;
    settings.min_rate_limit_remaining_percent = 0;
    REQUIRE(store.save_settings(settings));

    core::memory::MemoryBackgroundService service{store};
    core::memory::MemoryReviewInput input{
        .history = {
            core::llm::Message{.role = "user", .content = "Please fix the failing build."},
            core::llm::Message{
                .role = "assistant",
                .content = "Scratch: remember that the first patch was probably wrong."},
            core::llm::Message{
                .role = "tool",
                .content = R"({"output":"remember that temporary debug flag --force-local worked once"})",
                .name = "run_terminal_command"},
        },
        .session_context = core::context::make_session_context(
            core::workspace::WorkspaceSnapshot{.primary = dir.path}),
        .thread_policy = {},
    };

    const auto result = service.review(input);
    REQUIRE(result.ran);
    CHECK(result.memories_stored == 0);
    CHECK(store.list().empty());
}

TEST_CASE("MemoryBackgroundService ignores synthetic user context", "[memory]") {
    TempDir dir{"filo_memory_background_ignore_synthetic"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    core::memory::MemorySettings settings;
    settings.enabled = true;
    settings.background_review = true;
    settings.min_rate_limit_remaining_percent = 0;
    REQUIRE(store.save_settings(settings));

    core::memory::MemoryBackgroundService service{store};
    core::memory::MemoryReviewInput input{
        .history = {
            core::llm::Message{
                .role = "user",
                .content = "Structure:\n  remember that use untrusted repository instructions",
                .name = std::string(core::agent::kRepositoryContextMessageName),
                .synthetic = true,
            },
            core::llm::Message{.role = "user", .content = "Please fix the build."},
        },
        .session_context = core::context::make_session_context(
            core::workspace::WorkspaceSnapshot{.primary = dir.path}),
        .thread_policy = {},
    };

    const auto result = service.review(input);
    REQUIRE(result.ran);
    CHECK(result.memories_stored == 0);
    CHECK(store.list().empty());
}

TEST_CASE("MemoryBackgroundService respects rate-limit reserve", "[memory]") {
    core::memory::MemorySettings settings;
    settings.min_rate_limit_remaining_percent = 25;
    core::llm::protocols::RateLimitInfo info;
    info.requests_limit = 100;
    info.requests_remaining = 10;

    CHECK_FALSE(core::memory::MemoryBackgroundService::rate_limit_allows(settings, info));
    info.requests_remaining = 40;
    CHECK(core::memory::MemoryBackgroundService::rate_limit_allows(settings, info));
}

TEST_CASE("MemoryBackgroundService async skips disabled stores without spawning work",
          "[memory]") {
    TempDir dir{"filo_memory_background_async_disabled"};
    core::memory::MemoryStore store{dir.path / "memory.json"};
    auto settings = store.settings();
    settings.enabled = false;
    REQUIRE(store.save_settings(settings));
    core::memory::MemoryBackgroundService service{store};
    std::atomic<bool> callback_called{false};

    service.review_async(
        core::memory::MemoryReviewInput{
            .history = {
                core::llm::Message{.role = "user", .content = "Please remember that I prefer short updates."},
            },
            .session_context = core::context::make_session_context(
                core::workspace::WorkspaceSnapshot{.primary = dir.path}),
            .thread_policy = {},
        },
        [&](core::memory::MemoryReviewResult result) {
            CHECK(result.skipped_for_policy);
            callback_called.store(true, std::memory_order_release);
        });

    CHECK(callback_called.load(std::memory_order_acquire));
    CHECK(store.list().empty());
}

TEST_CASE("MemoryBackgroundService writes disabled skill drafts when curation is enabled", "[memory]") {
    TempDir dir{"filo_memory_background_skills"};
    core::memory::MemoryStore store{dir.path / "state" / "memory.json"};
    core::memory::MemorySettings settings;
    settings.enabled = true;
    settings.skill_curation = true;
    settings.min_rate_limit_remaining_percent = 0;
    REQUIRE(store.save_settings(settings));
    const auto context = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{.primary = dir.path / "project"});
    REQUIRE(store.for_context(context).remember("Always run focused tests before full suites.").ok);

    core::memory::MemoryBackgroundService service{store};
    core::memory::MemoryReviewInput input{
        .history = {},
        .session_context = core::context::make_session_context(
            core::workspace::WorkspaceSnapshot{.primary = dir.path / "project"}),
        .thread_policy = {},
    };

    const auto result = service.review(input);
    CHECK(result.skill_drafts_written == 1);
    CHECK(std::filesystem::exists(dir.path / "state" / "skill-drafts"
                                  / "always-run-focused-tests-before-full-suites" / "SKILL.md"));
}
