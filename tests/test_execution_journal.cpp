#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include "TestSessionContext.hpp"
#include "core/agent/Agent.hpp"
#include "core/session/ExecutionJournal.hpp"
#include "core/utils/InterprocessFile.hpp"
#include <simdjson.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <future>
#include <thread>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {
struct JournalFixture {
    std::filesystem::path root;
    JournalFixture() {
        auto pattern = (std::filesystem::temp_directory_path() / "filo-journal-XXXXXX").string();
        auto* directory = ::mkdtemp(pattern.data());
        REQUIRE(directory);
        root = directory;
    }
    ~JournalFixture() { std::error_code ec; std::filesystem::remove_all(root, ec); }
    auto journal() const { return std::make_shared<core::session::ExecutionJournal>(root / "journals"); }
    auto context() const {
        return test_support::make_session_context({.primary = root, .enforce = true},
            core::context::SessionTransport::cli, "journal-test");
    }
};
std::vector<std::string> lines(const std::filesystem::path& file) {
    std::ifstream stream(file);
    std::vector<std::string> result;
    std::string line;
    while (std::getline(stream, line)) result.push_back(line);
    return result;
}
class LargeResultTool final : public core::tools::Tool {
public:
    core::tools::ToolDefinition get_definition() const override {
        return {.name = "journal_large_result", .description = "Return a large result.",
            .input_schema = R"({"type":"object","properties":{},"additionalProperties":false})",
            .annotations = {.read_only_hint = true}};
    }
    std::string execute(const std::string&, const core::context::SessionContext&) override {
        return R"({"output":")" + std::string(200'000, 'x') + "END_OF_FULL_RESULT\"}";
    }
};
class JournalProvider final : public core::llm::LLMProvider {
public:
    int calls = 0;
    void stream_response(const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        set_last_usage(100, 20);
        if (calls++ == 0) {
            core::llm::StreamChunk chunk;
            chunk.tools = {core::llm::ToolCall{.index = 0, .id = "raw-call",
                .function = {.name = "journal_large_result", .arguments = "{}"}}};
            callback(chunk);
        } else callback(core::llm::StreamChunk::make_content("Finished."));
        auto final = core::llm::StreamChunk::make_final();
        final.stop_reason = calls == 1 ? "tool_use" : "end_turn";
        callback(final);
    }
    bool should_estimate_cost() const override { return false; }
};
}

TEST_CASE("Execution journal appends private ordered records and recovers a torn tail", "[journal]") {
    JournalFixture fixture;
    auto journal = fixture.journal();
    REQUIRE(journal->append("abc123", "agent", "", "message", R"({"content":"first\nmessage"})"));
    const auto path = journal->path_for("abc123");
    const auto original = core::utils::read_file(path).value();
    std::ofstream(path, std::ios::app) << R"({"broken":)";
    REQUIRE(journal->append("abc123", "child", "call-1", "clear", "{}"));
    auto records = lines(path);
    REQUIRE(records.size() == 2);
    CHECK(core::utils::read_file(path)->starts_with(original));
    simdjson::dom::parser parser;
    auto doc = parser.parse(records.back());
    CHECK(int64_t(doc["offset"]) == static_cast<int64_t>(original.size()));
    CHECK(std::string_view(doc["actor"]) == "child");
    CHECK(std::string_view(doc["parent_tool_call_id"]) == "call-1");
    struct stat status{};
    REQUIRE(::stat(path.c_str(), &status) == 0);
    CHECK((status.st_mode & 0777) == 0600);
    CHECK_FALSE(journal->append("../escape", "agent", "", "clear", "{}"));
    std::ofstream(fixture.root / "untouched") << "original";
    std::filesystem::create_symlink(fixture.root / "untouched", journal->path_for("symlink"));
    CHECK_FALSE(journal->append("symlink", "agent", "", "clear", "{}"));
    CHECK(core::utils::read_file(fixture.root / "untouched") == "original");
    std::filesystem::create_hard_link(fixture.root / "untouched", journal->path_for("hardlink"));
    CHECK_FALSE(journal->append("hardlink", "agent", "", "clear", "{}"));
    CHECK(core::utils::read_file(fixture.root / "untouched") == "original");
}

TEST_CASE("Execution journal serializes concurrent writers without interleaving", "[journal]") {
    JournalFixture fixture;
    std::atomic<int> failures = 0;
    std::vector<std::thread> threads;
    for (int worker = 0; worker < 4; ++worker) {
        threads.emplace_back([&fixture, &failures, worker] {
            auto journal = fixture.journal(); // independent instances, same file
            for (int i = 0; i < 12; ++i)
                if (!journal->append("shared", std::to_string(worker), "", "message", R"({"content":"payload"})"))
                    ++failures;
        });
    }
    for (auto& thread : threads) thread.join();
    REQUIRE(failures == 0);
    auto records = lines(fixture.journal()->path_for("shared"));
    REQUIRE(records.size() == 48);
    int64_t expected_offset = 0;
    for (const auto& line : records) {
        simdjson::dom::parser parser;
        auto doc = parser.parse(line);
        CHECK(int64_t(doc["offset"]) == expected_offset);
        expected_offset += static_cast<int64_t>(line.size() + 1);
    }
}

TEST_CASE("Agent compaction rewind clear and resume never replace execution history", "[journal][agent]") {
    JournalFixture fixture;
    auto& tools = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(nullptr, tools, fixture.context(), fixture.root / "results");
    auto journal = fixture.journal();
    agent->set_execution_journal(journal);
    agent->append_history_message({.role = "user", .content = "original request"});
    agent->append_history_message({.role = "assistant", .content = "details discarded by compaction"});
    agent->compact_history("compact summary");
    const auto compacted = agent->get_history();
    REQUIRE(std::ranges::none_of(compacted, [](const auto& m) {
        return m.content == "details discarded by compaction";
    }));
    agent->append_history_message({.role = "user", .content = "rewound request"});
    agent->append_history_message({.role = "assistant", .content = "rewound answer"});
    agent->undo_last();
    agent->clear_history();
    agent->load_history(compacted, "compact summary", "BUILD");
    const auto before = core::utils::read_file(journal->path_for("journal-test")).value();
    agent.reset();
    agent = std::make_shared<core::agent::Agent>(nullptr, tools, fixture.context(), fixture.root / "results");
    agent->set_execution_journal(journal);
    agent->load_history(compacted, "compact summary", "BUILD");
    agent->append_history_message({.role = "user", .content = "after restart"});
    const auto full = core::utils::read_file(journal->path_for("journal-test")).value();
    CHECK(full.starts_with(before));
    for (const auto& expected : {"original request", "details discarded by compaction", "rewound request",
                                  "rewound answer", "compaction", "rewind", "clear", "resume", "after restart"})
        CHECK_THAT(full, Catch::Matchers::ContainsSubstring(expected));
}

TEST_CASE("Agent journal keeps full tool output and usage before context compression", "[journal][agent]") {
    JournalFixture fixture;
    auto& tools = core::tools::ToolManager::get_instance();
    tools.register_tool(std::make_shared<LargeResultTool>());
    struct Unregister {
        core::tools::ToolManager& tools;
        ~Unregister() { (void)tools.unregister_tool("journal_large_result"); }
    } unregister{tools};
    auto provider = std::make_shared<JournalProvider>();
    auto agent = std::make_shared<core::agent::Agent>(provider, tools, fixture.context(), fixture.root / "results");
    auto journal = fixture.journal();
    agent->set_execution_journal(journal);
    agent->set_mode("BUILD");
    agent->set_auto_compact_threshold(0);
    std::promise<void> done;
    auto completion = done.get_future();
    agent->send_message("Run the tool", [](const std::string&) {},
        [](const std::string&, const std::string&) {}, [&] { done.set_value(); },
        {.allow_background_memory_review = false});
    REQUIRE(completion.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    const auto records = lines(journal->path_for("journal-test"));
    int raw_results = 0;
    int usages = 0;
    for (const auto& record : records) {
        simdjson::dom::parser parser;
        auto doc = parser.parse(record);
        const std::string_view event = doc["event"];
        if (event == "tool_result") {
            ++raw_results;
            const std::string_view content = doc["data"]["content"];
            CHECK(content.size() > 200'000);
            CHECK(content.ends_with("END_OF_FULL_RESULT\"}"));
        }
        if (event == "usage") {
            ++usages;
            CHECK(std::string_view(doc["data"]["stop_reason"]) == (usages == 1 ? "tool_use" : "end_turn"));
            CHECK_FALSE(bool(doc["data"]["incomplete_tool_call"]));
        }
    }
    CHECK(raw_results == 1);
    CHECK(usages == 2);
    const auto history = agent->get_history();
    auto tool = std::ranges::find_if(history, [](const auto& m) { return m.role == "tool"; });
    REQUIRE(tool != history.end());
    CHECK(tool->content.size() < 200'000);
}

TEST_CASE("Deleting a saved session also deletes its execution journal", "[journal][session]") {
    JournalFixture fixture;
    core::session::SessionStore store(fixture.root);
    core::session::SessionData data;
    data.session_id = "abcd1234";
    data.created_at = core::session::SessionStore::now_iso8601();
    data.last_active_at = data.created_at;
    REQUIRE(store.save(data));
    auto journal = fixture.journal();
    REQUIRE(journal->append(data.session_id, "agent", "", "clear", "{}"));
    REQUIRE(store.remove(data.session_id));
    CHECK_FALSE(std::filesystem::exists(journal->path_for(data.session_id)));
    CHECK_FALSE(store.load_by_id(data.session_id));
}

TEST_CASE("Execution journal coordinates separate processes", "[journal]") {
    JournalFixture fixture;
    std::vector<pid_t> children;
    for (int worker = 0; worker < 2; ++worker) {
        const auto child = ::fork();
        REQUIRE(child >= 0);
        if (child == 0) {
            auto journal = fixture.journal();
            for (int i = 0; i < 8; ++i)
                if (!journal->append("processes", std::to_string(worker), "", "clear", "{}"))
                    ::_exit(1);
            ::_exit(0);
        }
        children.push_back(child);
    }
    for (const auto child : children) {
        int status = 0;
        REQUIRE(::waitpid(child, &status, 0) == child);
        REQUIRE(WIFEXITED(status));
        CHECK(WEXITSTATUS(status) == 0);
    }
    CHECK(lines(fixture.journal()->path_for("processes")).size() == 16);
}
