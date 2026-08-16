#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/Agent.hpp"
#include "core/agent/ToolRecovery.hpp"
#include "core/memory/MemorySystem.hpp"
#include "core/memory/ToolRecoveryMemory.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/tools/Tool.hpp"
#include "core/tools/ToolManager.hpp"
#include "TestSessionContext.hpp"

#include <simdjson.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;

namespace rec = core::agent::recovery;

namespace {

class TempDir {
public:
    TempDir()
        : path_(std::filesystem::temp_directory_path()
                / std::format("filo-recovery-{}", std::chrono::steady_clock::now().time_since_epoch().count())) {
        std::filesystem::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

core::tools::ToolDefinition search_definition() {
    return {
        .name = "search",
        .description = "Search files",
        .input_schema =
            R"({"type":"object","properties":{"pattern":{"type":"string"},"limit":{"type":"integer"},"mode":{"enum":["fast","deep"],"type":"string"}},"required":["pattern"],"additionalProperties":false})",
    };
}

core::tools::schema::ArgumentIssue require_issue(
    const core::tools::ToolDefinition& definition,
    std::string_view arguments) {
    const auto result =
        core::tools::schema::validate_arguments(definition, arguments);
    REQUIRE_FALSE(result.has_value());
    return result.error();
}

rec::RecoveryKey search_key(std::string parameter,
                            std::string issue_code = "unknown_argument") {
    return rec::RecoveryKey{
        .tool = "search",
        .schema_fingerprint = rec::schema_fingerprint(search_definition()),
        .issue_code = std::move(issue_code),
        .parameter = std::move(parameter),
    };
}

// ---------------------------------------------------------------------
// Agent-level harness
// ---------------------------------------------------------------------

constexpr std::string_view kValidationToolName = "recovery_search_tool";
constexpr std::string_view kRuntimeToolName = "recovery_runtime_tool";

class RecoverySearchTool final : public core::tools::Tool {
public:
    [[nodiscard]] core::tools::ToolDefinition get_definition() const override {
        return {
            .name = std::string(kValidationToolName),
            .title = "Recovery Search",
            .description = "Searches files for a pattern.",
            .input_schema =
                R"({"type":"object","properties":{"pattern":{"type":"string"}},"required":["pattern"],"additionalProperties":false})",
            .annotations = {.read_only_hint = true, .idempotent_hint = true},
        };
    }

    [[nodiscard]] std::string execute(
        const std::string&,
        const core::context::SessionContext&) override {
        return R"({"ok":true})";
    }
};

class RecoveryRuntimeTool final : public core::tools::Tool {
public:
    [[nodiscard]] core::tools::ToolDefinition get_definition() const override {
        return {
            .name = std::string(kRuntimeToolName),
            .title = "Recovery Runtime",
            .description = "Fails when the verbose flag is used.",
            .input_schema =
                R"({"type":"object","properties":{"term":{"type":"string"},"verbose":{"type":"boolean"}},"additionalProperties":false})",
            .annotations = {.read_only_hint = true, .idempotent_hint = true},
        };
    }

    [[nodiscard]] std::string execute(
        const std::string& json_args,
        const core::context::SessionContext&) override {
        if (json_args.find("\"verbose\"") != std::string::npos) {
            return R"({"error":"verbose mode is not supported"})";
        }
        return R"({"ok":true})";
    }
};

/// Replays a fixed script of tool calls, one model step per entry, then ends
/// the turn with plain text.
class ScriptedToolCallProvider final : public core::llm::LLMProvider {
public:
    struct Step {
        std::string tool;
        std::string arguments;
        /// Extra same-tool calls emitted in the same model step.
        std::vector<std::string> extra_arguments{};
    };

    explicit ScriptedToolCallProvider(std::vector<Step> steps)
        : steps_(std::move(steps)) {}

    ScriptedToolCallProvider(std::string tool_name,
                             std::vector<std::string> argument_scripts) {
        steps_.reserve(argument_scripts.size());
        for (auto& arguments : argument_scripts) {
            steps_.push_back(Step{tool_name, std::move(arguments)});
        }
    }

    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        const auto call = static_cast<std::size_t>(
            calls_.fetch_add(1, std::memory_order_acq_rel));
        if (call < steps_.size()) {
            const auto& step = steps_[call];
            std::vector<core::llm::ToolCall> tools;
            auto append_call = [&](const std::string& arguments, int index) {
                core::llm::ToolCall tool_call;
                tool_call.index = index;
                tool_call.id = std::format("recovery-call-{}-{}", call, index);
                tool_call.type = "function";
                tool_call.function.name = step.tool;
                tool_call.function.arguments = arguments;
                tools.push_back(std::move(tool_call));
            };
            append_call(step.arguments, 0);
            int index = 1;
            for (const auto& extra : step.extra_arguments) {
                append_call(extra, index++);
            }

            core::llm::StreamChunk chunk;
            chunk.tools = std::move(tools);
            chunk.is_final = true;
            callback(chunk);
            return;
        }
        callback(core::llm::StreamChunk::make_content("done"));
        callback(core::llm::StreamChunk::make_final());
    }

private:
    std::vector<Step> steps_;
    std::atomic<int> calls_{0};
};

class RecordingToolRecoveryMemory final : public core::memory::ToolRecoveryMemory {
public:
    explicit RecordingToolRecoveryMemory(std::string fixed_hint = {})
        : fixed_hint_(std::move(fixed_hint)) {}

    [[nodiscard]] std::optional<std::string> recall(
        const rec::RecoveryKey& key) override {
        std::lock_guard lock(mutex_);
        recalled_keys_.push_back(key);
        if (!fixed_hint_.empty()) {
            return fixed_hint_;
        }
        return std::nullopt;
    }

    [[nodiscard]] std::vector<std::string> recall_runtime_hints(
        const std::string&, const std::string&) override {
        return {};
    }

    void record(const rec::RecoveryLesson& lesson) override {
        std::lock_guard lock(mutex_);
        lessons_.push_back(lesson);
    }

    [[nodiscard]] std::vector<rec::RecoveryLesson> lessons() const {
        std::lock_guard lock(mutex_);
        return lessons_;
    }

    [[nodiscard]] std::vector<rec::RecoveryKey> recalled_keys() const {
        std::lock_guard lock(mutex_);
        return recalled_keys_;
    }

private:
    std::string fixed_hint_;
    mutable std::mutex mutex_;
    std::vector<rec::RecoveryLesson> lessons_;
    std::vector<rec::RecoveryKey> recalled_keys_;
};

std::vector<core::llm::Message> run_turn(
    const std::shared_ptr<core::llm::LLMProvider>& provider,
    const std::shared_ptr<core::memory::ToolRecoveryMemory>& recovery) {
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        core::tools::ToolManager::get_instance(),
        test_support::make_workspace_session_context(),
        core::agent::ToolResultStore::default_root(),
        std::shared_ptr<core::power::SleepInhibitor>{},
        std::shared_ptr<core::session::SessionStatsRegistry>{},
        nullptr,
        core::memory::make_memory_system(recovery));

    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;
    agent->send_message(
        "use the tool",
        [](const std::string&) {},
        [](const std::string&, const std::string&) {},
        [&] {
            {
                std::lock_guard lock(done_mutex);
                done = true;
            }
            done_cv.notify_one();
        });
    {
        std::unique_lock lock(done_mutex);
        REQUIRE(done_cv.wait_for(
            lock, std::chrono::seconds(10), [&] { return done; }));
    }
    return agent->get_history();
}

} // namespace

// -----------------------------------------------------------------------
// L0 — structured argument issues
// -----------------------------------------------------------------------

TEST_CASE("validate_arguments classifies unknown arguments with candidates",
          "[agent][recovery][schema]") {
    const auto issue = require_issue(search_definition(), R"({"query":"swift"})");
    CHECK(issue.code == core::tools::schema::ArgumentIssueCode::UnknownArgument);
    CHECK(issue.parameter == "query");
    CHECK(issue.allowed == std::vector<std::string>{"limit", "mode", "pattern"});
    CHECK_THAT(issue.message, ContainsSubstring("Unknown argument 'query'"));
}

TEST_CASE("validate_arguments classifies missing required parameters",
          "[agent][recovery][schema]") {
    const auto issue = require_issue(search_definition(), "{}");
    CHECK(issue.code == core::tools::schema::ArgumentIssueCode::MissingRequired);
    CHECK(issue.parameter == "pattern");
}

TEST_CASE("validate_arguments classifies type and enum mismatches",
          "[agent][recovery][schema]") {
    const auto type_issue =
        require_issue(search_definition(), R"({"pattern":5})");
    CHECK(type_issue.code == core::tools::schema::ArgumentIssueCode::TypeMismatch);
    CHECK(type_issue.parameter == "pattern");
    CHECK(type_issue.allowed == std::vector<std::string>{"string"});

    const auto enum_issue = require_issue(
        search_definition(), R"({"pattern":"x","mode":"turbo"})");
    CHECK(enum_issue.code == core::tools::schema::ArgumentIssueCode::EnumMismatch);
    CHECK(enum_issue.parameter == "mode");
    REQUIRE(enum_issue.allowed.size() == 2);
    CHECK(enum_issue.allowed[0] == "fast");
    CHECK(enum_issue.allowed[1] == "deep");
}

TEST_CASE("validate_arguments classifies malformed payloads",
          "[agent][recovery][schema]") {
    const auto json_issue = require_issue(search_definition(), "{oops");
    CHECK(json_issue.code == core::tools::schema::ArgumentIssueCode::InvalidJson);

    const auto object_issue = require_issue(search_definition(), "[1,2]");
    CHECK(object_issue.code == core::tools::schema::ArgumentIssueCode::NotAnObject);
}

TEST_CASE("normalize_arguments preserves the historical message contract",
          "[agent][recovery][schema]") {
    const auto legacy = core::tools::schema::normalize_arguments(
        search_definition(), R"({"query":"swift"})");
    REQUIRE_FALSE(legacy.has_value());
    CHECK(legacy.error()
          == require_issue(search_definition(), R"({"query":"swift"})").message);
}

// -----------------------------------------------------------------------
// L1 — fingerprint, advisor, lesson policies
// -----------------------------------------------------------------------

TEST_CASE("schema fingerprint ignores descriptions but tracks structure",
          "[agent][recovery]") {
    const core::tools::ToolDefinition described{
        .name = "search",
        .input_schema =
            R"({"type":"object","properties":{"pattern":{"type":"string","description":"the text"}},"additionalProperties":false})",
    };
    const core::tools::ToolDefinition reworded{
        .name = "search",
        .input_schema =
            R"({"type":"object","properties":{"pattern":{"description":"changed wording","type":"string"}},"additionalProperties":false})",
    };
    const core::tools::ToolDefinition structural{
        .name = "search",
        .input_schema =
            R"({"type":"object","properties":{"query":{"type":"string","description":"the text"}},"additionalProperties":false})",
    };

    CHECK(rec::schema_fingerprint(described)
          == rec::schema_fingerprint(reworded));
    CHECK(rec::schema_fingerprint(described)
          != rec::schema_fingerprint(structural));
    CHECK(rec::schema_fingerprint(described)
          == rec::schema_fingerprint(described));

    const core::tools::ToolDefinition named_description{
        .name = "search",
        .input_schema =
            R"({"type":"object","properties":{"description":{"type":"string","description":"the text"}},"additionalProperties":false})",
    };
    const core::tools::ToolDefinition renamed_description{
        .name = "search",
        .input_schema =
            R"({"type":"object","properties":{"summary":{"type":"string","description":"the text"}},"additionalProperties":false})",
    };
    CHECK(rec::schema_fingerprint(named_description)
          != rec::schema_fingerprint(renamed_description));
    CHECK(rec::schema_fingerprint(named_description)
          != rec::schema_fingerprint(described));
    CHECK_THAT(
        core::tools::schema::structural_input_schema(named_description),
        ContainsSubstring(R"("description":{"type":"string"})"));
}

TEST_CASE("advisor suggests renames, removals, and repairs deterministically",
          "[agent][recovery]") {
    const auto definition = search_definition();

    const auto typo = rec::advise(
        require_issue(definition, R"({"patern":"x"})"), definition);
    REQUIRE(typo.has_value());
    CHECK(typo->instruction == "Replace 'patern' with 'pattern'.");
    CHECK_FALSE(typo->learned);

    const auto distant = rec::advise(
        require_issue(definition, R"({"query":"x"})"), definition);
    REQUIRE(distant.has_value());
    CHECK(distant->instruction
          == "Remove 'query'; this tool accepts: limit, mode, pattern.");

    const auto missing = rec::advise(require_issue(definition, "{}"), definition);
    REQUIRE(missing.has_value());
    CHECK(missing->instruction == "Add the required parameter 'pattern'.");

    const auto enumeration = rec::advise(
        require_issue(definition, R"({"pattern":"x","mode":"turbo"})"),
        definition);
    REQUIRE(enumeration.has_value());
    CHECK(enumeration->instruction == "'mode' accepts one of: fast, deep.");

    const auto typed = rec::advise(
        require_issue(definition, R"({"pattern":5})"), definition);
    REQUIRE(typed.has_value());
    CHECK(typed->instruction == "'pattern' must be of type string.");

    CHECK_FALSE(rec::advise(
        require_issue(definition, "{oops"), definition)
        .has_value());
}

TEST_CASE("validation lessons infer renames from fail→success evidence",
          "[agent][recovery]") {
    const auto definition = search_definition();
    const auto issue = require_issue(definition, R"({"query":"swift"})");

    const auto rename = rec::derive_validation_lesson(
        "search",
        definition,
        issue,
        R"({"query":"swift","limit":10})",
        R"({"pattern":"swift","limit":10})");
    REQUIRE(rename.has_value());
    CHECK(rename->hint == "Replace 'query' with 'pattern'.");
    CHECK(rename->key == search_key("query"));

    const auto removal = rec::derive_validation_lesson(
        "search",
        definition,
        issue,
        R"({"pattern":"x","query":"y"})",
        R"({"pattern":"x"})");
    REQUIRE(removal.has_value());
    CHECK(removal->hint == "Remove the unsupported 'query' parameter.");

    // Ambiguous: two new parameters appear alongside the corrected one.
    const auto ambiguous = rec::derive_validation_lesson(
        "search",
        definition,
        issue,
        R"({"query":"x"})",
        R"({"pattern":"x","mode":"fast"})");
    REQUIRE(ambiguous.has_value());
    CHECK(ambiguous->hint == "Remove the unsupported 'query' parameter.");

    // Non-learnable issue codes never yield lessons.
    CHECK_FALSE(rec::derive_validation_lesson(
        "search",
        definition,
        require_issue(definition, "{}"),
        "{}",
        R"({"pattern":"x"})")
        .has_value());
}

TEST_CASE("runtime lessons require a minimal structural correction",
          "[agent][recovery]") {
    const auto definition = search_definition();

    const auto removal = rec::derive_runtime_lesson(
        "search",
        definition,
        R"({"pattern":"x","limit":10})",
        R"({"pattern":"x"})");
    REQUIRE(removal.has_value());
    CHECK(removal->hint == "Remove the 'limit' parameter.");
    CHECK(removal->key.issue_code == "runtime_failure");
    CHECK(removal->key.parameter == "limit");

    const auto rename = rec::derive_runtime_lesson(
        "search",
        definition,
        R"({"query":"swift"})",
        R"({"pattern":"swift"})");
    REQUIRE(rename.has_value());
    CHECK(rename->hint == "Replace 'query' with 'pattern'.");

    // The value changed too: the evidence no longer isolates the rename.
    CHECK_FALSE(rec::derive_runtime_lesson(
        "search",
        definition,
        R"({"query":"swift"})",
        R"({"pattern":"other"})")
        .has_value());

    // More than one parameter changed: ambiguous, not learnable.
    CHECK_FALSE(rec::derive_runtime_lesson(
        "search",
        definition,
        R"({"pattern":"a","limit":1})",
        R"({"pattern":"b","limit":2})")
        .has_value());

    // Identical arguments carry no structural lesson.
    CHECK_FALSE(rec::derive_runtime_lesson(
        "search",
        definition,
        R"({"pattern":"a"})",
        R"({"pattern":"a"})")
        .has_value());
}

TEST_CASE("error payloads are detected and augmented safely",
          "[agent][recovery]") {
    CHECK(rec::result_indicates_error(R"({"error":"boom"})"));
    CHECK(rec::result_indicates_error(R"({"isError":true,"output":"x"})"));
    CHECK_FALSE(rec::result_indicates_error(R"({"ok":true})"));
    CHECK_FALSE(rec::result_indicates_error("not json"));

    const std::string augmented =
        rec::augment_error_payload(R"({"error":"boom"})", "Remove 'verbose'.");
    CHECK_THAT(augmented, ContainsSubstring(R"("error":"boom")"));
    CHECK_THAT(augmented, ContainsSubstring(R"("recovery_hint":"Remove 'verbose'.")"));

    // Existing hints are never overwritten; non-objects pass through.
    CHECK(rec::augment_error_payload(
              R"({"error":"x","recovery_hint":"kept"})", "ignored")
          == R"({"error":"x","recovery_hint":"kept"})");
    CHECK(rec::augment_error_payload("plain text", "ignored") == "plain text");
}

// -----------------------------------------------------------------------
// L2 — persistent store
// -----------------------------------------------------------------------

TEST_CASE("file recovery memory gates recall on evidence",
          "[agent][recovery][memory]") {
    TempDir dir;
    core::memory::FileToolRecoveryMemory store(dir.path() / "recovery.json");
    const rec::RecoveryLesson lesson{
        .key = search_key("query"),
        .hint = "Replace 'query' with 'pattern'.",
    };

    store.record(lesson);
    CHECK_FALSE(store.recall(lesson.key).has_value());

    store.record(lesson);
    const auto recalled = store.recall(lesson.key);
    REQUIRE(recalled.has_value());
    CHECK(*recalled == "Replace 'query' with 'pattern'.");
}

TEST_CASE("file recovery memory resets evidence on conflicting hints",
          "[agent][recovery][memory]") {
    TempDir dir;
    core::memory::FileToolRecoveryMemory store(dir.path() / "recovery.json");
    const rec::RecoveryKey key = search_key("query");

    store.record({.key = key, .hint = "first"});
    store.record({.key = key, .hint = "first"});
    REQUIRE(store.recall(key).has_value());

    store.record({.key = key, .hint = "second"});
    CHECK_FALSE(store.recall(key).has_value());

    store.record({.key = key, .hint = "second"});
    const auto recalled = store.recall(key);
    REQUIRE(recalled.has_value());
    CHECK(*recalled == "second");
}

TEST_CASE("file recovery memory survives restarts and corrupt files",
          "[agent][recovery][memory]") {
    TempDir dir;
    const std::filesystem::path path = dir.path() / "recovery.json";
    {
        core::memory::FileToolRecoveryMemory store(path);
        store.record({.key = search_key("query"), .hint = "hint-a"});
        store.record({.key = search_key("query"), .hint = "hint-a"});
    }
    {
        core::memory::FileToolRecoveryMemory store(path);
        const auto recalled = store.recall(search_key("query"));
        REQUIRE(recalled.has_value());
        CHECK(*recalled == "hint-a");
    }

    {
        std::ofstream corrupt(path, std::ios::binary | std::ios::trunc);
        corrupt << "{not json";
    }
    core::memory::FileToolRecoveryMemory corrupt_store(path);
    CHECK(corrupt_store.entry_count() == 0);
    corrupt_store.record({.key = search_key("mode"), .hint = "hint-b"});
    CHECK(corrupt_store.entry_count() == 1);
}

TEST_CASE("file recovery memory evicts weakest lessons first",
          "[agent][recovery][memory]") {
    TempDir dir;
    core::memory::FileToolRecoveryMemory store(dir.path() / "recovery.json");

    const rec::RecoveryLesson proven{
        .key = search_key("query"),
        .hint = "proven lesson",
    };
    store.record(proven);
    store.record(proven);
    REQUIRE(store.recall(proven.key).has_value());

    for (int i = 0;
         i < static_cast<int>(core::memory::FileToolRecoveryMemory::kMaxEntries);
         ++i) {
        store.record({
            .key = search_key(std::format("filler-{}", i)),
            .hint = std::format("filler {}", i),
        });
    }

    CHECK(store.entry_count()
          == core::memory::FileToolRecoveryMemory::kMaxEntries);
    const auto survivor = store.recall(proven.key);
    REQUIRE(survivor.has_value());
    CHECK(*survivor == "proven lesson");
}

TEST_CASE("file recovery memory merges concurrent writers instead of clobbering",
          "[agent][recovery][memory]") {
    TempDir dir;
    const std::filesystem::path path = dir.path() / "recovery.json";

    // Two independent stores model the real topology: the TUI agent plus one
    // store per delegated subagent, all pointed at the same file.
    core::memory::FileToolRecoveryMemory first(path);
    core::memory::FileToolRecoveryMemory second(path);

    first.record({.key = search_key("query"), .hint = "from-first"});
    second.record({.key = search_key("mode"), .hint = "from-second"});

    // Neither writer erased the other's lesson.
    CHECK(first.entry_count() == 2);
    CHECK(second.entry_count() == 2);

    // Evidence accumulated across instances promotes the lesson.
    second.record({.key = search_key("query"), .hint = "from-first"});
    const auto recalled = first.recall(search_key("query"));
    REQUIRE(recalled.has_value());
    CHECK(*recalled == "from-first");
}

TEST_CASE("file recovery memory tolerates hostile and future-versioned stores",
          "[agent][recovery][memory]") {
    TempDir dir;
    const std::filesystem::path path = dir.path() / "recovery.json";

    SECTION("a newer on-disk version is ignored rather than misparsed") {
        const std::string newer =
            R"({"version":99,"lessons":[{"tool":"search","fingerprint":"f",)"
            R"("issue":"unknown_argument","parameter":"query","hint":"x","evidence":9}]})";
        std::ofstream(path, std::ios::binary | std::ios::trunc) << newer;
        core::memory::FileToolRecoveryMemory store(path);
        CHECK(store.entry_count() == 0);
        store.record({.key = search_key("query"), .hint = "must not overwrite"});
        store.record({.key = search_key("query"), .hint = "must not overwrite"});
        CHECK(store.entry_count() == 0);
        std::ifstream in(path, std::ios::binary);
        const std::string on_disk((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
        CHECK_THAT(on_disk, ContainsSubstring(R"("version":99)"));
        CHECK_THAT(on_disk, ContainsSubstring(R"("hint":"x")"));
    }

    SECTION("duplicate keys collapse to a single entry") {
        std::ofstream(path, std::ios::binary | std::ios::trunc)
            << R"({"version":1,"lessons":[)"
               R"({"tool":"t","fingerprint":"f","issue":"i","parameter":"p","hint":"first","evidence":5},)"
               R"({"tool":"t","fingerprint":"f","issue":"i","parameter":"p","hint":"second","evidence":5}]})";
        core::memory::FileToolRecoveryMemory store(path);
        CHECK(store.entry_count() == 1);
        const auto recalled = store.recall(rec::RecoveryKey{
            .tool = "t", .schema_fingerprint = "f",
            .issue_code = "i", .parameter = "p"});
        REQUIRE(recalled.has_value());
        CHECK(*recalled == "first");
    }

    SECTION("an oversized stored hint is clamped on load") {
        const std::string huge(rec::kMaxHintChars * 4, 'x');
        std::ofstream(path, std::ios::binary | std::ios::trunc)
            << R"({"version":1,"lessons":[{"tool":"t","fingerprint":"f",)"
               R"("issue":"i","parameter":"p","hint":")" << huge
            << R"(","evidence":5}]})";
        core::memory::FileToolRecoveryMemory store(path);
        const auto recalled = store.recall(rec::RecoveryKey{
            .tool = "t", .schema_fingerprint = "f",
            .issue_code = "i", .parameter = "p"});
        REQUIRE(recalled.has_value());
        CHECK(recalled->size() <= rec::kMaxHintChars + 4);
    }
}

TEST_CASE("file recovery memory does not mutate the store when the lock fails",
          "[agent][recovery][memory]") {
    TempDir dir;
    const auto blocker = dir.path() / "blocked";
    {
        std::ofstream(blocker, std::ios::binary) << "not-a-directory";
    }
    core::memory::FileToolRecoveryMemory store(blocker / "recovery.json");
    store.record({.key = search_key("query"), .hint = "unreachable"});
    store.record({.key = search_key("query"), .hint = "unreachable"});
    CHECK(store.entry_count() == 0);
    CHECK_FALSE(std::filesystem::exists(blocker / "recovery.json"));
}

TEST_CASE("file recovery memory caps the runtime hints it merges",
          "[agent][recovery][memory]") {
    TempDir dir;
    core::memory::FileToolRecoveryMemory store(dir.path() / "recovery.json");
    const std::string fingerprint = rec::schema_fingerprint(search_definition());

    for (int i = 0; i < 5; ++i) {
        const rec::RecoveryLesson lesson{
            .key = search_key(std::format("p{}", i), "runtime_failure"),
            .hint = std::format("runtime hint {}", i),
        };
        store.record(lesson);
        store.record(lesson);
    }

    const auto hints = store.recall_runtime_hints("search", fingerprint);
    CHECK(hints.size() == core::memory::FileToolRecoveryMemory::kMaxRuntimeHints);
}

TEST_CASE("hints are clamped without splitting a UTF-8 sequence",
          "[agent][recovery]") {
    CHECK(rec::clamp_hint("short") == "short");

    const std::string clamped = rec::clamp_hint(std::string(500, 'a'));
    CHECK(clamped.size() <= rec::kMaxHintChars + 4);
    CHECK(clamped.ends_with("…"));

    // A run of multi-byte characters must not be cut mid-sequence.
    std::string multibyte;
    while (multibyte.size() < 400) multibyte += "é";
    const std::string clamped_multibyte = rec::clamp_hint(multibyte);
    simdjson::dom::parser parser;
    const std::string as_json =
        "{\"h\":\"" + clamped_multibyte + "\"}";
    simdjson::padded_string padded(as_json);
    simdjson::dom::element root;
    CHECK(parser.parse(padded).get(root) == simdjson::SUCCESS);
}

TEST_CASE("the recovery memory factory honours the configuration switch",
          "[agent][recovery][memory]") {
    const auto disabled = core::memory::make_tool_recovery_memory(false);
    REQUIRE(disabled != nullptr);
    disabled->record({.key = search_key("query"), .hint = "never stored"});
    disabled->record({.key = search_key("query"), .hint = "never stored"});
    CHECK_FALSE(disabled->recall(search_key("query")).has_value());

    const auto enabled = core::memory::make_tool_recovery_memory(true);
    REQUIRE(enabled != nullptr);
    CHECK(dynamic_cast<core::memory::FileToolRecoveryMemory*>(enabled.get())
          != nullptr);
}

// -----------------------------------------------------------------------
// L3 — agent loop integration
// -----------------------------------------------------------------------

TEST_CASE("agent records a lesson from failure followed by corrected success",
          "[agent][recovery][integration]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<RecoverySearchTool>());

    auto memory = std::make_shared<RecordingToolRecoveryMemory>();
    const auto history = run_turn(
        std::make_shared<ScriptedToolCallProvider>(
            std::string(kValidationToolName),
            std::vector<std::string>{
                R"({"query":"swift"})",
                R"({"pattern":"swift"})",
            }),
        memory);

    const auto lessons = memory->lessons();
    REQUIRE(lessons.size() == 1);
    CHECK(lessons[0].hint == "Replace 'query' with 'pattern'.");
    CHECK(lessons[0].key.parameter == "query");
    CHECK(lessons[0].key.issue_code == "unknown_argument");

    bool saw_structured_rejection = false;
    for (const auto& message : history) {
        if (message.role != "tool") continue;
        if (message.content.find("Invalid tool arguments") != std::string::npos) {
            saw_structured_rejection = true;
            CHECK_THAT(message.content,
                       ContainsSubstring(R"("parameter":"query")"));
            CHECK_THAT(message.content, ContainsSubstring("recovery_hint"));
        }
    }
    CHECK(saw_structured_rejection);
}

TEST_CASE("agent recalls a proven lesson inside the failure payload",
          "[agent][recovery][integration]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<RecoverySearchTool>());

    const auto history = run_turn(
        std::make_shared<ScriptedToolCallProvider>(
            std::string(kValidationToolName),
            std::vector<std::string>{R"({"query":"swift"})"}),
        std::make_shared<RecordingToolRecoveryMemory>(
            "Replace 'query' with 'pattern'."));

    bool saw_learned_hint = false;
    for (const auto& message : history) {
        if (message.role != "tool") continue;
        if (message.content.find("Invalid tool arguments") != std::string::npos) {
            CHECK_THAT(
                message.content,
                ContainsSubstring(
                    R"("recovery_hint":"Replace 'query' with 'pattern'.")"));
            saw_learned_hint = true;
        }
    }
    CHECK(saw_learned_hint);
}

TEST_CASE("agent ignores a correction that arrives many steps after the failure",
          "[agent][recovery][integration]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<RecoverySearchTool>());
    tool_manager.register_tool(std::make_shared<RecoveryRuntimeTool>());

    // The failure and the eventual success are separated by unrelated work, so
    // pairing them would attribute another task's arguments to the mistake.
    std::vector<ScriptedToolCallProvider::Step> steps{
        {std::string(kValidationToolName), R"({"query":"swift"})"},
    };
    for (int i = 0; i <= rec::kMaxObservationStepDistance; ++i) {
        steps.push_back({std::string(kRuntimeToolName), R"({"term":"x"})"});
    }
    steps.push_back({std::string(kValidationToolName), R"({"pattern":"swift"})"});

    auto memory = std::make_shared<RecordingToolRecoveryMemory>();
    run_turn(std::make_shared<ScriptedToolCallProvider>(std::move(steps)), memory);

    CHECK(memory->lessons().empty());
}

TEST_CASE("agent records runtime lessons from tool-reported failures",
          "[agent][recovery][integration]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<RecoveryRuntimeTool>());

    auto memory = std::make_shared<RecordingToolRecoveryMemory>();
    run_turn(
        std::make_shared<ScriptedToolCallProvider>(
            std::string(kRuntimeToolName),
            std::vector<std::string>{
                R"({"term":"x","verbose":true})",
                R"({"term":"x"})",
            }),
        memory);

    const auto lessons = memory->lessons();
    REQUIRE(lessons.size() == 1);
    CHECK(lessons[0].hint == "Remove the 'verbose' parameter.");
    CHECK(lessons[0].key.issue_code == "runtime_failure");
    CHECK(lessons[0].key.parameter == "verbose");
}

TEST_CASE("agent does not learn from same-step parallel runtime outcomes",
          "[agent][recovery][integration]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<RecoveryRuntimeTool>());

    SECTION("a sibling success in the failing step is not a correction") {
        auto memory = std::make_shared<RecordingToolRecoveryMemory>();
        run_turn(
            std::make_shared<ScriptedToolCallProvider>(
                std::vector<ScriptedToolCallProvider::Step>{{
                    .tool = std::string(kRuntimeToolName),
                    .arguments = R"({"term":"x","verbose":true})",
                    .extra_arguments = {R"({"term":"x"})"},
                }}),
            memory);
        CHECK(memory->lessons().empty());
    }

    SECTION("two same-step failures make a later success ambiguous") {
        auto memory = std::make_shared<RecordingToolRecoveryMemory>();
        run_turn(
            std::make_shared<ScriptedToolCallProvider>(
                std::vector<ScriptedToolCallProvider::Step>{
                    {
                        .tool = std::string(kRuntimeToolName),
                        .arguments = R"({"term":"a","verbose":true})",
                        .extra_arguments = {R"({"term":"b","verbose":true})"},
                    },
                    {
                        .tool = std::string(kRuntimeToolName),
                        .arguments = R"({"term":"x"})",
                    },
                }),
            memory);
        CHECK(memory->lessons().empty());
    }
}
