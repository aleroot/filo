#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/session/SessionData.hpp"
#include "core/session/SessionReport.hpp"
#include "core/session/SessionStats.hpp"
#include "core/session/SessionStore.hpp"
#include "core/session/TodoUtils.hpp"
#include <ftxui/screen/string.hpp>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
namespace {

core::session::SessionData make_test_session(std::string_view id) {
    core::session::SessionData data;
    data.session_id        = std::string(id);
    data.created_at        = "2026-03-22T10:15:30Z";
    data.last_active_at    = "2026-03-22T11:00:00Z";
    data.working_dir       = "/home/user/project";
    data.provider          = "grok";
    data.model             = "grok-4";
    data.mode              = "BUILD";
    data.context_summary   = "";

    data.messages.push_back({"user",      "Hello, Filo!", "", "", {}});
    data.messages.push_back({"assistant", "Hello! How can I help?", "", "", {}});

    data.stats.prompt_tokens      = 1234;
    data.stats.completion_tokens  = 567;
    data.stats.cost_usd           = 0.0123;
    data.stats.turn_count         = 1;
    data.stats.tool_calls_total   = 3;
    data.stats.tool_calls_success = 2;
    return data;
}

/// RAII helper: deletes a directory on destruction.
struct TempDir {
    std::filesystem::path path;
    explicit TempDir(std::filesystem::path p) : path(std::move(p)) {
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

struct ScopedEnvVar {
    std::string key;
    std::string old_value;
    bool had_old = false;

    ScopedEnvVar(const char* k, const char* v) : key(k) {
        if (const char* old = std::getenv(k); old != nullptr) {
            had_old = true;
            old_value = old;
        }
        ::setenv(k, v, 1);
    }

    ~ScopedEnvVar() {
        if (had_old) {
            ::setenv(key.c_str(), old_value.c_str(), 1);
        } else {
            ::unsetenv(key.c_str());
        }
    }
};

std::string capture_session_report(const core::budget::BudgetTracker& budget,
                                   const core::session::SessionStats::Snapshot& snap,
                                   std::string_view session_id,
                                   std::string_view session_path) {
    std::ostringstream capture;
    auto* old_buf = std::cout.rdbuf(capture.rdbuf());
    core::session::SessionReport::print(budget, snap, session_id, session_path);
    std::cout.rdbuf(old_buf);
    return capture.str();
}

} // namespace

// ---------------------------------------------------------------------------
// SessionStore::generate_id
// ---------------------------------------------------------------------------

TEST_CASE("SessionStore generates 8-char hex IDs", "[session][store]") {
    const std::string id = core::session::SessionStore::generate_id();
    REQUIRE(id.size() == 8);
    for (const char c : id) {
        const bool hex = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        REQUIRE(hex);
    }
}

TEST_CASE("SessionStore generates unique IDs across calls", "[session][store]") {
    const std::string a = core::session::SessionStore::generate_id();
    const std::string b = core::session::SessionStore::generate_id();
    const std::string c = core::session::SessionStore::generate_id();
    // With 8 hex chars (32-bit entropy) the probability of a collision is negligible.
    REQUIRE(a != b);
    REQUIRE(b != c);
}

// ---------------------------------------------------------------------------
// SessionStore::now_iso8601
// ---------------------------------------------------------------------------

TEST_CASE("now_iso8601 produces valid ISO 8601 format", "[session][store]") {
    const std::string ts = core::session::SessionStore::now_iso8601();
    // "YYYY-MM-DDTHH:MM:SSZ"
    REQUIRE(ts.size() == 20);
    REQUIRE(ts[4]  == '-');
    REQUIRE(ts[7]  == '-');
    REQUIRE(ts[10] == 'T');
    REQUIRE(ts[13] == ':');
    REQUIRE(ts[16] == ':');
    REQUIRE(ts[19] == 'Z');
}

// ---------------------------------------------------------------------------
// JSON round-trip
// ---------------------------------------------------------------------------

TEST_CASE("SessionStore serializes and deserializes a session", "[session][json]") {
    TempDir tmp{std::filesystem::temp_directory_path() / "filo_test_session_rt"};
    core::session::SessionStore store{tmp.path};

    const auto original = make_test_session("abcd1234");
    REQUIRE(store.save(original));

    const auto loaded = store.load_by_id("abcd1234");
    REQUIRE(loaded.has_value());

    const auto& d = *loaded;
    CHECK(d.session_id        == original.session_id);
    CHECK(d.created_at        == original.created_at);
    CHECK(d.last_active_at    == original.last_active_at);
    CHECK(d.working_dir       == original.working_dir);
    CHECK(d.provider          == original.provider);
    CHECK(d.model             == original.model);
    CHECK(d.mode              == original.mode);
    CHECK(d.stats.prompt_tokens      == original.stats.prompt_tokens);
    CHECK(d.stats.completion_tokens  == original.stats.completion_tokens);
    CHECK(d.stats.turn_count         == original.stats.turn_count);
    CHECK(d.stats.tool_calls_total   == original.stats.tool_calls_total);
    CHECK(d.stats.tool_calls_success == original.stats.tool_calls_success);
    CHECK(d.messages.size() == 2);
    CHECK(d.messages[0].role    == "user");
    CHECK(d.messages[0].content == "Hello, Filo!");
    CHECK(d.messages[1].role    == "assistant");
}

TEST_CASE("SessionStore round-trips messages with tool calls", "[session][json]") {
    TempDir tmp{std::filesystem::temp_directory_path() / "filo_test_session_tc"};
    core::session::SessionStore store{tmp.path};

    core::session::SessionData data = make_test_session("tooltest");
    core::llm::ToolCall tc;
    tc.id   = "call_001";
    tc.type = "function";
    tc.function.name      = "read_file";
    tc.function.arguments = R"({"path":"/etc/hosts"})";
    data.messages.push_back({"assistant", "", "", "", {tc}});
    data.messages.push_back({"tool", R"({"content":"127.0.0.1 localhost"})", "read_file", "call_001", {}});

    REQUIRE(store.save(data));
    const auto loaded = store.load_by_id("tooltest");
    REQUIRE(loaded.has_value());

    const auto& msgs = loaded->messages;
    REQUIRE(msgs.size() == 4);
    REQUIRE(msgs[2].role == "assistant");
    REQUIRE(msgs[2].tool_calls.size() == 1);
    CHECK(msgs[2].tool_calls[0].id              == "call_001");
    CHECK(msgs[2].tool_calls[0].function.name   == "read_file");
    REQUIRE(msgs[3].role         == "tool");
    CHECK(msgs[3].tool_call_id   == "call_001");
}

TEST_CASE("SessionStore round-trips messages with special JSON characters", "[session][json]") {
    TempDir tmp{std::filesystem::temp_directory_path() / "filo_test_session_esc"};
    core::session::SessionStore store{tmp.path};

    core::session::SessionData data = make_test_session("escape01");
    // Content with backslash, quote, and newlines.
    data.messages[0].content = "He said \"hello\"\nLine2\\nBackslash";
    REQUIRE(store.save(data));

    const auto loaded = store.load_by_id("escape01");
    REQUIRE(loaded.has_value());
    CHECK(loaded->messages[0].content == "He said \"hello\"\nLine2\\nBackslash");
}

TEST_CASE("SessionStore round-trips session todos", "[session][json]") {
    TempDir tmp{std::filesystem::temp_directory_path() / "filo_test_session_todos"};
    core::session::SessionStore store{tmp.path};

    core::session::SessionData data = make_test_session("todo1234");
    data.todos.push_back(core::session::SessionTodoItem{
        .id = "1",
        .text = "Investigate steering prompt",
        .completed = false,
        .created_at = "2026-03-22T10:20:00Z",
        .completed_at = {},
    });
    data.todos.push_back(core::session::SessionTodoItem{
        .id = "2",
        .text = "Ship MCP command",
        .completed = true,
        .created_at = "2026-03-22T10:21:00Z",
        .completed_at = "2026-03-22T10:30:00Z",
    });

    REQUIRE(store.save(data));
    const auto loaded = store.load_by_id("todo1234");
    REQUIRE(loaded.has_value());
    REQUIRE(loaded->todos.size() == 2);
    CHECK(loaded->todos[0].text == "Investigate steering prompt");
    CHECK_FALSE(loaded->todos[0].completed);
    CHECK(loaded->todos[1].completed);
    CHECK(loaded->todos[1].completed_at == "2026-03-22T10:30:00Z");
}

TEST_CASE("Todo utils generate stable prefixed IDs across legacy sessions", "[session][todos]") {
    std::vector<core::session::SessionTodoItem> todos = {
        {.id = "1", .text = "Legacy one"},
        {.id = "t3", .text = "Modern three"},
    };

    CHECK(core::session::todo::next_id(todos) == "t4");
}

TEST_CASE("Todo utils prefer exact IDs over row indexes for legacy numeric IDs", "[session][todos]") {
    std::vector<core::session::SessionTodoItem> todos = {
        {.id = "2", .text = "Second legacy todo"},
        {.id = "3", .text = "Third legacy todo"},
    };

    REQUIRE(core::session::todo::resolve_index(todos, "2").has_value());
    CHECK(*core::session::todo::resolve_index(todos, "2") == 0);
    REQUIRE(core::session::todo::resolve_index(todos, "{3}").has_value());
    CHECK(*core::session::todo::resolve_index(todos, "{3}") == 1);
}

TEST_CASE("Todo utils fall back to row indexes when IDs are prefixed", "[session][todos]") {
    std::vector<core::session::SessionTodoItem> todos = {
        {.id = "t2", .text = "First visible row"},
        {.id = "t3", .text = "Second visible row"},
    };

    REQUIRE(core::session::todo::resolve_index(todos, "2").has_value());
    CHECK(*core::session::todo::resolve_index(todos, "2") == 1);
    REQUIRE(core::session::todo::resolve_index(todos, "t2").has_value());
    CHECK(*core::session::todo::resolve_index(todos, "t2") == 0);
}

// ---------------------------------------------------------------------------
// SessionStore list / load_by_index
// ---------------------------------------------------------------------------

TEST_CASE("SessionStore list returns sessions sorted most-recent first", "[session][store]") {
    TempDir tmp{std::filesystem::temp_directory_path() / "filo_test_session_list"};
    core::session::SessionStore store{tmp.path};

    auto a = make_test_session("aaaaaaaa");
    a.last_active_at = "2026-03-20T08:00:00Z";
    auto b = make_test_session("bbbbbbbb");
    b.last_active_at = "2026-03-22T12:00:00Z";
    auto c = make_test_session("cccccccc");
    c.last_active_at = "2026-03-21T15:30:00Z";

    REQUIRE(store.save(a));
    REQUIRE(store.save(b));
    REQUIRE(store.save(c));

    const auto infos = store.list();
    REQUIRE(infos.size() == 3);
    CHECK(infos[0].session_id == "bbbbbbbb"); // most recent
    CHECK(infos[1].session_id == "cccccccc");
    CHECK(infos[2].session_id == "aaaaaaaa");
}

TEST_CASE("SessionStore load_by_index resolves 1-based index", "[session][store]") {
    TempDir tmp{std::filesystem::temp_directory_path() / "filo_test_session_idx"};
    core::session::SessionStore store{tmp.path};

    auto a = make_test_session("idx1aaaa");
    a.last_active_at = "2026-03-21T10:00:00Z";
    auto b = make_test_session("idx2bbbb");
    b.last_active_at = "2026-03-22T10:00:00Z";
    REQUIRE(store.save(a));
    REQUIRE(store.save(b));

    const auto first = store.load_by_index(1);
    REQUIRE(first.has_value());
    CHECK(first->session_id == "idx2bbbb"); // most recent = index 1

    const auto second = store.load_by_index(2);
    REQUIRE(second.has_value());
    CHECK(second->session_id == "idx1aaaa");

    const auto oob = store.load_by_index(3);
    CHECK(!oob.has_value());
}

TEST_CASE("SessionStore::load dispatches by integer index or ID", "[session][store]") {
    TempDir tmp{std::filesystem::temp_directory_path() / "filo_test_session_disp"};
    core::session::SessionStore store{tmp.path};

    auto d = make_test_session("dispatch1");
    d.last_active_at = "2026-03-22T10:00:00Z";
    REQUIRE(store.save(d));

    const auto by_id  = store.load("dispatch1");
    REQUIRE(by_id.has_value());
    CHECK(by_id->session_id == "dispatch1");

    const auto by_idx = store.load("1");
    REQUIRE(by_idx.has_value());
    CHECK(by_idx->session_id == "dispatch1");
}

// ---------------------------------------------------------------------------
// SessionStore::remove
// ---------------------------------------------------------------------------

TEST_CASE("SessionStore removes a session by ID", "[session][store]") {
    TempDir tmp{std::filesystem::temp_directory_path() / "filo_test_session_del"};
    core::session::SessionStore store{tmp.path};

    const auto d = make_test_session("del00001");
    REQUIRE(store.save(d));
    REQUIRE(store.load_by_id("del00001").has_value());

    REQUIRE(store.remove("del00001"));
    CHECK(!store.load_by_id("del00001").has_value());
}

// ---------------------------------------------------------------------------
// SessionStats
// ---------------------------------------------------------------------------

TEST_CASE("SessionStats accumulates turns and per-model stats", "[session][stats]") {
    auto& stats = core::session::SessionStats::get_instance();
    stats.reset();

    core::llm::TokenUsage u1{.prompt_tokens = 100, .completion_tokens = 50, .total_tokens = 150};
    core::llm::TokenUsage u2{.prompt_tokens = 200, .completion_tokens = 80, .total_tokens = 280};

    stats.record_turn("grok-4", u1);
    stats.record_turn("grok-4", u2);
    stats.record_turn("grok-code-fast-1", u1);

    const auto snap = stats.snapshot();
    CHECK(snap.turn_count == 3);
    REQUIRE(snap.per_model.size() == 2);

    // per_model is sorted by call_count descending.
    CHECK(snap.per_model[0].model == "grok-4");
    CHECK(snap.per_model[0].call_count == 2);
    CHECK(snap.per_model[0].prompt_tokens == 300);
    CHECK(snap.per_model[0].completion_tokens == 130);
    CHECK(snap.per_model[1].model == "grok-code-fast-1");
    CHECK(snap.per_model[1].call_count == 1);
}

TEST_CASE("SessionStats can skip cost estimation for local providers", "[session][stats]") {
    auto& stats = core::session::SessionStats::get_instance();
    stats.reset();

    stats.record_turn("grok-4", {.prompt_tokens = 1'000'000, .completion_tokens = 1'000'000}, false);

    const auto snap = stats.snapshot();
    REQUIRE(snap.turn_count == 1);
    REQUIRE(snap.per_model.size() == 1);
    CHECK(snap.per_model[0].model == "grok-4");
    CHECK(snap.per_model[0].cost_usd == 0.0);
}

TEST_CASE("SessionStats accumulates tool call success and failure", "[session][stats]") {
    auto& stats = core::session::SessionStats::get_instance();
    stats.reset();

    stats.record_tool_call(true);
    stats.record_tool_call(true);
    stats.record_tool_call(false);

    const auto snap = stats.snapshot();
    CHECK(snap.tool_calls_total   == 3);
    CHECK(snap.tool_calls_success == 2);
}

TEST_CASE("SessionStats accumulates API call success and failure", "[session][stats]") {
    auto& stats = core::session::SessionStats::get_instance();
    stats.reset();

    stats.record_api_call(true);
    stats.record_api_call(true);
    stats.record_api_call(true);
    stats.record_api_call(false);
    stats.record_api_call(false);

    const auto snap = stats.snapshot();
    CHECK(snap.api_calls_total   == 5);
    CHECK(snap.api_calls_success == 3);
}

TEST_CASE("SessionStats reset clears all counters", "[session][stats]") {
    auto& stats = core::session::SessionStats::get_instance();
    stats.record_turn("grok-4", {100, 50, 150});
    stats.record_tool_call(true);
    stats.record_api_call(true);
    stats.record_api_call(false);

    stats.reset();
    const auto snap = stats.snapshot();
    CHECK(snap.turn_count        == 0);
    CHECK(snap.tool_calls_total  == 0);
    CHECK(snap.api_calls_total   == 0);
    CHECK(snap.api_calls_success == 0);
    CHECK(snap.per_model.empty());
}

TEST_CASE("SessionStats is safe for concurrent record_turn calls", "[session][stats][threading]") {
    auto& stats = core::session::SessionStats::get_instance();
    stats.reset();

    constexpr int kThreads = 8;
    constexpr int kPerThread = 100;

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&stats, t]() {
            const std::string model = (t % 2 == 0) ? "model-a" : "model-b";
            const core::llm::TokenUsage u{10, 5, 15};
            for (int i = 0; i < kPerThread; ++i) {
                stats.record_turn(model, u);
            }
        });
    }
    for (auto& th : threads) th.join();

    const auto snap = stats.snapshot();
    CHECK(snap.turn_count == kThreads * kPerThread);

    int32_t total_calls = 0;
    for (const auto& m : snap.per_model) total_calls += m.call_count;
    CHECK(total_calls == kThreads * kPerThread);
}

// ---------------------------------------------------------------------------
// SessionStore::compute_path
// ---------------------------------------------------------------------------

TEST_CASE("SessionStore::compute_path produces sortable filenames", "[session][store]") {
    const core::session::SessionStore store{"/tmp/filo_test"};
    core::session::SessionData data = make_test_session("path1234");
    data.created_at = "2026-03-22T10:15:30Z";

    const auto p = store.compute_path(data);
    const std::string fname = p.filename().string();

    CHECK(fname == "session-20260322-101530-path1234.json");
}

// ---------------------------------------------------------------------------
// SessionStore::default_sessions_dir
// ---------------------------------------------------------------------------

TEST_CASE("SessionStore::default_sessions_dir uses XDG_DATA_HOME when set", "[session][store]") {
    // Temporarily set XDG_DATA_HOME
    ::setenv("XDG_DATA_HOME", "/custom/data", 1);
    const auto dir = core::session::SessionStore::default_sessions_dir();
    ::unsetenv("XDG_DATA_HOME");
    CHECK(dir.string() == "/custom/data/filo/sessions");
}

// ---------------------------------------------------------------------------
// SessionReport
// ---------------------------------------------------------------------------

TEST_CASE("SessionReport keeps rows aligned and prints --resume command", "[session][report]") {
    ScopedEnvVar no_color{"NO_COLOR", "1"};

    auto& budget = core::budget::BudgetTracker::get_instance();
    budget.reset_session();
    budget.record({110300, 695, 110995}, "grok-code-fast-1");

    core::session::SessionStats::Snapshot snap{
        .started_at = std::chrono::system_clock::now() - std::chrono::seconds(460),
        .turn_count = 9,
        .tool_calls_total = 8,
        .tool_calls_success = 5,
        .api_calls_total = 10,
        .api_calls_success = 9,
        .per_model = {{
            .model = "grok-code-fast-1",
            .call_count = 9,
            .prompt_tokens = 110300,
            .completion_tokens = 695,
            .cost_usd = 0.023,
        }},
    };

    const std::string out = capture_session_report(
        budget, snap, "0e275b32",
        "/home/alessio/.local/share/filo/sessions/session-20260322-143010-0e275b32.json");

    CHECK_THAT(out, Catch::Matchers::ContainsSubstring("Resume with"));
    CHECK_THAT(out, Catch::Matchers::ContainsSubstring("filo --resume 0e275b32"));

    std::istringstream in{out};
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto width = ftxui::string_width(line);
        // 64 inner cells + 2 border cells.
        CHECK(width == 66);
    }
}
