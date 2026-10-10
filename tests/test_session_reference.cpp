#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/context/ContextMentions.hpp"
#include "core/session/SessionDigest.hpp"
#include "core/session/SessionReference.hpp"
#include "core/session/SessionReferenceCatalogue.hpp"
#include "core/session/SessionStore.hpp"
#include "core/session/ThreadCatalog.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;

namespace {

struct TempDir {
    fs::path path;
    explicit TempDir(const std::string& label)
        : path(fs::temp_directory_path()
               / (label + "_" + std::to_string(static_cast<long long>(std::rand())))) {
        fs::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

core::llm::Message message(std::string role, std::string content) {
    core::llm::Message m;
    m.role = std::move(role);
    m.content = std::move(content);
    return m;
}

core::session::SessionData make_session(std::string id, std::string name = {}) {
    core::session::SessionData data;
    data.session_id = std::move(id);
    data.name = std::move(name);
    data.created_at = "2026-09-10T09:00:00Z";
    data.last_active_at = "2026-09-10T10:00:00Z";
    data.working_dir = "/home/user/payments";
    data.provider = "grok";
    data.model = "grok-4";
    data.stats.turn_count = 2;

    auto first = message("user", "expanded body with [Context for src/a.cpp] noise");
    first.input_text = "Design the retry policy for @src/a.cpp";
    data.messages.push_back(first);

    auto assistant = message("assistant", "I will read the client first.");
    core::llm::ToolCall call;
    call.function.name = "read";
    call.function.arguments = R"({"path":"src/a.cpp"})";
    assistant.tool_calls.push_back(call);
    data.messages.push_back(assistant);

    auto tool = message("tool", "SECRET TOOL OUTPUT THAT MUST NOT LEAK");
    tool.tool_call_id = "1";
    data.messages.push_back(tool);

    auto synthetic = message("user", "internal synthetic context record");
    synthetic.synthetic = true;
    data.messages.push_back(synthetic);

    data.messages.push_back(message("assistant", "Decision: exponential backoff, 5 attempts."));
    return data;
}

} // namespace

TEST_CASE("session references render a text digest without tool output",
          "[session][reference]") {
    const auto data = make_session("a1b2c3d4", "Retry policy");
    const std::string rendered = core::session::render_session_digest(data, "#a1b2c3d4");

    CHECK_THAT(rendered, ContainsSubstring("[Context for #a1b2c3d4]"));
    CHECK_THAT(rendered, ContainsSubstring("\"Retry policy\" (id a1b2c3d4)"));
    CHECK_THAT(rendered, ContainsSubstring("Project: /home/user/payments"));
    // The prompt as typed, not the expanded @file payload.
    CHECK_THAT(rendered, ContainsSubstring("User: Design the retry policy for @src/a.cpp"));
    CHECK_THAT(rendered, !ContainsSubstring("[Context for src/a.cpp]"));
    CHECK_THAT(rendered, ContainsSubstring("Tools: read(src/a.cpp)"));
    CHECK_THAT(rendered, ContainsSubstring("Assistant: Decision: exponential backoff"));
    CHECK_THAT(rendered, !ContainsSubstring("SECRET TOOL OUTPUT"));
    CHECK_THAT(rendered, !ContainsSubstring("internal synthetic"));
    CHECK(rendered.ends_with("[/Context]\n"));
}

TEST_CASE("session references keep the opening request and newest turns under budget",
          "[session][reference]") {
    auto data = make_session("0badcafe");
    for (int i = 0; i < 200; ++i) {
        data.messages.push_back(message("user", "follow-up number " + std::to_string(i)
                                                    + std::string(200, 'x')));
    }
    data.messages.push_back(message("assistant", "LATEST ANSWER"));

    const core::session::SessionDigestOptions options{.max_bytes = 4096};
    const std::string rendered =
        core::session::render_session_digest(data, "#0badcafe", options);

    CHECK(rendered.size() <= options.max_bytes);
    CHECK_THAT(rendered, ContainsSubstring("User: Design the retry policy"));
    CHECK_THAT(rendered, ContainsSubstring("earlier messages omitted"));
    CHECK_THAT(rendered, ContainsSubstring("Assistant: LATEST ANSWER"));
    CHECK_THAT(rendered, !ContainsSubstring("follow-up number 0x"));
}

namespace {

std::optional<core::session::SessionReferenceToken> token_at(std::string_view input) {
    return core::session::parse_session_reference_token(input, input.find('#'));
}

} // namespace

TEST_CASE("# tokens name a session only as an 8-hex id or a quoted name",
          "[session][reference]") {
    auto id = token_at("see #DEADBEEF, then");
    REQUIRE(id.has_value());
    CHECK(id->target == "deadbeef");
    CHECK_FALSE(id->quoted);
    CHECK(id->trailing_suffix == ",");

    auto named = token_at(R"(see #"Retry policy" first)");
    REQUIRE(named.has_value());
    CHECK(named->target == "Retry policy");
    CHECK(named->quoted);

    // Ordinary uses of '#' are never references.
    CHECK_FALSE(token_at("# Heading").has_value());
    CHECK_FALSE(token_at("fixes #123").has_value());
    CHECK_FALSE(token_at("#include <vector>").has_value());
    CHECK_FALSE(token_at("color #fff").has_value());
    CHECK_FALSE(token_at("C#deadbeef").has_value());       // not at a boundary
    CHECK_FALSE(token_at("#deadbeef0").has_value());       // 9 chars
    CHECK_FALSE(token_at("#\"unterminated\nname\"").has_value());
}

TEST_CASE("session references resolve by id or quoted name, never the current one",
          "[session][reference]") {
    TempDir tmp("filo_session_reference");
    const core::session::SessionStore store{tmp.path};
    REQUIRE(store.save(make_session("a1b2c3d4", "Retry policy")));
    REQUIRE(store.save(make_session("deadbeef")));

    const auto by_id = core::session::resolve_session_reference(store, *token_at("#deadbeef"), "");
    REQUIRE(by_id.has_value());
    CHECK(by_id->session_id == "deadbeef");

    const auto by_name =
        core::session::resolve_session_reference(store, *token_at(R"(#"Retry policy")"), "");
    REQUIRE(by_name.has_value());
    CHECK(by_name->session_id == "a1b2c3d4");

    CHECK_FALSE(core::session::resolve_session_reference(store, *token_at("#0000ffff"), "")
                    .has_value());
    CHECK_FALSE(core::session::resolve_session_reference(store, *token_at("#deadbeef"),
                                                         "deadbeef")
                    .has_value());
}

TEST_CASE("expand_prompt inlines #<id> references and leaves other # text alone",
          "[session][reference][context]") {
    TempDir tmp("filo_session_reference_expand");
    const core::session::SessionStore store{tmp.path / "sessions"};
    REQUIRE(store.save(make_session("deadbeef", "Retry policy")));

    const std::string prompt =
        "# Plan\nApply what we decided in #deadbeef, then close #123 and #0000ffff.";
    const auto expanded = core::context::expand_prompt(
        prompt, tmp.path, {.session_store = &store, .current_session_id = "11111111"});
    CHECK(expanded.display_text.starts_with("# Plan\nApply what we decided in \n"));
    CHECK_THAT(expanded.display_text, ContainsSubstring("[Context for #deadbeef]"));
    CHECK_THAT(expanded.display_text, ContainsSubstring("Decision: exponential backoff"));
    CHECK_THAT(expanded.display_text,
               ContainsSubstring("[/Context],\n then close #123 and #0000ffff."));

    // Inside a fence, and without a store, '#' is plain text.
    const std::string fenced = "```\n#deadbeef\n```";
    CHECK(core::context::expand_prompt(fenced, tmp.path, {.session_store = &store}).display_text
          == fenced);
    CHECK(core::context::expand_prompt(prompt, tmp.path).display_text == prompt);
}

TEST_CASE("the # picker opens on a title search and closes after a reference",
          "[session][reference]") {
    using core::session::find_active_session_reference;

    const std::string bare = "look at #";
    auto active = find_active_session_reference(bare, bare.size());
    REQUIRE(active.has_value());
    CHECK(active->query.empty());
    CHECK(active->replace_begin == 8);

    const std::string words = "look at #retry pol";
    active = find_active_session_reference(words, words.size());
    REQUIRE(active.has_value());
    CHECK(active->query == "retry pol");
    CHECK(active->replace_end == words.size());

    CHECK_FALSE(find_active_session_reference("# Heading", 9).has_value());
    CHECK_FALSE(find_active_session_reference("## Heading", 2).has_value());
    CHECK_FALSE(find_active_session_reference("#retry ", 7).has_value());
    CHECK_FALSE(find_active_session_reference("#retry\nnext", 11).has_value());
    CHECK_FALSE(find_active_session_reference("C#", 2).has_value());
    CHECK_FALSE(find_active_session_reference("```\n#retry", 10).has_value());

    const auto completed = core::session::apply_session_reference_completion(
        words, *active, "a1b2c3d4");
    CHECK(completed.text == "look at #a1b2c3d4 ");
    CHECK(completed.cursor == completed.text.size());
    CHECK_FALSE(find_active_session_reference(completed.text, completed.cursor).has_value());

    // Completing mid-text keeps what follows and does not double the space.
    const std::string middle = "use #ret and go";
    const auto mid = find_active_session_reference(middle, 8);
    REQUIRE(mid.has_value());
    const auto mid_done = core::session::apply_session_reference_completion(middle, *mid, "a1b2c3d4");
    CHECK(mid_done.text == "use #a1b2c3d4 and go");
    CHECK(mid_done.cursor == 14);
}

TEST_CASE("the # picker ranks id prefix, title prefix, word starts, then anywhere",
          "[session][reference]") {
    std::vector<core::session::SessionInfo> sessions(6);
    sessions[0] = {.session_id = "11112222", .name = "Fix the payments retry",
                   .last_active_at = "2026-09-12T08:00:00Z", .working_dir = "/w/payments",
                   .provider = "grok", .model = "grok-4", .turn_count = 7};
    sessions[1] = {.session_id = "33334444", .preview = "payments retry design"};
    sessions[2] = {.session_id = "55556666", .preview = "prepayments retrying"};
    sessions[3] = {.session_id = "current1", .name = "payments retry current"};
    sessions[4] = {.session_id = "77778888", .preview = "unrelated"};
    sessions[5] = {.session_id = "paa00000", .preview = "something else"};

    const core::session::SessionReferenceCatalogue catalogue{sessions, "", "current1"};
    const auto rows = catalogue.search("payments retry", 10);
    REQUIRE(rows.size() == 3);
    CHECK(rows[0].session_id == "33334444"); // title prefix
    CHECK(rows[1].session_id == "11112222"); // every term at a word start
    CHECK(rows[2].session_id == "55556666"); // substring
    CHECK(rows[1].title == "Fix the payments retry");
    CHECK(rows[1].project == "payments");
    CHECK(rows[1].model == "grok/grok-4");
    CHECK(rows[1].turn_count == 7);
    CHECK(rows[1].ordinal == 0); // no project, so no positional shortcut

    CHECK(catalogue.search("paa", 10).front().session_id == "paa00000");
    CHECK(catalogue.search("", 3).size() == 3);
    CHECK(catalogue.search("", 0).empty());
    CHECK(catalogue.search("zzz", 10).empty());
    CHECK(catalogue.search("123", 10).empty());
    // Prose after an accepted reference is not a new search.
    CHECK(catalogue.search("11112222 fix", 10).empty());
}

TEST_CASE("session ordinals parse like /resume indexes", "[session][reference]") {
    using core::session::parse_session_ordinal;
    CHECK(parse_session_ordinal("1") == 1);
    CHECK(parse_session_ordinal("42") == 42);
    CHECK(parse_session_ordinal("-1") == -1);
    CHECK(parse_session_ordinal("9999") == 9999);
    CHECK_FALSE(parse_session_ordinal("").has_value());
    CHECK_FALSE(parse_session_ordinal("-").has_value());
    CHECK_FALSE(parse_session_ordinal("0").has_value());
    CHECK_FALSE(parse_session_ordinal("-0").has_value());
    CHECK_FALSE(parse_session_ordinal("01").has_value());
    CHECK_FALSE(parse_session_ordinal("12345").has_value());
    CHECK_FALSE(parse_session_ordinal("1a").has_value());
    CHECK_FALSE(parse_session_ordinal("+1").has_value());
    CHECK_FALSE(parse_session_ordinal("--1").has_value());
    CHECK_FALSE(parse_session_ordinal("1 fix").has_value());

    CHECK(core::session::session_reference_terms("-1").empty());
    CHECK(core::session::session_reference_terms("Retry  Pol")
          == std::vector<std::string>{"retry", "pol"});
}

TEST_CASE("Enter accepts a # row only when that is clearly what was meant",
          "[session][reference]") {
    using core::session::session_reference_accepts_on_enter;
    CHECK(session_reference_accepts_on_enter("retry", "a1b2c3d4"));
    CHECK(session_reference_accepts_on_enter("-1", "a1b2c3d4"));
    // A positive number may be an issue at the end of a sentence: Tab only.
    CHECK_FALSE(session_reference_accepts_on_enter("2", "a1b2c3d4"));
    // The id is already typed out: Enter sends.
    CHECK_FALSE(session_reference_accepts_on_enter("a1b2c3d4", "a1b2c3d4"));
}

TEST_CASE("thread model labels and activity stamps fall back sensibly",
          "[session][reference]") {
    using core::session::thread_model_label;
    CHECK(thread_model_label("grok", "grok-4") == "grok/grok-4");
    CHECK(thread_model_label("", "grok-4") == "grok-4");
    CHECK(thread_model_label("grok", "").empty());

    core::session::SessionInfo info;
    info.created_at = "2026-01-01T00:00:00Z";
    CHECK(core::session::thread_activity_timestamp(info) == info.created_at);
    info.last_active_at = "2026-01-02T00:00:00Z";
    CHECK(core::session::thread_activity_timestamp(info) == info.last_active_at);
}

TEST_CASE("#N picks this project's conversations newest first, #-N from the oldest",
          "[session][reference]") {
    TempDir tmp("filo_session_ordinal");
    const auto project = (tmp.path / "app").string();
    const auto elsewhere = (tmp.path / "other").string();
    fs::create_directories(project);
    fs::create_directories(elsewhere);

    // SessionStore::list() order: most recent first.
    const std::vector<core::session::SessionInfo> sessions{
        {.session_id = "aaaa0001", .working_dir = project, .preview = "current"},
        {.session_id = "aaaa0002", .working_dir = project + "/", .preview = "newest here"},
        {.session_id = "bbbb0001", .working_dir = elsewhere, .preview = "other project"},
        {.session_id = "aaaa0003", .working_dir = project, .preview = "middle here"},
        {.session_id = "cccc0001", .preview = "unknown dir"},
        {.session_id = "aaaa0004", .working_dir = project, .preview = "oldest here"},
    };
    const core::session::SessionReferenceCatalogue catalogue{sessions, project, "aaaa0001"};
    REQUIRE(catalogue.project_size() == 3);

    const auto only = [&](std::string_view query) {
        const auto rows = catalogue.search(query, 10);
        return rows.size() == 1 ? rows.front().session_id : std::string{};
    };
    CHECK(only("1") == "aaaa0002"); // the current session never counts
    CHECK(only("2") == "aaaa0003");
    CHECK(only("3") == "aaaa0004");
    CHECK(only("-1") == "aaaa0004");
    CHECK(only("-3") == "aaaa0002");
    CHECK(catalogue.search("4", 10).empty());
    CHECK(catalogue.search("-4", 10).empty());
    CHECK(catalogue.search("12345", 10).empty());

    const auto all = catalogue.search("", 10);
    REQUIRE(all.size() == 5);
    CHECK(all[0].ordinal == 1);
    CHECK(all[1].ordinal == 0); // other project: reachable by title, not by number
    CHECK(all[2].ordinal == 2);
    CHECK(all[3].ordinal == 0);
    CHECK(all[4].ordinal == 3);

    // Positions are a picker affordance: a sent `#2` stays plain text.
    CHECK_FALSE(core::session::parse_session_reference_token("see #2", 4).has_value());
    CHECK_FALSE(core::session::parse_session_reference_token("see #-1", 4).has_value());
}

TEST_CASE("picker timestamps scale from clock time to a dated label",
          "[session][reference]") {
    const auto now = std::chrono::system_clock::now();
    const auto iso = [](std::chrono::system_clock::time_point when) {
        return core::session::SessionStore::to_iso8601(when);
    };
    using std::chrono::hours;

    const std::string today = core::session::format_session_moment(iso(now), now);
    CHECK(today.size() == 5);
    CHECK(today[2] == ':');
    CHECK(core::session::format_session_moment(iso(now - hours(24 * 400)), now).size() > 6);
    CHECK(core::session::format_session_moment("", now).empty());

    using std::chrono::minutes;
    CHECK(core::session::format_session_age(iso(now - minutes(5)), now) == "5m ago");
    CHECK(core::session::format_session_age(iso(now - hours(24 * 9)), now) == "9d ago");
    CHECK(core::session::format_session_age(iso(now - hours(24 * 800)), now) == "2y ago");
    CHECK(core::session::format_session_age("", now).empty());
}
