#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/context/ContextMentions.hpp"
#include "core/session/SessionDigest.hpp"
#include "core/session/SessionReference.hpp"
#include "core/session/SessionReferenceCatalogue.hpp"
#include "core/session/SessionStore.hpp"
#include "core/session/SessionTranscriptReader.hpp"
#include "core/session/ThreadCatalog.hpp"
#include "core/tools/ConversationTool.hpp"
#include "core/tools/BuiltinToolRegistry.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/WebSearchTool.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"
#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/llm/protocols/GeminiProtocol.hpp"
#include "core/llm/protocols/OllamaProtocol.hpp"
#include "core/agent/Agent.hpp"
#include "core/agent/ToolOutputHistory.hpp"
#include "core/context/SteeringGuard.hpp"
#include "core/mcp/McpDispatcher.hpp"
#include "TestSessionContext.hpp"
#include <simdjson.h>
#include <algorithm>
#include <condition_variable>
#include <format>
#include <memory>
#include <mutex>
#include <utility>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
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

struct ScopedToolRegistration {
    explicit ScopedToolRegistration(std::shared_ptr<core::tools::Tool> tool)
        : name(tool->get_definition().name) {
        core::tools::ToolManager::get_instance().register_tool(std::move(tool));
    }

    ~ScopedToolRegistration() {
        (void)core::tools::ToolManager::get_instance().unregister_tool(name);
    }

    std::string name;
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
    // Numbered like the reader, and honest that nothing was left out.
    CHECK_THAT(rendered, ContainsSubstring("[3] Assistant: Decision"));
    CHECK_THAT(rendered, ContainsSubstring("Complete: all 3 messages shown in full"));
    CHECK_THAT(rendered, !ContainsSubstring("Partial"));
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
    // The closing line names the gap and where the reader should resume.
    CHECK_THAT(rendered, ContainsSubstring("Partial: messages 2-"));
    CHECK_THAT(rendered, ContainsSubstring(R"(session_id="0badcafe", from=2.)"));
    CHECK_THAT(rendered, !ContainsSubstring("Complete:"));
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

TEST_CASE("reference completion preserves quoted ranges, punctuation and Unicode",
          "[session][reference][regression]") {
    using namespace core::session;
    const std::string quoted = "use #\"Retry policy\" and go";
    for (const auto cursor : {quoted.find(" policy"), quoted.find(" and")}) {
        const auto active = find_active_session_reference(quoted, cursor);
        REQUIRE(active);
        const auto done = apply_session_reference_completion(quoted, *active, "deadbeef");
        CHECK(done.text == "use #deadbeef and go");
        CHECK(done.cursor == std::string_view("use #deadbeef ").size());
        CHECK_FALSE(find_active_session_reference(done.text, done.cursor));
    }
    const std::string punct = "use (#retry), then go";
    for (const auto cursor : {punct.find(')'), punct.find(" then")}) {
        const auto active = find_active_session_reference(punct, cursor);
        REQUIRE(active);
        const auto done = apply_session_reference_completion(punct, *active, "deadbeef");
        CHECK(done.text == "use (#deadbeef), then go");
        CHECK(done.cursor == std::string_view("use (#deadbeef), ").size());
    }
    const std::string unicode = "Vedi #\"Politica café\"\npoi";
    const auto active = find_active_session_reference(unicode, unicode.find(" café"));
    REQUIRE(active);
    const auto done = apply_session_reference_completion(unicode, *active, "deadbeef");
    CHECK(done.text == "Vedi #deadbeef \npoi");
    CHECK_FALSE(find_active_session_reference("#\"Retry policy\" and", 19));
    const auto escaped = token_at(R"(#"Retry \"policy\" \\ notes")");
    REQUIRE(escaped);
    CHECK(escaped->target == "Retry \"policy\" \\ notes");
    CHECK_FALSE(token_at("#\"unterminated"));
    const std::string nested = R"(use #"Fix #123 parser" next)";
    const auto named = find_active_session_reference(nested, nested.find(" next"));
    REQUIRE(named);
    CHECK(named->replace_begin == 4);
    CHECK(apply_session_reference_completion(nested, *named, "deadbeef").text
          == "use #deadbeef next");
}

TEST_CASE("reference picker and expansion share Markdown code boundaries",
          "[session][reference][context][regression]") {
    TempDir tmp("filo_reference_code");
    const core::session::SessionStore store(tmp.path / "sessions");
    REQUIRE(store.save(make_session("deadbeef", "Retry policy")));
    const std::vector<std::string> literal{
        "```cpp\n#deadbeef\n```", "~~~cpp\n#deadbeef\n~~~",
        "````markdown\n```cpp\n#deadbeef\n```\n````",
        "```cpp\n~~~\n#deadbeef\n```",
        "  ~~~~cpp\r\n#deadbeef\r\n  ~~~~\r\n",
        "```cpp\n``` still code\n#deadbeef\n```",
        "Explain `use #deadbeef here`", "Explain ``use ` #deadbeef here``",
        "Explain `line one\n#deadbeef here`", "~~~cpp\n#deadbeef",
    };
    for (const auto& input : literal) {
        INFO(input);
        CHECK(core::context::expand_prompt(input, tmp.path, {.session_store = &store}).display_text
              == input);
        const auto cursor = input.find("#deadbeef") + std::string_view("#deadbeef").size();
        CHECK_FALSE(core::session::find_active_session_reference(input, cursor));
    }
    for (const auto& input : {
        std::string("Use `literal ``` text` then #deadbeef"),
        std::string("```cpp\ncode\n```\n#deadbeef"),
        std::string("~~~cpp\ncode\n~~~~\n#deadbeef"),
        std::string("Unmatched ` then #deadbeef"),
        std::string("Escaped \\` then #deadbeef"),
        std::string("`paragraph\n\n#deadbeef"),
    }) {
        INFO(input);
        CHECK_THAT(core::context::expand_prompt(input, tmp.path, {.session_store = &store}).display_text,
                   ContainsSubstring("[Context for #deadbeef]"));
        CHECK(core::session::find_active_session_reference(input, input.size()));
    }
}

TEST_CASE("digest enforces every budget and preserves UTF-8 framing",
          "[session][reference][regression]") {
    auto data = make_session("deadbeef", "Retry policy");
    data.working_dir = std::string(9000, 'p');
    data.context_summary = std::string(9000, 's');
    data.messages.front().input_text = std::string(9000, 'x') + "café 😀";
    data.messages.push_back(message("assistant", std::string(9000, 'a') + "LATEST café 😀"));
    for (const auto limit : {std::size_t{0}, 1UL, 16UL, 40UL, 80UL, 128UL, 1024UL, 4096UL, 32768UL}) {
        INFO(limit);
        const auto digest = core::session::render_session_digest(data, "#deadbeef", {.max_bytes = limit});
        CHECK(digest.size() <= limit);
        if (limit == 32768) CHECK_THAT(digest, ContainsSubstring("from=0."));
        if (!digest.empty()) {
            CHECK(digest.starts_with("\n[Context for #deadbeef]\n"));
            CHECK(digest.ends_with("[/Context]\n"));
            CHECK(simdjson::validate_utf8(digest.data(), digest.size()));
        }
    }
    data.working_dir.clear();
    data.context_summary.clear();
    data.messages = {message("user", "Opening request"),
                     message("assistant", std::string(9000, 'a'))};
    const auto digest = core::session::render_session_digest(data, "#deadbeef", {.max_bytes = 1024});
    CHECK_THAT(digest, ContainsSubstring("User: Opening request"));
    CHECK_THAT(digest, ContainsSubstring("Assistant:"));
    data.messages.clear();
    CHECK(core::session::render_session_digest(data, "#deadbeef", {.max_bytes = 128}).size() <= 128);
    data.messages = {message("user", "ééé😀😀😀")};
    for (std::size_t n = 0; n <= 24; ++n) {
        const auto text = core::session::render_session_digest(data, "#deadbeef", {.max_message_bytes = n});
        CHECK(simdjson::validate_utf8(text.data(), text.size()));
    }
}

TEST_CASE("prompt references deduplicate ids and respect aggregate limits",
          "[session][reference][context][regression]") {
    TempDir tmp("filo_reference_limits");
    const core::session::SessionStore store(tmp.path / "sessions");
    REQUIRE(store.save(make_session("deadbeef", "Retry policy")));
    REQUIRE(store.save(make_session("a1b2c3d4", "Payments")));
    const auto expanded = core::context::expand_prompt(
        "#deadbeef #\"Retry policy\" #a1b2c3d4", tmp.path,
        {.session_store = &store, .max_total_session_bytes = 1024});
    CHECK(expanded.display_text.find("[Context for #deadbeef]")
          == expanded.display_text.rfind("[Context for #deadbeef]"));
    CHECK_THAT(expanded.display_text, ContainsSubstring("conversation context included above"));
    std::size_t digest_bytes = 0;
    for (std::size_t at = 0; (at = expanded.display_text.find("\n[Context for #", at)) != std::string::npos;) {
        const auto end = expanded.display_text.find("[/Context]\n", at);
        REQUIRE(end != std::string::npos);
        digest_bytes += end + std::string_view("[/Context]\n").size() - at;
        at = end + std::string_view("[/Context]\n").size();
    }
    CHECK(digest_bytes <= 1024);
    const auto capped = core::context::expand_prompt("#deadbeef #a1b2c3d4", tmp.path,
        {.session_store = &store, .max_session_references = 1});
    CHECK_THAT(capped.display_text, ContainsSubstring("use conversation with action=read"));
    CHECK_THAT(capped.display_text, !ContainsSubstring("[Context for #a1b2c3d4]"));
    const auto tiny = core::context::expand_prompt("#deadbeef", tmp.path,
        {.session_store = &store, .max_total_session_bytes = 1});
    CHECK_THAT(tiny.display_text, ContainsSubstring("Digest budget too small"));
}

namespace {

struct ToolPage {
    std::string content;
    std::string next_cursor;
    std::int64_t first_message = -1;
    std::int64_t last_message = -1;
    std::int64_t total_messages = -1;
    bool complete = false;
};

ToolPage parse_page(const std::string& response) {
    INFO(response);
    simdjson::dom::parser parser;
    simdjson::dom::object page;
    REQUIRE(parser.parse(response).get(page) == simdjson::SUCCESS);
    std::string_view content, next_cursor;
    ToolPage out;
    REQUIRE(page["content"].get(content) == simdjson::SUCCESS);
    REQUIRE(page["next_cursor"].get(next_cursor) == simdjson::SUCCESS);
    REQUIRE(page["first_message"].get(out.first_message) == simdjson::SUCCESS);
    REQUIRE(page["last_message"].get(out.last_message) == simdjson::SUCCESS);
    REQUIRE(page["total_messages"].get(out.total_messages) == simdjson::SUCCESS);
    REQUIRE(page["complete"].get(out.complete) == simdjson::SUCCESS);
    out.content = std::string(content);
    out.next_cursor = std::string(next_cursor);
    return out;
}

/// Follows next_cursor from @p first_args until complete; returns every page's text.
std::string read_all(core::tools::ConversationTool& tool,
                     const core::context::SessionContext& context,
                     std::string_view first_args,
                     std::size_t limit) {
    std::string collected;
    auto page = parse_page(tool.execute(std::string(first_args), context));
    CHECK(page.content.size() <= limit + 4096);  // the first page adds the header
    collected += page.content;
    for (int i = 0; i < 2000 && !page.complete; ++i) {
        page = parse_page(tool.execute(std::format(
            R"({{"action":"read","session_id":"deadbeef","limit":{},"cursor":"{}"}})",
            limit, page.next_cursor), context));
        CHECK(page.content.size() <= limit);
        CHECK(simdjson::validate_utf8(page.content.data(), page.content.size()));
        collected += page.content;
    }
    REQUIRE(page.complete);
    CHECK(page.next_cursor.empty());
    return collected;
}

} // namespace

TEST_CASE("conversation pages complete saved text by message cursor",
          "[session][reference][tools][regression]") {
    TempDir tmp("filo_reference_read");
    const core::session::SessionStore store(tmp.path / "sessions");
    auto data = make_session("deadbeef", "Retry policy");
    std::string long_text;
    for (int i = 0; i < 2000; ++i) long_text += "History café 😀\n";
    data.messages.push_back(message("assistant", long_text));
    REQUIRE(store.save(data));
    core::tools::ConversationTool tool(store);
    const auto context = test_support::make_workspace_session_context(
        core::context::SessionTransport::cli, "11111111");

    const auto collected = read_all(tool, context,
        R"({"action":"read","session_id":"DEADBEEF","limit":127})", 127);
    CHECK(collected == core::session::render_session_transcript(data));
    CHECK(collected.starts_with(core::session::render_transcript_header(data)));
    CHECK_THAT(collected, ContainsSubstring(long_text));
    CHECK_THAT(collected, ContainsSubstring("[4] Assistant: History café"));
    CHECK_THAT(collected, !ContainsSubstring("SECRET TOOL OUTPUT"));
    CHECK_THAT(collected, !ContainsSubstring("internal synthetic"));

    // A live conversation keeps growing: an outstanding cursor stays valid and
    // the continuation picks up the new message.
    const auto first = parse_page(tool.execute(
        R"({"action":"read","session_id":"deadbeef","limit":127})", context));
    REQUIRE_FALSE(first.complete);
    CHECK(first.first_message == 1);
    CHECK(first.total_messages == 4);
    data.messages.push_back(message("assistant", "APPENDED LATER"));
    REQUIRE(store.save(data));
    auto page = first;
    std::string rest;
    while (!page.complete) {
        page = parse_page(tool.execute(std::format(
            R"({{"action":"read","session_id":"deadbeef","limit":32768,"cursor":"{}"}})",
            page.next_cursor), context));
        rest += page.content;
    }
    CHECK(page.total_messages == 5);
    CHECK(page.last_message == 5);
    CHECK(first.content + rest == core::session::render_session_transcript(data));
    CHECK(rest.ends_with("[5] Assistant: APPENDED LATER\n"));

    // Rewriting what was already read (a compaction) invalidates the cursor.
    data.context_summary = "Compacted: exponential backoff chosen.";
    data.messages.erase(data.messages.begin(), data.messages.begin() + 2);
    REQUIRE(store.save(data));
    CHECK_THAT(tool.execute(std::format(
        R"({{"action":"read","session_id":"deadbeef","cursor":"{}"}})", first.next_cursor), context),
        ContainsSubstring("rewritten"));

    for (const auto& args : {"[]", "{}", R"({"session_id":"deadbeef"})",
        R"({"action":"delete","session_id":"deadbeef"})", R"({"action":"read","session_id":"../file"})",
        R"({"action":"read","session_id":"deadbeef","from":"1"})",
        R"({"action":"read","session_id":"deadbeef","from":99})",
        R"({"action":"read","session_id":"deadbeef","cursor":7})",
        R"({"action":"read","session_id":"deadbeef","cursor":"garbage"})",
        R"({"action":"read","session_id":"deadbeef","cursor":"1.0.0.0123456789abcdef"})",
        R"({"action":"read","session_id":"deadbeef","from":1,"cursor":"1.0.0.0123456789abcdef"})",
        R"({"action":"read","session_id":"deadbeef","include_tool_output":"yes"})",
        R"({"action":"read","session_id":"deadbeef","limit":0})",
        R"({"action":"read","session_id":"deadbeef","limit":-1})",
        R"({"action":"read","session_id":"deadbeef","limit":32769})",
        R"({"action":"read","session_id":"0000ffff"})"}) {
        INFO(args);
        CHECK_THAT(tool.execute(args, context), ContainsSubstring("error"));
    }
    const auto self = test_support::make_workspace_session_context(
        core::context::SessionTransport::cli, "deadbeef");
    CHECK_THAT(tool.execute(R"({"action":"read","session_id":"deadbeef"})", self), ContainsSubstring("error"));
    CHECK(core::tools::agent_builtin_tool_options().include_conversation);
    CHECK_FALSE(core::tools::mcp_builtin_tool_options().include_conversation);

    // The advertised page bounds are the reader's.
    simdjson::dom::parser parser;
    const auto definition = tool.get_definition();
    simdjson::dom::element schema;
    REQUIRE(parser.parse(definition.input_schema).get(schema) == simdjson::SUCCESS);
    CHECK(schema["properties"]["limit"]["maximum"].get_int64().value()
          == static_cast<std::int64_t>(core::session::kMaxTranscriptPageBytes));
    CHECK(schema["properties"]["limit"]["minimum"].get_int64().value()
          == static_cast<std::int64_t>(core::session::kMinTranscriptPageBytes));
}

TEST_CASE("conversation reads from a message number, from the end, or the summary",
          "[session][reference][regression]") {
    TempDir tmp("filo_reference_from");
    const core::session::SessionStore store(tmp.path / "sessions");
    auto data = make_session("deadbeef", "Retry policy");
    data.context_summary = "Earlier: chose backoff.";
    data.messages.push_back(message("user", "Fourth message"));
    data.messages.push_back(message("assistant", "Fifth message"));
    REQUIRE(store.save(data));
    const core::session::SessionTranscriptReader reader(store);

    const auto whole = reader.read("deadbeef", "11111111");
    REQUIRE(whole);
    CHECK(whole->complete);
    CHECK(whole->first_message == 0);
    CHECK(whole->last_message == 5);
    CHECK(whole->total_messages == 5);
    CHECK_THAT(whole->content, ContainsSubstring(
        "[0] Summary of compacted earlier history:\nEarlier: chose backoff.\n[1] User:"));

    const auto third = reader.read("deadbeef", "11111111", {.from = 3});
    REQUIRE(third);
    CHECK(third->first_message == 3);
    CHECK_THAT(third->content, ContainsSubstring("[3] Assistant: Decision"));
    CHECK_THAT(third->content, !ContainsSubstring("[2] "));

    const auto newest = reader.read("deadbeef", "11111111", {.from = -1});
    REQUIRE(newest);
    CHECK(newest->first_message == 5);
    CHECK_THAT(newest->content, ContainsSubstring("[5] Assistant: Fifth message"));
    CHECK_THAT(newest->content, !ContainsSubstring("[4] "));

    const auto clamped = reader.read("deadbeef", "11111111", {.from = -99});
    REQUIRE(clamped);
    CHECK(clamped->first_message == 1);

    const auto summary = reader.read("deadbeef", "11111111", {.from = 0, .limit = 16});
    REQUIRE(summary);
    CHECK(summary->first_message == 0);
    CHECK_FALSE(summary->complete);

    const auto past = reader.read("deadbeef", "11111111", {.from = 6});
    REQUIRE_FALSE(past);
    CHECK_THAT(past.error(), ContainsSubstring("past the last message (5)"));

    // The digest cites the same numbers the reader accepts.
    const auto digest = core::session::render_session_digest(data, "#deadbeef");
    CHECK_THAT(digest, ContainsSubstring("[3] Assistant: Decision"));
    CHECK_THAT(digest, ContainsSubstring("[0] Summary of compacted earlier history:"));
}

TEST_CASE("conversation includes tool output only on request and never with steering withheld",
          "[session][reference][tools][steering][regression]") {
    TempDir tmp("filo_reference_tool_output");
    const core::session::SessionStore store(tmp.path / "sessions");
    auto data = make_session("deadbeef", "Retry policy");
    data.messages[1].tool_calls.front().id = "call-1";
    data.messages[2].tool_call_id = "call-1";
    data.messages[2].content = "SECRET TOOL OUTPUT THAT MUST NOT LEAK " + std::string(5000, 'z');
    REQUIRE(store.save(data));
    core::tools::ConversationTool tool(store);
    const auto context = test_support::make_workspace_session_context(
        core::context::SessionTransport::cli, "11111111");

    const auto plain = parse_page(tool.execute(
        R"({"action":"read","session_id":"deadbeef"})", context));
    CHECK_THAT(plain.content, !ContainsSubstring("SECRET TOOL OUTPUT"));
    CHECK_THAT(plain.content, ContainsSubstring("Tools: read(src/a.cpp)"));

    const auto with_output = core::session::render_session_transcript(
        data, {.include_tool_output = true});
    CHECK_THAT(with_output, ContainsSubstring(
        "[2] Assistant: I will read the client first.\nTool read(src/a.cpp) returned:\n"
        "SECRET TOOL OUTPUT THAT MUST NOT LEAK zzz"));
    // Clamped per result, and numbering is unchanged by the option.
    CHECK_THAT(with_output, !ContainsSubstring(std::string(core::session::kMaxTranscriptToolOutputBytes, 'z')));
    CHECK_THAT(with_output, ContainsSubstring(" [...]\n[3] Assistant: Decision"));
    // The cursor carries the option, so later pages need not repeat it.
    CHECK(read_all(tool, context,
        R"({"action":"read","session_id":"deadbeef","limit":256,"include_tool_output":true})", 256)
          == with_output);

    // A saved result may be a withheld steering file: refused, not filtered.
    TempDir project("filo_reference_steering");
    {
        std::ofstream(project.path / "AGENTS.md") << "PROJECT RULES\n";
    }
    auto withheld = core::context::make_session_context(
        core::workspace::WorkspaceSnapshot{
            .primary = project.path, .additional = {}, .enforce = true, .version = 1},
        core::context::SessionTransport::cli, "11111111");
    withheld.steering_policy = core::context::SteeringPolicy{.mode = core::context::SteeringMode::None};
    REQUIRE_FALSE(core::context::SteeringGuard::for_context(withheld).empty());
    const auto refused = tool.execute(
        R"({"action":"read","session_id":"deadbeef","include_tool_output":true})", withheld);
    CHECK_THAT(refused, ContainsSubstring("steering policy"));
    CHECK_THAT(refused, !ContainsSubstring("SECRET TOOL OUTPUT"));
    const auto continued = parse_page(tool.execute(
        R"({"action":"read","session_id":"deadbeef","limit":16,"include_tool_output":true})", context));
    CHECK_THAT(tool.execute(std::format(
        R"({{"action":"read","session_id":"deadbeef","cursor":"{}"}})", continued.next_cursor), withheld),
        ContainsSubstring("steering policy"));
    CHECK_THAT(tool.execute(R"({"action":"read","session_id":"deadbeef"})", withheld),
               ContainsSubstring("Decision: exponential backoff"));
}

TEST_CASE("a full conversation page reaches the model unclipped",
          "[session][reference][agent][tool-history][regression]") {
    TempDir tmp("filo_reference_history");
    const core::session::SessionStore store(tmp.path / "sessions");
    auto data = make_session("deadbeef", "Retry policy");
    // Every byte a quote, backslash or newline: each JSON-escapes to two,
    // the worst case short of control characters.
    std::string escaped_heavy;
    while (escaped_heavy.size() < 3 * core::session::kMaxTranscriptPageBytes) {
        escaped_heavy += "\"\\\"\\\n";
    }
    data.messages.push_back(message("assistant", escaped_heavy));
    REQUIRE(store.save(data));
    core::tools::ConversationTool tool(store);
    const auto context = test_support::make_workspace_session_context(
        core::context::SessionTransport::cli, "11111111");
    const std::string arguments = std::format(
        R"({{"action":"read","session_id":"deadbeef","from":4,"limit":{}}})",
        core::session::kMaxTranscriptPageBytes);
    const auto raw = tool.execute(arguments, context);
    const auto page = parse_page(raw);
    REQUIRE_FALSE(page.complete);
    REQUIRE(raw.size() > 2 * 8 * 1024);  // larger than any generic history limit

    namespace history = core::agent::tool_output_history;
    const std::string session_id = "11111111";
    for (const std::size_t token_limit : std::vector<std::size_t>{0, 512, 1024, 3072, 8192}) {
        const auto limits = history::limits_for_tool("conversation", token_limit);
        // Agent offloads and clamps only results above max_chars.
        CHECK(raw.size() <= limits.max_chars);
        for (const auto* mode : {"off", "light", "full", "ultra"}) {
            INFO(token_limit << " " << mode);
            CHECK(history::clamp_for_history("conversation", raw, limits, mode,
                      history::Context{.tool_arguments = arguments, .session_id = session_id})
                  == raw);
        }
    }
}

TEST_CASE("conversation is hidden and blocked by MCP even in a shared registry",
          "[session][reference][mcp][regression]") {
    TempDir tmp("filo_reference_mcp");
    const core::session::SessionStore store(tmp.path / "sessions");
    REQUIRE(store.save(make_session("deadbeef", "Private conversation")));
    auto& dispatcher = core::mcp::McpDispatcher::get_instance();
    auto& manager = core::tools::ToolManager::get_instance();
    const auto tool = std::make_shared<core::tools::ConversationTool>(store);
    ScopedToolRegistration registration(tool);
    REQUIRE(manager.has_tool("conversation"));
    REQUIRE(std::ranges::any_of(manager.get_all_tools(), [](const auto& t) {
        return t.function.name == "conversation";
    }));
    for (const auto transport : {core::context::SessionTransport::mcp_stdio,
                                 core::context::SessionTransport::mcp_http}) {
        const auto context = test_support::make_workspace_session_context(transport, "mcp-test");
        CHECK(std::ranges::none_of(manager.get_all_tools(transport), [](const auto& t) {
            return t.function.name == "conversation";
        }));
        for (const auto mode : {core::mcp::McpProtocolMode::legacy,
                               core::mcp::McpProtocolMode::stateless}) {
            const auto listing = dispatcher.dispatch(
                R"({"jsonrpc":"2.0","id":100,"method":"tools/list","params":{}})", context, mode);
            CHECK_THAT(listing, !ContainsSubstring(R"("name":"conversation")"));
            for (const auto name : {"conversation", "CONVERSATION", " conversation "}) {
                const auto response = dispatcher.dispatch(std::format(
                    R"({{"jsonrpc":"2.0","id":101,"method":"tools/call","params":{{"name":"{}","arguments":{{"action":"read","session_id":"deadbeef"}}}}}})", name),
                    context, mode);
                CHECK_THAT(response, ContainsSubstring("-32602"));
                CHECK_THAT(response, !ContainsSubstring("Private conversation"));
            }
        }
        CHECK_THAT(manager.execute_tool("conversation",
            R"({"action":"read","session_id":"deadbeef"})", context), ContainsSubstring("error"));
        CHECK_THAT(tool->execute(R"({"action":"read","session_id":"deadbeef"})", context),
                   ContainsSubstring("error"));
    }
}

TEST_CASE("referenced text survives save reload and every provider wire format",
          "[session][reference][provider][regression]") {
    TempDir tmp("filo_reference_protocol");
    const core::session::SessionStore store(tmp.path / "sessions");
    REQUIRE(store.save(make_session("deadbeef", "Retry policy")));
    const auto expanded = core::context::expand_prompt("Follow #deadbeef", tmp.path,
        {.session_store = &store});
    auto destination = make_session("a1b2c3d4", "New work");
    auto user = message("user", expanded.display_text);
    user.input_text = "Follow #deadbeef";
    user.content_parts = expanded.content_parts;
    destination.messages = {user};
    destination.provider = "claude";
    REQUIRE(store.save(destination));
    const auto loaded = store.load_by_id("a1b2c3d4");
    REQUIRE(loaded);
    REQUIRE(loaded->messages.front().content == expanded.display_text);
    core::llm::ChatRequest request;
    request.model = "test-model";
    request.messages = loaded->messages;
    const core::tools::ConversationTool tool(store);
    request.tools.push_back({.type = "function", .function = tool.get_definition()});
    using namespace core::llm::protocols;
    const std::vector<std::string> payloads{
        OpenAIProtocol{}.serialize(request), OpenAIResponsesProtocol{}.serialize(request),
        AnthropicProtocol{}.serialize(request), GeminiProtocol{}.serialize(request),
        OllamaProtocol{}.serialize(request),
    };
    for (const auto& payload : payloads) {
        simdjson::dom::parser parser;
        CHECK(parser.parse(payload).error() == simdjson::SUCCESS);
        CHECK_THAT(payload, ContainsSubstring("Decision: exponential backoff"));
        CHECK_THAT(payload, ContainsSubstring("conversation"));
        CHECK_THAT(payload, !ContainsSubstring("SECRET TOOL OUTPUT"));
    }
}

namespace {

class ReferenceCapturingProvider final : public core::llm::LLMProvider {
public:
    explicit ReferenceCapturingProvider(bool read = false) : read_(read) {}
    void stream_response(const core::llm::ChatRequest& request,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        bool call_reader;
        {
            std::lock_guard lock(mutex_);
            call_reader = read_ && requests_.empty();
            requests_.push_back(request);
        }
        if (call_reader) {
            core::llm::ToolCall call;
            call.id = "read_referenced_conversation";
            call.function.name = "conversation";
            call.function.arguments = R"({"action":"read","session_id":"deadbeef","limit":8192})";
            callback(core::llm::StreamChunk::make_tools({call}));
        } else {
            callback(core::llm::StreamChunk::make_content("Read the referenced conversation."));
        }
        callback(core::llm::StreamChunk::make_final());
    }
    std::vector<core::llm::ChatRequest> requests() const {
        std::lock_guard lock(mutex_);
        return requests_;
    }
private:
    bool read_;
    mutable std::mutex mutex_;
    std::vector<core::llm::ChatRequest> requests_;
};

} // namespace

TEST_CASE("switching providers retains reference context and executes its reader",
          "[session][reference][agent][provider][regression]") {
    TempDir tmp("filo_reference_switch");
    const core::session::SessionStore store(tmp.path / "sessions");
    auto source = make_session("deadbeef", "Retry policy");
    source.messages.push_back(message("assistant", "EARLIER DETAIL AVAILABLE THROUGH READER"));
    REQUIRE(store.save(source));
    const auto expanded = core::context::expand_prompt("Follow #deadbeef", tmp.path,
        {.session_store = &store, .max_session_bytes = 700});
    auto& manager = core::tools::ToolManager::get_instance();
    ScopedToolRegistration registration(std::make_shared<core::tools::ConversationTool>(store));
    const auto first = std::make_shared<ReferenceCapturingProvider>();
    const auto second = std::make_shared<ReferenceCapturingProvider>(true);
    const auto context = test_support::make_workspace_session_context(
        core::context::SessionTransport::cli, "11111111");
    const auto agent = std::make_shared<core::agent::Agent>(first, manager, context, tmp.path / "results");
    const auto send = [&](core::llm::Message user) {
        std::mutex mutex;
        std::condition_variable cv;
        bool done = false;
        agent->send_message(std::move(user), [](const auto&) {}, [](const auto&, const auto&) {}, [&] {
            {
                std::lock_guard lock(mutex);
                done = true;
            }
            cv.notify_one();
        });
        std::unique_lock lock(mutex);
        REQUIRE(cv.wait_for(lock, std::chrono::seconds(10), [&] { return done; }));
        REQUIRE_FALSE(agent->last_turn_failed());
    };
    auto initial = message("user", expanded.display_text);
    initial.input_text = "Follow #deadbeef";
    send(initial);
    agent->set_provider(second);
    agent->set_active_provider_name("claude");
    agent->set_active_model("claude-test");
    send(message("user", "Read the earlier details of that conversation."));
    const auto requests = second->requests();
    REQUIRE(requests.size() == 2);
    CHECK(std::ranges::any_of(requests.front().messages, [&](const auto& m) {
        return m.role == "user" && m.content == expanded.display_text;
    }));
    CHECK(std::ranges::any_of(requests.front().tools, [](const auto& t) {
        return t.function.name == "conversation";
    }));
    CHECK(std::ranges::any_of(requests.back().messages, [](const auto& m) {
        return m.role == "tool" && m.content.contains("EARLIER DETAIL AVAILABLE THROUGH READER");
    }));
    CHECK(std::ranges::none_of(requests.back().messages, [](const auto& m) {
        return m.content.contains("SECRET TOOL OUTPUT");
    }));
}

TEST_CASE("saved transcript reader exposes typed pages independently of tools",
          "[session][reference][regression]") {
    TempDir tmp("filo_transcript_reader");
    const core::session::SessionStore store(tmp.path / "sessions");
    const auto source = make_session("deadbeef", "Retry policy");
    REQUIRE(store.save(source));
    const core::session::SessionTranscriptReader reader(store);
    const auto first = reader.read("DEADBEEF", "11111111", {.limit = 16});
    REQUIRE(first);
    CHECK(first->session_id == "deadbeef");
    CHECK(first->content.starts_with("Conversation \"Retry policy\" (id deadbeef)\n"));
    CHECK(first->first_message == 1);
    CHECK(first->last_message == 1);
    CHECK(first->total_messages == 3);
    CHECK_FALSE(first->complete);
    CHECK_FALSE(first->includes_tool_output);
    const auto rest = reader.read("deadbeef", "11111111", {
        .cursor = first->next_cursor,
        .limit = core::session::kMaxTranscriptPageBytes,
    });
    REQUIRE(rest);
    REQUIRE(rest->complete);
    CHECK(rest->next_cursor.empty());
    CHECK(rest->first_message == 1);  // continues the entry the first page cut
    CHECK(rest->last_message == 3);
    CHECK(first->content + rest->content == core::session::render_session_transcript(source));
    CHECK_FALSE(reader.read("../file", "11111111"));
    CHECK_FALSE(reader.read("deadbeef", "deadbeef"));
    CHECK_FALSE(reader.read("deadbeef", "11111111", {.limit = 0}));
    CHECK_FALSE(reader.read("deadbeef", "11111111",
                            {.limit = core::session::kMaxTranscriptPageBytes + 1}));
    CHECK_FALSE(reader.read("deadbeef", "11111111", {.cursor = "not-a-cursor"}));
}

namespace {

class AgentOnlyExposureProbe final : public core::tools::Tool {
public:
    core::tools::ToolDefinition get_definition() const override {
        return {
            .name = "internal_exposure_probe",
            .description = "Internal test tool with a name unrelated to conversation.",
            .agent_only = true,
        };
    }

    std::string execute(const std::string&, const core::context::SessionContext&) override {
        return R"({"executed":true})";
    }
};

} // namespace

TEST_CASE("tool metadata controls MCP exposure without a name allowlist",
          "[session][reference][tools][mcp][regression]") {
    ScopedToolRegistration registration(std::make_shared<AgentOnlyExposureProbe>());
    auto& manager = core::tools::ToolManager::get_instance();
    const auto native_search = core::tools::WebSearchTool{}.get_definition();
    CHECK(native_search.agent_only);
    CHECK_FALSE(core::tools::tool_allows_transport(native_search, core::context::SessionTransport::mcp_stdio));
    CHECK_FALSE(core::tools::tool_allows_transport(native_search, core::context::SessionTransport::mcp_http));
    const auto cli = test_support::make_workspace_session_context(core::context::SessionTransport::cli);
    CHECK_THAT(manager.execute_tool("internal_exposure_probe", "{}", cli),
               ContainsSubstring(R"("executed":true)"));
    for (const auto transport : {core::context::SessionTransport::mcp_stdio,
                                 core::context::SessionTransport::mcp_http}) {
        CHECK_FALSE(manager.get_tool_definition("internal_exposure_probe", transport));
        CHECK(std::ranges::none_of(manager.get_all_tools(transport), [](const auto& tool) {
            return tool.function.name == "internal_exposure_probe";
        }));
        const auto context = test_support::make_workspace_session_context(transport);
        CHECK_THAT(manager.execute_tool("internal_exposure_probe", "{}", context),
                   ContainsSubstring("error"));
        const auto response = core::mcp::McpDispatcher::get_instance().dispatch(
            R"({"jsonrpc":"2.0","id":102,"method":"tools/call","params":{"name":"internal_exposure_probe","arguments":{}}})",
            context);
        CHECK_THAT(response, ContainsSubstring("-32602"));
    }
}
