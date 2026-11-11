#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/ClaudeBilling.hpp"
#include "core/llm/continuation/ThreadLedger.hpp"
#include "core/llm/protocols/ClaudeMessageThreads.hpp"
#include "core/llm/protocols/ClaudeProtocol.hpp"

#include <simdjson.h>

#include <chrono>
#include <memory>
#include <string>
#include <vector>

using namespace core::llm;
using namespace core::llm::continuation;
using namespace core::llm::protocols;
using Catch::Matchers::ContainsSubstring;

namespace {

[[nodiscard]] WireTranscript transcript_of(std::initializer_list<WireMessage> messages,
                                           std::uint64_t create_only = 7, std::uint64_t fixed = 9) {
    return {.messages = messages, .create_only_digest = create_only, .fixed_digest = fixed};
}

struct FakeClock {
    Clock::time_point now = Clock::time_point{} + std::chrono::hours(1);
};

[[nodiscard]] ThreadLedger ledger_with(const std::shared_ptr<FakeClock>& clock) {
    return ThreadLedger(ThreadPolicy{.now = [clock] { return clock->now; }});
}

const WireMessage kUser1{.digest = 11};
const WireMessage kReply1{.digest = 12, .reply_digest = 120};
const WireMessage kUser2{.digest = 13};
const WireMessage kReply2{.digest = 14, .reply_digest = 140};
const WireMessage kUser3{.digest = 15};

/// Stores the conversation [user1] answered by reply1 as `msg_1`.
void store_first_turn(const ThreadLedger& ledger, std::string_view scope = "s") {
    auto attempt = ledger.open(scope, transcript_of({kUser1}));
    REQUIRE(std::holds_alternative<Create>(attempt.plan()));
    attempt.observe_response_id("msg_1");
    attempt.commit(kReply1.reply_digest);
}

} // namespace

TEST_CASE("ThreadLedger continues only from a stored, matching prefix", "[threads][ledger]") {
    auto clock = std::make_shared<FakeClock>();
    const auto ledger = ledger_with(clock);

    {
        auto attempt = ledger.open("s", transcript_of({kUser1}));
        const auto* create = std::get_if<Create>(&attempt.plan());
        REQUIRE(create);
        CHECK(create->reason == CreateReason::first_request);
        attempt.observe_response_id("msg_1");
        attempt.commit(kReply1.reply_digest);
    }
    CHECK(ledger.size() == 1);

    auto attempt = ledger.open("s", transcript_of({kUser1, kReply1, kUser2}));
    const auto* next = std::get_if<Continue>(&attempt.plan());
    REQUIRE(next);
    CHECK(next->previous_message_id == "msg_1");
    CHECK(next->first_new_message == 2);

    // The continue is stored too, so the turn after it continues from msg_2.
    attempt.observe_response_id("msg_2");
    attempt.commit(kReply2.reply_digest);
    auto third = ledger.open("s", transcript_of({kUser1, kReply1, kUser2, kReply2, kUser3}));
    const auto* again = std::get_if<Continue>(&third.plan());
    REQUIRE(again);
    CHECK(again->previous_message_id == "msg_2");
    CHECK(again->first_new_message == 4);
}

TEST_CASE("ThreadLedger starts a new thread whenever the request differs", "[threads][ledger]") {
    auto clock = std::make_shared<FakeClock>();
    const auto ledger = ledger_with(clock);
    const auto reason_for = [&](WireTranscript transcript) {
        store_first_turn(ledger);
        auto attempt = ledger.open("s", std::move(transcript));
        const auto* create = std::get_if<Create>(&attempt.plan());
        REQUIRE(create);
        return create->reason;
    };

    CHECK(reason_for(transcript_of({kUser1, kReply1, kUser2}, 7, 10)) == CreateReason::fixed_fields_changed);
    CHECK(reason_for(transcript_of({kUser1, kReply1, kUser2}, 8, 9)) == CreateReason::create_only_changed);
    CHECK(reason_for(transcript_of({kUser2, kReply1, kUser3})) == CreateReason::transcript_drift);
    CHECK(reason_for(transcript_of({kUser1})) == CreateReason::transcript_drift);
    CHECK(reason_for(transcript_of({kUser1, kReply2, kUser2})) == CreateReason::anchor_rewritten);
    CHECK(reason_for(transcript_of({kUser1, kReply1})) == CreateReason::nothing_new);

    store_first_turn(ledger);
    clock->now += std::chrono::seconds(289);
    {
        auto attempt = ledger.open("s", transcript_of({kUser1, kReply1, kUser2}));
        CHECK(std::holds_alternative<Continue>(attempt.plan()));
        attempt.abandon();
    }
    store_first_turn(ledger);
    clock->now += std::chrono::seconds(290);
    auto attempt = ledger.open("s", transcript_of({kUser1, kReply1, kUser2}));
    REQUIRE(std::holds_alternative<Create>(attempt.plan()));
    CHECK(std::get<Create>(attempt.plan()).reason == CreateReason::expired);
}

TEST_CASE("ThreadLedger never lets two requests share a scope", "[threads][ledger]") {
    const ThreadLedger ledger;
    store_first_turn(ledger);
    auto first = ledger.open("s", transcript_of({kUser1, kReply1, kUser2}));
    REQUIRE(std::holds_alternative<Continue>(first.plan()));
    auto second = ledger.open("s", transcript_of({kUser1, kReply1, kUser2}));
    CHECK_FALSE(is_threaded(second.plan()));
    // The concurrent unthreaded request does not disturb the first one.
    second = {};
    first.observe_response_id("msg_2");
    first.commit(kReply2.reply_digest);
    CHECK(ledger.size() == 1);

    CHECK_FALSE(is_threaded(ledger.open("", transcript_of({kUser1})).plan()));
}

TEST_CASE("ThreadLedger forgets a scope whose response was never settled", "[threads][ledger]") {
    const ThreadLedger ledger;
    SECTION("destroyed unsettled") {
        store_first_turn(ledger);
        { auto attempt = ledger.open("s", transcript_of({kUser1, kReply1, kUser2})); }
    }
    SECTION("abandoned") {
        store_first_turn(ledger);
        ledger.open("s", transcript_of({kUser1, kReply1, kUser2})).abandon();
    }
    SECTION("committed without an id") {
        auto attempt = ledger.open("s", transcript_of({kUser1}));
        attempt.commit(kReply1.reply_digest);
    }
    SECTION("committed without a reply digest") {
        auto attempt = ledger.open("s", transcript_of({kUser1}));
        attempt.observe_response_id("msg_1");
        attempt.commit(0);
    }
    SECTION("replanned unthreaded") {
        store_first_turn(ledger);
        auto attempt = ledger.open("s", transcript_of({kUser1, kReply1, kUser2}));
        attempt.replan(Unthreaded{UnthreadedReason::rejected});
        attempt.observe_response_id("msg_2");
        attempt.commit(kReply2.reply_digest);
    }
    CHECK(ledger.size() == 0);
    auto next = ledger.open("s", transcript_of({kUser1, kReply1, kUser2}));
    REQUIRE(std::holds_alternative<Create>(next.plan()));
    CHECK(std::get<Create>(next.plan()).reason == CreateReason::first_request);
}

TEST_CASE("ThreadLedger bounds the scopes it remembers", "[threads][ledger]") {
    const ThreadLedger ledger(ThreadPolicy{.max_scopes = 2});
    store_first_turn(ledger, "a");
    store_first_turn(ledger, "b");
    store_first_turn(ledger, "c");
    CHECK(ledger.size() == 2);
}

TEST_CASE("Attempt cannot thread a request once it gave its scope back", "[threads][ledger]") {
    const ThreadLedger ledger;
    store_first_turn(ledger);
    auto attempt = ledger.open("s", transcript_of({kUser1, kReply1, kUser2}));
    REQUIRE(std::holds_alternative<Continue>(attempt.plan()));
    attempt.abandon();

    attempt.replan(Create{CreateReason::replay_after_rejection});
    CHECK_FALSE(is_threaded(attempt.plan()));
    CHECK(std::get<Unthreaded>(attempt.plan()).reason == UnthreadedReason::dropped);
    // The scope is free for the next request, which starts over.
    auto next = ledger.open("s", transcript_of({kUser1, kReply1, kUser2}));
    CHECK(std::holds_alternative<Create>(next.plan()));
}

TEST_CASE("Attempt replays a rejected continue as a create on the same scope", "[threads][ledger]") {
    const ThreadLedger ledger;
    store_first_turn(ledger);
    auto attempt = ledger.open("s", transcript_of({kUser1, kReply1, kUser2}));
    attempt.replan(Create{CreateReason::replay_after_rejection});
    REQUIRE(std::holds_alternative<Create>(attempt.plan()));
    // Still in flight: a concurrent request of the scope stays unthreaded.
    CHECK(std::get<Unthreaded>(ledger.open("s", transcript_of({kUser1})).plan()).reason
          == UnthreadedReason::scope_busy);
    attempt.observe_response_id("msg_2");
    attempt.commit(kReply2.reply_digest);
    CHECK(ledger.size() == 1);
}

namespace {

const std::string kBilling =
    std::string(claude::kBillingMarker) + " cc_version=2.1.322.abc; cc_entrypoint=cli; cch=00000;";

[[nodiscard]] std::string body(std::string_view messages, std::string_view system_text = "You are Filo",
                               std::string_view extra = "") {
    return std::format(
        R"({{"model":"claude-sonnet-5-5","messages":{},"system":[{{"type":"text","text":"{}"}},)"
        R"({{"type":"text","text":"{}","cache_control":{{"type":"ephemeral"}}}}],)"
        R"("tools":[{{"name":"read","input_schema":{{"type":"object"}}}}],"max_tokens":100{},"stream":true}})",
        messages, kBilling, system_text, extra);
}

const std::string kTwoTurns =
    R"([{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"text","text":"hello"}]},)"
    R"({"role":"user","content":"more"}])";

} // namespace

TEST_CASE("claude_threads digests ignore the per-request billing block", "[threads][claude]") {
    const auto first = claude_threads::wire_transcript(body(kTwoTurns), "");
    auto other_billing = body(kTwoTurns);
    other_billing.replace(other_billing.find("abc"), 3, "xyz");
    const auto second = claude_threads::wire_transcript(other_billing, "");
    REQUIRE(first);
    REQUIRE(second);
    CHECK(first->create_only_digest == second->create_only_digest);
    CHECK(first->fixed_digest == second->fixed_digest);
    REQUIRE(first->messages.size() == 3);
    CHECK(first->messages[0].reply_digest == 0);
    CHECK(first->messages[1].reply_digest
          == claude_threads::reply_digest(R"({"content":[{"type":"text","text":"hello"}]})"));

    const auto changed_system = claude_threads::wire_transcript(body(kTwoTurns, "Other"), "");
    CHECK(changed_system->create_only_digest != first->create_only_digest);
    const auto changed_effort = claude_threads::wire_transcript(
        body(kTwoTurns, "You are Filo", R"(,"output_config":{"effort":"xhigh"})"), "");
    CHECK(changed_effort->fixed_digest != first->fixed_digest);
    CHECK(claude_threads::wire_transcript(body(kTwoTurns), "1m=1")->fixed_digest != first->fixed_digest);

    // A string system mixes attribution into the prompt: never threaded.
    CHECK_FALSE(claude_threads::wire_transcript(R"({"messages":[],"system":"x"})", ""));
    CHECK_FALSE(claude_threads::wire_transcript("not json", ""));
    // Someone else already threads it: never planned, so never cut.
    CHECK_FALSE(claude_threads::wire_transcript(R"({"messages":[],"thread":{"type":"create"}})", ""));
}

TEST_CASE("claude_threads cuts the body for each continue shape", "[threads][claude]") {
    const auto full = body(kTwoTurns);
    const Plan next = Continue{.previous_message_id = "msg_1", .first_new_message = 2};

    const auto billing = claude_threads::threaded_body(full, next, claude_threads::InheritedFields::billing_only);
    REQUIRE(billing);
    CHECK_THAT(*billing, ContainsSubstring(R"("messages":[{"role":"user","content":"more"}])"));
    CHECK_THAT(*billing, ContainsSubstring("cch=00000;"));
    CHECK_THAT(*billing, !ContainsSubstring("You are Filo"));
    CHECK_THAT(*billing, !ContainsSubstring(R"("tools")"));
    CHECK_THAT(*billing, ContainsSubstring(R"("thread":{"type":"continue","previous_message_id":"msg_1"})"));
    CHECK_THAT(*billing, ContainsSubstring(R"("max_tokens":100)"));

    const auto omitted = claude_threads::threaded_body(full, next, claude_threads::InheritedFields::omitted);
    REQUIRE(omitted);
    CHECK_THAT(*omitted, !ContainsSubstring(R"("system")"));
    CHECK_THAT(*omitted, !ContainsSubstring(R"("tools")"));

    const auto everything = claude_threads::threaded_body(full, next, claude_threads::InheritedFields::full);
    REQUIRE(everything);
    CHECK_THAT(*everything, ContainsSubstring("You are Filo"));
    CHECK_THAT(*everything, ContainsSubstring(R"("tools")"));

    const auto create = claude_threads::threaded_body(full, Create{}, claude_threads::InheritedFields::billing_only);
    REQUIRE(create);
    CHECK_THAT(*create, ContainsSubstring(R"("thread":{"type":"create"})"));
    CHECK_THAT(*create, ContainsSubstring("You are Filo"));
    CHECK_THAT(*create, ContainsSubstring(R"({"role":"user","content":"hi"})"));

    for (const auto& threaded : {*billing, *omitted, *everything, *create}) {
        simdjson::dom::parser parser;
        const simdjson::padded_string bytes(threaded);
        CHECK(parser.parse(bytes).error() == simdjson::SUCCESS);
    }

    CHECK_FALSE(claude_threads::threaded_body(full, Unthreaded{}, claude_threads::InheritedFields::billing_only));
    CHECK_FALSE(claude_threads::threaded_body(
        full, Continue{.previous_message_id = "msg_1", .first_new_message = 3},
        claude_threads::InheritedFields::billing_only));
}

TEST_CASE("claude_threads classifies thread rejections", "[threads][claude]") {
    using claude_threads::Rejection;
    const auto error = [](std::string_view code, std::string_view message = "x") {
        return std::format(
            R"({{"type":"error","error":{{"type":"invalid_request_error","message":"{}","details":{{"error_code":"{}"}}}}}})",
            message, code);
    };
    CHECK(claude_threads::classify_rejection(404, error("thread_not_found")).kind == Rejection::not_found);
    CHECK(claude_threads::classify_rejection(400, error("thread_already_continued")).kind
          == Rejection::already_continued);
    CHECK(claude_threads::classify_rejection(400, error("thread_continue_in_flight")).kind
          == Rejection::continue_in_flight);
    CHECK(claude_threads::classify_rejection(400, error("thread_unsupported_request")).kind
          == Rejection::unsupported_request);
    CHECK(claude_threads::classify_rejection(400, R"({"error":{"message":"bad"}})").kind == Rejection::other);
    CHECK(claude_threads::classify_rejection(404, "not json").kind == Rejection::other);
    CHECK(claude_threads::classify_rejection(429, error("thread_not_found")).kind == Rejection::none);
    CHECK(claude_threads::classify_rejection(500, "").kind == Rejection::none);

    const auto fingerprint = claude_threads::classify_rejection(
        400, error("thread_fingerprint_mismatch", "differing field(s): output_config"));
    CHECK(fingerprint.kind == Rejection::fingerprint_mismatch);
    CHECK_FALSE(fingerprint.names_inherited_fields);
    CHECK(claude_threads::classify_rejection(
              400, error("thread_fingerprint_mismatch", "differing field(s): system, tools"))
              .names_inherited_fields);
    CHECK(claude_threads::classify_rejection(400, error("thread_fingerprint_mismatch", "unreadable"))
              .names_inherited_fields);
}

namespace {

[[nodiscard]] ChatRequest agent_request(std::vector<Message> messages) {
    ChatRequest request;
    request.model = "claude-sonnet-5-5";
    request.session_id = "threads-protocol-test";
    request.continuation_scope = "threads-protocol-test";
    request.messages = std::move(messages);
    request.prompt_plan.append({
        .kind = core::context::ContextLayerKind::RuntimeInstructions,
        .stability = core::context::PromptStability::Stable,
        .name = "runtime",
        .content = "You are Filo",
    });
    request.tools.push_back(Tool{.function = {.name = "read", .description = "Read a file"}});
    return request;
}

[[nodiscard]] std::string sse(std::string_view type, std::string_view data) {
    return std::format("event: {}\ndata: {}\n\n", type, data);
}

/// Streams a text reply through @p protocol and returns the assistant message
/// the agent would append to its history.
Message stream_reply(ClaudeProtocol& protocol, std::string_view id, std::string_view text,
                     std::string_view stop_reason = "end_turn") {
    Message reply{.role = "assistant"};
    const std::vector<std::string> events{
        sse("message_start", std::format(
            R"({{"type":"message_start","message":{{"id":"{}","usage":{{"input_tokens":5}}}}}})", id)),
        sse("content_block_start",
            R"({"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}})"),
        sse("content_block_delta", std::format(
            R"({{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":"{}"}}}})", text)),
        sse("content_block_stop", R"({"type":"content_block_stop","index":0})"),
        sse("message_delta", std::format(
            R"({{"type":"message_delta","delta":{{"stop_reason":"{}"}},"usage":{{"output_tokens":2}}}})", stop_reason)),
        sse("message_stop", R"({"type":"message_stop"})"),
    };
    for (const auto& event : events) {
        for (const auto& chunk : protocol.parse_event(event).chunks) {
            reply.content += chunk.content;
            reply.continuation_items.insert(reply.continuation_items.end(),
                                            chunk.continuation_items.begin(), chunk.continuation_items.end());
        }
    }
    return reply;
}

struct ProtocolRun {
    std::unique_ptr<ApiProtocolBase> protocol;
    std::string payload;
    cpr::Header headers;
};

[[nodiscard]] ProtocolRun send(const ClaudeProtocol& base, ChatRequest request) {
    ProtocolRun run{.protocol = base.clone()};
    run.protocol->prepare_request(request);
    run.payload = run.protocol->serialize(request);
    core::auth::AuthInfo auth;
    auth.properties["oauth"] = "1";
    run.headers = run.protocol->build_headers(auth);
    run.protocol->prepare_headers(run.headers, request, "https://api.anthropic.com");
    return run;
}

[[nodiscard]] std::string beta_header(const ProtocolRun& run) {
    const auto it = run.headers.find("anthropic-beta");
    return it == run.headers.end() ? std::string{} : it->second;
}

} // namespace

TEST_CASE("ClaudeProtocol creates a thread, then sends only the new messages", "[threads][claude][protocol]") {
    const ClaudeProtocol base({}, kAnthropicDefaultMaxTokens, PromptCacheTtlPolicy{},
                              std::make_shared<claude_threads::ThreadSession>());
    std::vector<Message> history{{.role = "user", .content = "hi"}};

    auto first = send(base, agent_request(history));
    CHECK_THAT(first.payload, ContainsSubstring(R"("thread":{"type":"create"})"));
    CHECK_THAT(first.payload, !ContainsSubstring("cch=00000;"));
    CHECK_THAT(beta_header(first), ContainsSubstring(std::string(claude_threads::kBeta)));
    history.push_back(stream_reply(static_cast<ClaudeProtocol&>(*first.protocol), "msg_1", "hello"));
    REQUIRE_FALSE(history.back().continuation_items.empty());
    history.push_back({.role = "user", .content = "more"});

    auto second = send(base, agent_request(history));
    CHECK_THAT(second.payload, ContainsSubstring(R"("thread":{"type":"continue","previous_message_id":"msg_1"})"));
    CHECK_THAT(second.payload, ContainsSubstring(R"("messages":[{"role":"user","content":"more"}])"));
    CHECK_THAT(second.payload, !ContainsSubstring("You are Filo"));
    CHECK_THAT(second.payload, !ContainsSubstring(R"("tools")"));
    CHECK_THAT(second.payload, ContainsSubstring(std::string(claude::kBillingMarker)));
    CHECK_THAT(second.payload, !ContainsSubstring("cch=00000;"));
    CHECK_THAT(beta_header(second), ContainsSubstring(std::string(claude_threads::kBeta)));
}

TEST_CASE("ClaudeProtocol sends whole requests when threads cannot apply", "[threads][claude][protocol]") {
    const auto plain_of = [](const ClaudeProtocol& base, ChatRequest request) {
        auto run = send(base, std::move(request));
        CHECK_THAT(run.payload, !ContainsSubstring(R"("thread")"));
        CHECK_THAT(beta_header(run), !ContainsSubstring(std::string(claude_threads::kBeta)));
        return run;
    };
    std::vector<Message> history{{.role = "user", .content = "hi"}};

    SECTION("threads turned off") {
        const ClaudeProtocol base;
        (void)plain_of(base, agent_request(history));
    }
    SECTION("a request outside the agent loop") {
        const ClaudeProtocol base({}, kAnthropicDefaultMaxTokens, PromptCacheTtlPolicy{},
                                  std::make_shared<claude_threads::ThreadSession>());
        auto request = agent_request(history);
        request.continuation_scope.clear();
        (void)plain_of(base, request);
    }
    SECTION("a model the server refused threads for") {
        auto session = std::make_shared<claude_threads::ThreadSession>();
        session->suspend("claude-sonnet-5-5");
        const ClaudeProtocol base({}, kAnthropicDefaultMaxTokens, PromptCacheTtlPolicy{}, session);
        (void)plain_of(base, agent_request(history));
    }
    SECTION("the unthreaded body is the ordinary one") {
        const ClaudeProtocol threaded({}, kAnthropicDefaultMaxTokens, PromptCacheTtlPolicy{},
                                      std::make_shared<claude_threads::ThreadSession>());
        auto request = agent_request(history);
        request.continuation_scope.clear();
        auto run = plain_of(threaded, request);
        auto ordinary = send(ClaudeProtocol{}, request);
        // Signatures and lineage differ per request; the shape does not.
        CHECK(run.payload.size() == ordinary.payload.size());
    }
}

TEST_CASE("ClaudeProtocol abandons a thread whose reply must not be continued", "[threads][claude][protocol]") {
    auto session = std::make_shared<claude_threads::ThreadSession>();
    const ClaudeProtocol base({}, kAnthropicDefaultMaxTokens, PromptCacheTtlPolicy{}, session);
    std::vector<Message> history{{.role = "user", .content = "hi"}};
    auto first = send(base, agent_request(history));
    history.push_back(stream_reply(static_cast<ClaudeProtocol&>(*first.protocol), "msg_1", "no", "refusal"));
    history.push_back({.role = "user", .content = "more"});
    CHECK(session->ledger().size() == 0);
    auto second = send(base, agent_request(history));
    CHECK_THAT(second.payload, ContainsSubstring(R"("thread":{"type":"create"})"));
}

TEST_CASE("ClaudeProtocol walks the fallback ladder down to the plain request", "[threads][claude][protocol]") {
    auto session = std::make_shared<claude_threads::ThreadSession>();
    const ClaudeProtocol base({}, kAnthropicDefaultMaxTokens, PromptCacheTtlPolicy{}, session);
    std::vector<Message> history{{.role = "user", .content = "hi"}};
    auto first = send(base, agent_request(history));
    history.push_back(stream_reply(static_cast<ClaudeProtocol&>(*first.protocol), "msg_1", "hello"));
    history.push_back({.role = "user", .content = "more"});

    auto request = agent_request(history);
    auto run = send(base, request);
    auto& protocol = *run.protocol;
    REQUIRE_THAT(run.payload, ContainsSubstring(R"("type":"continue")"));
    const cpr::Header no_headers;
    const auto reject = [&](int status, std::string body) {
        return protocol.rebuilds_after_rejection(HttpResponse{status, body, no_headers});
    };

    SECTION("an expired thread is replayed as a create") {
        REQUIRE(reject(404, R"({"error":{"type":"not_found_error","details":{"error_code":"thread_not_found"}}})"));
        const auto replay = protocol.serialize(request);
        CHECK_THAT(replay, ContainsSubstring(R"("thread":{"type":"create"})"));
        CHECK_THAT(replay, ContainsSubstring("You are Filo"));
        CHECK_THAT(replay, ContainsSubstring(R"("content":"hi")"));
    }
    SECTION("an unexplained 400 tries the documented shape, then the plain request") {
        REQUIRE(reject(400, R"({"error":{"type":"invalid_request_error","message":"system not allowed"}})"));
        CHECK(session->inherited_fields() == claude_threads::InheritedFields::omitted);
        const auto documented = protocol.serialize(request);
        CHECK_THAT(documented, ContainsSubstring(R"("type":"continue")"));
        CHECK_THAT(documented, !ContainsSubstring(R"("system")"));

        REQUIRE(reject(400, R"({"error":{"type":"invalid_request_error","message":"still no"}})"));
        const auto plain = protocol.serialize(request);
        CHECK_THAT(plain, !ContainsSubstring(R"("thread")"));
        CHECK_THAT(plain, ContainsSubstring("You are Filo"));
        cpr::Header headers;
        protocol.prepare_headers(headers, request, "https://api.anthropic.com");
        CHECK(headers.find("anthropic-beta") == headers.end());
        // Nothing left to undo: the ordinary error handling takes over.
        CHECK_FALSE(reject(400, R"({"error":{"message":"bad"}})"));
        // The plain request passed, so this model stays unthreaded.
        CHECK_FALSE(session->suspended("claude-sonnet-5-5"));
        protocol.on_response(HttpResponse{200, "", no_headers});
        CHECK(session->suspended("claude-sonnet-5-5"));
    }
    SECTION("an unsupported request suspends the model at once") {
        REQUIRE(reject(400, R"({"error":{"details":{"error_code":"thread_unsupported_request"}}})"));
        CHECK(session->suspended("claude-sonnet-5-5"));
        CHECK_THAT(protocol.serialize(request), !ContainsSubstring(R"("thread")"));
    }
    SECTION("a fingerprint naming system sends full fields from then on") {
        REQUIRE(reject(400, R"({"error":{"message":"differing field(s): system","details":{"error_code":"thread_fingerprint_mismatch"}}})"));
        CHECK(session->inherited_fields() == claude_threads::InheritedFields::full);
        CHECK_THAT(protocol.serialize(request), ContainsSubstring(R"("thread":{"type":"create"})"));
    }
    SECTION("ordinary failures are left to the transport") {
        CHECK_FALSE(reject(429, R"({"error":{"type":"rate_limit_error"}})"));
        CHECK_FALSE(reject(529, ""));
        CHECK_FALSE(reject(401, ""));
    }
}
