#include <catch2/catch_test_macros.hpp>

#include "core/llm/protocols/SseUtils.hpp"

#include <random>
#include <string>
#include <string_view>
#include <vector>

using core::llm::protocols::sse::EnvelopeView;
using core::llm::protocols::sse::ParsedEventView;
using core::llm::protocols::sse::is_done_sentinel;
using core::llm::protocols::sse::parse_event_payload;
using core::llm::protocols::sse::parse_envelope;

TEST_CASE("SSE envelope parser extracts event and data lines", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope(
        "event: response.output_text.delta\n"
        "data: {\"delta\":\"hello\"}\n",
        out,
        scratch);

    REQUIRE(ok);
    REQUIRE(out.event == "response.output_text.delta");
    REQUIRE(out.data == "{\"delta\":\"hello\"}");
}

TEST_CASE("SSE envelope parser accepts no-space field separators", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope(
        "event:response.output_text.delta\n"
        "data:{\"delta\":\"hello\"}\n",
        out,
        scratch);

    REQUIRE(ok);
    REQUIRE(out.event == "response.output_text.delta");
    REQUIRE(out.data == "{\"delta\":\"hello\"}");
}

TEST_CASE("SSE envelope parser trims event and data values", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope(
        "event:   message.delta   \n"
        "data:   payload   \n",
        out,
        scratch);

    REQUIRE(ok);
    REQUIRE(out.event == "message.delta");
    REQUIRE(out.data == "payload");
}

TEST_CASE("SSE envelope parser handles CRLF framing", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope(
        "event: message.delta\r\n"
        "data: payload\r\n",
        out,
        scratch);

    REQUIRE(ok);
    REQUIRE(out.event == "message.delta");
    REQUIRE(out.data == "payload");
}

TEST_CASE("SSE envelope parser joins multiline data payloads with newline", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope(
        "data: {\"chunks\":[\n"
        "data: \"a\",\"b\"]}\n",
        out,
        scratch);

    REQUIRE(ok);
    REQUIRE(out.event.empty());
    REQUIRE(out.data == "{\"chunks\":[\n\"a\",\"b\"]}");
}

TEST_CASE("SSE envelope parser ignores non-event non-data lines", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope(
        "id: 42\n"
        "retry: 1000\n"
        "data: payload\n",
        out,
        scratch);

    REQUIRE(ok);
    REQUIRE(out.event.empty());
    REQUIRE(out.data == "payload");
}

TEST_CASE("SSE envelope parser normalizes empty first data line", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope(
        "data:\n"
        "data: content\n",
        out,
        scratch);

    REQUIRE(ok);
    REQUIRE(out.data == "content");
}

TEST_CASE("SSE envelope parser rejects whitespace-only data payload", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope("data:   \n", out, scratch);

    REQUIRE_FALSE(ok);
}

TEST_CASE("SSE envelope parser accepts unframed JSON payload by default", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope("{\"type\":\"delta\",\"delta\":\"x\"}", out, scratch);

    REQUIRE(ok);
    REQUIRE(out.event.empty());
    REQUIRE(out.data == "{\"type\":\"delta\",\"delta\":\"x\"}");
}

TEST_CASE("SSE envelope parser accepts unframed [DONE] payload by default", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope("[DONE]", out, scratch);

    REQUIRE(ok);
    REQUIRE(out.event.empty());
    REQUIRE(out.data == "[DONE]");
}

TEST_CASE("SSE envelope parser can reject unframed payloads when disabled", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope("{\"type\":\"delta\"}", out, scratch, /*allow_unframed_payload=*/false);

    REQUIRE_FALSE(ok);
}

TEST_CASE("SSE envelope parser rejects event-only payload", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope("event: message.delta\n", out, scratch);

    REQUIRE_FALSE(ok);
}

TEST_CASE("SSE envelope parser clears scratch state between calls", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;

    REQUIRE(parse_envelope("data: first\n"
                           "data: second\n",
                           out,
                           scratch));
    REQUIRE(out.data == "first\nsecond");

    REQUIRE(parse_envelope("data: final\n", out, scratch));
    REQUIRE(out.data == "final");
}

TEST_CASE("SSE envelope parser accepts event/data without trailing newline", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope("event: message.delta\ndata: payload", out, scratch);

    REQUIRE(ok);
    REQUIRE(out.event == "message.delta");
    REQUIRE(out.data == "payload");
}

TEST_CASE("SSE envelope parser handles comment lines and blank lines", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope(
        ": keepalive\n"
        "\n"
        "data: payload\n",
        out,
        scratch);

    REQUIRE(ok);
    REQUIRE(out.event.empty());
    REQUIRE(out.data == "payload");
}

TEST_CASE("SSE envelope parser uses last event field when repeated", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope(
        "event: first\n"
        "event: second\n"
        "data: payload\n",
        out,
        scratch);

    REQUIRE(ok);
    REQUIRE(out.event == "second");
    REQUIRE(out.data == "payload");
}

TEST_CASE("SSE envelope parser preserves empty middle data lines", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope(
        "data: first\n"
        "data:\n"
        "data: third\n",
        out,
        scratch);

    REQUIRE(ok);
    REQUIRE(out.data == "first\n\nthird");
}

TEST_CASE("SSE envelope parser rejects unframed [DONE] when disabled", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;
    const bool ok = parse_envelope("[DONE]", out, scratch, /*allow_unframed_payload=*/false);

    REQUIRE_FALSE(ok);
}

TEST_CASE("SSE done sentinel detection trims surrounding whitespace", "[sse][utils]") {
    REQUIRE(is_done_sentinel("[DONE]"));
    REQUIRE(is_done_sentinel(" \t[DONE]\r\n"));
    REQUIRE_FALSE(is_done_sentinel("[done]"));
    REQUIRE_FALSE(is_done_sentinel("[1,2,3]"));
}

TEST_CASE("SSE parsed event helper classifies done events", "[sse][utils]") {
    ParsedEventView out;
    std::string scratch;
    const bool ok = parse_event_payload("data: [DONE]\n", out, scratch);

    REQUIRE(ok);
    REQUIRE(out.event.empty());
    REQUIRE(out.data == "[DONE]");
    REQUIRE(out.is_done);
}

TEST_CASE("SSE parsed event helper does not mark JSON array as done", "[sse][utils]") {
    ParsedEventView out;
    std::string scratch;
    const bool ok = parse_event_payload("data: [1,2,3]\n", out, scratch);

    REQUIRE(ok);
    REQUIRE(out.data == "[1,2,3]");
    REQUIRE_FALSE(out.is_done);
}

TEST_CASE("SSE parsed event helper thread-local overload resets state between calls", "[sse][utils]") {
    ParsedEventView out;

    REQUIRE(parse_event_payload("data: one\n"
                                "data: two\n",
                                out));
    REQUIRE(out.data == "one\ntwo");
    REQUIRE_FALSE(out.is_done);

    REQUIRE(parse_event_payload("data: final\n", out));
    REQUIRE(out.data == "final");
    REQUIRE_FALSE(out.is_done);
}

TEST_CASE("SSE envelope parser preserves colons in data payload", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;

    REQUIRE(parse_envelope("data: key:value:still-data\n", out, scratch));
    REQUIRE(out.data == "key:value:still-data");
}

TEST_CASE("SSE envelope parser accepts framed done sentinel when unframed is disabled", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;

    REQUIRE(parse_envelope("data: [DONE]\n", out, scratch, /*allow_unframed_payload=*/false));
    REQUIRE(out.data == "[DONE]");
}

TEST_CASE("SSE envelope parser rejects comment-only events", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;

    REQUIRE_FALSE(parse_envelope(": ping\n: pong\n", out, scratch));
}

TEST_CASE("SSE envelope parser trims tab-prefixed data payloads", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;

    REQUIRE(parse_envelope("data:\tpayload\n", out, scratch));
    REQUIRE(out.data == "payload");
}

TEST_CASE("SSE envelope parser treats indented fields as plain text", "[sse][utils]") {
    EnvelopeView out;
    std::string scratch;

    REQUIRE_FALSE(parse_envelope(" data: payload\n", out, scratch, /*allow_unframed_payload=*/false));
}

TEST_CASE("SSE parsed event helper matches explicit and thread-local scratch behavior", "[sse][utils]") {
    const std::vector<std::string_view> corpus = {
        "",
        "data: payload\n",
        "event: message.delta\ndata: payload\n",
        "event: message.delta\r\ndata: payload\r\n",
        "data:\n"
        "data: second\n",
        "data: [DONE]\n",
        "{\"type\":\"delta\"}",
        "   [DONE]   ",
        "id: 42\nretry: 1\n",
        " data: not-a-field\n",
    };

    for (std::string_view raw : corpus) {
        ParsedEventView explicit_out;
        ParsedEventView tls_out;
        std::string scratch;

        const bool explicit_ok = parse_event_payload(raw, explicit_out, scratch);
        const bool tls_ok = parse_event_payload(raw, tls_out);

        REQUIRE(explicit_ok == tls_ok);
        if (!explicit_ok) continue;

        REQUIRE(explicit_out.event == tls_out.event);
        REQUIRE(explicit_out.data == tls_out.data);
        REQUIRE(explicit_out.is_done == tls_out.is_done);
    }
}

TEST_CASE("SSE parser remains stable over high-volume events", "[sse][utils][perf]") {
    ParsedEventView out;
    std::string scratch;

    constexpr std::string_view raw =
        "event: response.output_text.delta\n"
        "data: {\"delta\":\"abcdefghijklmnopqrstuvwxyz\"}\n";
    bool stable = true;

    for (int i = 0; i < 20000; ++i) {
        if (!parse_event_payload(raw, out, scratch)
            || out.is_done
            || out.event != "response.output_text.delta"
            || out.data != "{\"delta\":\"abcdefghijklmnopqrstuvwxyz\"}") {
            stable = false;
            break;
        }
    }

    REQUIRE(stable);
    REQUIRE(parse_event_payload("data: [DONE]\n", out, scratch));
    REQUIRE(out.is_done);
}

TEST_CASE("SSE parser handles many multiline data lines", "[sse][utils][perf]") {
    EnvelopeView out;
    std::string scratch;
    std::string raw;
    std::string expected;

    for (int i = 0; i < 256; ++i) {
        const std::string chunk = "chunk_" + std::to_string(i);
        raw += "data: " + chunk + "\n";
        if (!expected.empty()) expected.push_back('\n');
        expected += chunk;
    }

    REQUIRE(parse_envelope(raw, out, scratch));
    REQUIRE(out.data == expected);
}

TEST_CASE("SSE parser randomized corpus matches explicit and thread-local scratch", "[sse][utils][fuzz]") {
    std::mt19937 rng(1337);
    std::uniform_int_distribution<int> line_count_dist(0, 8);
    std::uniform_int_distribution<int> line_kind_dist(0, 8);
    std::bernoulli_distribution crlf_dist(0.35);
    std::bernoulli_distribution trailing_newline_dist(0.8);
    std::bernoulli_distribution allow_unframed_dist(0.7);

    auto build_random_event = [&]() {
        std::string raw;
        const int lines = line_count_dist(rng);
        for (int i = 0; i < lines; ++i) {
            switch (line_kind_dist(rng)) {
                case 0: raw += "data: payload_" + std::to_string(i); break;
                case 1: raw += "data:" + std::to_string(i); break;
                case 2: raw += "event: message.delta_" + std::to_string(i); break;
                case 3: raw += "id: " + std::to_string(i); break;
                case 4: raw += "retry: " + std::to_string(1000 + i); break;
                case 5: raw += ": keepalive"; break;
                case 6: raw += "garbage_line_" + std::to_string(i); break;
                case 7: raw += "data: [DONE]"; break;
                case 8: raw += "data: {\"i\":" + std::to_string(i) + "}"; break;
                default: break;
            }
            if (crlf_dist(rng)) raw.push_back('\r');
            raw.push_back('\n');
        }
        if (!trailing_newline_dist(rng) && !raw.empty()) raw.pop_back();
        return raw;
    };

    for (int i = 0; i < 2000; ++i) {
        const std::string raw = build_random_event();
        const bool allow_unframed = allow_unframed_dist(rng);

        ParsedEventView explicit_out;
        ParsedEventView tls_out;
        std::string scratch;

        const bool explicit_ok = parse_event_payload(raw, explicit_out, scratch, allow_unframed);
        const bool tls_ok = parse_event_payload(raw, tls_out, allow_unframed);

        REQUIRE(explicit_ok == tls_ok);
        if (!explicit_ok) continue;

        REQUIRE(explicit_out.event == tls_out.event);
        REQUIRE(explicit_out.data == tls_out.data);
        REQUIRE(explicit_out.is_done == tls_out.is_done);
    }
}
