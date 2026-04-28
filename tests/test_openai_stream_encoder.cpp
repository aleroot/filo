#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <simdjson.h>

#include "exec/http/SseStream.hpp"
#include "exec/openai/OpenAIChatCompletionStreamEncoder.hpp"

using Catch::Matchers::ContainsSubstring;

namespace {

simdjson::dom::element parse_json(simdjson::dom::parser& parser, const std::string& json) {
    simdjson::dom::element doc;
    REQUIRE(parser.parse(json).get(doc) == simdjson::SUCCESS);
    return doc;
}

exec::gateway::openai::StreamChunkContext context(bool include_usage = false) {
    return exec::gateway::openai::StreamChunkContext{
        .response_id = "chatcmpl-test",
        .created = 12345,
        .model = "filo-test-model",
        .include_usage = include_usage,
    };
}

} // namespace

TEST_CASE("OpenAI stream encoder emits documented assistant role chunk",
          "[openai][stream][encoder]") {
    simdjson::dom::parser parser;
    const auto doc = parse_json(
        parser,
        exec::gateway::openai::role_chunk(context(true)));

    std::string_view object;
    REQUIRE(doc["object"].get(object) == simdjson::SUCCESS);
    CHECK(object == "chat.completion.chunk");

    std::string_view id;
    REQUIRE(doc["id"].get(id) == simdjson::SUCCESS);
    CHECK(id == "chatcmpl-test");

    int64_t created = 0;
    REQUIRE(doc["created"].get(created) == simdjson::SUCCESS);
    CHECK(created == 12345);

    std::string_view model;
    REQUIRE(doc["model"].get(model) == simdjson::SUCCESS);
    CHECK(model == "filo-test-model");

    std::string_view role;
    REQUIRE(doc["choices"].at(0)["delta"]["role"].get(role) == simdjson::SUCCESS);
    CHECK(role == "assistant");

    simdjson::dom::element finish_reason;
    REQUIRE(doc["choices"].at(0)["finish_reason"].get(finish_reason) == simdjson::SUCCESS);
    CHECK(finish_reason.type() == simdjson::dom::element_type::NULL_VALUE);

    simdjson::dom::element usage;
    REQUIRE(doc["usage"].get(usage) == simdjson::SUCCESS);
    CHECK(usage.type() == simdjson::dom::element_type::NULL_VALUE);
}

TEST_CASE("OpenAI stream encoder omits usage null unless include_usage was requested",
          "[openai][stream][encoder]") {
    simdjson::dom::parser parser;
    const auto doc = parse_json(
        parser,
        exec::gateway::openai::content_chunk(context(false), "Hello"));

    std::string_view content;
    REQUIRE(doc["choices"].at(0)["delta"]["content"].get(content) == simdjson::SUCCESS);
    CHECK(content == "Hello");

    simdjson::dom::element usage;
    CHECK(doc["usage"].get(usage) == simdjson::NO_SUCH_FIELD);
}

TEST_CASE("OpenAI stream encoder emits documented tool call delta",
          "[openai][stream][encoder]") {
    core::llm::ToolCall call;
    call.index = 0;
    call.id = "call_123";
    call.type = "function";
    call.function.name = "lookup";
    call.function.arguments = R"({"query":"filo"})";

    simdjson::dom::parser parser;
    const auto doc = parse_json(
        parser,
        exec::gateway::openai::tool_calls_chunk(context(), {call}));

    int64_t index = -1;
    REQUIRE(doc["choices"].at(0)["delta"]["tool_calls"].at(0)["index"].get(index)
            == simdjson::SUCCESS);
    CHECK(index == 0);

    std::string_view id;
    REQUIRE(doc["choices"].at(0)["delta"]["tool_calls"].at(0)["id"].get(id)
            == simdjson::SUCCESS);
    CHECK(id == "call_123");

    std::string_view type;
    REQUIRE(doc["choices"].at(0)["delta"]["tool_calls"].at(0)["type"].get(type)
            == simdjson::SUCCESS);
    CHECK(type == "function");

    std::string_view name;
    REQUIRE(doc["choices"].at(0)["delta"]["tool_calls"].at(0)["function"]["name"].get(name)
            == simdjson::SUCCESS);
    CHECK(name == "lookup");

    std::string_view arguments;
    REQUIRE(doc["choices"].at(0)["delta"]["tool_calls"].at(0)["function"]["arguments"].get(arguments)
            == simdjson::SUCCESS);
    CHECK(arguments == R"({"query":"filo"})");
}

TEST_CASE("OpenAI stream encoder emits finish and final usage chunks",
          "[openai][stream][encoder]") {
    simdjson::dom::parser parser;
    const auto finish_doc = parse_json(
        parser,
        exec::gateway::openai::finish_chunk(context(true), "tool_calls"));

    std::string_view finish_reason;
    REQUIRE(finish_doc["choices"].at(0)["finish_reason"].get(finish_reason)
            == simdjson::SUCCESS);
    CHECK(finish_reason == "tool_calls");

    simdjson::dom::parser usage_parser;
    const auto usage_doc = parse_json(
        usage_parser,
        exec::gateway::openai::usage_chunk(
            context(true),
            core::llm::TokenUsage{
                .prompt_tokens = 4,
                .completion_tokens = 7,
                .total_tokens = 11,
            }));

    simdjson::dom::array choices;
    REQUIRE(usage_doc["choices"].get(choices) == simdjson::SUCCESS);
    CHECK(choices.size() == 0);

    int64_t prompt_tokens = 0;
    REQUIRE(usage_doc["usage"]["prompt_tokens"].get(prompt_tokens) == simdjson::SUCCESS);
    CHECK(prompt_tokens == 4);

    int64_t completion_tokens = 0;
    REQUIRE(usage_doc["usage"]["completion_tokens"].get(completion_tokens) == simdjson::SUCCESS);
    CHECK(completion_tokens == 7);

    int64_t total_tokens = 0;
    REQUIRE(usage_doc["usage"]["total_tokens"].get(total_tokens) == simdjson::SUCCESS);
    CHECK(total_tokens == 11);
}

TEST_CASE("SSE helper emits data frames terminated by a blank line",
          "[openai][stream][sse]") {
    CHECK(exec::http::sse_data_frame(R"({"x":1})") == "data: {\"x\":1}\n\n");
    CHECK(exec::http::sse_data_frame("[DONE]") == "data: [DONE]\n\n");
}
