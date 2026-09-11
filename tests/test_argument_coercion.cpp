#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/llm/ToolCallAssembly.hpp"
#include "core/tools/ArgumentCoercion.hpp"
#include "core/tools/SearchReplaceTool.hpp"
#include "core/tools/ToolSchema.hpp"

using Catch::Matchers::ContainsSubstring;

namespace {

/// The shape that model output most often gets wrong: a required list of
/// two-field objects next to a plain string.
core::tools::ToolDefinition edit_tool() {
    return {
        .name = "edit",
        .description = "Apply edits",
        .parameters = {
            {
                .name = "file_path",
                .type = "string",
                .description = "File path.",
                .required = true,
            },
            {
                .name = "edits",
                .type = "array",
                .description = "Ordered edits.",
                .required = true,
                .items_schema =
                    R"({"type":"object","properties":{"old_string":{"type":"string"},)"
                    R"("new_string":{"type":"string"}},)"
                    R"("required":["old_string","new_string"],"additionalProperties":false})",
            },
        },
    };
}

core::tools::ToolDefinition read_tool() {
    return {
        .name = "read",
        .description = "Read a file",
        .parameters = {
            {.name = "path", .type = "string", .required = true},
            {.name = "offset_line", .type = "integer"},
            {.name = "exact", .type = "boolean"},
        },
    };
}

} // namespace

TEST_CASE("a list serialized into a string is decoded", "[tools][coercion]") {
    const auto normalized = core::tools::schema::normalize_arguments(
        edit_tool(),
        R"({"file_path":"a.cpp","edits":"[{\"old_string\":\"x\",\"new_string\":\"y\"}]"})");

    REQUIRE(normalized.has_value());
    CHECK(*normalized ==
          R"({"edits":[{"new_string":"y","old_string":"x"}],"file_path":"a.cpp"})");
}

TEST_CASE("a lone item is wrapped into the declared list", "[tools][coercion]") {
    const auto normalized = core::tools::schema::normalize_arguments(
        edit_tool(),
        R"({"file_path":"a.cpp","edits":{"old_string":"x","new_string":"y"}})");

    REQUIRE(normalized.has_value());
    CHECK(*normalized ==
          R"({"edits":[{"new_string":"y","old_string":"x"}],"file_path":"a.cpp"})");
}

TEST_CASE("list-item fields flattened onto the root are adopted",
          "[tools][coercion]") {
    const auto normalized = core::tools::schema::normalize_arguments(
        edit_tool(),
        R"({"file_path":"a.cpp","old_string":"x","new_string":"y"})");

    REQUIRE(normalized.has_value());
    CHECK(*normalized ==
          R"({"edits":[{"new_string":"y","old_string":"x"}],"file_path":"a.cpp"})");
}

TEST_CASE("stringified items inside a real list are decoded",
          "[tools][coercion]") {
    const auto normalized = core::tools::schema::normalize_arguments(
        edit_tool(),
        R"({"file_path":"a.cpp","edits":["{\"old_string\":\"x\",\"new_string\":\"y\"}"]})");

    REQUIRE(normalized.has_value());
    CHECK(*normalized ==
          R"({"edits":[{"new_string":"y","old_string":"x"}],"file_path":"a.cpp"})");
}

TEST_CASE("doubly encoded payloads are decoded", "[tools][coercion]") {
    // A gateway that re-stringifies an already stringified argument.
    const auto normalized = core::tools::schema::normalize_arguments(
        edit_tool(),
        R"({"file_path":"a.cpp","edits":"\"[{\\\"old_string\\\":\\\"x\\\",\\\"new_string\\\":\\\"y\\\"}]\""})");

    REQUIRE(normalized.has_value());
    CHECK(*normalized ==
          R"({"edits":[{"new_string":"y","old_string":"x"}],"file_path":"a.cpp"})");
}

TEST_CASE("quoted scalars are decoded to their declared type",
          "[tools][coercion]") {
    const auto number = core::tools::schema::normalize_arguments(
        read_tool(), R"({"path":"a.cpp","offset_line":"700"})");
    REQUIRE(number.has_value());
    CHECK(*number == R"({"offset_line":700,"path":"a.cpp"})");

    const auto boolean = core::tools::schema::normalize_arguments(
        read_tool(), R"({"path":"a.cpp","exact":"true"})");
    REQUIRE(boolean.has_value());
    CHECK(*boolean == R"({"exact":true,"path":"a.cpp"})");
}

TEST_CASE("malformed serialized payloads are rejected, never smuggled through",
          "[tools][coercion]") {
    // Truncated mid-string: the replacement text is simply not there, so
    // accepting anything here would apply an edit the model never wrote.
    const auto truncated = core::tools::schema::normalize_arguments(
        edit_tool(),
        R"({"file_path":"a.cpp","edits":"[{\"old_string\":\"unterminated"})");
    REQUIRE_FALSE(truncated.has_value());
    CHECK_THAT(truncated.error(), ContainsSubstring("$.edits must be array, got string"));
    CHECK_THAT(truncated.error(), ContainsSubstring("truncated"));

    // Escaping applied at the wrong level: valid-looking, unparsable JSON.
    const auto mis_escaped = core::tools::schema::normalize_arguments(
        edit_tool(),
        R"({"file_path":"a.cpp","edits":"[{\"old_string\": \"a\"}, \"new_string\": \"b\"}]"})");
    REQUIRE_FALSE(mis_escaped.has_value());
    CHECK_THAT(mis_escaped.error(), ContainsSubstring("malformed"));
    CHECK_THAT(mis_escaped.error(),
               ContainsSubstring("Send the array as a real JSON value"));
}

TEST_CASE("a quoted scalar with trailing text is rejected with guidance",
          "[tools][coercion]") {
    // Observed model output: two line numbers crammed into one argument.
    const auto issue = core::tools::schema::validate_arguments(
        read_tool(), R"({"path":"a.cpp","offset_line":"821, 1025"})");

    REQUIRE_FALSE(issue.has_value());
    CHECK(issue.error().parameter == "offset_line");
    CHECK_THAT(issue.error().message,
               ContainsSubstring("$.offset_line must be integer, got string"));
    CHECK_THAT(issue.error().message,
               ContainsSubstring("send a bare JSON integer value"));
}

TEST_CASE("coercion leaves conforming and string-typed arguments untouched",
          "[tools][coercion]") {
    // A string parameter whose value happens to look like JSON must survive
    // verbatim: it already satisfies the contract, so coercion never runs.
    const auto normalized = core::tools::schema::normalize_arguments(
        read_tool(), R"({"path":"[{\"not\":\"json\"}]"})");
    REQUIRE(normalized.has_value());
    CHECK(*normalized == R"({"path":"[{\"not\":\"json\"}]"})");

    const auto untouched = core::tools::schema::normalize_arguments(
        edit_tool(),
        R"({"edits":[{"new_string":"y","old_string":"x"}],"file_path":"a.cpp"})");
    REQUIRE(untouched.has_value());
    CHECK(*untouched ==
          R"({"edits":[{"new_string":"y","old_string":"x"}],"file_path":"a.cpp"})");
}

TEST_CASE("coercion never invents a value the schema cannot accept",
          "[tools][coercion]") {
    // A bare string cannot become an {old_string,new_string} object, and must
    // not be packed into the list either.
    const auto nonsense = core::tools::schema::normalize_arguments(
        edit_tool(), R"({"file_path":"a.cpp","edits":"replace foo with bar"})");
    REQUIRE_FALSE(nonsense.has_value());

    // Right shape, wrong member: the item schema forbids it.
    const auto wrong_member = core::tools::schema::normalize_arguments(
        edit_tool(),
        R"({"file_path":"a.cpp","edits":"[{\"old\":\"x\",\"new\":\"y\"}]"})");
    REQUIRE_FALSE(wrong_member.has_value());

    // Partial flattened item: required item fields are missing.
    const auto partial = core::tools::schema::normalize_arguments(
        edit_tool(), R"({"file_path":"a.cpp","old_string":"x"})");
    REQUIRE_FALSE(partial.has_value());
}

TEST_CASE("the shipped search_replace contract tolerates a stringified list",
          "[tools][coercion]") {
    const core::tools::SearchReplaceTool tool;
    const auto normalized = core::tools::schema::normalize_arguments(
        tool.get_definition(),
        R"({"file_path":"/tmp/a.cpp","edits":"[{\"old_string\":\"int a;\",)"
        R"(\"new_string\":\"int b;\"}]"})");

    REQUIRE(normalized.has_value());
    CHECK_THAT(*normalized, ContainsSubstring(R"("edits":[{"new_string":"int b;")"));
}

TEST_CASE("coercion is inert for the fields a safety gate inspects",
          "[tools][coercion][safety]") {
    // Permission prompts and PreToolUse hooks key off string-typed fields such
    // as a shell command or a target path. There is no rule that *produces* a
    // string, so such a field is either already valid (and untouched) or
    // rejected — coercion can never rewrite what a gate judged.
    const core::tools::ToolDefinition shell{
        .name = "run_terminal_command",
        .description = "Run a command",
        .parameters = {
            {.name = "command", .type = "string", .required = true},
            {.name = "timeout_seconds", .type = "integer"},
        },
    };

    // A command that looks like JSON stays the exact command it was.
    const auto json_shaped = core::tools::schema::normalize_arguments(
        shell, R"({"command":"[{\"rm\":\"-rf /\"}]"})");
    REQUIRE(json_shaped.has_value());
    CHECK(*json_shaped == R"({"command":"[{\"rm\":\"-rf /\"}]"})");

    // A non-string command is rejected outright: nothing is coerced *into* a
    // string, so a gate can never be handed a value it did not see.
    CHECK_FALSE(core::tools::schema::normalize_arguments(
                    shell, R"({"command":["rm","-rf","/"]})").has_value());
    CHECK_FALSE(core::tools::schema::normalize_arguments(
                    shell, R"({"command":{"run":"rm -rf /"}})").has_value());

    // Only the mis-typed neighbour is repaired; the command is byte-identical.
    const auto neighbour = core::tools::schema::normalize_arguments(
        shell, R"({"command":"ls -la","timeout_seconds":"30"})");
    REQUIRE(neighbour.has_value());
    CHECK(*neighbour == R"({"command":"ls -la","timeout_seconds":30})");
}

TEST_CASE("unparsable arguments are diagnosed as a cut-off call",
          "[tools][coercion]") {
    const auto truncated = core::tools::schema::validate_arguments(
        edit_tool(), R"({"file_path":"a.cpp","edits":)");
    REQUIRE_FALSE(truncated.has_value());
    CHECK(truncated.error().code
          == core::tools::schema::ArgumentIssueCode::InvalidJson);
    CHECK_THAT(truncated.error().message,
               ContainsSubstring("arguments are not valid JSON"));
    CHECK_THAT(truncated.error().message, ContainsSubstring("cut off"));

    // Damage that is not truncation keeps the bare verdict rather than
    // guessing at a cause.
    const auto garbage = core::tools::schema::validate_arguments(
        edit_tool(), "not json at all");
    REQUIRE_FALSE(garbage.has_value());
    CHECK_THAT(garbage.error().message,
               ContainsSubstring("arguments are not valid JSON"));
    CHECK_THAT(garbage.error().message, !ContainsSubstring("cut off"));
}

TEST_CASE("an empty-object placeholder never corrupts streamed arguments",
          "[tools][coercion][streaming]") {
    // Observed from an OpenAI-compatible provider: the call is announced with
    // a placeholder `{}` payload before the real object streams in. Plain
    // concatenation produced `{}{"path":"..."}` — unparsable, and the call was
    // rejected outright.
    std::vector<core::llm::ToolCall> accumulated;
    core::llm::ToolCall announce;
    announce.index = 0;
    announce.id = "call_1";
    announce.function.name = "list_directory";
    announce.function.arguments = "{}";
    core::llm::merge_tool_call_fragment(accumulated, announce);

    core::llm::ToolCall first_fragment;
    first_fragment.index = 0;
    first_fragment.function.arguments = R"({"path":"src)";
    core::llm::merge_tool_call_fragment(accumulated, first_fragment);

    core::llm::ToolCall second_fragment;
    second_fragment.index = 0;
    second_fragment.function.arguments = R"(/core"})";
    core::llm::merge_tool_call_fragment(accumulated, second_fragment);

    REQUIRE(accumulated.size() == 1);
    CHECK(accumulated[0].id == "call_1");
    CHECK(accumulated[0].function.name == "list_directory");
    CHECK(accumulated[0].function.arguments == R"({"path":"src/core"})");
}

TEST_CASE("streamed argument fragments are otherwise concatenated verbatim",
          "[tools][coercion][streaming]") {
    std::string accumulated;
    core::llm::append_arguments_fragment(accumulated, R"({"a":"{}")");
    core::llm::append_arguments_fragment(accumulated, R"(,"b":1})");
    CHECK(accumulated == R"({"a":"{}","b":1})");

    // A trailing placeholder after real content is dropped, not appended.
    core::llm::append_arguments_fragment(accumulated, "{}");
    CHECK(accumulated == R"({"a":"{}","b":1})");

    // A genuinely empty argument object still survives on its own.
    std::string empty;
    core::llm::append_arguments_fragment(empty, "{}");
    CHECK(empty == "{}");
}

TEST_CASE("coerce_arguments reports when nothing can be repaired",
          "[tools][coercion]") {
    const std::string schema =
        core::tools::schema::canonical_input_schema(edit_tool());

    CHECK_FALSE(core::tools::schema::coerce_arguments(
        R"({"file_path":"a.cpp","edits":"[{"})", schema).has_value());
    // Already valid: nothing to do, so nothing is returned.
    CHECK_FALSE(core::tools::schema::coerce_arguments(
        R"({"file_path":"a.cpp","edits":[{"old_string":"x","new_string":"y"}]})",
        schema).has_value());
    // Unparsable input is a different failure and is not coercion's business.
    CHECK_FALSE(core::tools::schema::coerce_arguments("{not json", schema)
                    .has_value());
}
