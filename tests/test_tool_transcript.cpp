#include <catch2/catch_test_macros.hpp>

#include "core/llm/ToolTranscript.hpp"

#include <vector>

TEST_CASE("terminal tool-call normalization only persists replayable calls",
          "[tool-transcript][streaming][replay]") {
    std::vector<core::llm::ToolCall> calls{
        {
            .id = "",
            .type = "function",
            .function = {.name = "read", .arguments = "{}"},
        },
        {
            .id = "call_missing_name",
            .type = "function",
            .function = {.name = "", .arguments = "{}"},
        },
        {
            .id = "call_existing",
            .type = "function",
            .function = {.name = "search", .arguments = "{}"},
        },
    };

    const auto normalization = core::llm::normalize_completed_tool_calls(
        calls,
        [] { return "call_generated"; });

    CHECK(normalization.assigned_ids == 1);
    CHECK(normalization.discarded_calls == 1);
    REQUIRE(calls.size() == 2);
    CHECK(calls[0].id == "call_generated");
    CHECK(calls[0].function.name == "read");
    CHECK(calls[1].id == "call_existing");
}

TEST_CASE("tool transcript repair preserves valid siblings and later history",
          "[tool-transcript][history][replay]") {
    std::vector<core::llm::Message> history{
        {.role = "user", .content = "Inspect the project."},
        {
            .role = "assistant",
            .content = "I will inspect the project.",
            .tool_calls = {
                {
                    .id = "call_valid",
                    .type = "function",
                    .function = {.name = "read", .arguments = "{}"},
                },
                {
                    .id = "call_nameless",
                    .type = "function",
                    .function = {.name = "", .arguments = "{}"},
                },
            },
        },
        {
            .role = "tool",
            .content = R"({"output":"README"})",
            .tool_call_id = "call_valid",
        },
        {
            .role = "tool",
            .content = R"({"error":"tool not found"})",
            .tool_call_id = "call_nameless",
        },
        {.role = "assistant", .content = "The README is present."},
    };

    const auto repair = core::llm::repair_tool_transcript(history);

    CHECK(repair.removed_calls == 1);
    CHECK(repair.removed_results == 1);
    CHECK(repair.removed_empty_messages == 0);
    REQUIRE(history.size() == 4);
    REQUIRE(history[1].tool_calls.size() == 1);
    CHECK(history[1].tool_calls.front().id == "call_valid");
    CHECK(history.back().content == "The README is present.");
    CHECK_FALSE(core::llm::validate_tool_transcript(history).has_value());
}

TEST_CASE("tool transcript validation reports orphaned results",
          "[tool-transcript][history]") {
    const std::vector<core::llm::Message> history{
        {
            .role = "tool",
            .content = "orphaned output",
            .tool_call_id = "call_orphaned",
        },
    };

    const auto issue = core::llm::validate_tool_transcript(history);

    REQUIRE(issue.has_value());
    CHECK(issue->message_index == 0);
}
