#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "tui/Conversation.hpp"
#include "tui/SessionReplay.hpp"

TEST_CASE("Session replay rebuilds UI transcript and tool status", "[tui][session_replay]") {
    core::session::SessionData data;
    data.session_id = "sess_123";
    data.created_at = "2026-04-05T11:22:33Z";
    data.last_active_at = "2026-04-05T12:00:00Z";
    data.provider = "openai";
    data.model = "gpt-5";
    data.stats.turn_count = 3;

    core::llm::Message user;
    user.role = "user";
    user.content = "check the config";
    data.messages.push_back(user);

    core::llm::Message assistant;
    assistant.role = "assistant";
    assistant.content = "I will inspect the file.";
    assistant.tool_calls.push_back(core::llm::ToolCall{
        .index = 0,
        .id = "call_1",
        .type = "function",
        .function = {.name = "read_file", .arguments = R"({"path":"config.toml"})"},
    });
    data.messages.push_back(assistant);

    core::llm::Message tool_result;
    tool_result.role = "tool";
    tool_result.tool_call_id = "call_1";
    tool_result.content = R"({"output":"loaded"})";
    data.messages.push_back(tool_result);

    const auto messages = tui::build_resumed_ui_messages(
        data,
        tui::SessionReplayOptions{.include_continue_hint = true});

    REQUIRE(messages.size() == 3);
    REQUIRE(messages[0].type == tui::MessageType::System);
    REQUIRE_THAT(messages[0].text, Catch::Matchers::ContainsSubstring("Resumed session sess_123"));
    REQUIRE_THAT(messages[0].text, Catch::Matchers::ContainsSubstring("Type a message to continue"));

    REQUIRE(messages[1].type == tui::MessageType::User);
    REQUIRE(messages[1].text == "check the config");

    REQUIRE(messages[2].type == tui::MessageType::Assistant);
    REQUIRE(messages[2].tools.size() == 1);
    REQUIRE(messages[2].tools[0].name == "read_file");
    REQUIRE(messages[2].tools[0].status == tui::ToolActivity::Status::Succeeded);
    REQUIRE(messages[2].tools[0].result.summary == "loaded");
}

TEST_CASE("Session replay handles failed tool calls and multiple assistant messages", "[tui][session_replay]") {
    core::session::SessionData data;
    data.session_id = "sess_456";
    
    // Turn 1: Assistant calls tool, it fails.
    core::llm::Message assistant1;
    assistant1.role = "assistant";
    assistant1.tool_calls.push_back(core::llm::ToolCall{
        .id = "call_fail",
        .function = {.name = "bad_tool", .arguments = "{}"},
    });
    data.messages.push_back(assistant1);

    core::llm::Message tool_fail;
    tool_fail.role = "tool";
    tool_fail.tool_call_id = "call_fail";
    tool_fail.content = R"({"error":"failed"})";
    data.messages.push_back(tool_fail);

    // Turn 2: Assistant responds with final text.
    core::llm::Message assistant2;
    assistant2.role = "assistant";
    assistant2.content = "I failed to run that.";
    data.messages.push_back(assistant2);

    const auto messages = tui::build_resumed_ui_messages(data);

    REQUIRE(messages.size() == 3); // System banner + 2 assistant messages
    REQUIRE(messages[1].type == tui::MessageType::Assistant);
    REQUIRE(messages[1].tools.size() == 1);
    REQUIRE(messages[1].tools[0].status == tui::ToolActivity::Status::Failed);
    REQUIRE(messages[1].tools[0].result.summary == "failed");

    REQUIRE(messages[2].type == tui::MessageType::Assistant);
    REQUIRE(messages[2].text == "I failed to run that.");
}
