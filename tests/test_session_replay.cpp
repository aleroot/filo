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
        .function = {.name = "read", .arguments = R"({"path":"config.toml"})"},
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
    REQUIRE(messages[2].tools[0].name == "read");
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

TEST_CASE("live multi-step tool turn has the same assistant boundaries as replay",
          "[tui][session_replay][live_timeline][regression]") {
    std::vector<tui::UiMessage> live;
    live.push_back(tui::make_user_message("Inspect, then report."));
    live.push_back(tui::make_assistant_message("", "", true));
    tui::LiveAssistantTimeline timeline(live.back().id);

    auto* first = timeline.begin_step(live);
    REQUIRE(first != nullptr);
    first->text = "I will inspect the file.";
    first->assistant_source_text = first->text;
    first->reasoning_text = "Locate the relevant code.";
    first->reasoning_active = true;
    first->activity_recorded = true;
    first->tools.push_back(tui::make_tool_activity(
        "call_read",
        "read",
        R"({"path":"config.toml"})",
        "config.toml"));
    tui::apply_tool_result(first->tools.back(), R"({"output":"loaded"})");

    auto* second = timeline.begin_step(live, "1s");
    REQUIRE(second != nullptr);
    second->text = "I will run the checks.";
    second->assistant_source_text = second->text;
    second->tools.push_back(tui::make_tool_activity(
        "call_test",
        "run_terminal_command",
        R"({"command":"ctest"})",
        "cmd: ctest"));
    tui::apply_tool_result(
        second->tools.back(),
        R"({"output":"All tests passed","exit_code":0})");

    auto* third = timeline.begin_step(live, "1s");
    REQUIRE(third != nullptr);
    third->text = "The configuration is valid.";
    third->assistant_source_text = third->text;
    timeline.finish(live, "1s", false);

    core::session::SessionData data;
    data.session_id = "sess_parity";
    data.messages.push_back({
        .role = "user",
        .content = "Inspect, then report.",
    });
    core::llm::Message persisted_first{
        .role = "assistant",
        .content = "I will inspect the file.",
        .reasoning_content = "Locate the relevant code.",
        .reasoning_elapsed = "1s",
    };
    persisted_first.tool_calls.push_back(core::llm::ToolCall{
        .id = "call_read",
        .function = {
            .name = "read",
            .arguments = R"({"path":"config.toml"})",
        },
    });
    data.messages.push_back(std::move(persisted_first));
    data.messages.push_back({
        .role = "tool",
        .content = R"({"output":"loaded"})",
        .tool_call_id = "call_read",
    });
    core::llm::Message persisted_second{
        .role = "assistant",
        .content = "I will run the checks.",
    };
    persisted_second.tool_calls.push_back(core::llm::ToolCall{
        .id = "call_test",
        .function = {
            .name = "run_terminal_command",
            .arguments = R"({"command":"ctest"})",
        },
    });
    data.messages.push_back(std::move(persisted_second));
    data.messages.push_back({
        .role = "tool",
        .content = R"({"output":"All tests passed","exit_code":0})",
        .tool_call_id = "call_test",
    });
    data.messages.push_back({
        .role = "assistant",
        .content = "The configuration is valid.",
    });

    const auto replayed = tui::build_resumed_ui_messages(data);
    REQUIRE(live.size() == 4);
    REQUIRE(replayed.size() == 5); // Resume banner + the same four transcript messages.

    for (std::size_t live_index = 0; live_index < live.size(); ++live_index) {
        const auto& live_message = live[live_index];
        const auto& replayed_message = replayed[live_index + 1];
        CHECK(live_message.type == replayed_message.type);
        CHECK(live_message.text == replayed_message.text);
        CHECK(live_message.pending == replayed_message.pending);
        CHECK(live_message.finalized == replayed_message.finalized);
        CHECK(live_message.reasoning_text == replayed_message.reasoning_text);
        CHECK(live_message.reasoning_elapsed == replayed_message.reasoning_elapsed);
        CHECK(live_message.activity_recorded == replayed_message.activity_recorded);
        REQUIRE(live_message.tools.size() == replayed_message.tools.size());
        if (!live_message.tools.empty()) {
            CHECK(live_message.tools[0].id == replayed_message.tools[0].id);
            CHECK(live_message.tools[0].name == replayed_message.tools[0].name);
            CHECK(live_message.tools[0].status == replayed_message.tools[0].status);
            CHECK(live_message.tools[0].result.summary
                  == replayed_message.tools[0].result.summary);
        }
    }
}

TEST_CASE("Session replay restores reasoning duration for the disclosure",
          "[tui][session_replay][reasoning]") {
    core::session::SessionData data;
    data.session_id = "sess_reasoning";

    core::llm::Message assistant;
    assistant.role = "assistant";
    assistant.content = "Here is the answer.";
    assistant.reasoning_content = "Private chain of thought.";
    assistant.reasoning_elapsed = "7s";
    data.messages.push_back(assistant);

    const auto messages = tui::build_resumed_ui_messages(data);
    REQUIRE(messages.size() == 2);  // System banner + assistant
    REQUIRE(messages[1].type == tui::MessageType::Assistant);
    // The finished trace must carry both the text and its duration so the
    // disclosure renders "Thought for 7s" instead of a bare "Thought".
    REQUIRE(messages[1].reasoning_text == "Private chain of thought.");
    REQUIRE(messages[1].reasoning_elapsed == "7s");
    REQUIRE(messages[1].activity_recorded);
    REQUIRE(!messages[1].reasoning_active);
    REQUIRE(messages[1].reasoning_kind == tui::UiMessage::ActivityKind::Thinking);
}

TEST_CASE("Session replay does not render synthetic user context as prompts",
          "[tui][session_replay][rewind]") {
    core::session::SessionData data;
    data.session_id = "sess_synthetic";
    data.messages = {
        {.role = "user", .content = "visible prompt"},
        {.role = "assistant", .content = "visible answer"},
        {.role = "user", .content = "automatic continuation", .synthetic = true},
        {.role = "assistant", .content = "continued answer"},
    };

    const auto messages = tui::build_resumed_ui_messages(data);
    REQUIRE(messages.size() == 4);
    CHECK(messages[1].type == tui::MessageType::User);
    CHECK(messages[1].text == "visible prompt");
    CHECK(messages[2].type == tui::MessageType::Assistant);
    CHECK(messages[3].type == tui::MessageType::Assistant);
}

TEST_CASE("Session replay restores the diff of a past edit",
          "[tui][session_replay][diff][regression]") {
    // Regression: replay deliberately skipped diff construction, so resuming a
    // session silently dropped every edit's diff — the one thing a file
    // modification card exists to show. The diff comes from the recorded call
    // arguments, not from disk, so it is reconstructible and stays true to what
    // was applied at the time.
    core::session::SessionData data;
    data.session_id = "sess_edit";

    core::llm::Message assistant;
    assistant.role = "assistant";
    assistant.tool_calls.push_back(core::llm::ToolCall{
        .id = "call_edit",
        .function = {
            .name = "replace",
            .arguments = R"({"file_path":"notes.md","old_string":"alpha\nbeta",)"
                         R"("new_string":"alpha\ngamma\ndelta"})",
        },
    });
    data.messages.push_back(assistant);

    core::llm::Message tool_result;
    tool_result.role = "tool";
    tool_result.tool_call_id = "call_edit";
    tool_result.content = R"({"result":"Done"})";
    data.messages.push_back(tool_result);

    const auto messages = tui::build_resumed_ui_messages(data);
    REQUIRE(messages.size() == 2);
    REQUIRE(messages[1].tools.size() == 1);

    const auto& diff = messages[1].tools[0].diff_preview;
    REQUIRE_FALSE(diff.empty());
    CHECK(diff.title == "notes.md");
    CHECK(diff.deleted_count == 2);
    CHECK(diff.added_count == 3);
    // Short enough to open on its own, exactly as it did in the live session.
    CHECK(tui::tool_disclosure_defaults_expanded(messages[1].tools[0]));
}

TEST_CASE("Session replay leaves non-file tools without a diff",
          "[tui][session_replay][diff]") {
    core::session::SessionData data;
    data.session_id = "sess_plain";

    core::llm::Message assistant;
    assistant.role = "assistant";
    assistant.tool_calls.push_back(core::llm::ToolCall{
        .id = "call_read",
        .function = {.name = "read", .arguments = R"({"path":"notes.md"})"},
    });
    data.messages.push_back(assistant);

    const auto messages = tui::build_resumed_ui_messages(data);
    REQUIRE(messages[1].tools.size() == 1);
    CHECK(messages[1].tools[0].diff_preview.empty());
}
