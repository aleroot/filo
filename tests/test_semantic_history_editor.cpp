#include <catch2/catch_test_macros.hpp>

#include "core/agent/SemanticHistoryEditor.hpp"

#include <string>
#include <vector>

namespace {

core::llm::Message read_call(std::string id, std::string arguments) {
    core::llm::ToolCall call;
    call.id = std::move(id);
    call.function.name = "read_file";
    call.function.arguments = std::move(arguments);
    return core::llm::Message{
        .role = "assistant",
        .tool_calls = {std::move(call)},
    };
}

core::llm::Message result(std::string id, std::string content) {
    return core::llm::Message{
        .role = "tool",
        .content = std::move(content),
        .name = "read_file",
        .tool_call_id = std::move(id),
    };
}

std::string large_payload(char fill) {
    return R"({"output":")" + std::string(4096, fill) + R"("})";
}

} // namespace

TEST_CASE("semantic history editor removes only superseded observation payloads",
          "[agent][context-edit]") {
    const std::string old_payload = large_payload('a');
    const std::string latest_payload = large_payload('b');
    const std::vector<core::llm::Message> history{
        {.role = "user", .content = "Inspect this file"},
        read_call("old", R"({"path":"src/a.cpp"})"),
        result("old", old_payload),
        {.role = "assistant", .content = "I will check it again"},
        read_call("new", R"({ "path" : "src/a.cpp" })"),
        result("new", latest_payload),
    };

    const auto edited = core::agent::SemanticHistoryEditor::edit(
        history,
        {.protected_tail_messages = 0, .minimum_result_chars = 128});

    REQUIRE(edited.superseded_results == 1);
    CHECK(edited.messages[2].content.find("superseded_tool_result") != std::string::npos);
    CHECK(edited.messages[5].content == latest_payload);
    CHECK(history[2].content == old_payload); // durable transcript is untouched
}

TEST_CASE("semantic history editor preserves errors recent results and opaque state",
          "[agent][context-edit]") {
    const std::string large_error = R"({"error":")" + std::string(4096, 'e') + R"("})";
    auto opaque_result = result("opaque", large_payload('x'));
    opaque_result.continuation_items.push_back({
        .provider = "test",
        .kind = "signed",
        .payload = R"({"signature":"keep"})",
    });

    const std::vector<core::llm::Message> history{
        read_call("error", R"({"path":"src/error.cpp"})"),
        result("error", large_error),
        read_call("error2", R"({"path":"src/error.cpp"})"),
        result("error2", large_payload('n')),
        read_call("opaque", R"({"path":"src/opaque.cpp"})"),
        opaque_result,
        read_call("opaque2", R"({"path":"src/opaque.cpp"})"),
        result("opaque2", large_payload('z')),
    };

    const auto edited = core::agent::SemanticHistoryEditor::edit(
        history,
        {.protected_tail_messages = 0, .minimum_result_chars = 128});

    CHECK(edited.messages[1].content == large_error);
    CHECK(edited.messages[5].content == opaque_result.content);
    CHECK(edited.messages[5].continuation_items.size() == 1);
}
