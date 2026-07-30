#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/HistoryCompactionPlanner.hpp"
#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/llm/protocols/DashScopeProtocol.hpp"
#include "core/llm/protocols/GeminiCodeAssistProtocol.hpp"
#include "core/llm/protocols/GeminiProtocol.hpp"
#include "core/llm/protocols/GrokProtocol.hpp"
#include "core/llm/protocols/KimiProtocol.hpp"
#include "core/llm/protocols/MistralProtocol.hpp"
#include "core/llm/protocols/OllamaProtocol.hpp"
#include "core/llm/protocols/OpenAIProtocol.hpp"
#include "core/llm/protocols/OpenAIResponsesProtocol.hpp"

#include <array>
#include <memory>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;

namespace {

core::llm::ToolCall tool_call(
    std::string id,
    std::string name,
    std::string arguments) {
    core::llm::ToolCall call;
    call.id = std::move(id);
    call.function.name = std::move(name);
    call.function.arguments = std::move(arguments);
    return call;
}

} // namespace

TEST_CASE("HistoryCompactionPlanner produces provider-neutral retained context",
          "[agent][compaction][planner]") {
    std::vector<core::llm::Message> history{
        {.role = "system", .content = "runtime system prompt"},
        {
            .role = "user",
            .content = "Inspect the attached architecture.",
            .content_parts = {
                core::llm::ContentPart::make_text("Inspect the attached architecture."),
                core::llm::ContentPart::make_image("/tmp/architecture.png", "image/png"),
            },
        },
        {
            .role = "assistant",
            .content = "",
            .tool_calls = {tool_call("call-1", "read_file", R"({"path":"src/main.cpp"})")},
            .reasoning_content = "provider-private reasoning",
            .continuation_items = {
                {.provider = "openai", .kind = "reasoning", .payload = R"({"id":"r1"})"},
            },
        },
        {
            .role = "tool",
            .content = R"({"content":"int main() {}","offload":{"reference":"s/c.result"}})",
            .name = "read_file",
            .tool_call_id = "call-1",
        },
        {.role = "assistant", .content = "The entry point is small."},
        {.role = "user", .content = "Keep the API compatible."},
    };

    const auto plan = core::agent::HistoryCompactionPlanner::plan(
        history,
        "Earlier checkpoint.");

    REQUIRE(plan.summary_history.size() == 1);
    CHECK(plan.summary_history.front().role == "user");
    CHECK(plan.summary_history.front().synthetic);
    CHECK_THAT(
        plan.summary_history.front().content,
        ContainsSubstring("Earlier checkpoint."));
    CHECK_THAT(
        plan.summary_history.front().content,
        ContainsSubstring("read_file"));

    REQUIRE(plan.retained_history.size() == 2);
    for (const auto& message : plan.retained_history) {
        CHECK(message.role == "user");
        CHECK_FALSE(message.synthetic);
        CHECK(message.tool_calls.empty());
        CHECK(message.continuation_items.empty());
        CHECK(message.content_parts.empty());
        CHECK(message.reasoning_content.empty());
    }
    CHECK_THAT(plan.retained_history.front().content, ContainsSubstring("architecture"));
    CHECK_THAT(plan.retained_history.back().content, ContainsSubstring("API compatible"));
    CHECK_THAT(plan.execution_checkpoint, ContainsSubstring("call-1"));
    CHECK_THAT(plan.execution_checkpoint, ContainsSubstring("s/c.result"));
}

TEST_CASE("HistoryCompactionPlanner keeps bounded oldest and newest user intent",
          "[agent][compaction][planner]") {
    std::vector<core::llm::Message> history{
        {.role = "user", .content = "ORIGINAL-" + std::string(8'000, 'a')},
        {.role = "assistant", .content = "middle"},
        {.role = "user", .content = "MIDDLE-" + std::string(8'000, 'b')},
        {.role = "assistant", .content = "middle response"},
        {.role = "user", .content = std::string(8'000, 'c') + "-LATEST"},
    };

    const auto plan = core::agent::HistoryCompactionPlanner::plan(
        history,
        {},
        {
            .retained_user_tokens = 1'000,
            .retained_head_user_tokens = 100,
            .execution_checkpoint_tokens = 0,
        });

    REQUIRE(plan.retained_history.size() == 2);
    CHECK_THAT(plan.retained_history.front().content, ContainsSubstring("ORIGINAL-"));
    CHECK_THAT(plan.retained_history.back().content, ContainsSubstring("-LATEST"));
    CHECK(plan.retained_history.front().content.find("MIDDLE-") == std::string::npos);
}

TEST_CASE("HistoryCompactionPlanner scales retained state for small context windows",
          "[agent][compaction][planner]") {
    const auto policy =
        core::agent::HistoryCompactionPlanner::policy_for_context_window(8'000);

    CHECK(policy.summary_input_tokens == 5'333);
    CHECK(policy.retained_user_tokens == 800);
    CHECK(policy.retained_head_user_tokens == 80);
    CHECK(policy.execution_checkpoint_tokens == 400);
}

TEST_CASE("HistoryCompactionPlanner bounds summarizer input while preserving endpoints",
          "[agent][compaction][planner]") {
    const auto plan = core::agent::HistoryCompactionPlanner::plan(
        {
            {.role = "user", .content = "ORIGINAL-" + std::string(8'000, 'a')},
            {.role = "assistant", .content = std::string(8'000, 'b')},
            {.role = "user", .content = std::string(8'000, 'c') + "-LATEST"},
        },
        {},
        {
            .summary_input_tokens = 1'000,
            .retained_user_tokens = 100,
            .retained_head_user_tokens = 10,
            .execution_checkpoint_tokens = 0,
        });

    REQUIRE(plan.summary_history.size() == 1);
    const auto& content = plan.summary_history.front().content;
    CHECK_THAT(content, ContainsSubstring("ORIGINAL-"));
    CHECK_THAT(content, ContainsSubstring("-LATEST"));
    CHECK_THAT(content, ContainsSubstring("compaction retention truncated"));
    CHECK(content.size() < 4'500);
}

TEST_CASE("Provider wire families serialize compaction input without protocol state",
          "[agent][compaction][planner][providers]") {
    const auto plan = core::agent::HistoryCompactionPlanner::plan({
        {.role = "user", .content = "Investigate the build."},
        {
            .role = "assistant",
            .tool_calls = {tool_call("call-7", "run_terminal_command", R"({"command":"cmake --build build"})")},
        },
        {
            .role = "tool",
            .content = R"({"exit_code":0})",
            .name = "run_terminal_command",
            .tool_call_id = "call-7",
        },
    });

    core::llm::ChatRequest request;
    request.model = "generic-model";
    request.messages = {
        {.role = "system", .content = "Create a checkpoint."},
    };
    request.messages.insert(
        request.messages.end(),
        plan.summary_history.begin(),
        plan.summary_history.end());
    request.messages.push_back({
        .role = "user",
        .content = "Summarize now.",
    });

    core::llm::ChatRequest replay_request;
    replay_request.model = "generic-model";
    replay_request.messages = {
        {.role = "system", .content = "Continue the task."},
    };
    replay_request.messages.insert(
        replay_request.messages.end(),
        plan.retained_history.begin(),
        plan.retained_history.end());
    replay_request.messages.push_back({
        .role = "user",
        .content = "Continue from the checkpoint.",
    });

    std::array<std::unique_ptr<core::llm::protocols::ApiProtocolBase>, 16> protocols{
        std::make_unique<core::llm::protocols::OpenAIProtocol>(),
        std::make_unique<core::llm::protocols::ZaiProtocol>(),
        std::make_unique<core::llm::protocols::ZaiCodingProtocol>(),
        std::make_unique<core::llm::protocols::GrokProtocol>(),
        std::make_unique<core::llm::protocols::OpenAIResponsesProtocol>(),
        std::make_unique<core::llm::protocols::CodexResponsesProtocol>(),
        std::make_unique<core::llm::protocols::GrokResponsesProtocol>(),
        std::make_unique<core::llm::protocols::DashScopeProtocol>(),
        std::make_unique<core::llm::protocols::DashScopeResponsesProtocol>(),
        std::make_unique<core::llm::protocols::DashScopeTokenPlanProtocol>(),
        std::make_unique<core::llm::protocols::AnthropicProtocol>(),
        std::make_unique<core::llm::protocols::GeminiProtocol>(),
        std::make_unique<core::llm::protocols::GeminiCodeAssistProtocol>(),
        std::make_unique<core::llm::protocols::OllamaProtocol>(),
        std::make_unique<core::llm::protocols::KimiProtocol>(),
        std::make_unique<core::llm::protocols::MistralProtocol>(),
    };

    for (const auto& protocol : protocols) {
        CAPTURE(protocol->name());
        const auto payload = protocol->serialize(request);
        CHECK_FALSE(payload.empty());
        CHECK_THAT(payload, ContainsSubstring("conversation_transcript"));

        const auto replay_payload = protocol->serialize(replay_request);
        CHECK_FALSE(replay_payload.empty());
        CHECK_THAT(replay_payload, ContainsSubstring("Continue from the checkpoint."));
    }
}
