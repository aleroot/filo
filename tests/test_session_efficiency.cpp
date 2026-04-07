#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/session/SessionEfficiencyController.hpp"
#include "core/session/SessionHandoff.hpp"

TEST_CASE("SessionEfficiencyController rotates remote sessions after large waste growth",
          "[session][efficiency]") {
    core::session::SessionEfficiencyController controller;

    for (int i = 0; i < 4; ++i) {
        controller.record_turn({
            .prompt_tokens = 4'000,
            .completion_tokens = 600,
            .estimated_history_tokens = static_cast<std::size_t>(18'000 + i * 2'000),
            .max_context_tokens = 200'000,
            .provider_is_local = false,
            .provider_supports_prompt_caching = false,
        });
    }
    for (int i = 0; i < 6; ++i) {
        controller.record_turn({
            .prompt_tokens = 25'000,
            .completion_tokens = 800,
            .estimated_history_tokens = static_cast<std::size_t>(48'000 + i * 4'000),
            .max_context_tokens = 200'000,
            .provider_is_local = false,
            .provider_supports_prompt_caching = false,
        });
    }

    const auto decision = controller.current_decision();
    REQUIRE(decision.action == core::session::SessionEfficiencyDecision::Action::Rotate);
    REQUIRE(decision.turn_count == 10);
    REQUIRE(decision.waste_factor > 5.0);
    REQUIRE_THAT(decision.reason, Catch::Matchers::ContainsSubstring("baseline"));
}

TEST_CASE("SessionEfficiencyController rotates local sessions on high context pressure",
          "[session][efficiency]") {
    core::session::SessionEfficiencyController controller;

    for (int i = 0; i < 8; ++i) {
        controller.record_turn({
            .prompt_tokens = 5'000 + i * 300,
            .completion_tokens = 500,
            .estimated_history_tokens = static_cast<std::size_t>(48'000 + i * 6'000),
            .max_context_tokens = 96'000,
            .provider_is_local = true,
            .provider_supports_prompt_caching = false,
        });
    }

    const auto decision = controller.current_decision();
    REQUIRE(decision.action == core::session::SessionEfficiencyDecision::Action::Rotate);
    REQUIRE(decision.context_utilization > 0.72);
}

TEST_CASE("SessionEfficiencyController is less eager when prompt caching is available",
          "[session][efficiency]") {
    core::session::SessionEfficiencyController controller;

    for (int i = 0; i < 4; ++i) {
        controller.record_turn({
            .prompt_tokens = 4'000,
            .completion_tokens = 400,
            .estimated_history_tokens = static_cast<std::size_t>(16'000 + i * 1'000),
            .max_context_tokens = 200'000,
            .provider_is_local = false,
            .provider_supports_prompt_caching = true,
        });
    }
    for (int i = 0; i < 6; ++i) {
        controller.record_turn({
            .prompt_tokens = 24'000,
            .completion_tokens = 700,
            .estimated_history_tokens = static_cast<std::size_t>(42'000 + i * 2'000),
            .max_context_tokens = 200'000,
            .provider_is_local = false,
            .provider_supports_prompt_caching = true,
        });
    }

    const auto decision = controller.current_decision();
    REQUIRE(decision.action == core::session::SessionEfficiencyDecision::Action::None);
}

TEST_CASE("Session handoff builder prefers existing summaries", "[session][handoff]") {
    std::vector<core::llm::Message> messages{
        {.role = "user", .content = "old task"},
        {.role = "assistant", .content = "done"},
    };

    const std::string summary = core::session::build_handoff_summary(
        messages,
        "Carry forward the refactor summary.",
        "BUILD");

    REQUIRE(summary == "Carry forward the refactor summary.");
}

TEST_CASE("Session handoff builder captures recent files and commands", "[session][handoff]") {
    core::session::SessionData data;
    data.provider = "mock";
    data.model = "mock-model";
    data.mode = "BUILD";
    data.context_summary = "Carry forward the rotation work.";

    core::llm::ToolCall write_call;
    write_call.id = "write-1";
    write_call.function.name = "write_file";
    write_call.function.arguments = R"({"file_path":"/tmp/src/core/agent/Agent.cpp","content":"patched"})";

    core::llm::ToolCall read_call;
    read_call.id = "read-1";
    read_call.function.name = "read_file";
    read_call.function.arguments = R"({"path":"/tmp/src/core/session/SessionStore.cpp"})";

    core::llm::ToolCall shell_call;
    shell_call.id = "shell-1";
    shell_call.function.name = "run_terminal_command";
    shell_call.function.arguments = R"({"command":"ctest --output-on-failure"})";

    data.messages = {
        {.role = "user", .content = "Make session rotation transparent."},
        {.role = "assistant", .tool_calls = {write_call, read_call, shell_call}},
        {.role = "tool",
         .content = R"({"success":true,"file_path":"/tmp/src/core/agent/Agent.cpp"})",
         .name = "write_file",
         .tool_call_id = "write-1"},
        {.role = "tool",
         .content = R"({"content":"session store contents"})",
         .name = "read_file",
         .tool_call_id = "read-1"},
        {.role = "tool",
         .content = R"({"output":"All tests passed\n","exit_code":0})",
         .name = "run_terminal_command",
         .tool_call_id = "shell-1"},
        {.role = "assistant", .content = "Rotation now happens between tool steps."},
    };

    const std::string summary = core::session::build_handoff_summary(data);

    REQUIRE_THAT(summary, Catch::Matchers::ContainsSubstring("Carry forward the rotation work."));
    REQUIRE_THAT(summary, Catch::Matchers::ContainsSubstring("Make session rotation transparent."));
    REQUIRE_THAT(summary, Catch::Matchers::ContainsSubstring("Agent.cpp"));
    REQUIRE_THAT(summary, Catch::Matchers::ContainsSubstring("SessionStore.cpp"));
    REQUIRE_THAT(summary, Catch::Matchers::ContainsSubstring("ctest --output-on-failure"));
    REQUIRE_THAT(summary, Catch::Matchers::ContainsSubstring("All tests passed"));
    REQUIRE_THAT(summary, Catch::Matchers::ContainsSubstring("mock / mock-model"));
}

TEST_CASE("Session handoff builder captures apply_patch file targets", "[session][handoff]") {
    core::session::SessionData data;
    data.provider = "mock";
    data.model = "mock-model";
    data.mode = "BUILD";

    core::llm::ToolCall patch_call;
    patch_call.id = "patch-1";
    patch_call.function.name = "apply_patch";
    patch_call.function.arguments =
        R"({"patch":"--- a/src/core/session/SessionHandoff.cpp\n+++ b/src/core/session/SessionHandoff.cpp\n@@ -1 +1 @@\n-old\n+new\n"})";

    data.messages = {
        {.role = "user", .content = "Keep the handoff aware of patch edits."},
        {.role = "assistant", .tool_calls = {patch_call}},
        {.role = "tool",
         .content = R"({"success":true,"output":"patching file src/core/session/SessionHandoff.cpp\n"})",
         .name = "apply_patch",
         .tool_call_id = "patch-1"},
        {.role = "assistant", .content = "The patch path is now tracked."},
    };

    const std::string summary = core::session::build_handoff_summary(data);

    REQUIRE_THAT(summary, Catch::Matchers::ContainsSubstring("SessionHandoff.cpp"));
}
