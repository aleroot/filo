#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/Agent.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/tools/Tool.hpp"
#include "core/tools/ToolManager.hpp"
#include "TestSessionContext.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

[[nodiscard]] std::vector<core::llm::Message> visible_history(
    const std::shared_ptr<core::agent::Agent>& agent) {
    auto history = agent->get_history();
    std::erase_if(history, [](const core::llm::Message& message) {
        return message.synthetic;
    });
    return history;
}

class ToolCallThenTextProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        const int call = ++calls_;
        if (call == 1) {
            core::llm::ToolCall first;
            first.index = 0;
            first.id = "tc-1";
            first.type = "function";
            first.function.name = "write_file";
            first.function.arguments = R"({"file_path":"tmp.txt","content":"x"})";

            core::llm::ToolCall second;
            second.index = 1;
            second.id = "tc-2";
            second.type = "function";
            second.function.name = "run_terminal_command";
            second.function.arguments = R"({"command":"echo should_not_run"})";

            core::llm::StreamChunk chunk;
            chunk.tools = {first, second};
            chunk.is_final = true;
            callback(chunk);
            return;
        }

        core::llm::StreamChunk final_chunk;
        final_chunk.content = "follow-up step";
        final_chunk.is_final = true;
        callback(final_chunk);
    }

    [[nodiscard]] int call_count() const noexcept {
        return calls_.load(std::memory_order_acquire);
    }

private:
    std::atomic<int> calls_{0};
};

class BlockedToolProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        const int call = ++calls_;
        if (call > 1) {
            core::llm::StreamChunk final_chunk;
            final_chunk.content = "blocked tool handled";
            final_chunk.is_final = true;
            callback(final_chunk);
            return;
        }

        core::llm::ToolCall tool_call;
        tool_call.index = 0;
        tool_call.id = "tc-blocked";
        tool_call.type = "function";
        tool_call.function.name = "run_terminal_command";
        tool_call.function.arguments = R"({"command":"echo should_not_run"})";

        core::llm::StreamChunk chunk;
        chunk.tools = {tool_call};
        chunk.is_final = true;
        callback(chunk);
    }

private:
    std::atomic<int> calls_{0};
};

class SingleToolCallProvider final : public core::llm::LLMProvider {
public:
    SingleToolCallProvider(std::string tool_name, std::string arguments)
        : tool_name_(std::move(tool_name))
        , arguments_(std::move(arguments)) {}

    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        core::llm::ToolCall tool_call;
        tool_call.index = 0;
        tool_call.id = "tc-single";
        tool_call.type = "function";
        tool_call.function.name = tool_name_;
        tool_call.function.arguments = arguments_;

        core::llm::StreamChunk chunk;
        chunk.tools = {tool_call};
        chunk.is_final = true;
        callback(chunk);
    }

private:
    std::string tool_name_;
    std::string arguments_;
};

class AnnotatedNoopTool final : public core::tools::Tool {
public:
    explicit AnnotatedNoopTool(std::string name) : name_(std::move(name)) {}

    core::tools::ToolDefinition get_definition() const override {
        return {
            .name = name_,
            .description = "No-op destructive test tool.",
            .parameters = {
                {"value", "string", "A test value.", false},
            },
            .annotations = {
                .destructive_hint = true,
                .open_world_hint = true,
            },
        };
    }

    std::string execute(
        const std::string&,
        const core::context::SessionContext&) override {
        return R"({"success":true})";
    }

private:
    std::string name_;
};

class ThrowingProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)>) override {
        throw std::runtime_error(
            "Claude token refresh failed (400): {\"error\":\"invalid_scope\"}");
    }
};

} // namespace

TEST_CASE("Agent stops current loop after user denies a tool call", "[agent][permission]") {
    auto provider = std::make_shared<ToolCallThenTextProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    std::atomic<int> permission_checks{0};
    agent->set_permission_fn([&](std::string_view, std::string_view) {
        ++permission_checks;
        return false; // deny the first sensitive call
    });

    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;

    agent->send_message(
        "Trigger two tool calls.",
        [](const std::string&) {},
        [](const std::string&, const std::string&) {},
        [&]() {
            {
                std::lock_guard lock(done_mutex);
                done = true;
            }
            done_cv.notify_one();
        });

    {
        std::unique_lock lock(done_mutex);
        REQUIRE(done_cv.wait_for(lock, std::chrono::seconds(3), [&]() { return done; }));
    }

    REQUIRE(provider->call_count() == 1);
    REQUIRE(permission_checks.load(std::memory_order_acquire) == 1);

    const auto history = visible_history(agent);
    REQUIRE(history.size() == 4);
    REQUIRE(history[0].role == "user");
    REQUIRE(history[1].role == "assistant");
    REQUIRE(history[2].role == "tool");
    REQUIRE(history[3].role == "tool");
    REQUIRE_THAT(history[2].content, Catch::Matchers::ContainsSubstring("denied by user"));
    REQUIRE_THAT(history[3].content, Catch::Matchers::ContainsSubstring("skipped after a previous denial"));
}

TEST_CASE("Agent still gates destructive tools in EXECUTE mode",
          "[agent][permission]") {
    const std::string tool_name = "permission_test_execute_destructive_tool";
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<AnnotatedNoopTool>(tool_name));

    auto provider = std::make_shared<SingleToolCallProvider>(
        tool_name,
        R"({"value":"test"})");
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());
    agent->set_mode("EXECUTE");

    std::atomic<int> permission_checks{0};
    agent->set_permission_fn([&](std::string_view name, std::string_view) {
        REQUIRE(name == tool_name);
        ++permission_checks;
        return false;
    });

    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;

    agent->send_message(
        "Run the test tool.",
        [](const std::string&) {},
        [](const std::string&, const std::string&) {},
        [&]() {
            {
                std::lock_guard lock(done_mutex);
                done = true;
            }
            done_cv.notify_one();
        });

    {
        std::unique_lock lock(done_mutex);
        REQUIRE(done_cv.wait_for(lock, std::chrono::seconds(3), [&]() { return done; }));
    }

    REQUIRE(permission_checks.load(std::memory_order_acquire) == 1);
    const auto history = visible_history(agent);
    REQUIRE(history.size() == 3);
    REQUIRE(history[2].role == "tool");
    REQUIRE_THAT(history[2].content, Catch::Matchers::ContainsSubstring("denied by user"));
}

TEST_CASE("Agent distinguishes turn tool allow-list blocks from user denials",
          "[agent][permission][constraints]") {
    auto provider = std::make_shared<BlockedToolProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;

    agent->send_message(
        "Trigger a blocked tool call.",
        [](const std::string&) {},
        [](const std::string&, const std::string&) {},
        [&]() {
            {
                std::lock_guard lock(done_mutex);
                done = true;
            }
            done_cv.notify_one();
        },
        core::agent::Agent::TurnCallbacks{
            .allowed_tools = {"read_file"},
        });

    {
        std::unique_lock lock(done_mutex);
        REQUIRE(done_cv.wait_for(lock, std::chrono::seconds(3), [&]() { return done; }));
    }

    const auto history = visible_history(agent);
    REQUIRE(history.size() == 4);
    REQUIRE(history[2].role == "tool");
    REQUIRE(history[3].role == "assistant");
    REQUIRE_THAT(
        history[2].content,
        Catch::Matchers::ContainsSubstring("blocked by the turn tool allow-list"));
}

TEST_CASE("Agent turns provider startup exceptions into terminal assistant errors",
          "[agent][error]") {
    auto provider = std::make_shared<ThrowingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;
    std::string streamed_text;

    REQUIRE_NOTHROW(agent->send_message(
        "Hello",
        [&](const std::string& chunk) { streamed_text += chunk; },
        [](const std::string&, const std::string&) {},
        [&]() {
            {
                std::lock_guard lock(done_mutex);
                done = true;
            }
            done_cv.notify_one();
        }));

    {
        std::unique_lock lock(done_mutex);
        REQUIRE(done_cv.wait_for(lock, std::chrono::seconds(3), [&]() { return done; }));
    }

    REQUIRE_THAT(streamed_text, Catch::Matchers::ContainsSubstring("Provider startup error"));
    REQUIRE_THAT(streamed_text, Catch::Matchers::ContainsSubstring("invalid_scope"));

    const auto history = visible_history(agent);
    REQUIRE(history.size() == 2);
    REQUIRE(history[0].role == "user");
    REQUIRE(history[1].role == "assistant");
    REQUIRE_THAT(history[1].content, Catch::Matchers::ContainsSubstring("invalid_scope"));
}
