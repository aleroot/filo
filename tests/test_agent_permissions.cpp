#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/Agent.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/tools/ToolManager.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <vector>

namespace {

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

} // namespace

TEST_CASE("Agent stops current loop after user denies a tool call", "[agent][permission]") {
    auto provider = std::make_shared<ToolCallThenTextProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(provider, tool_manager);

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

    const auto history = agent->get_history();
    REQUIRE(history.size() == 4);
    REQUIRE(history[0].role == "user");
    REQUIRE(history[1].role == "assistant");
    REQUIRE(history[2].role == "tool");
    REQUIRE(history[3].role == "tool");
    REQUIRE_THAT(history[2].content, Catch::Matchers::ContainsSubstring("denied by user"));
    REQUIRE_THAT(history[3].content, Catch::Matchers::ContainsSubstring("skipped after a previous denial"));
}
