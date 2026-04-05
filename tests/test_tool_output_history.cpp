#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/Agent.hpp"
#include "core/agent/ToolOutputHistory.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/tools/Tool.hpp"
#include "core/tools/ToolManager.hpp"

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::string_view kLargeOutputToolName = "test_large_output_history_tool";
constexpr std::string_view kLargeErrorToolName = "test_large_error_history_tool";

class LargeOutputTool final : public core::tools::Tool {
public:
    [[nodiscard]] core::tools::ToolDefinition get_definition() const override {
        return {
            .name = std::string(kLargeOutputToolName),
            .title = "Large Output Test Tool",
            .description = "Returns an intentionally large payload for history truncation tests.",
            .parameters = {},
            .annotations = {
                .read_only_hint = true,
                .idempotent_hint = true,
            },
        };
    }

    [[nodiscard]] std::string execute(const std::string&) override {
        return std::string(R"({"output":")") + std::string(120 * 1024, 'x') + R"("})";
    }
};

class LargeErrorTool final : public core::tools::Tool {
public:
    [[nodiscard]] core::tools::ToolDefinition get_definition() const override {
        return {
            .name = std::string(kLargeErrorToolName),
            .title = "Large Error Test Tool",
            .description = "Returns an intentionally large error payload for truncation tests.",
            .parameters = {},
            .annotations = {
                .read_only_hint = true,
                .idempotent_hint = true,
            },
        };
    }

    [[nodiscard]] std::string execute(const std::string&) override {
        return std::string(R"({"error":")") + std::string(120 * 1024, 'e') + R"("})";
    }
};

class ToolThenFinalProvider final : public core::llm::LLMProvider {
public:
    explicit ToolThenFinalProvider(std::string tool_name)
        : tool_name_(std::move(tool_name)) {}

    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        ++calls_;
        if (calls_ == 1) {
            core::llm::ToolCall tc;
            tc.index = 0;
            tc.id = "tc-1";
            tc.type = "function";
            tc.function.name = tool_name_;
            tc.function.arguments = "{}";

            core::llm::StreamChunk chunk;
            chunk.tools = {tc};
            chunk.is_final = true;
            callback(chunk);
            return;
        }

        callback(core::llm::StreamChunk::make_content("final"));
        callback(core::llm::StreamChunk::make_final());
    }

private:
    std::string tool_name_;
    int calls_ = 0;
};

void send_and_wait(const std::shared_ptr<core::agent::Agent>& agent,
                   std::string_view prompt) {
    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;

    agent->send_message(
        std::string(prompt),
        [](const std::string&) {},
        [](const std::string&, const std::string&) {},
        [&]() {
            {
                std::lock_guard lock(done_mutex);
                done = true;
            }
            done_cv.notify_one();
        });

    std::unique_lock lock(done_mutex);
    REQUIRE(done_cv.wait_for(lock, std::chrono::seconds(5), [&]() { return done; }));
}

} // namespace

TEST_CASE("ToolOutputHistory leaves compact outputs unchanged", "[agent][tool-history]") {
    const std::string small = R"({"output":"ok"})";
    const std::string clamped = core::agent::tool_output_history::clamp_for_history("read_file", small);
    CHECK(clamped == small);
}

TEST_CASE("ToolOutputHistory truncates oversized non-error outputs", "[agent][tool-history]") {
    const std::string large = std::string(R"({"output":")") + std::string(80 * 1024, 'a') + R"("})";
    const std::string clamped = core::agent::tool_output_history::clamp_for_history("write_file", large);

    CHECK(clamped.size() < large.size());
    CHECK_THAT(clamped, Catch::Matchers::ContainsSubstring(R"("truncated":true)"));
    CHECK_THAT(clamped, Catch::Matchers::ContainsSubstring(R"("original_chars":)"));
    CHECK_THAT(clamped, Catch::Matchers::ContainsSubstring(R"("digest_fnv1a64":)"));
    CHECK_THAT(clamped, Catch::Matchers::ContainsSubstring(R"("head":)"));
}

TEST_CASE("ToolOutputHistory preserves error semantics when truncating", "[agent][tool-history]") {
    const std::string large_error = std::string(R"({"error":")") + std::string(80 * 1024, 'z') + R"("})";
    const std::string clamped = core::agent::tool_output_history::clamp_for_history("replace", large_error);

    CHECK(clamped.size() < large_error.size());
    CHECK_THAT(clamped, Catch::Matchers::ContainsSubstring(R"("error":"Tool output truncated for history)"));
    CHECK_THAT(clamped, Catch::Matchers::ContainsSubstring(R"("original_chars":)"));
}

TEST_CASE("Agent stores oversized tool output in compact history format", "[agent][tool-history]") {
    auto provider = std::make_shared<ToolThenFinalProvider>(std::string(kLargeOutputToolName));
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<LargeOutputTool>());

    auto agent = std::make_shared<core::agent::Agent>(provider, tool_manager);
    send_and_wait(agent, "Run large output tool");

    const auto history = agent->get_history();
    bool found = false;
    for (const auto& msg : history) {
        if (msg.role == "tool" && msg.name == kLargeOutputToolName) {
            found = true;
            CHECK(msg.content.size() < 40 * 1024);
            CHECK_THAT(msg.content, Catch::Matchers::ContainsSubstring(R"("truncated":true)"));
            CHECK_THAT(msg.content, Catch::Matchers::ContainsSubstring(R"("original_chars":)"));
        }
    }
    REQUIRE(found);
}

TEST_CASE("Agent keeps error marker when compacting oversized error payloads", "[agent][tool-history]") {
    auto provider = std::make_shared<ToolThenFinalProvider>(std::string(kLargeErrorToolName));
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<LargeErrorTool>());

    auto agent = std::make_shared<core::agent::Agent>(provider, tool_manager);
    send_and_wait(agent, "Run large error tool");

    const auto history = agent->get_history();
    bool found = false;
    for (const auto& msg : history) {
        if (msg.role == "tool" && msg.name == kLargeErrorToolName) {
            found = true;
            CHECK(msg.content.size() < 40 * 1024);
            CHECK_THAT(msg.content, Catch::Matchers::ContainsSubstring(R"("error":"Tool output truncated for history)"));
        }
    }
    REQUIRE(found);
}

