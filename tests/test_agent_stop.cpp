#include <catch2/catch_test_macros.hpp>
#include "core/agent/Agent.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/tools/ToolManager.hpp"
#include "TestSessionContext.hpp"
#include <atomic>
#include <string>
#include <vector>
#include <iostream>

namespace {

class MultiChunkProvider : public core::llm::LLMProvider {
public:
    std::atomic<int> cancel_calls{0};

    void stream_response(const core::llm::ChatRequest&,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        for (int i = 0; i < 10; ++i) {
            core::llm::StreamChunk chunk;
            chunk.content = "chunk " + std::to_string(i) + " ";
            chunk.is_final = false;
            callback(chunk);
        }
        core::llm::StreamChunk final_chunk;
        final_chunk.is_final = true;
        callback(final_chunk);
    }

    void cancel() override {
        cancel_calls.fetch_add(1, std::memory_order_relaxed);
    }
};

class CancelThenFinalProvider : public core::llm::LLMProvider {
public:
    std::atomic<int> cancel_calls{0};

    void stream_response(const core::llm::ChatRequest&,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        core::llm::StreamChunk chunk;
        chunk.content = "partial ";
        chunk.is_final = false;
        callback(chunk);

        if (cancel_calls.load(std::memory_order_relaxed) > 0) {
            core::llm::StreamChunk final_chunk;
            final_chunk.is_final = true;
            callback(final_chunk);
            return;
        }

        core::llm::StreamChunk next_chunk;
        next_chunk.content = "complete";
        next_chunk.is_final = false;
        callback(next_chunk);
        core::llm::StreamChunk final_chunk;
        final_chunk.is_final = true;
        callback(final_chunk);
    }

    void cancel() override {
        cancel_calls.fetch_add(1, std::memory_order_relaxed);
    }
};

} // namespace

TEST_CASE("Agent stop request causes repeated stop messages", "[agent][stop]") {
    auto provider = std::make_shared<MultiChunkProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    std::vector<std::string> outputs;
    int done_calls = 0;

    auto text_callback = [&](const std::string& text) {
        outputs.push_back(text);
        if (outputs.size() == 3) {
            agent->request_stop();
        }
    };

    auto done_callback = [&]() {
        done_calls++;
    };

    agent->send_message("test", text_callback, [](auto, auto){}, done_callback);

    // After the fix, we expect exactly 1 "[Generation stopped by user]" message.
    int stop_messages = 0;
    for (const auto& out : outputs) {
        if (out.find("Generation stopped by user") != std::string::npos) {
            stop_messages++;
        }
    }

    CHECK(stop_messages == 1);
    CHECK(done_calls == 1);
    CHECK(provider->cancel_calls.load(std::memory_order_relaxed) == 1);

    const auto history = agent->get_history();
    REQUIRE(history.size() >= 2);
    CHECK(history.back().role == "assistant");
    CHECK(history.back().content == "chunk 0 chunk 1 chunk 2 ");
}

TEST_CASE("Agent stop request handles immediate final cancellation", "[agent][stop]") {
    auto provider = std::make_shared<CancelThenFinalProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    std::vector<std::string> outputs;
    int done_calls = 0;

    auto text_callback = [&](const std::string& text) {
        outputs.push_back(text);
        if (text == "partial ") {
            agent->request_stop();
        }
    };

    agent->send_message("test", text_callback, [](auto, auto){}, [&] {
        done_calls++;
    });

    int stop_messages = 0;
    for (const auto& out : outputs) {
        if (out.find("Generation stopped by user") != std::string::npos) {
            stop_messages++;
        }
    }

    CHECK(stop_messages == 1);
    CHECK(done_calls == 1);
    CHECK(provider->cancel_calls.load(std::memory_order_relaxed) == 1);

    const auto history = agent->get_history();
    REQUIRE(history.size() >= 2);
    CHECK(history.back().role == "assistant");
    CHECK(history.back().content == "partial ");
}
