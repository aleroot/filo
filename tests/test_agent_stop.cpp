#include <catch2/catch_test_macros.hpp>
#include "core/agent/Agent.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/tools/ToolManager.hpp"
#include <atomic>
#include <string>
#include <vector>
#include <iostream>

namespace {

class MultiChunkProvider : public core::llm::LLMProvider {
public:
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
};

} // namespace

TEST_CASE("Agent stop request causes repeated stop messages", "[agent][stop]") {
    auto provider = std::make_shared<MultiChunkProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(provider, tool_manager);

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
}
