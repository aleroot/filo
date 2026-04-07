#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/Agent.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/session/SessionStore.hpp"
#include "core/tools/ToolManager.hpp"
#include "TestSessionContext.hpp"
#include <filesystem>
#include <future>
#include <thread>

namespace {

class MockProvider : public core::llm::LLMProvider {
public:
    core::llm::ProviderCapabilities caps;
    std::function<void(const core::llm::ChatRequest&, std::function<void(const core::llm::StreamChunk&)>)> on_stream;

    MockProvider() {
        caps.supports_tool_calls = true;
    }

    core::llm::ProviderCapabilities capabilities() const override {
        return caps;
    }

    void stream_response(
        const core::llm::ChatRequest& req,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        if (on_stream) {
            on_stream(req, callback);
        } else {
            callback(core::llm::StreamChunk::make_final());
        }
    }
};

struct TempDir {
    std::filesystem::path path;
    explicit TempDir(std::filesystem::path p) : path(std::move(p)) {
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

} // namespace

TEST_CASE("Agent auto-compaction triggers when threshold exceeded", "[agent][session]") {
    auto provider = std::make_shared<MockProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    // 1. Configure agent with low threshold
    agent->set_auto_compact_threshold(50); // very low threshold (chars * 4)

    // 2. Setup mock provider to handle normal turns and summarization
    std::promise<void> summary_called;
    auto summary_future = summary_called.get_future();
    bool summary_triggered = false;

    provider->on_stream = [&](const core::llm::ChatRequest& req, auto callback) {
        if (!req.messages.empty() && req.messages.back().content.find("Summarise the conversation") != std::string::npos) {
            summary_triggered = true;
            summary_called.set_value();
            callback(core::llm::StreamChunk::make_content("This is a summary of the conversation."));
            callback(core::llm::StreamChunk::make_final());
        } else {
            callback(core::llm::StreamChunk::make_content("This is a long response that should trigger compaction because it exceeds the threshold."));
            callback(core::llm::StreamChunk::make_final());
        }
    };

    // 3. Send a message that triggers response
    std::promise<void> turn_done;
    agent->send_message(
        "Hello",
        [](const std::string&) {},
        [](const std::string&, const std::string&) {},
        [&]() { turn_done.set_value(); }
    );

    turn_done.get_future().wait();

    // 4. Wait for background compaction
    REQUIRE(summary_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    REQUIRE(summary_triggered);

    // 5. Verify agent state
    // Give a moment for the detached thread to call compact_history
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    CHECK(agent->get_context_summary() == "This is a summary of the conversation.");
    // History should be cleared (only system prompt remains)
    const auto history = agent->get_history();
    // System prompt + maybe new context summary injection? 
    // Agent::compact_history clears history and calls ensure_system_prompt.
    // ensure_system_prompt adds the system message.
    REQUIRE(history.empty()); // get_history excludes system prompt
}

TEST_CASE("SessionStore list/delete operations", "[session][store]") {
    TempDir tmp{std::filesystem::temp_directory_path() / "filo_test_lifecycle"};
    core::session::SessionStore store{tmp.path};

    // Create a few sessions
    core::session::SessionData s1;
    s1.session_id = "sess1";
    s1.created_at = "2023-01-01T12:00:00Z";
    store.save(s1);

    core::session::SessionData s2;
    s2.session_id = "sess2";
    s2.created_at = "2023-01-02T12:00:00Z";
    store.save(s2);

    // List
    auto sessions = store.list();
    REQUIRE(sessions.size() == 2);
    // Sorts by most recent
    CHECK(sessions[0].session_id == "sess2");
    CHECK(sessions[1].session_id == "sess1");

    // Delete
    std::string error;
    bool removed = store.remove("sess1", &error);
    REQUIRE(removed);
    
    sessions = store.list();
    REQUIRE(sessions.size() == 1);
    CHECK(sessions[0].session_id == "sess2");
}

TEST_CASE("Agent resumes from SessionData", "[agent][session]") {
    auto provider = std::make_shared<MockProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    std::vector<core::llm::Message> history = {
        {"user", "User msg 1", "", "", {}},
        {"assistant", "Asst msg 1", "", "", {}}
    };
    std::string summary = "Previous summary";
    std::string mode = "RESEARCH";

    agent->load_history(history, summary, mode);

    CHECK(agent->get_mode() == "RESEARCH");
    CHECK(agent->get_context_summary() == "Previous summary");
    
    const auto loaded = agent->get_history();
    REQUIRE(loaded.size() == 2);
    CHECK(loaded[0].content == "User msg 1");
}

TEST_CASE("Agent preserves multimodal user turns for retry", "[agent][session]") {
    auto provider = std::make_shared<MockProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    std::promise<void> done;
    agent->send_message(
        core::llm::Message{
            .role = "user",
            .content = "Please inspect [Attached image: /tmp/example.png]",
            .content_parts = {
                core::llm::ContentPart::make_text("Please inspect "),
                core::llm::ContentPart::make_image("/tmp/example.png", "image/png"),
            },
        },
        [](const std::string&) {},
        [](const std::string&, const std::string&) {},
        [&]() { done.set_value(); });

    done.get_future().wait();

    const auto last = agent->last_user_turn();
    REQUIRE(last.has_value());
    REQUIRE(last->content_parts.size() == 2);
    REQUIRE(last->content_parts[1].type == core::llm::ContentPartType::Image);
}
