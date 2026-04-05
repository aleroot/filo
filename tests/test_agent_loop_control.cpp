#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/Agent.hpp"
#include "core/context/SessionContext.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/tools/Tool.hpp"
#include "core/tools/ToolManager.hpp"
#include "TestSessionContext.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::string_view kLoopToolName = "test_loop_control_noop";

class ScopedCurrentPath {
public:
    explicit ScopedCurrentPath(std::filesystem::path target)
        : old_path_(std::filesystem::current_path()) {
        std::filesystem::current_path(std::move(target));
    }

    ~ScopedCurrentPath() {
        std::error_code ec;
        std::filesystem::current_path(old_path_, ec);
    }

private:
    std::filesystem::path old_path_;
};

class TempDir {
public:
    explicit TempDir(std::filesystem::path path)
        : path_(std::move(path)) {
        std::filesystem::create_directories(path_);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path_, ec);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    std::filesystem::path path_;
};

class NoopLoopTool final : public core::tools::Tool {
public:
    [[nodiscard]] core::tools::ToolDefinition get_definition() const override {
        return {
            .name = std::string(kLoopToolName),
            .title = "Loop Control No-op",
            .description = "No-op tool used for loop-control tests.",
            .parameters = {},
            .annotations = {
                .read_only_hint = true,
                .idempotent_hint = true,
            },
        };
    }

    [[nodiscard]] std::string execute(
        const std::string&,
        const core::context::SessionContext&) override {
        return R"({"ok":true})";
    }
};

class InfiniteToolLoopProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {

        const int call = calls_.fetch_add(1, std::memory_order_acq_rel) + 1;

        core::llm::ToolCall tool_call;
        tool_call.index = 0;
        tool_call.id = std::format("loop-call-{}", call);
        tool_call.type = "function";
        tool_call.function.name = std::string(kLoopToolName);
        tool_call.function.arguments = "{}";

        core::llm::StreamChunk chunk;
        chunk.tools = {tool_call};
        chunk.is_final = true;
        callback(chunk);
    }

    [[nodiscard]] int call_count() const noexcept {
        return calls_.load(std::memory_order_acquire);
    }

private:
    std::atomic<int> calls_{0};
};

class CapturingProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest& request,
        std::function<void(const core::llm::StreamChunk&)> callback) override {

        {
            std::lock_guard lock(mutex_);
            requests_.push_back(request);
        }

        callback(core::llm::StreamChunk::make_content("ok"));
        callback(core::llm::StreamChunk::make_final());
    }

    [[nodiscard]] std::vector<core::llm::ChatRequest> requests_snapshot() const {
        std::lock_guard lock(mutex_);
        return requests_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<core::llm::ChatRequest> requests_;
};

void send_and_wait(const std::shared_ptr<core::agent::Agent>& agent,
                   const std::string& prompt) {
    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;

    agent->send_message(
        prompt,
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

TEST_CASE("Agent enforces a per-turn model step bound", "[agent][loop]") {
    auto provider = std::make_shared<InfiniteToolLoopProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<NoopLoopTool>());

    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());
    agent->set_loop_limits({.max_steps_per_turn = 3});

    std::mutex output_mutex;
    std::string streamed_text;
    std::atomic<int> done_calls{0};
    std::promise<void> done_promise;
    auto done_future = done_promise.get_future();

    agent->send_message(
        "Run until bounded.",
        [&](const std::string& chunk) {
            std::lock_guard lock(output_mutex);
            streamed_text += chunk;
        },
        [](const std::string&, const std::string&) {},
        [&]() {
            if (done_calls.fetch_add(1, std::memory_order_acq_rel) == 0) {
                done_promise.set_value();
            }
        });

    REQUIRE(done_future.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    CHECK(done_calls.load(std::memory_order_acquire) == 1);
    CHECK(provider->call_count() == 3);
    CHECK_THAT(streamed_text, Catch::Matchers::ContainsSubstring("step limit"));

    const auto history = agent->get_history();
    std::size_t tool_messages = 0;
    bool found_limit_message = false;
    for (const auto& message : history) {
        if (message.role == "tool") {
            ++tool_messages;
        }
        if (message.role == "assistant"
            && message.content.find("step limit") != std::string::npos) {
            found_limit_message = true;
        }
    }

    CHECK(tool_messages == 3);
    CHECK(found_limit_message);
}

TEST_CASE("Agent keeps stable prompt prefix cached across turns", "[agent][prompt]") {
    TempDir tmp{
        std::filesystem::temp_directory_path()
        / std::format("filo_prompt_cache_test_{}", std::chrono::steady_clock::now().time_since_epoch().count())
    };
    ScopedCurrentPath scoped_cwd(tmp.path());

    {
        std::ofstream alpha(tmp.path() / "alpha.txt");
        alpha << "alpha\n";
    }

    auto provider = std::make_shared<CapturingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    const auto session_context = test_support::make_session_context(
        core::workspace::WorkspaceSnapshot{
            .primary = tmp.path(),
            .additional = {},
            .enforce = true,
            .version = 1,
        },
        core::context::SessionTransport::cli,
        "prompt-cache-session");
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        session_context);

    send_and_wait(agent, "first");

    {
        std::ofstream beta(tmp.path() / "beta.txt");
        beta << "beta\n";
    }

    send_and_wait(agent, "second");

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 2);
    REQUIRE_FALSE(requests[0].messages.empty());
    REQUIRE_FALSE(requests[1].messages.empty());
    REQUIRE(requests[0].messages[0].role == "system");
    REQUIRE(requests[1].messages[0].role == "system");

    const auto& first_system_prompt = requests[0].messages[0].content;
    const auto& second_system_prompt = requests[1].messages[0].content;

    CHECK(first_system_prompt == second_system_prompt);
    CHECK(first_system_prompt.find("alpha.txt") != std::string::npos);
    CHECK(second_system_prompt.find("beta.txt") == std::string::npos);
}

TEST_CASE("Agent project context follows the explicit SessionContext workspace",
          "[agent][prompt][workspace]") {
    const auto base = std::filesystem::temp_directory_path()
        / std::format("filo_agent_session_workspace_{}",
                      std::chrono::steady_clock::now().time_since_epoch().count());
    TempDir workspace_dir(base / "workspace");
    TempDir cwd_dir(base / "cwd");

    {
        std::ofstream workspace_file(workspace_dir.path() / "workspace_marker.txt");
        workspace_file << "workspace\n";
    }
    {
        std::ofstream cwd_file(cwd_dir.path() / "cwd_marker.txt");
        cwd_file << "cwd\n";
    }

    ScopedCurrentPath scoped_cwd(cwd_dir.path());

    auto provider = std::make_shared<CapturingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    const auto session_context = test_support::make_session_context(
        core::workspace::WorkspaceSnapshot{
            .primary = workspace_dir.path(),
            .additional = {},
            .enforce = true,
            .version = 1,
        },
        core::context::SessionTransport::cli,
        "agent-workspace-session");
    auto agent = std::make_shared<core::agent::Agent>(provider, tool_manager, session_context);

    send_and_wait(agent, "hello");

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 1);
    REQUIRE_FALSE(requests[0].messages.empty());
    REQUIRE(requests[0].messages[0].role == "system");

    const auto& system_prompt = requests[0].messages[0].content;
    CHECK(system_prompt.find("workspace_marker.txt") != std::string::npos);
    CHECK(system_prompt.find("cwd_marker.txt") == std::string::npos);
}
