#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/Agent.hpp"
#include "core/agent/RepositoryContextMessage.hpp"
#include "core/context/SessionContext.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/protocols/AnthropicProtocol.hpp"
#include "core/tools/Tool.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/ToolNames.hpp"
#include "TestSessionContext.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <fstream>
#include <future>
#include <iterator>
#include <mutex>
#include <ranges>
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

class NamedNoopTool final : public core::tools::Tool {
public:
    explicit NamedNoopTool(std::string name)
        : name_(std::move(name)) {}

    [[nodiscard]] core::tools::ToolDefinition get_definition() const override {
        return {
            .name = name_,
            .title = "Named No-op",
            .description = "No-op tool used for request filtering tests.",
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

private:
    std::string name_;
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

class RecoveringMaxOutputProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest& request,
        std::function<void(const core::llm::StreamChunk&)> callback) override {

        {
            std::lock_guard lock(mutex_);
            requests_.push_back(request);
        }

        const int call = calls_.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (call == 1) {
            callback(core::llm::StreamChunk::make_final("max_tokens", true));
            return;
        }

        callback(core::llm::StreamChunk::make_content("continued after recovery"));
        callback(core::llm::StreamChunk::make_final());
    }

    [[nodiscard]] int call_count() const noexcept {
        return calls_.load(std::memory_order_acquire);
    }

    [[nodiscard]] std::vector<core::llm::ChatRequest> requests_snapshot() const {
        std::lock_guard lock(mutex_);
        return requests_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<core::llm::ChatRequest> requests_;
    std::atomic<int> calls_{0};
};

class AlwaysMaxOutputProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest& request,
        std::function<void(const core::llm::StreamChunk&)> callback) override {

        {
            std::lock_guard lock(mutex_);
            requests_.push_back(request);
        }

        calls_.fetch_add(1, std::memory_order_acq_rel);
        callback(core::llm::StreamChunk::make_final("max_tokens", true));
    }

    [[nodiscard]] int call_count() const noexcept {
        return calls_.load(std::memory_order_acquire);
    }

    [[nodiscard]] std::vector<core::llm::ChatRequest> requests_snapshot() const {
        std::lock_guard lock(mutex_);
        return requests_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<core::llm::ChatRequest> requests_;
    std::atomic<int> calls_{0};
};

class BlockingProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {

        entered_.set_value();
        release_.get_future().wait();
        callback(core::llm::StreamChunk::make_content("first done"));
        callback(core::llm::StreamChunk::make_final());
    }

    void wait_until_entered() {
        entered_.get_future().wait();
    }

    void release() {
        release_.set_value();
    }

private:
    std::promise<void> entered_;
    std::promise<void> release_;
};

class RotatingToolLoopProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest& request,
        std::function<void(const core::llm::StreamChunk&)> callback) override {

        {
            std::lock_guard lock(mutex_);
            requests_.push_back(request);
        }

        const int call = calls_.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (call <= 10) {
            set_last_usage(call <= 4 ? 4'000 : 26'000, call <= 4 ? 400 : 800);

            core::llm::ToolCall tool_call;
            tool_call.index = 0;
            tool_call.id = std::format("rotate-{}", call);
            tool_call.type = "function";
            tool_call.function.name = std::string(kLoopToolName);
            tool_call.function.arguments = "{}";

            core::llm::StreamChunk chunk;
            chunk.tools = {tool_call};
            chunk.is_final = true;
            callback(chunk);
            return;
        }

        set_last_usage(3'000, 300);
        callback(core::llm::StreamChunk::make_content("rotation complete"));
        callback(core::llm::StreamChunk::make_final());
    }

    void reset_conversation_state() override {
        reset_calls_.fetch_add(1, std::memory_order_acq_rel);
    }

    [[nodiscard]] int call_count() const noexcept {
        return calls_.load(std::memory_order_acquire);
    }

    [[nodiscard]] int reset_count() const noexcept {
        return reset_calls_.load(std::memory_order_acquire);
    }

    [[nodiscard]] std::vector<core::llm::ChatRequest> requests_snapshot() const {
        std::lock_guard lock(mutex_);
        return requests_;
    }

private:
    mutable std::mutex mutex_;
    std::vector<core::llm::ChatRequest> requests_;
    std::atomic<int> calls_{0};
    std::atomic<int> reset_calls_{0};
};

class ParallelExploreProvider final : public core::llm::LLMProvider {
public:
    [[nodiscard]] core::llm::ProviderCapabilities capabilities() const override {
        return {
            .supports_tool_calls = true,
            .is_local = false,
            .supports_parallel_requests = true,
        };
    }

    void stream_response(
        const core::llm::ChatRequest& request,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        const bool parent_follow_up = std::ranges::any_of(
            request.messages,
            [](const core::llm::Message& message) { return message.role == "tool"; });
        if (parent_follow_up) {
            callback(core::llm::StreamChunk::make_content("parallel exploration complete"));
            callback(core::llm::StreamChunk::make_final());
            return;
        }

        const bool delegated_worker = std::ranges::any_of(
            request.messages,
            [](const core::llm::Message& message) {
                return message.role == "user" && message.content.contains("worker-");
            });
        if (delegated_worker) {
            run_worker(std::move(callback));
            return;
        }

        std::vector<core::llm::ToolCall> calls;
        calls.reserve(6);
        for (int i = 0; i < 6; ++i) {
            core::llm::ToolCall call;
            call.index = i;
            call.id = std::format("parallel-task-{}", i);
            call.type = "function";
            call.function.name = "task";
            call.function.arguments = std::format(
                R"({{"description":"worker {}","prompt":"worker-{} inspect independently","subagent_type":"explore"}})",
                i,
                i);
            calls.push_back(std::move(call));
        }
        core::llm::StreamChunk chunk;
        chunk.tools = std::move(calls);
        chunk.is_final = true;
        callback(chunk);
    }

    [[nodiscard]] int max_active() const {
        std::lock_guard lock(mutex_);
        return max_active_;
    }

    [[nodiscard]] int completed() const {
        std::lock_guard lock(mutex_);
        return completed_;
    }

private:
    void run_worker(std::function<void(const core::llm::StreamChunk&)> callback) {
        {
            std::unique_lock lock(mutex_);
            ++active_;
            max_active_ = std::max(max_active_, active_);
            if (max_active_ >= core::agent::SubagentOrchestrator::kMaxParallelReadOnlySubagents) {
                ready_.notify_all();
            }
            (void)ready_.wait_for(lock, std::chrono::milliseconds(500), [&] {
                return max_active_
                    >= core::agent::SubagentOrchestrator::kMaxParallelReadOnlySubagents;
            });
        }

        callback(core::llm::StreamChunk::make_content("worker complete"));
        callback(core::llm::StreamChunk::make_final());

        {
            std::lock_guard lock(mutex_);
            --active_;
            ++completed_;
        }
        ready_.notify_all();
    }

    mutable std::mutex mutex_;
    std::condition_variable ready_;
    int active_ = 0;
    int max_active_ = 0;
    int completed_ = 0;
};

void send_and_wait(const std::shared_ptr<core::agent::Agent>& agent,
                   const std::string& prompt,
                   core::agent::Agent::TurnCallbacks callbacks) {
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
        },
        std::move(callbacks));

    std::unique_lock lock(done_mutex);
    REQUIRE(done_cv.wait_for(lock, std::chrono::seconds(5), [&]() { return done; }));
}

void send_and_wait(const std::shared_ptr<core::agent::Agent>& agent,
                   const std::string& prompt) {
    send_and_wait(agent, prompt, core::agent::Agent::TurnCallbacks{});
}

} // namespace

TEST_CASE("Agent appends history-only user messages without model turn", "[agent][history]") {
    auto provider = std::make_shared<CapturingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    agent->append_history_message(core::llm::Message{
        .role = "user",
        .content =
            "I ran the following shell command:\n"
            "```sh\n"
            "pwd\n"
            "```\n\n"
            "This produced the following result:\n"
            "```\n"
            "/tmp\n"
            "```",
    });

    const auto history = agent->get_history();
    REQUIRE(history.size() == 1);
    CHECK(history[0].role == "user");
    REQUIRE_THAT(history[0].content,
                 Catch::Matchers::ContainsSubstring("I ran the following shell command"));
    REQUIRE_THAT(history[0].content,
                 Catch::Matchers::ContainsSubstring("This produced the following result"));
    CHECK(provider->requests_snapshot().empty());
}

TEST_CASE("Agent runs independent explore subagents concurrently with a bounded fan-out",
          "[agent][orchestration][parallel]") {
    auto provider = std::make_shared<ParallelExploreProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());
    std::atomic<int> permission_requests{0};
    agent->set_permission_profile(core::agent::PermissionProfile::Interactive);
    agent->set_permission_fn([&](std::string_view, std::string_view) {
        permission_requests.fetch_add(1, std::memory_order_relaxed);
        return false;
    });

    send_and_wait(agent, "Run independent parallel explorations");

    CHECK(provider->completed() == 6);
    CHECK(permission_requests.load(std::memory_order_relaxed) == 0);
    CHECK(provider->max_active()
          == core::agent::SubagentOrchestrator::kMaxParallelReadOnlySubagents);

    const auto history = agent->get_history();
    CHECK(std::ranges::count_if(history, [](const core::llm::Message& message) {
        return message.role == "tool" && message.name == "task";
    }) == 6);
}

TEST_CASE("Agent rejects overlapping turns", "[agent][loop]") {
    auto provider = std::make_shared<BlockingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    std::promise<void> first_done;
    std::thread first_turn([&]() {
        agent->send_message(
            "first",
            [](const std::string&) {},
            [](const std::string&, const std::string&) {},
            [&]() { first_done.set_value(); });
    });

    provider->wait_until_entered();

    std::string second_output;
    int second_done_count = 0;
    agent->send_message(
        "second",
        [&](const std::string& chunk) { second_output += chunk; },
        [](const std::string&, const std::string&) {},
        [&]() { ++second_done_count; });

    CHECK(second_done_count == 1);
    CHECK(second_output.find("another agent turn is already running") != std::string::npos);

    provider->release();
    REQUIRE(first_done.get_future().wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    first_turn.join();

    auto history = agent->get_history();
    std::erase_if(history, [](const core::llm::Message& message) {
        return message.synthetic;
    });
    REQUIRE(history.size() == 2);
    CHECK(history[0].role == "user");
    CHECK(history[0].content == "first");
    CHECK(history[1].role == "assistant");
    CHECK(history[1].content == "first done");
}

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

TEST_CASE("Agent automatically recovers from truncated Claude turns",
          "[agent][loop][claude]") {
    auto provider = std::make_shared<RecoveringMaxOutputProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());
    agent->set_active_provider_name("claude");

    std::mutex capture_mutex;
    std::string streamed_text;
    std::vector<std::string> status_logs;
    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;

    agent->send_message(
        "complete a tool-heavy task",
        [&](const std::string& chunk) {
            std::lock_guard lock(capture_mutex);
            streamed_text += chunk;
        },
        [](const std::string&, const std::string&) {},
        [&]() {
            {
                std::lock_guard lock(done_mutex);
                done = true;
            }
            done_cv.notify_one();
        },
        core::agent::Agent::TurnCallbacks{
            .on_status_log = [&](const std::string& status) {
                std::lock_guard lock(capture_mutex);
                status_logs.push_back(status);
            },
        });

    {
        std::unique_lock lock(done_mutex);
        REQUIRE(done_cv.wait_for(lock, std::chrono::seconds(5), [&]() { return done; }));
    }

    {
        std::lock_guard lock(capture_mutex);
        CHECK_THAT(streamed_text, Catch::Matchers::ContainsSubstring("continued after recovery"));
        CHECK(streamed_text.find("ended this turn with no visible text") == std::string::npos);
        REQUIRE(status_logs.size() == 1);
        CHECK_THAT(status_logs.front(),
                   Catch::Matchers::ContainsSubstring("continuing automatically (1/3)"));
    }

    CHECK(provider->call_count() == 2);

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 2);
    REQUIRE_FALSE(requests[1].messages.empty());
    const auto& recovery_message = requests[1].messages.back();
    CHECK(recovery_message.role == "user");
    CHECK_THAT(recovery_message.content,
               Catch::Matchers::ContainsSubstring("Output token limit hit"));

    const auto history = agent->get_history();
    bool found_recovery_prompt = false;
    bool found_final_assistant = false;
    bool found_empty_turn_notice = false;
    for (const auto& message : history) {
        if (message.role == "user"
            && message.content.find("Output token limit hit") != std::string::npos) {
            found_recovery_prompt = true;
        }
        if (message.role == "assistant"
            && message.content.find("continued after recovery") != std::string::npos) {
            found_final_assistant = true;
        }
        if (message.role == "assistant"
            && message.content.find("ended this turn with no visible text") != std::string::npos) {
            found_empty_turn_notice = true;
        }
    }

    CHECK(found_recovery_prompt);
    CHECK(found_final_assistant);
    CHECK_FALSE(found_empty_turn_notice);
}

TEST_CASE("Agent bounds automatic recovery for repeated truncated Claude turns",
          "[agent][loop][claude]") {
    auto provider = std::make_shared<AlwaysMaxOutputProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());
    agent->set_active_provider_name("claude");

    std::mutex capture_mutex;
    std::string streamed_text;
    std::vector<std::string> status_logs;
    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;

    agent->send_message(
        "repeat truncation",
        [&](const std::string& chunk) {
            std::lock_guard lock(capture_mutex);
            streamed_text += chunk;
        },
        [](const std::string&, const std::string&) {},
        [&]() {
            {
                std::lock_guard lock(done_mutex);
                done = true;
            }
            done_cv.notify_one();
        },
        core::agent::Agent::TurnCallbacks{
            .on_status_log = [&](const std::string& status) {
                std::lock_guard lock(capture_mutex);
                status_logs.push_back(status);
            },
        });

    {
        std::unique_lock lock(done_mutex);
        REQUIRE(done_cv.wait_for(lock, std::chrono::seconds(5), [&]() { return done; }));
    }

    CHECK(provider->call_count() == 4);

    {
        std::lock_guard lock(capture_mutex);
        REQUIRE(status_logs.size() == 3);
        CHECK_THAT(status_logs.back(),
                   Catch::Matchers::ContainsSubstring("continuing automatically (3/3)"));
        CHECK_THAT(streamed_text,
                   Catch::Matchers::ContainsSubstring("Claude ended this turn"));
        CHECK_THAT(streamed_text,
                   Catch::Matchers::ContainsSubstring("stream ended before the provider completed a tool call"));
    }

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 4);
    int recovery_prompt_count = 0;
    for (const auto& request : requests) {
        if (!request.messages.empty()
            && request.messages.back().role == "user"
            && request.messages.back().content.find("Output token limit hit") != std::string::npos) {
            ++recovery_prompt_count;
        }
    }
    CHECK(recovery_prompt_count == 3);
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

    {
        std::ofstream beta(tmp.path() / "beta.txt", std::ios::trunc);
        beta << "beta changed without changing repository facts\n";
    }

    send_and_wait(agent, "third");

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 3);
    REQUIRE_FALSE(requests[0].messages.empty());
    REQUIRE_FALSE(requests[1].messages.empty());
    REQUIRE_FALSE(requests[2].messages.empty());
    REQUIRE(requests[0].messages[0].role == "system");
    REQUIRE(requests[1].messages[0].role == "system");
    REQUIRE(requests[2].messages[0].role == "system");

    const auto& first_system_prompt = requests[0].messages[0].content;
    const auto& second_system_prompt = requests[1].messages[0].content;
    const auto& third_system_prompt = requests[2].messages[0].content;
    const auto first_prompt_plan = requests[0].prompt_plan.render();
    const auto second_prompt_plan = requests[1].prompt_plan.render();
    const auto third_prompt_plan = requests[2].prompt_plan.render();

    CHECK(first_system_prompt == second_system_prompt);
    CHECK(second_system_prompt == third_system_prompt);
    CHECK(first_prompt_plan == second_prompt_plan);
    CHECK(second_prompt_plan == third_prompt_plan);
    CHECK(first_prompt_plan == first_system_prompt);
    CHECK(first_system_prompt.find("[Project Context]") == std::string::npos);
    CHECK(first_system_prompt.find("alpha.txt") == std::string::npos);
    CHECK(second_system_prompt.find("beta.txt") == std::string::npos);
    CHECK(second_prompt_plan.find("beta.txt") == std::string::npos);

    const auto synthetic_messages = [](const core::llm::ChatRequest& request) {
        std::vector<core::llm::Message> messages;
        std::ranges::copy_if(
            request.messages,
            std::back_inserter(messages),
            [](const core::llm::Message& message) {
                return core::agent::is_repository_context_message(message);
            });
        return messages;
    };
    const auto first_context = synthetic_messages(requests[0]);
    const auto second_context = synthetic_messages(requests[1]);
    const auto third_context = synthetic_messages(requests[2]);
    REQUIRE(first_context.size() == 1);
    REQUIRE(second_context.size() == 2);
    REQUIRE(third_context.size() == 2);
    CHECK_THAT(first_context[0].content, Catch::Matchers::ContainsSubstring("alpha.txt"));
    CHECK_THAT(second_context[1].content,
               Catch::Matchers::ContainsSubstring("[Project Context Update]"));
    CHECK_THAT(second_context[1].content, Catch::Matchers::ContainsSubstring("beta.txt"));
    CHECK(second_context[1].content.find("Status:") == std::string::npos);

    REQUIRE(requests[1].messages.size() >= requests[0].messages.size());
    for (std::size_t i = 0; i < requests[0].messages.size(); ++i) {
        CHECK(requests[1].messages[i].role == requests[0].messages[i].role);
        CHECK(requests[1].messages[i].content == requests[0].messages[i].content);
        CHECK(requests[1].messages[i].synthetic == requests[0].messages[i].synthetic);
    }

    const auto first_payload =
        core::llm::protocols::AnthropicSerializer::serialize(requests[0]);
    const auto second_payload =
        core::llm::protocols::AnthropicSerializer::serialize(requests[1]);
    const auto third_payload =
        core::llm::protocols::AnthropicSerializer::serialize(requests[2]);
    const auto first_messages = first_payload.find(R"(,"messages":[)");
    const auto second_messages = second_payload.find(R"(,"messages":[)");
    const auto third_messages = third_payload.find(R"(,"messages":[)");
    REQUIRE(first_messages != std::string::npos);
    REQUIRE(second_messages != std::string::npos);
    REQUIRE(third_messages != std::string::npos);
    CHECK(first_payload.substr(0, first_messages)
          == second_payload.substr(0, second_messages));
    CHECK(second_payload.substr(0, second_messages)
          == third_payload.substr(0, third_messages));
}

TEST_CASE("Agent exposes write_todos and injects its current plan", "[agent][prompt][todos]") {
    auto provider = std::make_shared<CapturingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    const auto added = agent->add_todo("Inspect the request pipeline");
    REQUIRE(added.has_value());
    send_and_wait(agent, "continue");

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 1);
    CHECK(std::ranges::any_of(requests[0].tools, [](const core::llm::Tool& tool) {
        return tool.function.name == core::tools::names::kWriteTodos;
    }));
    const auto prompt = requests[0].prompt_plan.render();
    CHECK_THAT(prompt, Catch::Matchers::ContainsSubstring("Current task plan"));
    CHECK_THAT(prompt, Catch::Matchers::ContainsSubstring("t1: Inspect the request pipeline"));
}

TEST_CASE("Agent restores repository snapshot continuity across resume",
          "[agent][prompt][resume]") {
    TempDir tmp{
        std::filesystem::temp_directory_path()
        / std::format("filo_prompt_resume_test_{}",
                      std::chrono::steady_clock::now().time_since_epoch().count())
    };
    {
        std::ofstream alpha(tmp.path() / "alpha.txt");
        alpha << "alpha\n";
    }

    auto& tool_manager = core::tools::ToolManager::get_instance();
    const auto session_context = test_support::make_session_context(
        core::workspace::WorkspaceSnapshot{
            .primary = tmp.path(),
            .additional = {},
            .enforce = true,
            .version = 1,
        },
        core::context::SessionTransport::cli,
        "prompt-resume-session");

    auto first_provider = std::make_shared<CapturingProvider>();
    auto first_agent = std::make_shared<core::agent::Agent>(
        first_provider, tool_manager, session_context);
    send_and_wait(first_agent, "first");
    const auto saved_history = first_agent->get_history();
    const auto initial_context_count = std::ranges::count_if(
        saved_history,
        core::agent::is_repository_context_message);
    REQUIRE(initial_context_count == 1);
    const auto initial_context = std::ranges::find_if(
        saved_history,
        core::agent::is_repository_context_message);
    REQUIRE(initial_context != saved_history.end());
    CHECK_FALSE(initial_context->input_text.empty());

    auto resumed_provider = std::make_shared<CapturingProvider>();
    auto resumed_agent = std::make_shared<core::agent::Agent>(
        resumed_provider, tool_manager, session_context);
    resumed_agent->load_history(saved_history, {}, "BUILD");
    send_and_wait(resumed_agent, "second");

    {
        std::ofstream beta(tmp.path() / "beta.txt");
        beta << "beta\n";
    }
    send_and_wait(resumed_agent, "third");

    const auto requests = resumed_provider->requests_snapshot();
    REQUIRE(requests.size() == 2);
    CHECK(std::ranges::count_if(
              requests[0].messages,
              core::agent::is_repository_context_message) == 1);
    CHECK(std::ranges::count_if(
              requests[1].messages,
              core::agent::is_repository_context_message) == 2);
    CHECK(std::ranges::count_if(
              requests[1].messages,
              [](const core::llm::Message& message) {
                  return core::agent::is_repository_context_message(message)
                      && !message.input_text.empty();
              }) == 1);
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
    CHECK(system_prompt.find("workspace_marker.txt") == std::string::npos);
    const auto context_message = std::ranges::find_if(
        requests[0].messages,
        [](const core::llm::Message& message) {
            return core::agent::is_repository_context_message(message);
        });
    REQUIRE(context_message != requests[0].messages.end());
    CHECK(context_message->content.find("workspace_marker.txt") != std::string::npos);
    CHECK(context_message->content.find("cwd_marker.txt") == std::string::npos);
}

TEST_CASE("Agent loads FILO steering files into the system prompt",
          "[agent][prompt][steering]") {
    const auto base = std::filesystem::temp_directory_path()
        / std::format("filo_agent_steering_{}",
                      std::chrono::steady_clock::now().time_since_epoch().count());
    TempDir workspace_dir(base / "workspace");

    {
        std::ofstream filo_md(workspace_dir.path() / "FILO.md");
        filo_md << "Project steering from FILO.md\n";
    }
    {
        std::ofstream agents_md(workspace_dir.path() / "AGENTS.md");
        agents_md << "Repository rules from AGENTS.md\n";
    }
    {
        std::filesystem::create_directories(workspace_dir.path() / ".filo" / "steering");
        std::ofstream steering_md(workspace_dir.path() / ".filo" / "steering" / "backend.md");
        steering_md << "Backend steering document\n";
    }

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
        "agent-steering-session");
    auto agent = std::make_shared<core::agent::Agent>(provider, tool_manager, session_context);

    send_and_wait(agent, "hello");

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 1);
    REQUIRE_FALSE(requests[0].messages.empty());

    const auto& system_prompt = requests[0].messages[0].content;
    CHECK(system_prompt.find("[Project Steering]") != std::string::npos);
    CHECK(system_prompt.find("FILO.md") != std::string::npos);
    CHECK(system_prompt.find("AGENTS.md") != std::string::npos);
    CHECK(system_prompt.find("backend.md") != std::string::npos);
    CHECK(system_prompt.find("Project steering from FILO.md") != std::string::npos);
}

TEST_CASE("Agent loads AGENTS.md hierarchically from repo root to workspace root",
          "[agent][prompt][steering]") {
    const auto base = std::filesystem::temp_directory_path()
        / std::format("filo_agent_hierarchical_{}",
                      std::chrono::steady_clock::now().time_since_epoch().count());
    TempDir repo_dir(base / "repo");
    const auto workspace_dir = repo_dir.path() / "workspace" / "crate_a";
    std::filesystem::create_directories(workspace_dir);

    {
        std::filesystem::create_directories(repo_dir.path() / ".git");
    }
    {
        std::ofstream root_agents(repo_dir.path() / "AGENTS.md");
        root_agents << "Repository-wide rules\n";
    }
    {
        std::ofstream nested_agents(workspace_dir / "AGENTS.md");
        nested_agents << "Crate-specific rules\n";
    }

    auto provider = std::make_shared<CapturingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    const auto session_context = test_support::make_session_context(
        core::workspace::WorkspaceSnapshot{
            .primary = workspace_dir,
            .additional = {},
            .enforce = true,
            .version = 1,
        },
        core::context::SessionTransport::cli,
        "agent-hierarchical-steering-session");
    auto agent = std::make_shared<core::agent::Agent>(provider, tool_manager, session_context);

    send_and_wait(agent, "hello");

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 1);
    REQUIRE_FALSE(requests[0].messages.empty());

    const auto& system_prompt = requests[0].messages[0].content;
    const auto root_pos = system_prompt.find("Repository-wide rules");
    const auto nested_pos = system_prompt.find("Crate-specific rules");

    CHECK(root_pos != std::string::npos);
    CHECK(nested_pos != std::string::npos);
    CHECK(root_pos < nested_pos);
    CHECK(system_prompt.find("workspace/crate_a/AGENTS.md") != std::string::npos);
}

TEST_CASE("Agent enforces per-turn model and tool constraints",
          "[agent][turn][constraints]") {
    auto provider = std::make_shared<CapturingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<NamedNoopTool>("allowed_tool"));
    tool_manager.register_tool(std::make_shared<NamedNoopTool>("blocked_tool"));

    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    send_and_wait(
        agent,
        "use constrained tools",
        core::agent::Agent::TurnCallbacks{
            .model_override = "skill-model",
            .allowed_tools = {"allowed_tool"},
        });

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 1);
    CHECK(requests[0].model == "skill-model");

    std::vector<std::string> tool_names;
    tool_names.reserve(requests[0].tools.size());
    for (const auto& tool : requests[0].tools) {
        tool_names.push_back(tool.function.name);
    }

    CHECK(std::ranges::find(tool_names, std::string("allowed_tool")) != tool_names.end());
    CHECK(std::ranges::find(tool_names, std::string("blocked_tool")) == tool_names.end());
    CHECK(std::ranges::find(tool_names, std::string("task")) == tool_names.end());
}

TEST_CASE("Agent can rotate transparently between tool-loop steps", "[agent][loop][rotation]") {
    auto provider = std::make_shared<RotatingToolLoopProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<NoopLoopTool>());

    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    int rotations = 0;
    bool saw_rotate_decision = false;
    agent->set_efficiency_decision_fn(
        [agent, &rotations, &saw_rotate_decision](const core::session::SessionEfficiencyDecision& decision) {
            ++rotations;
            saw_rotate_decision =
                decision.action == core::session::SessionEfficiencyDecision::Action::Rotate;
            agent->compact_history("Carry forward the loop rotation state.");
        });

    send_and_wait(agent, std::string(200'000, 'x'));

    CHECK(rotations == 1);
    CHECK(saw_rotate_decision);
    CHECK(provider->call_count() == 11);
    CHECK(provider->reset_count() == 1);

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 11);

    bool third_request_has_user = false;
    for (const auto& message : requests.back().messages) {
        if (message.role == "user") {
            third_request_has_user = true;
            break;
        }
    }

    CHECK_FALSE(third_request_has_user);
}

TEST_CASE("Agent can gate efficiency rotation by minimum context utilization",
          "[agent][loop][rotation]") {
    auto provider = std::make_shared<RotatingToolLoopProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<NoopLoopTool>());

    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    int rotations = 0;
    agent->set_efficiency_decision_fn(
        [agent, &rotations](const core::session::SessionEfficiencyDecision&) {
            ++rotations;
            agent->compact_history("Rotation should be suppressed for this turn.");
        });

    send_and_wait(
        agent,
        std::string(200'000, 'x'),
        core::agent::Agent::TurnCallbacks{
            .min_context_utilization_for_rotation = 0.90,
        });

    CHECK(rotations == 0);
    CHECK(provider->reset_count() == 0);
    CHECK(provider->call_count() == 11);
}
