#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/SubagentOrchestrator.hpp"
#include "core/agent/DelegatedAgentRunner.hpp"
#include "core/tools/TaskTool.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/ProviderManager.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "core/tools/ReadTool.hpp"
#include "core/tools/FileSearchTool.hpp"
#include "core/tools/GrepSearchTool.hpp"
#include "core/tools/ListDirectoryTool.hpp"
#include "core/tools/ShellTool.hpp"
#include "TestSessionContext.hpp"

#include <simdjson.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <future>
#include <format>
#include <mutex>
#include <ranges>
#include <string>
#include <unordered_set>
#include <vector>

namespace {

class RecordingProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest& request,
        std::function<void(const core::llm::StreamChunk&)> callback) override {

        int call_index = 0;
        {
            std::lock_guard lock(mutex_);
            requests_.push_back(request);
            call_index = ++calls_;
        }

        // Return content and then signal completion
        core::llm::StreamChunk chunk;
        chunk.content = std::format("delegated-response-{}", call_index);
        chunk.is_final = true;
        callback(chunk);
    }

    [[nodiscard]] std::vector<core::llm::ChatRequest> requests_snapshot() const {
        std::lock_guard lock(mutex_);
        return requests_;
    }

private:
    mutable std::mutex mutex_;
    int calls_ = 0;
    std::vector<core::llm::ChatRequest> requests_;
};

class BudgetProvider final : public core::llm::LLMProvider {
public:
    std::vector<core::llm::ChatRequest> requests;
    bool ignore_tool_policy = false;

    void stream_response(
        const core::llm::ChatRequest& request,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        requests.push_back(request);
        core::llm::StreamChunk chunk;
        if (request.tools.empty() && !ignore_tool_policy) {
            chunk.content = "partial: found relevant files; further investigation needed";
        } else if (std::ranges::any_of(request.messages, [](const auto& message) {
                       return message.role == "assistant" && message.content.starts_with("partial:");
                   })) {
            chunk.content = "done: resumed with previous findings";
        } else {
            chunk.tools.push_back(core::llm::ToolCall{
                .index = 0,
                .id = std::format("budget_call_{}", requests.size()),
                .function = {.name = "list_directory", .arguments = R"({"path":"."})"},
            });
        }
        chunk.is_final = true;
        callback(chunk);
    }
};

class ControlledProvider final : public core::llm::LLMProvider {
public:
    explicit ControlledProvider(int block_call = 1) : block_call_(block_call) {}

    void stream_response(const core::llm::ChatRequest& request,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        int call = 0;
        {
            std::unique_lock lock(mutex_);
            requests_.push_back(request);
            call = static_cast<int>(requests_.size());
            cv_.notify_all();
            if (call == block_call_) cv_.wait(lock, [&] { return released_; });
        }
        callback(core::llm::StreamChunk::make_content(std::format("controlled-response-{}", call)));
        callback(core::llm::StreamChunk::make_final());
    }

    bool wait_for_call(int count) {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(3), [&] {
            return requests_.size() >= static_cast<std::size_t>(count);
        });
    }
    void release() {
        std::lock_guard lock(mutex_);
        released_ = true;
        cv_.notify_all();
    }
    std::vector<core::llm::ChatRequest> requests() const {
        std::lock_guard lock(mutex_);
        return requests_;
    }

private:
    int block_call_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool released_ = false;
    std::vector<core::llm::ChatRequest> requests_;
};

class ScriptedProvider final : public core::llm::LLMProvider {
public:
    std::vector<core::llm::StreamChunk> responses;
    std::size_t calls = 0;
    void stream_response(const core::llm::ChatRequest&,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        if (calls >= responses.size()) {
            callback(core::llm::StreamChunk::make_error("Unexpected extra model request"));
            return;
        }
        callback(responses[calls++]);
    }
};

std::string extract_task_id(const std::string& payload) {
    simdjson::dom::parser parser;
    simdjson::padded_string padded(payload);
    simdjson::dom::element doc;
    if (parser.parse(padded).get(doc) != simdjson::SUCCESS) return {};

    std::string_view task_id;
    if (doc["task_id"].get(task_id) != simdjson::SUCCESS) return {};
    return std::string(task_id);
}

std::unordered_set<std::string> tool_names(const core::llm::ChatRequest& request) {
    std::unordered_set<std::string> names;
    for (const auto& tool : request.tools) {
        names.insert(tool.function.name);
    }
    return names;
}

std::size_t visible_message_count(const core::llm::ChatRequest& request) {
    return static_cast<std::size_t>(std::ranges::count_if(
        request.messages,
        [](const core::llm::Message& message) { return !message.synthetic; }));
}

} // namespace

TEST_CASE("Bounded subagents summarize evidence and resume through task with a larger budget",
          "[agent][orchestration][budget]") {
    auto provider = std::make_shared<BudgetProvider>();
    auto& tools = core::tools::ToolManager::get_instance();
    tools.register_tool(std::make_shared<core::tools::ListDirectoryTool>());
    core::config::AppConfig config;
    config.subagents["explore"].max_steps = 1;
    core::agent::SubagentOrchestrator orchestrator(tools, &config);
    const auto context = test_support::make_workspace_session_context();
    const core::agent::SubagentOrchestrator::RunContext run_context{
        .active_model = "test-model",
        .parent_mode = "BUILD",
        .session_context = context,
        .permission_check = [](const auto&, const auto&) { return true; },
    };
    const auto first = orchestrator.execute_task(
        R"({"description":"investigate files","prompt":"inspect the repository","subagent_type":"explore","max_steps":3})",
        provider, run_context);
    INFO(first);
    REQUIRE_THAT(first, Catch::Matchers::ContainsSubstring("\"status\":\"partial\""));
    REQUIRE_THAT(first, Catch::Matchers::ContainsSubstring("\"steps\":3"));
    REQUIRE_THAT(first, Catch::Matchers::ContainsSubstring("\"tool_calls\":2"));
    REQUIRE_THAT(first, Catch::Matchers::ContainsSubstring("found relevant files"));
    REQUIRE_THAT(first, Catch::Matchers::ContainsSubstring("resume_hint"));
    REQUIRE(provider->requests.size() == 3);
    CHECK_THAT(provider->requests.front().messages.back().content,
               Catch::Matchers::ContainsSubstring("Budget: 3 model steps"));
    CHECK(provider->requests.back().tools.empty());
    CHECK(std::ranges::count_if(provider->requests.back().messages, [](const auto& message) {
        return message.role == "tool";
    }) == 2);
    CHECK_THAT(provider->requests.back().messages.back().content,
               Catch::Matchers::ContainsSubstring("final model step"));

    const auto task_id = extract_task_id(first);
    REQUIRE_FALSE(task_id.empty());
    const auto resumed = orchestrator.execute_task(std::format(
        R"({{"description":"investigate files","prompt":"finish using earlier findings","subagent_type":"explore","task_id":"{}","max_steps":5}})",
        task_id), provider, run_context);
    INFO(resumed);
    CHECK_THAT(resumed, Catch::Matchers::ContainsSubstring("\"status\":\"completed\""));
    CHECK_THAT(resumed, Catch::Matchers::ContainsSubstring("\"max_steps\":5"));
    CHECK(extract_task_id(resumed) == task_id);
    REQUIRE(provider->requests.size() == 4);
    CHECK_FALSE(provider->requests.back().tools.empty());
    CHECK(std::ranges::none_of(provider->requests.back().messages, [](const auto& message) {
        return message.content.contains("This is your final model step");
    }));
}

TEST_CASE("Subagent task rejects invalid budgets before calling the provider",
          "[agent][orchestration][budget]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tools = core::tools::ToolManager::get_instance();
    core::agent::SubagentOrchestrator orchestrator(tools);
    const auto context = test_support::make_workspace_session_context();
    for (const auto value : {"0", "-1", "65", "4294967297", "1.5", "\"invalid\""}) {
        const auto result = orchestrator.execute_task(std::format(
            R"({{"description":"bad budget","prompt":"inspect files","subagent_type":"explore","max_steps":{}}})",
            value), provider, {.parent_mode = "BUILD", .session_context = context});
        INFO(value);
        CHECK_THAT(result, Catch::Matchers::ContainsSubstring("max_steps must be an integer"));
    }
    CHECK(provider->requests_snapshot().empty());
}

TEST_CASE("Subagent resumes retain their chosen budget and reject foreign ids",
          "[agent][orchestration][budget]") {
    auto provider = std::make_shared<RecordingProvider>();
    core::agent::SubagentOrchestrator orchestrator(core::tools::ToolManager::get_instance());
    const auto context = test_support::make_workspace_session_context();
    const core::agent::SubagentOrchestrator::RunContext run_context{
        .parent_mode = "BUILD", .session_context = context};
    const auto first = orchestrator.execute_task(
        R"({"description":"inspect","prompt":"inspect files","subagent_type":"explore","max_steps":32})",
        provider, run_context);
    const auto id = extract_task_id(first);
    REQUIRE_FALSE(id.empty());
    const auto resumed = orchestrator.execute_task(std::format(
        R"({{"description":"inspect","prompt":"continue","subagent_type":"explore","task_id":"{}"}})", id),
        provider, run_context);
    CHECK_THAT(resumed, Catch::Matchers::ContainsSubstring(R"("max_steps":32)"));
    for (const auto foreign_id : {"work_0123abcd", "task_typo", "unknown-id"}) {
        const auto response = orchestrator.execute_task(std::format(
            R"({{"description":"inspect","prompt":"continue","subagent_type":"explore","task_id":"{}"}})", foreign_id),
            provider, run_context);
        INFO(response);
        CHECK_THAT(response, Catch::Matchers::ContainsSubstring("\"error\""));
    }
    CHECK(provider->requests_snapshot().size() == 2);
}

TEST_CASE("Cancelled queued subagents never start a provider request",
          "[agent][orchestration][cancellation]") {
    auto provider = std::make_shared<ControlledProvider>(2);
    core::agent::SubagentOrchestrator orchestrator(core::tools::ToolManager::get_instance());
    const auto context = test_support::make_workspace_session_context();
    const core::agent::SubagentOrchestrator::RunContext run_context{
        .parent_mode = "BUILD", .session_context = context};
    const auto first = orchestrator.execute_task(
        R"({"description":"inspect","prompt":"inspect files","subagent_type":"explore"})", provider, run_context);
    const auto args = std::format(
        R"({{"description":"inspect","prompt":"continue","subagent_type":"explore","task_id":"{}"}})", extract_task_id(first));
    auto owner = std::async(std::launch::async, [&] { return orchestrator.execute_task(args, provider, run_context); });
    const bool owner_started = provider->wait_for_call(2);
    std::atomic<bool> cancel{false};
    std::atomic<int> cancellation_checks{0};
    std::promise<void> waiting;
    auto waiting_future = waiting.get_future();
    auto queued_context = run_context;
    queued_context.cancellation_requested = [&] {
        if (cancellation_checks.fetch_add(1) == 2) waiting.set_value();
        return cancel.load();
    };
    std::string queued_args = args;
    SECTION("same mutable task") {}
    SECTION("different task waiting for a serial provider") {
        queued_args = R"({"description":"other","prompt":"inspect other files","subagent_type":"explore"})";
    }
    auto queued = std::async(std::launch::async, [&] {
        return orchestrator.execute_task(queued_args, provider, queued_context);
    });
    const bool did_wait = waiting_future.wait_for(std::chrono::seconds(2)) == std::future_status::ready;
    cancel.store(true);
    const bool cancelled_promptly = queued.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    provider->release();
    const auto queued_result = queued.get();
    const auto owner_result = owner.get();
    INFO(queued_result);
    CHECK(owner_started);
    CHECK(did_wait);
    CHECK(cancelled_promptly);
    CHECK_THAT(queued_result, Catch::Matchers::ContainsSubstring("cancelled while waiting"));
    CHECK(provider->requests().size() == 2);
}

TEST_CASE("Duplicate subagent resumes serialize and observe the latest history",
          "[agent][orchestration][concurrency]") {
    auto provider = std::make_shared<ControlledProvider>(2);
    core::agent::SubagentOrchestrator orchestrator(core::tools::ToolManager::get_instance());
    const auto context = test_support::make_workspace_session_context();
    const core::agent::SubagentOrchestrator::RunContext run_context{
        .parent_mode = "BUILD", .session_context = context};
    const auto first = orchestrator.execute_task(
        R"({"description":"inspect","prompt":"inspect files","subagent_type":"explore"})", provider, run_context);
    const auto args = std::format(
        R"({{"description":"inspect","prompt":"continue","subagent_type":"explore","task_id":"{}"}})", extract_task_id(first));
    auto owner = std::async(std::launch::async, [&] { return orchestrator.execute_task(args, provider, run_context); });
    const bool owner_started = provider->wait_for_call(2);
    auto queued = std::async(std::launch::async, [&] { return orchestrator.execute_task(args, provider, run_context); });
    const bool serialized = queued.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout;
    provider->release();
    const auto owner_result = owner.get();
    const auto queued_result = queued.get();
    CHECK(owner_started);
    CHECK(serialized);
    const auto requests = provider->requests();
    REQUIRE(requests.size() == 3);
    CHECK(std::ranges::any_of(requests.back().messages, [](const auto& message) {
        return message.role == "assistant" && message.content == "controlled-response-2";
    }));
}

TEST_CASE("Worker timeout closes events and exposes provider settlement",
          "[agent][orchestration][timeout]") {
    auto provider = std::make_shared<ControlledProvider>();
    std::mutex events_mutex;
    std::vector<core::agent::SubagentEvent> events;
    auto lease = std::make_shared<int>(1);
    std::weak_ptr<int> weak_lease = lease;
    const auto result = core::agent::DelegatedAgentRunner::run({
        .provider = provider,
        .tool_manager = core::tools::ToolManager::get_instance(),
        .session_context = test_support::make_workspace_session_context(),
        .max_steps = 3,
        .prompt = "inspect files",
        .timeout = std::chrono::milliseconds(200),
        .on_event = [&](const auto& event) { std::lock_guard lock(events_mutex); events.push_back(event); },
        .execution_lease = std::move(lease),
    });
    const bool pending = result.settled.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout;
    const bool lease_held_until_settlement = !weak_lease.expired();
    std::size_t terminal_count = 0;
    {
        std::lock_guard lock(events_mutex);
        terminal_count = events.size();
    }
    provider->release();
    const auto settled = result.settled.wait_for(std::chrono::seconds(3));
    CHECK(result.timed_out());
    CHECK(result.status() == "failed");
    CHECK(pending);
    CHECK(lease_held_until_settlement);
    CHECK(settled == std::future_status::ready);
    CHECK(weak_lease.expired());
    std::lock_guard lock(events_mutex);
    REQUIRE_FALSE(events.empty());
    CHECK(events.back().kind == core::agent::SubagentEvent::Kind::Failed);
    CHECK(events.size() == terminal_count);
}

TEST_CASE("Worker cancellation before admission survives the turn's stop reset",
          "[agent][orchestration][cancellation]") {
    auto provider = std::make_shared<RecordingProvider>();
    std::atomic<bool> cancel{false};
    const auto result = core::agent::DelegatedAgentRunner::run({
        .provider = provider,
        .tool_manager = core::tools::ToolManager::get_instance(),
        .session_context = test_support::make_workspace_session_context(),
        .max_steps = 3,
        .prompt = "inspect files",
        .on_agent_ready = [&](const auto&) { cancel.store(true); },
        .cancellation_requested = [&] { return cancel.load(); },
    });
    CHECK(result.cancelled());
    CHECK(provider->requests_snapshot().empty());
}

TEST_CASE("Final handoffs reject unsolicited tool calls", "[agent][orchestration][budget]") {
    auto provider = std::make_shared<BudgetProvider>();
    provider->ignore_tool_policy = true;
    auto& tools = core::tools::ToolManager::get_instance();
    tools.register_tool(std::make_shared<core::tools::ListDirectoryTool>());
    int permission_checks = 0;
    const auto result = core::agent::DelegatedAgentRunner::run({
        .provider = provider, .tool_manager = tools,
        .session_context = test_support::make_workspace_session_context(),
        .max_steps = 1, .prompt = "inspect files",
        .permission_check = [&](auto, auto) { ++permission_checks; return true; },
    });
    CHECK(result.status() == "partial");
    CHECK(result.reason() == "step_limit");
    CHECK(permission_checks == 0);
    CHECK(result.tool_calls_total == 1);
    CHECK(result.tool_calls_success == 0);
}

TEST_CASE("Interrupted resumes cannot recycle an earlier final answer as completion",
          "[agent][orchestration][resume]") {
    auto provider = std::make_shared<ScriptedProvider>();
    auto first_response = core::llm::StreamChunk::make_content("earlier final answer");
    first_response.is_final = true;
    core::llm::StreamChunk tool_response;
    tool_response.is_final = true;
    tool_response.tools.push_back(core::llm::ToolCall{
        .index = 0, .id = "denied-write", .function = {.name = "write_file", .arguments = R"({"file_path":"never-write.txt","content":"must be denied"})"}});
    provider->responses = {first_response, tool_response};
    auto& tools = core::tools::ToolManager::get_instance();
    tools.register_tool(std::make_shared<core::tools::WriteFileTool>());
    const auto context = test_support::make_workspace_session_context();
    const auto first = core::agent::DelegatedAgentRunner::run({
        .provider = provider, .tool_manager = tools, .session_context = context,
        .max_steps = 5, .prompt = "inspect files"});
    REQUIRE(first.status() == "completed");
    const auto resumed = core::agent::DelegatedAgentRunner::run({
        .provider = provider, .tool_manager = tools, .session_context = context,
        .max_steps = 5, .prompt = "inspect more files",
        .resume_state = core::agent::DelegatedAgentRunner::ResumeState{.messages = first.history},
        .permission_check = [](auto, auto) { return false; },
    });
    INFO(resumed.final_text);
    CHECK(resumed.status() == "partial");
    CHECK(resumed.reason() == "interrupted");
    CHECK_FALSE(resumed.final_text.contains("earlier final answer"));
    CHECK(provider->calls == 2);
}

TEST_CASE("SubagentOrchestrator executes delegated tasks and returns a task_id", "[agent][orchestration]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    core::agent::SubagentOrchestrator orchestrator(tool_manager);
    const auto session_context = test_support::make_workspace_session_context();
    std::vector<core::agent::SubagentEvent> events;

    const auto result = orchestrator.execute_task(
        R"({"description":"investigate startup","prompt":"inspect startup sequence","subagent_type":"general"})",
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
            .parent_tool_call_id = "parent-task-call",
            .on_subagent_event = [&](const core::agent::SubagentEvent& event) {
                events.push_back(event);
            },
        });

    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"task_id\":\"task_"));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"subagent_type\":\"general\""));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("delegated-response-1"));
    REQUIRE_FALSE(result.contains("\"error\""));
    REQUIRE_FALSE(events.empty());
    REQUIRE(events.front().kind == core::agent::SubagentEvent::Kind::Started);
    REQUIRE(events.front().parent_tool_call_id == "parent-task-call");
    REQUIRE(events.front().worker_name == "general");
    REQUIRE(events.front().description == "investigate startup");
    REQUIRE(events.back().kind == core::agent::SubagentEvent::Kind::Finished);
    REQUIRE(events.back().summary == "delegated-response-1");
}

TEST_CASE("SubagentOrchestrator forwards profile response schemas to the provider", "[agent][orchestration][structured]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    core::config::AppConfig config;
    core::config::SubagentConfig extractor;
    extractor.response_format = core::llm::ResponseFormat{
        .type = core::llm::ResponseFormat::Type::JsonSchema,
        .schema = R"({"type":"object","properties":{"summary":{"type":"string"}},"required":["summary"],"additionalProperties":false})",
    };
    config.subagents.emplace("extractor", std::move(extractor));
    core::agent::SubagentOrchestrator orchestrator(tool_manager, &config);
    const auto session_context = test_support::make_workspace_session_context();

    const auto result = orchestrator.execute_task(
        R"({"description":"extract result","prompt":"return the result","subagent_type":"extractor"})",
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
        });

    REQUIRE_FALSE(result.contains("\"error\""));
    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 1);
    CHECK(requests[0].response_format.type == core::llm::ResponseFormat::Type::JsonSchema);
    CHECK(requests[0].response_format.schema.contains("summary"));
}

TEST_CASE("SubagentOrchestrator resumes an existing task_id and keeps history", "[agent][orchestration]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    core::agent::SubagentOrchestrator orchestrator(tool_manager);
    const auto session_context = test_support::make_workspace_session_context();

    const auto first = orchestrator.execute_task(
        R"({"description":"trace auth","prompt":"find auth flow","subagent_type":"general"})",
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
        });

    const std::string task_id = extract_task_id(first);
    REQUIRE_FALSE(task_id.empty());

    const auto second = orchestrator.execute_task(
        std::format(
            R"({{"description":"trace auth","prompt":"continue auth flow","subagent_type":"general","task_id":"{}"}})",
            task_id),
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
        });

    REQUIRE_THAT(second, Catch::Matchers::ContainsSubstring(task_id));
    REQUIRE_THAT(second, Catch::Matchers::ContainsSubstring("delegated-response-2"));

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 2);
    REQUIRE(visible_message_count(requests[0]) == 2);  // system + first delegated user prompt
    REQUIRE(visible_message_count(requests[1]) >= 4);  // system + previous turn + new user prompt

    bool found_previous_assistant_message = false;
    for (const auto& message : requests[1].messages) {
        if (message.role == "assistant"
            && message.content.find("delegated-response-1") != std::string::npos) {
            found_previous_assistant_message = true;
            break;
        }
    }
    REQUIRE(found_previous_assistant_message);
}

TEST_CASE("SubagentOrchestrator ignores a provider-invented UUID on an initial PLAN task",
          "[agent][orchestration][plan][grok]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    core::agent::SubagentOrchestrator orchestrator(tool_manager);
    const auto session_context = test_support::make_workspace_session_context();
    constexpr std::string_view kProviderName = "grok-task-id-regression";
    core::llm::ProviderManager::get_instance().register_provider(
        std::string(kProviderName), provider);

    const auto result = orchestrator.execute_task(
        R"({"description":"explore footer","prompt":"find footer rendering","subagent_type":"explore","task_id":"a6e1cefb-4395-4683-9894-9110dc24407d"})",
        provider,
        {
            .active_provider_name = std::string(kProviderName),
            .active_model = "grok-4.5",
            .parent_mode = "PLAN",
            .session_context = session_context,
            .permission_check = {},
        });

    REQUIRE_FALSE(result.contains("\"error\""));
    const std::string task_id = extract_task_id(result);
    REQUIRE(task_id.starts_with("task_"));
    REQUIRE(task_id != "a6e1cefb-4395-4683-9894-9110dc24407d");
    REQUIRE(provider->requests_snapshot().size() == 1);
}

TEST_CASE("SubagentOrchestrator still rejects an unknown Filo task_id",
          "[agent][orchestration]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    core::agent::SubagentOrchestrator orchestrator(tool_manager);
    const auto session_context = test_support::make_workspace_session_context();

    const auto result = orchestrator.execute_task(
        R"({"description":"resume footer","prompt":"continue footer rendering","subagent_type":"explore","task_id":"task_deadbeef"})",
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "PLAN",
            .session_context = session_context,
            .permission_check = {},
        });

    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("Task session 'task_deadbeef' was not found"));
    REQUIRE(provider->requests_snapshot().empty());
}

TEST_CASE("SubagentOrchestrator task schema says task_id is output-derived",
          "[agent][orchestration][schema]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    core::agent::SubagentOrchestrator orchestrator(tool_manager);

    const auto definition = orchestrator.task_tool_definition();
    const auto parameter = std::ranges::find_if(
        definition.function.parameters,
        [](const core::tools::ToolParameter& value) { return value.name == "task_id"; });

    REQUIRE(parameter != definition.function.parameters.end());
    CHECK_FALSE(parameter->required);
    CHECK_THAT(parameter->description, Catch::Matchers::ContainsSubstring("Omit this field"));
    CHECK_THAT(parameter->description, Catch::Matchers::ContainsSubstring("never invent"));
    CHECK_THAT(definition.function.description, Catch::Matchers::ContainsSubstring("never invent"));
}

TEST_CASE("SubagentOrchestrator explore profile enforces read-only tool filtering", "[agent][orchestration]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();

    tool_manager.register_tool(std::make_shared<core::tools::WriteFileTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ReadTool>());
    tool_manager.register_tool(std::make_shared<core::tools::FileSearchTool>());
    tool_manager.register_tool(std::make_shared<core::tools::GrepSearchTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ListDirectoryTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ShellTool>(tool_manager.shell_sessions()));

    core::agent::SubagentOrchestrator orchestrator(tool_manager);
    const auto session_context = test_support::make_workspace_session_context();

    const auto result = orchestrator.execute_task(
        R"({"description":"scan repo","prompt":"find route handlers","subagent_type":"explore"})",
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
        });

    REQUIRE_FALSE(result.contains("\"error\""));

    const auto requests = provider->requests_snapshot();
    REQUIRE_FALSE(requests.empty());

    const auto names = tool_names(requests.front());
    REQUIRE(names.contains("read"));
    REQUIRE(names.contains("file_search"));
    REQUIRE(names.contains("grep_search"));
    REQUIRE(names.contains("list_directory"));

    REQUIRE_FALSE(names.contains("write_file"));
    REQUIRE_FALSE(names.contains("run_terminal_command"));
    REQUIRE_FALSE(names.contains("task"));
    REQUIRE(names.contains("read_tool_result"));
    REQUIRE(orchestrator.task_is_read_only(
        R"({"description":"scan repo","prompt":"find routes","subagent_type":"explore"})",
        "BUILD"));
    REQUIRE_FALSE(orchestrator.task_is_read_only(
        R"({"description":"change repo","prompt":"make changes","subagent_type":"general"})",
        "BUILD"));
}

TEST_CASE("SubagentOrchestrator does not spawn the internal reader profile", "[agent][orchestration][read]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    core::config::AppConfig config;
    core::config::SubagentConfig reader;
    reader.provider = "local-reader";
    reader.model = "test-reader";
    reader.enabled = true;
    config.subagents.emplace("reader", std::move(reader));
    core::agent::SubagentOrchestrator orchestrator(tool_manager, &config);
    const auto session_context = test_support::make_workspace_session_context();

    const auto result = orchestrator.execute_task(
        R"({"description":"read files","prompt":"summarize","subagent_type":"reader"})",
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
        });

    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"error\""));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("Unknown subagent_type"));
    REQUIRE(provider->requests_snapshot().empty());
}

TEST_CASE("SubagentOrchestrator rejects unknown subagent_type", "[agent][orchestration]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    core::agent::SubagentOrchestrator orchestrator(tool_manager);
    const auto session_context = test_support::make_workspace_session_context();

    const auto result = orchestrator.execute_task(
        R"({"description":"bad","prompt":"noop","subagent_type":"unknown_worker"})",
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
        });

    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"error\""));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("Unknown subagent_type"));
}

TEST_CASE("SubagentOrchestrator applies model override from subagent profile config", "[agent][orchestration]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();

    core::config::AppConfig config;
    core::config::SubagentConfig general;
    general.model = "model-from-subagent-profile";
    config.subagents["general"] = std::move(general);

    core::agent::SubagentOrchestrator orchestrator(tool_manager, &config);
    const auto session_context = test_support::make_workspace_session_context();

    const auto result = orchestrator.execute_task(
        R"({"description":"profile model","prompt":"run task","subagent_type":"general"})",
        provider,
        {
            .active_model = "parent-model",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
        });

    REQUIRE_FALSE(result.contains("\"error\""));

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 1);
    REQUIRE(requests.front().model == "model-from-subagent-profile");
}

TEST_CASE("SubagentOrchestrator sends profile prompt to delegated worker", "[agent][orchestration]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();

    core::config::AppConfig config;
    core::config::SubagentConfig general;
    general.prompt = "Use the custom worker rubric before answering.";
    config.subagents["general"] = std::move(general);

    core::agent::SubagentOrchestrator orchestrator(tool_manager, &config);
    const auto session_context = test_support::make_workspace_session_context();

    const auto result = orchestrator.execute_task(
        R"({"description":"profile prompt","prompt":"inspect the queue","subagent_type":"general"})",
        provider,
        {
            .active_model = "parent-model",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
        });

    REQUIRE_FALSE(result.contains("\"error\""));

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 1);
    REQUIRE_FALSE(requests.front().messages.empty());
    const auto& user_message = requests.front().messages.back().content;
    REQUIRE_THAT(user_message, Catch::Matchers::ContainsSubstring("Use the custom worker rubric before answering."));
    REQUIRE_THAT(user_message, Catch::Matchers::ContainsSubstring("inspect the queue"));
}

TEST_CASE("SubagentOrchestrator reloads profile overrides live", "[agent][orchestration]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();

    core::config::AppConfig initial_config;
    core::config::SubagentConfig initial_general;
    initial_general.model = "model-from-initial-profile";
    initial_config.subagents["general"] = std::move(initial_general);

    core::agent::SubagentOrchestrator orchestrator(tool_manager, &initial_config);
    const auto session_context = test_support::make_workspace_session_context();

    const auto first = orchestrator.execute_task(
        R"({"description":"reload profile","prompt":"first run","subagent_type":"general"})",
        provider,
        {
            .active_model = "parent-model",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
        });
    REQUIRE_FALSE(first.contains("\"error\""));

    core::config::AppConfig reloaded_config;
    core::config::SubagentConfig reloaded_general;
    reloaded_general.model = "model-from-reloaded-profile";
    reloaded_config.subagents["general"] = std::move(reloaded_general);
    orchestrator.reload_profiles(reloaded_config);

    const auto second = orchestrator.execute_task(
        R"({"description":"reload profile","prompt":"second run","subagent_type":"general"})",
        provider,
        {
            .active_model = "parent-model",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
        });
    REQUIRE_FALSE(second.contains("\"error\""));

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 2);
    REQUIRE(requests[0].model == "model-from-initial-profile");
    REQUIRE(requests[1].model == "model-from-reloaded-profile");
}

TEST_CASE("SubagentOrchestrator routes a configured execution profile through its provider and model",
          "[agent][orchestration]") {
    auto parent_provider = std::make_shared<RecordingProvider>();
    auto worker_provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();

    constexpr std::string_view kWorkerAlias = "test-luna-explore-provider";
    core::llm::ProviderManager::get_instance().register_provider(std::string(kWorkerAlias), worker_provider);

    core::config::AppConfig config;
    core::config::SubagentConfig luna_explore;
    luna_explore.description = "Read-only GPT-5.6 Luna investigator.";
    luna_explore.provider = std::string(kWorkerAlias);
    luna_explore.model = "gpt-5.6-luna";
    config.subagents["luna-explore"] = std::move(luna_explore);

    core::agent::SubagentOrchestrator orchestrator(tool_manager, &config);
    const auto session_context = test_support::make_workspace_session_context();
    const auto definition = orchestrator.task_tool_definition();
    CHECK_THAT(definition.function.description,
               Catch::Matchers::ContainsSubstring("test-luna-explore-provider"));
    CHECK_THAT(definition.function.description,
               Catch::Matchers::ContainsSubstring("gpt-5.6-luna"));

    const auto result = orchestrator.execute_task(
        R"({"description":"provider override","prompt":"run task","subagent_type":"luna-explore"})",
        parent_provider,
        {
            .active_model = "qwen3-max",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
        });

    REQUIRE_FALSE(result.contains("\"error\""));
    REQUIRE(parent_provider->requests_snapshot().empty());
    const auto requests = worker_provider->requests_snapshot();
    REQUIRE(requests.size() == 1);
    CHECK(requests.front().model == "gpt-5.6-luna");
}

TEST_CASE("SubagentOrchestrator fails clearly when provider override is unavailable", "[agent][orchestration]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();

    core::config::AppConfig config;
    core::config::SubagentConfig general;
    general.provider = "provider-that-does-not-exist";
    config.subagents["general"] = std::move(general);

    core::agent::SubagentOrchestrator orchestrator(tool_manager, &config);
    const auto session_context = test_support::make_workspace_session_context();

    const auto result = orchestrator.execute_task(
        R"({"description":"missing provider","prompt":"run task","subagent_type":"general"})",
        provider,
        {
            .active_model = "parent-model",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
        });

    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"error\""));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("provider-that-does-not-exist"));
}

TEST_CASE("AUTO parent mode is not inherited by delegated workers",
          "[agent][orchestration][auto]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();

    tool_manager.register_tool(std::make_shared<core::tools::WriteFileTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ReadTool>());
    tool_manager.register_tool(std::make_shared<core::tools::FileSearchTool>());
    tool_manager.register_tool(std::make_shared<core::tools::GrepSearchTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ListDirectoryTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ShellTool>(tool_manager.shell_sessions()));

    core::agent::SubagentOrchestrator orchestrator(tool_manager);
    const auto session_context = test_support::make_workspace_session_context();

    const auto explore = orchestrator.execute_task(
        R"({"description":"scan repo","prompt":"find the scheduler","subagent_type":"explore"})",
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "AUTO",
            .session_context = session_context,
            .permission_check = {},
        });
    REQUIRE_FALSE(explore.contains("\"error\""));

    const auto general = orchestrator.execute_task(
        R"({"description":"implement fix","prompt":"apply the fix","subagent_type":"general"})",
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "AUTO",
            .session_context = session_context,
            .permission_check = {},
        });
    REQUIRE_FALSE(general.contains("\"error\""));

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() >= 2);
    const auto system_text = [](const core::llm::ChatRequest& request) {
        if (request.messages.empty()) return std::string{};
        return request.messages.front().content;
    };
    CHECK_THAT(system_text(requests[0]), Catch::Matchers::ContainsSubstring("RESEARCH"));
    CHECK_THAT(system_text(requests[0]), !Catch::Matchers::ContainsSubstring("AUTO"));
    CHECK_THAT(system_text(requests[1]), Catch::Matchers::ContainsSubstring("BUILD"));
    CHECK_THAT(system_text(requests[1]), !Catch::Matchers::ContainsSubstring("AUTO"));
}
