#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/SubagentOrchestrator.hpp"
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
    tool_manager.register_tool(std::make_shared<core::tools::ShellTool>());

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

TEST_CASE("SubagentOrchestrator routes through provider override when configured", "[agent][orchestration]") {
    auto parent_provider = std::make_shared<RecordingProvider>();
    auto worker_provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();

    constexpr std::string_view kWorkerAlias = "test-subagent-worker-provider";
    core::llm::ProviderManager::get_instance().register_provider(std::string(kWorkerAlias), worker_provider);

    core::config::AppConfig config;
    core::config::SubagentConfig general;
    general.provider = std::string(kWorkerAlias);
    config.subagents["general"] = std::move(general);

    core::agent::SubagentOrchestrator orchestrator(tool_manager, &config);
    const auto session_context = test_support::make_workspace_session_context();

    const auto result = orchestrator.execute_task(
        R"({"description":"provider override","prompt":"run task","subagent_type":"general"})",
        parent_provider,
        {
            .active_model = "parent-model",
            .parent_mode = "BUILD",
            .session_context = session_context,
            .permission_check = {},
        });

    REQUIRE_FALSE(result.contains("\"error\""));
    REQUIRE(parent_provider->requests_snapshot().empty());
    REQUIRE(worker_provider->requests_snapshot().size() == 1);
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
    tool_manager.register_tool(std::make_shared<core::tools::ShellTool>());

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
