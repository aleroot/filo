#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "core/agent/SubagentOrchestrator.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/ProviderManager.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "core/tools/ReadFileTool.hpp"
#include "core/tools/FileSearchTool.hpp"
#include "core/tools/GrepSearchTool.hpp"
#include "core/tools/ListDirectoryTool.hpp"
#include "core/tools/ShellTool.hpp"

#include <simdjson.h>

#include <format>
#include <mutex>
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

} // namespace

TEST_CASE("SubagentOrchestrator executes delegated tasks and returns a task_id", "[agent][orchestration]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    core::agent::SubagentOrchestrator orchestrator(tool_manager);

    const auto result = orchestrator.execute_task(
        R"({"description":"investigate startup","prompt":"inspect startup sequence","subagent_type":"general"})",
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "BUILD",
            .permission_check = {},
        });

    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"task_id\":\"task_"));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"subagent_type\":\"general\""));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("delegated-response-1"));
    REQUIRE_FALSE(result.contains("\"error\""));
}

TEST_CASE("SubagentOrchestrator resumes an existing task_id and keeps history", "[agent][orchestration]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    core::agent::SubagentOrchestrator orchestrator(tool_manager);

    const auto first = orchestrator.execute_task(
        R"({"description":"trace auth","prompt":"find auth flow","subagent_type":"general"})",
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "BUILD",
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
            .permission_check = {},
        });

    REQUIRE_THAT(second, Catch::Matchers::ContainsSubstring(task_id));
    REQUIRE_THAT(second, Catch::Matchers::ContainsSubstring("delegated-response-2"));

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 2);
    REQUIRE(requests[0].messages.size() == 2);  // system + first delegated user prompt
    REQUIRE(requests[1].messages.size() >= 4);  // system + previous turn + new user prompt

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

TEST_CASE("SubagentOrchestrator explore profile enforces read-only tool filtering", "[agent][orchestration]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();

    tool_manager.register_tool(std::make_shared<core::tools::WriteFileTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ReadFileTool>());
    tool_manager.register_tool(std::make_shared<core::tools::FileSearchTool>());
    tool_manager.register_tool(std::make_shared<core::tools::GrepSearchTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ListDirectoryTool>());
    tool_manager.register_tool(std::make_shared<core::tools::ShellTool>());

    core::agent::SubagentOrchestrator orchestrator(tool_manager);

    const auto result = orchestrator.execute_task(
        R"({"description":"scan repo","prompt":"find route handlers","subagent_type":"explore"})",
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "BUILD",
            .permission_check = {},
        });

    REQUIRE_FALSE(result.contains("\"error\""));

    const auto requests = provider->requests_snapshot();
    REQUIRE_FALSE(requests.empty());

    const auto names = tool_names(requests.front());
    REQUIRE(names.contains("read_file"));
    REQUIRE(names.contains("file_search"));
    REQUIRE(names.contains("grep_search"));
    REQUIRE(names.contains("list_directory"));

    REQUIRE_FALSE(names.contains("write_file"));
    REQUIRE_FALSE(names.contains("run_terminal_command"));
    REQUIRE_FALSE(names.contains("task"));
}

TEST_CASE("SubagentOrchestrator rejects unknown subagent_type", "[agent][orchestration]") {
    auto provider = std::make_shared<RecordingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    core::agent::SubagentOrchestrator orchestrator(tool_manager);

    const auto result = orchestrator.execute_task(
        R"({"description":"bad","prompt":"noop","subagent_type":"unknown_worker"})",
        provider,
        {
            .active_model = "gpt-4o",
            .parent_mode = "BUILD",
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

    const auto result = orchestrator.execute_task(
        R"({"description":"profile model","prompt":"run task","subagent_type":"general"})",
        provider,
        {
            .active_model = "parent-model",
            .parent_mode = "BUILD",
            .permission_check = {},
        });

    REQUIRE_FALSE(result.contains("\"error\""));

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 1);
    REQUIRE(requests.front().model == "model-from-subagent-profile");
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

    const auto result = orchestrator.execute_task(
        R"({"description":"provider override","prompt":"run task","subagent_type":"general"})",
        parent_provider,
        {
            .active_model = "parent-model",
            .parent_mode = "BUILD",
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

    const auto result = orchestrator.execute_task(
        R"({"description":"missing provider","prompt":"run task","subagent_type":"general"})",
        provider,
        {
            .active_model = "parent-model",
            .parent_mode = "BUILD",
            .permission_check = {},
        });

    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("\"error\""));
    REQUIRE_THAT(result, Catch::Matchers::ContainsSubstring("provider-that-does-not-exist"));
}
