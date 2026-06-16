#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestSessionContext.hpp"
#include "core/config/ConfigManager.hpp"
#include "core/agent/DelegatedAgentRunner.hpp"
#include "core/agent/SubagentOrchestrator.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/ProviderManager.hpp"
#include "core/mcp/McpDispatcher.hpp"
#include "core/task/WorkStore.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/WriteFileTool.hpp"
#include "core/workspace/Workspace.hpp"

#include <simdjson.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <ranges>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace Catch::Matchers;
namespace fs = std::filesystem;

class RecordingProvider final : public core::llm::LLMProvider {
public:
    explicit RecordingProvider(std::string response_text)
        : response_text_(std::move(response_text)) {}

    void stream_response(
        const core::llm::ChatRequest& request,
        std::function<void(const core::llm::StreamChunk&)> callback) override
    {
        {
            std::lock_guard lock(mutex_);
            requests_.push_back(request);
        }

        core::llm::StreamChunk chunk;
        chunk.content = response_text_;
        chunk.is_final = true;
        callback(chunk);
    }

    [[nodiscard]] std::vector<core::llm::ChatRequest> requests_snapshot() const {
        std::lock_guard lock(mutex_);
        return requests_;
    }

private:
    std::string response_text_;
    mutable std::mutex mutex_;
    std::vector<core::llm::ChatRequest> requests_;
};

class BlockingProvider final : public core::llm::LLMProvider {
public:
    BlockingProvider(int expected_starts, std::string response_text)
        : expected_starts_(expected_starts)
        , response_text_(std::move(response_text)) {}

    void stream_response(
        const core::llm::ChatRequest& request,
        std::function<void(const core::llm::StreamChunk&)> callback) override
    {
        {
            std::unique_lock lock(mutex_);
            requests_.push_back(request);
            ++started_;
            started_cv_.notify_all();
            release_cv_.wait(lock, [&] { return released_; });
        }

        core::llm::StreamChunk chunk;
        chunk.content = response_text_;
        chunk.is_final = true;
        callback(chunk);
    }

    [[nodiscard]] bool wait_until_all_started(std::chrono::milliseconds timeout) {
        std::unique_lock lock(mutex_);
        return started_cv_.wait_for(lock, timeout, [&] { return started_ >= expected_starts_; });
    }

    void release_all() {
        std::lock_guard lock(mutex_);
        released_ = true;
        release_cv_.notify_all();
    }

private:
    int expected_starts_;
    std::string response_text_;
    std::mutex mutex_;
    std::condition_variable started_cv_;
    std::condition_variable release_cv_;
    int started_ = 0;
    bool released_ = false;
    std::vector<core::llm::ChatRequest> requests_;
};

class ToolCallingProvider final : public core::llm::LLMProvider {
public:
    explicit ToolCallingProvider(std::string write_file_args)
        : write_file_args_(std::move(write_file_args)) {}

    void stream_response(
        const core::llm::ChatRequest& request,
        std::function<void(const core::llm::StreamChunk&)> callback) override
    {
        bool saw_tool_result = false;
        {
            std::lock_guard lock(mutex_);
            requests_.push_back(request);
            saw_tool_result = std::ranges::any_of(request.messages, [](const core::llm::Message& message) {
                return message.role == "tool";
            });
        }

        core::llm::StreamChunk chunk;
        chunk.is_final = true;
        if (saw_tool_result) {
            chunk.content = "worker-saw-denial";
        } else {
            chunk.tools.push_back(core::llm::ToolCall{
                .index = 0,
                .id = "write-file-call",
                .type = "function",
                .function = {
                    .name = "write_file",
                    .arguments = write_file_args_,
                },
            });
        }
        callback(chunk);
    }

    [[nodiscard]] std::vector<core::llm::ChatRequest> requests_snapshot() const {
        std::lock_guard lock(mutex_);
        return requests_;
    }

private:
    std::string write_file_args_;
    mutable std::mutex mutex_;
    std::vector<core::llm::ChatRequest> requests_;
};

class ScopedEnvVar {
public:
    ScopedEnvVar(std::string name, std::string value)
        : name_(std::move(name))
    {
        if (const char* current = std::getenv(name_.c_str())) {
            old_value_ = std::string(current);
        }
        ::setenv(name_.c_str(), value.c_str(), 1);
    }

    ~ScopedEnvVar() {
        if (old_value_.has_value()) {
            ::setenv(name_.c_str(), old_value_->c_str(), 1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }

private:
    std::string name_;
    std::optional<std::string> old_value_;
};

class ScopedCurrentPath {
public:
    explicit ScopedCurrentPath(const fs::path& new_path)
        : old_path_(fs::current_path())
    {
        fs::current_path(new_path);
    }

    ~ScopedCurrentPath() {
        std::error_code ec;
        fs::current_path(old_path_, ec);
    }

    [[nodiscard]] const fs::path& old_path() const noexcept { return old_path_; }

private:
    fs::path old_path_;
};

[[nodiscard]] fs::path make_temp_dir(std::string_view tag) {
    const auto dir = fs::temp_directory_path() / ("filo_mcp_tasks_" + std::string(tag));
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir, ec);
    return dir;
}

void write_file(const fs::path& path, std::string_view content) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << content;
}

[[nodiscard]] core::mcp::McpDispatcher& dispatcher() {
    return core::mcp::McpDispatcher::get_instance();
}

[[nodiscard]] std::string dispatch_json(std::string_view request,
                                        const core::context::SessionContext& context) {
    return dispatcher().dispatch(std::string(request), context);
}

[[nodiscard]] simdjson::dom::element parse_json(const std::string& json) {
    static thread_local simdjson::dom::parser parser;
    simdjson::dom::element doc;
    REQUIRE(parser.parse(json).get(doc) == simdjson::SUCCESS);
    return doc;
}

[[nodiscard]] simdjson::dom::element element_at(
    const simdjson::dom::element& doc,
    std::initializer_list<std::string_view> keys)
{
    simdjson::dom::element current = doc;
    for (const auto key : keys) {
        const bool numeric =
            !key.empty()
            && std::ranges::all_of(key, [](const char ch) { return ch >= '0' && ch <= '9'; });
        if (numeric) {
            simdjson::dom::array array;
            REQUIRE(current.get(array) == simdjson::SUCCESS);
            const std::size_t index = static_cast<std::size_t>(std::stoul(std::string(key)));
            std::size_t i = 0;
            bool found = false;
            for (auto entry : array) {
                if (i++ == index) {
                    current = entry;
                    found = true;
                    break;
                }
            }
            REQUIRE(found);
        } else {
            current = current[key.data()];
        }
    }
    return current;
}

[[nodiscard]] std::string string_at(const simdjson::dom::element& doc,
                                    std::initializer_list<std::string_view> keys) {
    auto current = element_at(doc, keys);
    std::string_view value;
    REQUIRE(current.get(value) == simdjson::SUCCESS);
    return std::string(value);
}

[[nodiscard]] int64_t int_at(const simdjson::dom::element& doc,
                             std::initializer_list<std::string_view> keys) {
    auto current = element_at(doc, keys);
    int64_t value = 0;
    REQUIRE(current.get(value) == simdjson::SUCCESS);
    return value;
}

void initialize_task_capable_session(const core::context::SessionContext& context) {
    const auto response = dispatch_json(
        R"({"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2025-11-25","capabilities":{"extensions":{"io.modelcontextprotocol/tasks":{}}}},"id":100})",
        context);
    const auto doc = parse_json(response);
    REQUIRE(string_at(doc, {"result", "protocolVersion"}) == "2025-11-25");
    simdjson::dom::element extension;
    REQUIRE(doc["result"]["capabilities"]["extensions"]["io.modelcontextprotocol/tasks"].get(extension)
            == simdjson::SUCCESS);
}

[[nodiscard]] std::string task_client_meta() {
    return R"("_meta":{"io.modelcontextprotocol/clientCapabilities":{"extensions":{"io.modelcontextprotocol/tasks":{}}}})";
}

[[nodiscard]] std::string wait_for_terminal_task_response(
    std::string_view task_id,
    const core::context::SessionContext& context,
    int id)
{
    std::string response;
    for (int attempt = 0; attempt < 50; ++attempt) {
        response = dispatch_json(
            std::format(
                R"({{"jsonrpc":"2.0","method":"tasks/get","params":{{"taskId":"{}"}},"id":{}}})",
                task_id,
                id),
            context);
        const auto doc = parse_json(response);
        const auto status = string_at(doc, {"result", "status"});
        if (status == "completed" || status == "failed" || status == "cancelled") {
            return response;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    return response;
}

struct TaskTestSandbox {
    fs::path root;
    fs::path project_dir;
    std::unique_ptr<ScopedEnvVar> xdg_config_home;
    std::unique_ptr<ScopedEnvVar> xdg_data_home;
    std::unique_ptr<ScopedCurrentPath> cwd_guard;
    core::context::SessionContext context;

    explicit TaskTestSandbox(std::string_view tag)
        : root(make_temp_dir(tag))
        , project_dir(root / "project")
        , xdg_config_home(std::make_unique<ScopedEnvVar>(
            "XDG_CONFIG_HOME",
            (root / "xdg-config").string()))
        , xdg_data_home(std::make_unique<ScopedEnvVar>(
            "XDG_DATA_HOME",
            (root / "xdg-data").string()))
        , cwd_guard()
        , context([&] {
            fs::create_directories(project_dir);
            cwd_guard = std::make_unique<ScopedCurrentPath>(project_dir);
            core::workspace::Workspace::get_instance().initialize(project_dir, {}, true);
            return test_support::make_workspace_session_context(
                core::context::SessionTransport::mcp_http,
                std::string("mcp-task-session-") + std::string(tag));
        }())
    {
        initialize_task_capable_session(context);
    }

    ~TaskTestSandbox() {
        if (cwd_guard) {
            core::workspace::Workspace::get_instance().initialize(cwd_guard->old_path(), {}, false);
            core::config::ConfigManager::get_instance().load(cwd_guard->old_path());
        }
    }
};

void load_project_config(const fs::path& project_dir, std::string_view json) {
    write_file(project_dir / ".filo" / "config.json", json);
    core::config::ConfigManager::get_instance().load(project_dir);
}

} // namespace

TEST_CASE("MCP tasks are advertised through the standard extension capability", "[mcp][tasks]") {
    const auto sandbox = make_temp_dir("tool_metadata");
    const auto project_dir = sandbox / "project";
    fs::create_directories(project_dir);
    ScopedCurrentPath cwd(project_dir);
    auto context = test_support::make_session_context(
        core::workspace::WorkspaceSnapshot{
            .primary = project_dir,
            .additional = {},
            .enforce = true,
        },
        core::context::SessionTransport::mcp_http,
        "mcp-task-tool-metadata");
    initialize_task_capable_session(context);

    const auto response = dispatch_json(
        R"({"jsonrpc":"2.0","method":"tools/list","params":{},"id":201})",
        context);

    REQUIRE_THAT(response, ContainsSubstring(R"("name":"delegate_task")"));
    REQUIRE(response.find(R"("execution")") == std::string::npos);
}

TEST_CASE("MCP task methods require the task-capable protocol",
          "[mcp][tasks]") {
    TaskTestSandbox sandbox("legacy_protocol_tasks");

    const auto init_response = dispatch_json(
        R"({"jsonrpc":"2.0","method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{}},"id":211})",
        sandbox.context);
    REQUIRE_THAT(init_response, ContainsSubstring(R"("protocolVersion":"2024-11-05")"));

    const auto tools_response = dispatch_json(
        R"({"jsonrpc":"2.0","method":"tools/list","params":{},"id":212})",
        sandbox.context);
    REQUIRE_THAT(tools_response, ContainsSubstring(R"("name":"delegate_task")"));
    REQUIRE(tools_response.find(R"("execution")") == std::string::npos);

    const auto task_method_response = dispatch_json(
        R"({"jsonrpc":"2.0","method":"tasks/get","params":{"taskId":"missing-task"},"id":213})",
        sandbox.context);
    REQUIRE(int_at(parse_json(task_method_response), {"error", "code"}) == -32601);

    const auto task_call_response = dispatch_json(
        std::format(
            R"({{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"delegate_task","arguments":{{"action":"list"}},{}}},"id":214}})",
            task_client_meta()),
        sandbox.context);
    REQUIRE(string_at(parse_json(task_call_response), {"result", "structuredContent", "action"})
            == "list");
}

TEST_CASE("Delegated subagents honor the parent permission gate for destructive tools",
          "[mcp][tasks][permissions]") {
    TaskTestSandbox sandbox("subagent_permission_gate");
    (void)dispatcher();
    core::tools::ToolManager::get_instance().register_tool(
        std::make_shared<core::tools::WriteFileTool>());

    const auto denied_path = sandbox.project_dir / "denied.txt";
    auto provider = std::make_shared<ToolCallingProvider>(
        std::format(
            R"({{"file_path":"{}","content":"must not be written"}})",
            denied_path.string()));

    std::atomic<int> permission_calls{0};
    core::agent::SubagentOrchestrator orchestrator(
        core::tools::ToolManager::get_instance(),
        &core::config::ConfigManager::get_instance().get_config());

    const auto response = orchestrator.execute_task(
        R"({"description":"Denied write","prompt":"Try to write the file, then report the result.","subagent_type":"general"})",
        provider,
        core::agent::SubagentOrchestrator::RunContext{
            .active_model = "permission-model",
            .parent_mode = "BUILD",
            .session_context = sandbox.context,
            .permission_check = [&permission_calls](
                                    const std::string& tool_name,
                                    const std::string&) {
                if (tool_name == "write_file") {
                    permission_calls.fetch_add(1, std::memory_order_relaxed);
                }
                return false;
            },
        });

    REQUIRE(permission_calls.load(std::memory_order_acquire) == 1);
    REQUIRE_FALSE(fs::exists(denied_path));
    REQUIRE_THAT(response, ContainsSubstring(R"("failed_tool_calls":1)"));

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 1);
    REQUIRE(requests.front().messages.back().role == "user");
}

TEST_CASE("Delegated subagents do not inherit the persistent delegate task tool by default",
          "[mcp][tasks]") {
    TaskTestSandbox sandbox("subagent_delegate_task_filter");
    (void)dispatcher();

    auto provider = std::make_shared<RecordingProvider>("filter-check");
    core::agent::SubagentOrchestrator orchestrator(
        core::tools::ToolManager::get_instance(),
        &core::config::ConfigManager::get_instance().get_config());

    const auto plan = orchestrator.build_execution_plan(
        core::agent::SubagentOrchestrator::ExecutionRequest{
            .worker = "general",
            .parent_mode = "BUILD",
            .inherited_provider = provider,
            .inherited_model = "filter-model",
            .prefer_inherited_provider = true,
        });
    REQUIRE(plan.has_value());

    const auto names = plan->allowed_tool_names();
    REQUIRE(std::ranges::find(names, std::string("task")) == names.end());
    REQUIRE(std::ranges::find(names, std::string("delegate_task")) == names.end());
}

TEST_CASE("MCP keeps non-delegate tools synchronous when the client supports tasks",
          "[mcp][tasks]") {
    TaskTestSandbox sandbox("unsupported_task_execution");
    write_file(sandbox.project_dir / "notes.txt", "hello");

    const auto response = dispatch_json(
        std::format(
            R"({{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"read_file","arguments":{{"path":"notes.txt"}},{}}},"id":251}})",
            task_client_meta()),
        sandbox.context);
    const auto doc = parse_json(response);

    REQUIRE(string_at(doc, {"result", "structuredContent", "content"}) == "hello");
}

TEST_CASE("MCP task-capable tools/call returns a CreateTaskResult and tasks/get returns the tool result",
          "[mcp][tasks]") {
    TaskTestSandbox sandbox("task_augmented_roundtrip");

    auto provider = std::make_shared<RecordingProvider>("worker-finished-cleanly");
    core::llm::ProviderManager::get_instance().register_provider("mcp-task-default", provider);
    load_project_config(
        sandbox.project_dir,
        R"({
            "default_provider": "mcp-task-default",
            "providers": {
                "mcp-task-default": {
                    "api_type": "openai",
                    "base_url": "https://example.test/v1",
                    "model": "model-from-config"
                }
            }
        })");

    const auto create_response = dispatch_json(
        std::format(
            R"({{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"delegate_task","arguments":{{"action":"start","title":"Trace startup","instructions":"Inspect the startup sequence and summarize it.","worker":"general"}},{}}},"id":301}})",
            task_client_meta()),
        sandbox.context);
    const auto create_doc = parse_json(create_response);

    REQUIRE(string_at(create_doc, {"result", "resultType"}) == "task");
    const std::string task_id = string_at(create_doc, {"result", "task", "taskId"});
    REQUIRE_FALSE(task_id.empty());
    REQUIRE(int_at(create_doc, {"result", "task", "ttlMs"}) > 0);
    REQUIRE(int_at(create_doc, {"result", "task", "pollIntervalMs"}) > 0);
    REQUIRE_THAT(
        string_at(create_doc, {"result", "_meta", "io.modelcontextprotocol/model-immediate-response"}),
        ContainsSubstring("Task accepted"));

    const auto get_response = dispatch_json(
        std::format(
            R"({{"jsonrpc":"2.0","method":"tasks/get","params":{{"taskId":"{}"}},"id":302}})",
            task_id),
        sandbox.context);
    const auto get_doc = parse_json(get_response);
    const auto task_status = string_at(get_doc, {"result", "status"});
    REQUIRE((task_status == "working" || task_status == "completed"));

    const auto update_response = dispatch_json(
        std::format(
            R"({{"jsonrpc":"2.0","method":"tasks/update","params":{{"taskId":"{}","inputResponses":{{}}}},"id":304}})",
            task_id),
        sandbox.context);
    REQUIRE(update_response.find(R"("result":{})") != std::string::npos);

    const auto result_response = wait_for_terminal_task_response(
        task_id,
        sandbox.context,
        303);
    const auto result_doc = parse_json(result_response);

    REQUIRE(string_at(result_doc, {"result", "result", "_meta", "io.modelcontextprotocol/related-task", "taskId"})
            == task_id);
    REQUIRE(string_at(result_doc, {"result", "result", "structuredContent", "status"}) == "completed");
    REQUIRE(string_at(result_doc, {"result", "result", "structuredContent", "worker"}) == "general");
    REQUIRE(string_at(result_doc, {"result", "result", "structuredContent", "provider"}) == "mcp-task-default");
    REQUIRE(string_at(result_doc, {"result", "result", "structuredContent", "model"}) == "model-from-config");

    const auto sync_list_response = dispatch_json(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"delegate_task","arguments":{"action":"list"}},"id":305})",
        sandbox.context);
    const auto sync_list_doc = parse_json(sync_list_response);
    REQUIRE(string_at(sync_list_doc, {"result", "structuredContent", "items", "0", "worker"})
            == "general");
    REQUIRE(string_at(sync_list_doc, {"result", "structuredContent", "items", "0", "provider"})
            == "mcp-task-default");
}

TEST_CASE("MCP cancelled task results return promptly for non-cooperative tools",
          "[mcp][tasks][cancel]") {
    TaskTestSandbox sandbox("cancel_noncooperative_task");

    auto provider = std::make_shared<BlockingProvider>(1, "cancelled-worker-finished");
    core::llm::ProviderManager::get_instance().register_provider("mcp-task-cancel", provider);
    load_project_config(
        sandbox.project_dir,
        R"({
            "default_provider": "mcp-task-cancel",
            "providers": {
                "mcp-task-cancel": {
                    "api_type": "openai",
                    "base_url": "https://example.test/v1",
                    "model": "cancel-model"
                }
            }
        })");

    const auto create_response = dispatch_json(
        std::format(
            R"({{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"delegate_task","arguments":{{"action":"start","title":"Cancelable","instructions":"Block until cancelled.","worker":"general"}},{}}},"id":351}})",
            task_client_meta()),
        sandbox.context);
    const auto task_id = string_at(parse_json(create_response), {"result", "task", "taskId"});
    REQUIRE(provider->wait_until_all_started(std::chrono::milliseconds(2000)));

    const auto cancel_response = dispatch_json(
        std::format(
            R"({{"jsonrpc":"2.0","method":"tasks/cancel","params":{{"taskId":"{}"}},"id":352}})",
            task_id),
        sandbox.context);
    REQUIRE(cancel_response.find(R"("result":{})") != std::string::npos);

    const auto start = std::chrono::steady_clock::now();
    const auto result_response = wait_for_terminal_task_response(task_id, sandbox.context, 353);
    const auto elapsed = std::chrono::steady_clock::now() - start;
    provider->release_all();

    REQUIRE(elapsed < std::chrono::milliseconds(500));
    const auto result_doc = parse_json(result_response);
    REQUIRE(string_at(result_doc, {"result", "status"}) == "cancelled");
}

TEST_CASE("Delegated agent runner observes cancellation while provider start is blocked",
          "[mcp][tasks][cancel]") {
    TaskTestSandbox sandbox("cancel_blocked_provider");

    auto provider = std::make_shared<BlockingProvider>(1, "provider-eventually-finished");
    std::atomic<bool> cancel_requested{false};
    std::atomic<bool> provider_started{false};

    std::thread canceller([&] {
        if (provider->wait_until_all_started(std::chrono::milliseconds(2000))) {
            provider_started.store(true, std::memory_order_release);
            cancel_requested.store(true, std::memory_order_release);
        }
    });

    const auto start = std::chrono::steady_clock::now();
    const auto result = core::agent::DelegatedAgentRunner::run({
        .provider = provider,
        .tool_manager = core::tools::ToolManager::get_instance(),
        .session_context = sandbox.context,
        .mode = "BUILD",
        .model = "blocked-provider-model",
        .max_steps = 4,
        .prompt = "This provider will block before yielding a chunk.",
        .timeout = std::chrono::seconds(10),
        .cancellation_requested = [&] {
            return cancel_requested.load(std::memory_order_acquire);
        },
    });
    const auto elapsed = std::chrono::steady_clock::now() - start;

    provider->release_all();
    canceller.join();

    REQUIRE(provider_started.load(std::memory_order_acquire));
    REQUIRE(result.cancelled);
    REQUIRE_FALSE(result.timed_out);
    REQUIRE(elapsed < std::chrono::seconds(2));
}

TEST_CASE("MCP delegate task tool routes to the requested provider and model override", "[mcp][tasks]") {
    TaskTestSandbox sandbox("provider_override");

    auto default_provider = std::make_shared<RecordingProvider>("default-provider-response");
    auto alt_provider = std::make_shared<RecordingProvider>("alt-provider-response");
    core::llm::ProviderManager::get_instance().register_provider("mcp-task-primary", default_provider);
    core::llm::ProviderManager::get_instance().register_provider("mcp-task-alt", alt_provider);

    load_project_config(
        sandbox.project_dir,
        R"({
            "default_provider": "mcp-task-primary",
            "providers": {
                "mcp-task-primary": {
                    "api_type": "openai",
                    "base_url": "https://example.test/v1",
                    "model": "primary-model"
                },
                "mcp-task-alt": {
                    "api_type": "openai",
                    "base_url": "https://example.test/v1",
                    "model": "alt-default-model"
                }
            }
        })");

    const auto response = dispatch_json(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"delegate_task","arguments":{"action":"start","title":"Provider override","instructions":"Use the alternate provider and custom model.","provider":"mcp-task-alt","model":"alt-explicit-model"}},"id":401})",
        sandbox.context);
    const auto doc = parse_json(response);

    REQUIRE(string_at(doc, {"result", "structuredContent", "provider"}) == "mcp-task-alt");
    REQUIRE(string_at(doc, {"result", "structuredContent", "model"}) == "alt-explicit-model");

    const auto default_requests = default_provider->requests_snapshot();
    const auto alt_requests = alt_provider->requests_snapshot();
    REQUIRE(default_requests.empty());
    REQUIRE(alt_requests.size() == 1);
    REQUIRE(alt_requests.front().model == "alt-explicit-model");
}

TEST_CASE("MCP delegate tasks can run concurrently", "[mcp][tasks]") {
    TaskTestSandbox sandbox("parallel_delegate_tasks");

    auto provider = std::make_shared<BlockingProvider>(2, "parallel-worker-finished");
    core::llm::ProviderManager::get_instance().register_provider("mcp-task-parallel", provider);
    load_project_config(
        sandbox.project_dir,
        R"({
            "default_provider": "mcp-task-parallel",
            "providers": {
                "mcp-task-parallel": {
                    "api_type": "openai",
                    "base_url": "https://example.test/v1",
                    "model": "parallel-model"
                }
            }
        })");

    const auto create_response_a = dispatch_json(
        std::format(
            R"({{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"delegate_task","arguments":{{"action":"start","title":"Parallel A","instructions":"Do work item A.","worker":"general"}},{}}},"id":501}})",
            task_client_meta()),
        sandbox.context);
    const auto create_response_b = dispatch_json(
        std::format(
            R"({{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"delegate_task","arguments":{{"action":"start","title":"Parallel B","instructions":"Do work item B.","worker":"general"}},{}}},"id":502}})",
            task_client_meta()),
        sandbox.context);

    const bool all_started =
        provider->wait_until_all_started(std::chrono::milliseconds(2000));
    provider->release_all();
    REQUIRE(all_started);

    const auto task_id_a = string_at(parse_json(create_response_a), {"result", "task", "taskId"});
    const auto task_id_b = string_at(parse_json(create_response_b), {"result", "task", "taskId"});

    const auto result_response_a = wait_for_terminal_task_response(task_id_a, sandbox.context, 503);
    const auto result_response_b = wait_for_terminal_task_response(task_id_b, sandbox.context, 504);

    REQUIRE(string_at(parse_json(result_response_a), {"result", "result", "structuredContent", "status"})
            == "completed");
    REQUIRE(string_at(parse_json(result_response_b), {"result", "result", "structuredContent", "status"})
            == "completed");
}

TEST_CASE("MCP delegate work items are scoped to the current workspace", "[mcp][tasks]") {
    const auto root = make_temp_dir("workspace_scope");
    const auto shared_config = root / "xdg-config";
    const auto shared_data = root / "xdg-data";
    ScopedEnvVar config_home("XDG_CONFIG_HOME", shared_config.string());
    ScopedEnvVar data_home("XDG_DATA_HOME", shared_data.string());

    const auto project_a = root / "project-a";
    const auto project_b = root / "project-b";
    fs::create_directories(project_a);
    fs::create_directories(project_b);

    auto provider = std::make_shared<RecordingProvider>("workspace-scoped-result");
    core::llm::ProviderManager::get_instance().register_provider("mcp-task-scope", provider);

    load_project_config(
        project_a,
        R"({
            "default_provider": "mcp-task-scope",
            "providers": {
                "mcp-task-scope": {
                    "api_type": "openai",
                    "base_url": "https://example.test/v1",
                    "model": "scope-model"
                }
            }
        })");
    const auto context_a = test_support::make_session_context(
        core::workspace::WorkspaceSnapshot{
            .primary = project_a,
            .additional = {},
            .enforce = true,
        },
        core::context::SessionTransport::mcp_http,
        "workspace-scope-a");

    const auto create_response = dispatch_json(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"delegate_task","arguments":{"action":"start","title":"Scoped work","instructions":"Stay within workspace A.","worker":"general"}},"id":601})",
        context_a);
    const auto create_doc = parse_json(create_response);
    const auto work_id = string_at(create_doc, {"result", "structuredContent", "work_id"});

    load_project_config(
        project_b,
        R"({
            "default_provider": "mcp-task-scope",
            "providers": {
                "mcp-task-scope": {
                    "api_type": "openai",
                    "base_url": "https://example.test/v1",
                    "model": "scope-model"
                }
            }
        })");
    const auto context_b = test_support::make_session_context(
        core::workspace::WorkspaceSnapshot{
            .primary = project_b,
            .additional = {},
            .enforce = true,
        },
        core::context::SessionTransport::mcp_http,
        "workspace-scope-b");

    const auto list_response = dispatch_json(
        R"({"jsonrpc":"2.0","method":"tools/call","params":{"name":"delegate_task","arguments":{"action":"list"}},"id":602})",
        context_b);
    REQUIRE(list_response.find(work_id) == std::string::npos);

    const auto resume_response = dispatch_json(
        std::format(
            R"({{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"delegate_task","arguments":{{"action":"resume","work_id":"{}"}}}},"id":603}})",
            work_id),
        context_b);
    REQUIRE_THAT(resume_response, ContainsSubstring("Unknown work_id"));

    const auto malicious_work_id = std::format(
        "../{}/{}",
        core::task::WorkStore::default_work_dir(context_a).filename().string(),
        work_id);
    const auto traversal_response = dispatch_json(
        std::format(
            R"({{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"delegate_task","arguments":{{"action":"resume","work_id":"{}"}}}},"id":604}})",
            malicious_work_id),
        context_b);
    REQUIRE_THAT(traversal_response, ContainsSubstring("Invalid work_id"));
    REQUIRE_THAT(traversal_response, ContainsSubstring(R"("isError":true)"));
}

TEST_CASE("WorkStore rejects unsafe work ids before filesystem access", "[mcp][tasks]") {
    const auto root = make_temp_dir("work_store_ids") / "items";
    core::task::WorkStore store(root);

    REQUIRE(core::task::WorkStore::is_valid_work_id("work_0123abcd"));
    REQUIRE_FALSE(core::task::WorkStore::is_valid_work_id("../scope/work_0123abcd"));
    REQUIRE_FALSE(core::task::WorkStore::is_valid_work_id("work_0123abcd/evil"));
    REQUIRE_FALSE(core::task::WorkStore::is_valid_work_id("work_0123abcg"));

    core::task::WorkItem unsafe;
    unsafe.work_id = "../scope/work_0123abcd";
    std::string save_error;
    REQUIRE_FALSE(store.save(unsafe, &save_error));
    REQUIRE_THAT(save_error, ContainsSubstring("Invalid work_id"));
    REQUIRE_FALSE(store.load("../scope/work_0123abcd").has_value());

    const auto computed = store.compute_path("../scope/work_0123abcd");
    REQUIRE(computed.parent_path() == root);
    REQUIRE(computed.filename() == "__invalid_work_id__.json");
}

TEST_CASE("MCP task operations use spec-compliant errors for invalid and terminal task ids",
          "[mcp][tasks]") {
    TaskTestSandbox sandbox("task_error_codes");

    auto provider = std::make_shared<RecordingProvider>("task-error-codes-done");
    core::llm::ProviderManager::get_instance().register_provider("mcp-task-errors", provider);
    load_project_config(
        sandbox.project_dir,
        R"({
            "default_provider": "mcp-task-errors",
            "providers": {
                "mcp-task-errors": {
                    "api_type": "openai",
                    "base_url": "https://example.test/v1",
                    "model": "errors-model"
                }
            }
        })");

    const auto create_response = dispatch_json(
        std::format(
            R"({{"jsonrpc":"2.0","method":"tools/call","params":{{"name":"delegate_task","arguments":{{"action":"start","title":"Task error codes","instructions":"Finish quickly.","worker":"general"}},{}}},"id":701}})",
            task_client_meta()),
        sandbox.context);
    const auto task_id = string_at(parse_json(create_response), {"result", "task", "taskId"});

    const auto result_response = wait_for_terminal_task_response(task_id, sandbox.context, 702);
    REQUIRE(string_at(parse_json(result_response), {"result", "result", "structuredContent", "status"})
            == "completed");

    const auto cancel_terminal_response = dispatch_json(
        std::format(
            R"({{"jsonrpc":"2.0","method":"tasks/cancel","params":{{"taskId":"{}"}},"id":703}})",
            task_id),
        sandbox.context);
    REQUIRE(cancel_terminal_response.find(R"("result":{})") != std::string::npos);

    const auto missing_get_response = dispatch_json(
        R"({"jsonrpc":"2.0","method":"tasks/get","params":{"taskId":"missing-task"},"id":704})",
        sandbox.context);
    REQUIRE(int_at(parse_json(missing_get_response), {"error", "code"}) == -32602);

    const auto removed_result_response = dispatch_json(
        R"({"jsonrpc":"2.0","method":"tasks/result","params":{"taskId":"missing-task"},"id":705})",
        sandbox.context);
    REQUIRE(int_at(parse_json(removed_result_response), {"error", "code"}) == -32601);

    const auto missing_cancel_response = dispatch_json(
        R"({"jsonrpc":"2.0","method":"tasks/cancel","params":{"taskId":"missing-task"},"id":706})",
        sandbox.context);
    REQUIRE(int_at(parse_json(missing_cancel_response), {"error", "code"}) == -32602);
}
