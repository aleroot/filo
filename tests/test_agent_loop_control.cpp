#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "TestSessionContext.hpp"
#include "core/agent/Agent.hpp"
#include "core/agent/RepositoryContextMessage.hpp"
#include "core/scm/GitWorkspaceCoordinator.hpp"
#include "core/context/SessionContext.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/llm/protocols/ClaudeProtocol.hpp"
#include "core/memory/MemoryStore.hpp"
#include "core/memory/MemorySystem.hpp"
#include "core/memory/ToolRecoveryMemory.hpp"
#include "core/tools/ShellTool.hpp"
#include "core/tools/Tool.hpp"
#include "core/tools/ToolManager.hpp"
#include "core/tools/ToolNames.hpp"
#include "core/tools/MemoryTool.hpp"
#include "core/tools/TaskTool.hpp"
#include "core/tools/VerificationTool.hpp"
#include "core/tools/WriteFileTool.hpp"

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
    explicit NamedNoopTool(std::string name, bool destructive = false)
        : name_(std::move(name)), destructive_(destructive) {}

    [[nodiscard]] core::tools::ToolDefinition get_definition() const override {
        return {
            .name = name_,
            .title = "Named No-op",
            .description = "No-op tool used for request filtering tests.",
            .parameters = {},
            .annotations = {
                .read_only_hint = !destructive_,
                .destructive_hint = destructive_,
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
    bool destructive_;
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

class CountingTool final : public core::tools::Tool {
public:
    explicit CountingTool(std::string name)
        : name_(std::move(name)) {}

    [[nodiscard]] core::tools::ToolDefinition get_definition() const override {
        return {
            .name = name_,
            .title = "Counting tool",
            .description = "Records executions for failed-turn safety tests.",
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
        executions_.fetch_add(1, std::memory_order_acq_rel);
        return R"({"ok":true})";
    }

    [[nodiscard]] int execution_count() const noexcept {
        return executions_.load(std::memory_order_acquire);
    }

private:
    std::string name_;
    std::atomic<int> executions_{0};
};

class ToolThenErrorProvider final : public core::llm::LLMProvider {
public:
    explicit ToolThenErrorProvider(std::string tool_name)
        : tool_name_(std::move(tool_name)) {}

    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        calls_.fetch_add(1, std::memory_order_acq_rel);

        core::llm::ToolCall tool_call;
        tool_call.index = 0;
        tool_call.id = "failed-turn-call";
        tool_call.type = "function";
        tool_call.function.name = tool_name_;
        tool_call.function.arguments = "{}";

        core::llm::StreamChunk tool_chunk;
        tool_chunk.tools = {std::move(tool_call)};
        tool_chunk.continuation_items.push_back(core::llm::ContinuationItem{
            .provider = "openai",
            .kind = "reasoning",
            .payload = R"({"type":"reasoning","id":"failed-reasoning"})",
        });
        callback(tool_chunk);
        callback(core::llm::StreamChunk::make_error("\n[provider stream failed]"));
    }

    [[nodiscard]] int call_count() const noexcept {
        return calls_.load(std::memory_order_acquire);
    }

private:
    std::string tool_name_;
    std::atomic<int> calls_{0};
};

class MixedValidAndNamelessToolProvider final : public core::llm::LLMProvider {
public:
    explicit MixedValidAndNamelessToolProvider(std::string valid_tool_name)
        : valid_tool_name_(std::move(valid_tool_name)) {}

    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        const int call = calls_.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (call == 1) {
            core::llm::ToolCall valid;
            valid.index = 0;
            valid.id = "valid-tool-call";
            valid.type = "function";
            valid.function.name = valid_tool_name_;
            valid.function.arguments = "{}";

            core::llm::ToolCall nameless;
            nameless.index = 1;
            nameless.id = "nameless-tool-call";
            nameless.type = "function";
            nameless.function.arguments = "{}";

            callback(core::llm::StreamChunk{
                .tools = {std::move(valid), std::move(nameless)},
                .is_final = true,
            });
            return;
        }

        callback(core::llm::StreamChunk::make_content("finished"));
        callback(core::llm::StreamChunk::make_final());
    }

    [[nodiscard]] int call_count() const noexcept {
        return calls_.load(std::memory_order_acquire);
    }

private:
    std::string valid_tool_name_;
    std::atomic<int> calls_{0};
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
    int context_size = 0;

    [[nodiscard]] int max_context_size() const noexcept override { return context_size; }

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

class MemoryFailureProvider final : public core::llm::LLMProvider {
public:
    explicit MemoryFailureProvider(bool correct, bool unsolicited = false,
                                   bool denied = false)
        : correct_(correct), unsolicited_(unsolicited), denied_(denied) {}

    void stream_response(const core::llm::ChatRequest& request,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        requests_.push_back(request);
        const auto call = requests_.size();
        if (call <= 2 || (call == 3 && unsolicited_)) {
            core::llm::ToolCall tool;
            tool.index = 0;
            tool.id = std::format("memory-failure-{}", call);
            tool.type = "function";
            tool.function.name = std::string(core::tools::names::kMemory);
            tool.function.arguments = std::format(
                R"({{"action":"remember","content":"{}"}})",
                call == 2 && correct_ ? "A concise durable fact" : std::string(1275, 'x'));
            std::vector<core::llm::ToolCall> calls{std::move(tool)};
            if (call == 3 && denied_) {
                core::llm::ToolCall other;
                other.index = 1;
                other.id = "denied-companion";
                other.type = "function";
                other.function.name = "memory_denied_companion";
                other.function.arguments = "{}";
                calls.push_back(std::move(other));
            }
            callback(core::llm::StreamChunk{.tools = std::move(calls), .is_final = true});
        } else if (call == 3 && !correct_) {
            core::llm::ToolCall tool;
            tool.index = 0;
            tool.id = "other-work";
            tool.type = "function";
            tool.function.name = std::string(kLoopToolName);
            tool.function.arguments = "{}";
            callback(core::llm::StreamChunk{.tools = {std::move(tool)}, .is_final = true});
        } else {
            callback(core::llm::StreamChunk::make_content("The requested answer."));
            callback(core::llm::StreamChunk::make_final());
        }
    }

    std::vector<core::llm::ChatRequest> requests_;
private:
    bool correct_;
    bool unsolicited_;
    bool denied_;
};

class AutoQualityProvider final : public core::llm::LLMProvider {
public:
  void stream_response(
      const core::llm::ChatRequest &request,
      std::function<void(const core::llm::StreamChunk &)> callback) override {
    {
      std::lock_guard lock(mutex_);
      requests_.push_back(request);
    }

    if (std::ranges::any_of(
            request.messages, [](const core::llm::Message &message) {
              return message.content.contains(
                  "You are the planning module of an autonomous coding agent");
            })) {
      callback(core::llm::StreamChunk::make_content(
          R"({"nodes":[{"name":"implement","kind":"work","directive":"Implement the requested change","workspace_access":"exclusive_write"}],"edges":[]})"));
      callback(core::llm::StreamChunk::make_final());
      return;
    }

    const int call = calls_.fetch_add(1, std::memory_order_acq_rel) + 1;
    if (call == 1) {
      emit_tool(callback, "auto-write", core::tools::names::kWriteFile,
                R"({"file_path":"auto-quality.txt","content":"verified\n"})");
      return;
    }
    if (call == 2) {
      callback(
          core::llm::StreamChunk::make_content("Implementation complete."));
      callback(core::llm::StreamChunk::make_final());
      return;
    }
    callback(core::llm::StreamChunk::make_content("Unexpected extra turn."));
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
  static void
  emit_tool(const std::function<void(const core::llm::StreamChunk &)> &callback,
            std::string id, std::string_view name, std::string arguments) {
    core::llm::ToolCall tool_call;
    tool_call.index = 0;
    tool_call.id = std::move(id);
    tool_call.type = "function";
    tool_call.function.name = std::string(name);
    tool_call.function.arguments = std::move(arguments);

    core::llm::StreamChunk chunk;
    chunk.tools = {std::move(tool_call)};
    chunk.is_final = true;
    callback(chunk);
  }

  mutable std::mutex mutex_;
  std::vector<core::llm::ChatRequest> requests_;
  std::atomic<int> calls_{0};
};

class AutoExploreProvider final : public core::llm::LLMProvider {
public:
  void stream_response(
      const core::llm::ChatRequest &request,
      std::function<void(const core::llm::StreamChunk &)> callback) override {
    {
      std::lock_guard lock(mutex_);
      requests_.push_back(request);
    }

    const bool planning = std::ranges::any_of(
        request.messages, [](const core::llm::Message &message) {
          return message.content.contains(
              "You are the planning module of an autonomous coding agent");
        });
    if (planning) {
      callback(core::llm::StreamChunk::make_content(
          R"({"nodes":[
                {"name":"inspect-api","kind":"work","directive":"inspect api","workspace_access":"shared_read","parallel":true},
                {"name":"inspect-tests","kind":"work","directive":"inspect tests","workspace_access":"shared_read","parallel":true},
                {"name":"implement","kind":"work","directive":"implement the fix","workspace_access":"exclusive_write"}
              ],
              "edges":[
                {"from":"inspect-api","to":"implement","condition":"always"},
                {"from":"inspect-tests","to":"implement","condition":"always"}
              ]})"));
      callback(core::llm::StreamChunk::make_final());
      return;
    }

    if (std::ranges::any_of(request.messages, [](const auto &message) {
          return message.content.contains("You are Filo's independent BOOST reviewer");
        })) {
      callback(core::llm::StreamChunk::make_content(
          R"({"verdict":"pass","evidence":"Scheduler inspection covers the requested diagnosis; no changes to verify."})"));
      callback(core::llm::StreamChunk::make_final());
      return;
    }

    const bool delegated = std::ranges::any_of(
        request.messages, [](const core::llm::Message &message) {
          return message.role == "system" &&
                 message.content.contains("running in RESEARCH mode");
        });
    if (delegated) {
      explores_.fetch_add(1, std::memory_order_acq_rel);
      callback(core::llm::StreamChunk::make_content("Found the scheduler."));
      callback(core::llm::StreamChunk::make_final());
      return;
    }

    callback(core::llm::StreamChunk::make_content(
        "Used the preflight findings; no further edits."));
    callback(core::llm::StreamChunk::make_final());
  }

  [[nodiscard]] int explore_count() const noexcept {
    return explores_.load(std::memory_order_acquire);
  }

  [[nodiscard]] std::vector<core::llm::ChatRequest> requests_snapshot() const {
    std::lock_guard lock(mutex_);
    return requests_;
  }

private:
  mutable std::mutex mutex_;
  std::vector<core::llm::ChatRequest> requests_;
  std::atomic<int> explores_{0};
};

class ResetThenRecoverProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        callback(core::llm::StreamChunk::make_content("partial "));
        callback(core::llm::StreamChunk::make_tools({{
            .id = "provisional_call", .type = "function",
            .function = {.name = "provisional_tool", .arguments = "{}"},
        }}));
        core::llm::StreamChunk opaque;
        opaque.continuation_items.push_back({
            .provider = "openai", .kind = "reasoning",
            .payload = R"({"type":"reasoning","encrypted_content":"provisional"})",
        });
        callback(opaque);
        callback(core::llm::StreamChunk::make_attempt_reset(
            "[grok_responses retrying after a generation failure (1/3)]"));
        callback(core::llm::StreamChunk::make_content("recovered after generation failure"));
        callback(core::llm::StreamChunk::make_final());
    }
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

class DeferredFinalProvider final : public core::llm::LLMProvider {
public:
    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        {
            std::lock_guard lock(mutex_);
            entered_ = true;
        }
        cv_.notify_all();
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [&] { return released_; });
        }
        callback(core::llm::StreamChunk::make_content("late response"));
        callback(core::llm::StreamChunk::make_final());
    }

    void wait_until_entered() {
        std::unique_lock lock(mutex_);
        REQUIRE(cv_.wait_for(
            lock,
            std::chrono::seconds(3),
            [&] { return entered_; }));
    }

    void release() {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

private:
    std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_ = false;
    bool released_ = false;
};

class BlockingHistoryTool final : public core::tools::Tool {
public:
    explicit BlockingHistoryTool(std::string name)
        : name_(std::move(name)) {}

    [[nodiscard]] core::tools::ToolDefinition get_definition() const override {
        return {
            .name = name_,
            .title = "Blocking history tool",
            .description = "Waits until released by the transcript race test.",
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
        {
            std::lock_guard lock(mutex_);
            entered_ = true;
        }
        cv_.notify_all();
        {
            std::unique_lock lock(mutex_);
            cv_.wait(lock, [&] { return released_; });
        }
        return R"({"ok":true})";
    }

    void wait_until_entered() {
        std::unique_lock lock(mutex_);
        REQUIRE(cv_.wait_for(
            lock,
            std::chrono::seconds(3),
            [&] { return entered_; }));
    }

    void release() {
        {
            std::lock_guard lock(mutex_);
            released_ = true;
        }
        cv_.notify_all();
    }

private:
    std::string name_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool entered_ = false;
    bool released_ = false;
};

class FileEditingProvider final : public core::llm::LLMProvider {
public:
    explicit FileEditingProvider(std::vector<std::string> write_arguments)
        : write_arguments_(std::move(write_arguments)) {}

    void stream_response(
        const core::llm::ChatRequest&,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        const auto call = calls_.fetch_add(1, std::memory_order_acq_rel);
        if (call < write_arguments_.size()) {
            core::llm::ToolCall tool_call;
            tool_call.index = 0;
            tool_call.id = std::format("write-{}", call);
            tool_call.type = "function";
            tool_call.function.name = "write_file";
            tool_call.function.arguments = write_arguments_[call];

            core::llm::StreamChunk chunk;
            chunk.tools = {std::move(tool_call)};
            chunk.is_final = true;
            callback(chunk);
            return;
        }
        callback(core::llm::StreamChunk::make_content("edited"));
        callback(core::llm::StreamChunk::make_final());
    }

private:
    std::vector<std::string> write_arguments_;
    std::atomic<std::size_t> calls_{0};
};

class ToolThenTextCapturingProvider final : public core::llm::LLMProvider {
public:
    explicit ToolThenTextCapturingProvider(std::string tool_name)
        : tool_name_(std::move(tool_name)) {}

    void stream_response(
        const core::llm::ChatRequest& request,
        std::function<void(const core::llm::StreamChunk&)> callback) override {
        {
            std::lock_guard lock(mutex_);
            requests_.push_back(request);
        }
        const int call = calls_.fetch_add(1, std::memory_order_acq_rel) + 1;
        if (call == 1) {
            core::llm::ToolCall tool_call;
            tool_call.index = 0;
            tool_call.id = "history-race-tool-call";
            tool_call.type = "function";
            tool_call.function.name = tool_name_;
            tool_call.function.arguments = "{}";

            core::llm::StreamChunk chunk;
            chunk.tools = {std::move(tool_call)};
            chunk.is_final = true;
            callback(chunk);
            return;
        }
        callback(core::llm::StreamChunk::make_content("finished"));
        callback(core::llm::StreamChunk::make_final());
    }

    [[nodiscard]] int call_count() const noexcept {
        return calls_.load(std::memory_order_acquire);
    }

private:
    std::string tool_name_;
    std::mutex mutex_;
    std::vector<core::llm::ChatRequest> requests_;
    std::atomic<int> calls_{0};
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

// Mirrors LlamaCppProvider, which serializes inference with a non-recursive
// mutex it legitimately holds across every callback. A provider owns its own
// locking; what it cannot survive is the Agent re-entering stream_response()
// from inside that callback, which self-deadlocks on the very lock the provider
// is holding. Truncated-turn recovery and completion-hook continuations both
// used to chain the next step that way.
class LockAcrossCallbackProvider : public core::llm::LLMProvider {
public:
    void stream_response(const core::llm::ChatRequest&,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        std::scoped_lock lock(inference_mutex_);
        const int call = call_count_.fetch_add(1, std::memory_order_acq_rel);

        core::llm::StreamChunk partial;
        partial.content = "partial answer ";
        callback(partial);

        core::llm::StreamChunk terminal;
        terminal.is_final = true;
        if (call == 0) {
            // Truncated turn: the Agent recovers by chaining another step.
            terminal.stop_reason = "max_tokens";
        } else {
            terminal.content = "finished";
        }
        callback(terminal);
    }

    [[nodiscard]] int call_count() const noexcept {
        return call_count_.load(std::memory_order_acquire);
    }

private:
    std::mutex inference_mutex_;  // deliberately not recursive
    std::atomic<int> call_count_{0};
};

// Returns from stream_response() at once and delivers the response later from
// its own thread, which the LLMProvider contract allows. A continuation the
// Agent defers until stream_response() returns must still run when the final
// chunk only arrives after that.
class AsyncTruncatingProvider : public core::llm::LLMProvider {
public:
    void stream_response(const core::llm::ChatRequest&,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        const int call = call_count_.fetch_add(1, std::memory_order_acq_rel);
        std::lock_guard lock(workers_mutex_);
        workers_.emplace_back([callback = std::move(callback), call] {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            core::llm::StreamChunk partial;
            partial.content = "partial answer ";
            callback(partial);

            core::llm::StreamChunk terminal;
            terminal.is_final = true;
            if (call == 0) {
                terminal.stop_reason = "max_tokens";
            } else {
                terminal.content = "finished";
            }
            callback(terminal);
        });
    }

    [[nodiscard]] int call_count() const noexcept {
        return call_count_.load(std::memory_order_acquire);
    }

    // Joined by the test, never from a worker: the last reference to the Agent
    // (and so to this provider) may be released on one.
    void join_all() {
        for (std::size_t joined = 0;;) {
            std::thread worker;
            {
                std::lock_guard lock(workers_mutex_);
                if (joined == workers_.size()) return;
                worker = std::move(workers_[joined++]);
            }
            if (worker.joinable()) worker.join();
        }
    }

private:
    std::atomic<int> call_count_{0};
    std::mutex workers_mutex_;
    std::vector<std::thread> workers_;
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

TEST_CASE("Memory write failures cannot prevent the agent completing the task",
          "[agent][memory][loop][regression]") {
    bool correct = false;
    bool unsolicited = false;
    SECTION("a corrected write is accepted") { correct = true; }
    SECTION("repeated failures leave other tools available") {}
    SECTION("unsolicited memory calls are rejected and get a final response") { unsolicited = true; }

    TempDir temp(std::filesystem::temp_directory_path() /
        std::format("filo_memory_loop_{}", std::chrono::steady_clock::now().time_since_epoch().count()));
    core::tools::ToolManager tools;
    tools.register_tool(std::make_shared<core::tools::MemoryTool>(
        core::memory::MemoryStore{temp.path() / "memory.json"}));
    tools.register_tool(std::make_shared<NoopLoopTool>());
    auto provider = std::make_shared<MemoryFailureProvider>(correct, unsolicited);
    auto context = test_support::make_session_context(core::workspace::WorkspaceSnapshot{
        .primary = temp.path(), .enforce = true, .version = 1});
    auto agent = std::make_shared<core::agent::Agent>(provider, tools, context);
    agent->set_permission_fn([](std::string_view, std::string_view) { return true; });
    int loop_breaks = 0;
    agent->set_loop_break_fn([&](int) { ++loop_breaks; });
    send_and_wait(agent, "Answer the question.");
    CHECK(loop_breaks == 0);
    CHECK(std::ranges::any_of(agent->get_history(), [](const auto& message) {
        return message.role == "assistant" && message.content == "The requested answer.";
    }));
    REQUIRE(provider->requests_.size() == (correct ? 3 : 4));
    const auto has_tool = [](const auto& request, std::string_view name) {
        return std::ranges::any_of(request.tools, [&](const auto& tool) {
            return tool.function.name == name;
        });
    };
    CHECK(has_tool(provider->requests_[1], core::tools::names::kMemory));
    CHECK(has_tool(provider->requests_[2], core::tools::names::kMemory) == correct);
    if (correct) {
        CHECK(std::ranges::any_of(agent->get_history(), [](const auto& message) {
            return message.role == "tool" && message.name == core::tools::names::kMemory
                && message.content.contains("\"ok\":true");
        }));
    }
    if (!correct) {
        CHECK(has_tool(provider->requests_[2], kLoopToolName));
        CHECK(std::ranges::any_of(provider->requests_[2].messages, [](const auto& message) {
            return message.synthetic && message.content.contains("Memory writes failed repeatedly");
        }));
        if (unsolicited) CHECK(provider->requests_[3].tools.empty());
        const auto history = agent->get_history();
        CHECK(std::ranges::count_if(history, [](const auto& message) {
            return message.role == "tool" && message.content.contains("memory_content_too_long");
        }) == 2);
        if (unsolicited) CHECK(std::ranges::any_of(history, [](const auto& message) {
            return message.role == "tool" && message.content.contains("Memory is unavailable");
        }));
    }
    send_and_wait(agent, "A new turn.");
    CHECK(has_tool(provider->requests_.back(), core::tools::names::kMemory));
}

TEST_CASE("A denied tool pauses even alongside a disabled memory call",
          "[agent][memory][loop][permission]") {
    TempDir temp(std::filesystem::temp_directory_path() /
        std::format("filo_memory_denial_{}", std::chrono::steady_clock::now().time_since_epoch().count()));
    core::tools::ToolManager tools;
    tools.register_tool(std::make_shared<core::tools::MemoryTool>(
        core::memory::MemoryStore{temp.path() / "memory.json"}));
    tools.register_tool(std::make_shared<NamedNoopTool>("memory_denied_companion", true));
    auto provider = std::make_shared<MemoryFailureProvider>(false, true, true);
    auto agent = std::make_shared<core::agent::Agent>(provider, tools,
        test_support::make_session_context(core::workspace::WorkspaceSnapshot{
            .primary = temp.path(), .enforce = true, .version = 1}));
    agent->set_permission_fn([](std::string_view name, std::string_view) {
        return name != "memory_denied_companion";
    });

    send_and_wait(agent, "Answer the question.");
    CHECK(provider->requests_.size() == 3);
    const auto history = agent->get_history();
    CHECK(std::ranges::any_of(history, [](const auto& message) {
        return message.role == "tool" && message.content.contains("denied by user");
    }));
    CHECK(std::ranges::none_of(history, [](const auto& message) {
        return message.role == "assistant" && message.content == "The requested answer.";
    }));
}

TEST_CASE("Agent prepares owned high-context requests without losing history",
          "[agent][context-edit][regression]") {
    auto provider = std::make_shared<CapturingProvider>();
    provider->context_size = 200'000;
    core::tools::ToolManager tools;
    auto agent = std::make_shared<core::agent::Agent>(
        provider, tools, test_support::make_workspace_session_context());
    std::vector<core::llm::Message> history;
    bool expects_edit = false;

    SECTION("no eligible observation results") {
        for (int i = 0; i < 12; ++i)
            history.push_back({.role = "user", .content = std::string(44'000, 'a' + i)});
    }
    SECTION("only superseded request observations are edited") {
        expects_edit = true;
        for (const std::string id : {"old", "latest"}) {
            core::llm::ToolCall call;
            call.id = id;
            call.function.name = "read";
            call.function.arguments = R"({"path":"src/a.cpp"})";
            history.push_back({.role = "assistant", .tool_calls = {std::move(call)}});
            history.push_back({.role = "tool",
                .content = R"({"content":")" + std::string(200'000, 'x') + R"("})",
                .name = "read", .tool_call_id = id});
        }
        for (int i = 0; i < 8; ++i)
            history.push_back({.role = "assistant", .content = std::string(16'000, 'a' + i)});
    }

    agent->load_history(history, "", "BUILD");
    send_and_wait(agent, "Continue using the existing evidence.");
    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 1);
    REQUIRE(requests.front().messages.size() >= history.size());
    CHECK(std::ranges::any_of(requests.front().messages, [](const auto& message) {
        return message.content.contains("Continue using the existing evidence.");
    }));
    const auto durable = agent->get_history();
    for (const auto& original : history)
        CHECK(std::ranges::any_of(durable, [&](const auto& message) {
            return message.role == original.role && message.content == original.content
                && message.tool_call_id == original.tool_call_id;
        }));

    if (expects_edit) {
        const auto old = std::ranges::find_if(requests.front().messages, [](const auto& message) {
            return message.role == "tool" && message.tool_call_id == "old";
        });
        REQUIRE(old != requests.front().messages.end());
        CHECK(old->content.contains("superseded_tool_result"));
    } else {
        for (const auto& original : history)
            CHECK(std::ranges::any_of(requests.front().messages, [&](const auto& message) {
                return message.role == original.role && message.content == original.content;
            }));
    }
}

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

TEST_CASE("Agent never executes tool calls from a failed terminal response",
          "[agent][tool][error][regression]") {
    const std::string tool_name = "failed_terminal_counting_tool";
    auto tool = std::make_shared<CountingTool>(tool_name);
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(tool);
    auto provider = std::make_shared<ToolThenErrorProvider>(tool_name);
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    send_and_wait(agent, "Do not run a provisional tool after failure.");

    CHECK(provider->call_count() == 1);
    CHECK(tool->execution_count() == 0);
    const auto history = agent->get_history();
    CHECK(std::ranges::none_of(history, [](const core::llm::Message& message) {
        return !message.tool_calls.empty()
            || !message.continuation_items.empty()
            || message.role == "tool";
    }));
    REQUIRE_FALSE(history.empty());
    CHECK_THAT(history.back().content,
               Catch::Matchers::ContainsSubstring("provider stream failed"));
}

TEST_CASE("Agent drops terminal nameless tool calls before they enter history",
          "[agent][tool][history][replay][regression]") {
    const std::string valid_tool_name = "mixed_valid_and_nameless_tool";
    auto tool = std::make_shared<CountingTool>(valid_tool_name);
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(tool);
    auto provider = std::make_shared<MixedValidAndNamelessToolProvider>(valid_tool_name);
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    send_and_wait(agent, "Run the valid tool only.");

    CHECK(provider->call_count() == 2);
    CHECK(tool->execution_count() == 1);
    const auto history = agent->get_history();
    const auto tool_use = std::ranges::find_if(
        history,
        [](const core::llm::Message& message) {
            return message.role == "assistant" && !message.tool_calls.empty();
        });
    REQUIRE(tool_use != history.end());
    REQUIRE(tool_use->tool_calls.size() == 1);
    CHECK(tool_use->tool_calls.front().id == "valid-tool-call");
    CHECK(tool_use->tool_calls.front().function.name == valid_tool_name);
    CHECK(std::ranges::none_of(history, [](const core::llm::Message& message) {
        return message.role == "tool" && message.tool_call_id == "nameless-tool-call";
    }));
}

TEST_CASE("Clearing history invalidates a late provider callback",
          "[agent][history][generation][regression]") {
    auto provider = std::make_shared<DeferredFinalProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;
    std::jthread sender([&] {
        agent->send_message(
            "old session request",
            [](const std::string&) {},
            [](const std::string&, const std::string&) {},
            [&] {
                {
                    std::lock_guard lock(done_mutex);
                    done = true;
                }
                done_cv.notify_one();
            });
    });

    provider->wait_until_entered();
    REQUIRE(agent->turn_in_progress());
    agent->clear_history();
    provider->release();

    {
        std::unique_lock lock(done_mutex);
        REQUIRE(done_cv.wait_for(
            lock,
            std::chrono::seconds(3),
            [&] { return done; }));
    }
    sender.join();
    CHECK_FALSE(agent->turn_in_progress());
    CHECK(agent->get_history().empty());
}

TEST_CASE("Clearing history discards late tool results and prevents stale recursion",
          "[agent][history][tool][generation][regression]") {
    const std::string tool_name = "blocking_history_generation_tool";
    auto tool = std::make_shared<BlockingHistoryTool>(tool_name);
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(tool);
    auto provider = std::make_shared<ToolThenTextCapturingProvider>(tool_name);
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;
    agent->send_message(
        "run the blocking tool",
        [](const std::string&) {},
        [](const std::string&, const std::string&) {},
        [&] {
            {
                std::lock_guard lock(done_mutex);
                done = true;
            }
            done_cv.notify_one();
        });

    tool->wait_until_entered();
    const auto in_flight_snapshot = agent->get_history();
    CHECK(std::ranges::none_of(
        in_flight_snapshot,
        [](const core::llm::Message& message) {
            return !message.tool_calls.empty();
        }));

    agent->clear_history();
    tool->release();
    {
        std::unique_lock lock(done_mutex);
        REQUIRE(done_cv.wait_for(
            lock,
            std::chrono::seconds(3),
            [&] { return done; }));
    }

    CHECK(provider->call_count() == 1);
    CHECK(agent->get_history().empty());
}

namespace {

// Runs until its invocation is cancelled, like a tool that honours Ctrl+C.
class CancellableWaitTool final : public core::tools::Tool {
public:
    explicit CancellableWaitTool(std::string name) : name_(std::move(name)) {}

    [[nodiscard]] core::tools::ToolDefinition get_definition() const override {
        return {
            .name = name_,
            .title = "Cancellable wait tool",
            .description = "Waits until its invocation is cancelled.",
            .parameters = {},
            .annotations = {.read_only_hint = true, .idempotent_hint = true},
        };
    }

    [[nodiscard]] std::string execute(const std::string&,
                                      const core::context::SessionContext&) override {
        return R"({"error":"needs an invocation"})";
    }

    [[nodiscard]] std::string execute(
        const std::string&,
        const core::tools::ToolInvocationContext& invocation) override {
        entered_.store(true, std::memory_order_release);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!(invocation.cancellation_requested && invocation.cancellation_requested())) {
            if (std::chrono::steady_clock::now() > deadline) return R"({"output":"timed out"})";
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        return R"({"output":"partial"})";
    }

    void wait_until_entered() const {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
        while (!entered_.load(std::memory_order_acquire)) {
            REQUIRE(std::chrono::steady_clock::now() < deadline);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
    }

private:
    std::string name_;
    std::atomic<bool> entered_{false};
};

} // namespace

TEST_CASE("Interrupting running tools cancels the tool but not the turn",
          "[agent][tool][interrupt]") {
    const std::string tool_name = "cancellable_wait_interrupt_tool";
    auto tool = std::make_shared<CancellableWaitTool>(tool_name);
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(tool);
    auto provider = std::make_shared<ToolThenTextCapturingProvider>(tool_name);
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    CHECK_FALSE(agent->tools_running());
    CHECK_FALSE(agent->interrupt_running_tools());

    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;
    agent->send_message(
        "run the waiting tool",
        [](const std::string&) {},
        [](const std::string&, const std::string&) {},
        [&] {
            {
                std::lock_guard lock(done_mutex);
                done = true;
            }
            done_cv.notify_one();
        });

    tool->wait_until_entered();
    CHECK(agent->tools_running());
    CHECK(agent->interrupt_running_tools());
    {
        std::unique_lock lock(done_mutex);
        REQUIRE(done_cv.wait_for(lock, std::chrono::seconds(3), [&] { return done; }));
    }

    // The turn went on: the model saw the interrupted result and answered.
    CHECK_FALSE(agent->is_stop_requested());
    CHECK_FALSE(agent->tools_running());
    CHECK(provider->call_count() == 2);
    const auto history = agent->get_history();
    const auto tool_message = std::ranges::find(history, std::string("tool"), &core::llm::Message::role);
    REQUIRE(tool_message != history.end());
    CHECK_THAT(tool_message->content, Catch::Matchers::StartsWith(R"({"interrupted_by_user":")"));
    CHECK_THAT(tool_message->content, Catch::Matchers::ContainsSubstring(R"("output":"partial")"));
    CHECK(history.back().role == "assistant");
    CHECK(history.back().content == "finished");
}

TEST_CASE("Provider changes apply after the complete tool turn",
          "[agent][provider][tool][regression]") {
    const std::string tool_name = "blocking_provider_freeze_tool";
    auto tool = std::make_shared<BlockingHistoryTool>(tool_name);
    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(tool);
    auto original_provider =
        std::make_shared<ToolThenTextCapturingProvider>(tool_name);
    auto replacement_provider = std::make_shared<CapturingProvider>();
    auto agent = std::make_shared<core::agent::Agent>(
        original_provider,
        tool_manager,
        test_support::make_workspace_session_context());

    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;
    agent->send_message(
        "keep this turn on one provider",
        [](const std::string&) {},
        [](const std::string&, const std::string&) {},
        [&] {
            {
                std::lock_guard lock(done_mutex);
                done = true;
            }
            done_cv.notify_one();
        });

    tool->wait_until_entered();
    agent->set_provider(replacement_provider);
    agent->set_active_provider_name("replacement");
    agent->set_active_model("replacement-model");
    tool->release();
    {
        std::unique_lock lock(done_mutex);
        REQUIRE(done_cv.wait_for(
            lock,
            std::chrono::seconds(3),
            [&] { return done; }));
    }

    CHECK(original_provider->call_count() == 2);
    CHECK(replacement_provider->requests_snapshot().empty());
}

TEST_CASE("Loading a malformed tool transcript keeps only its complete prefix",
          "[agent][history][load][regression]") {
    auto provider = std::make_shared<CapturingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    core::llm::Message tool_use{
        .role = "assistant",
        .content = "starting",
    };
    tool_use.tool_calls.push_back(core::llm::ToolCall{
        .id = "missing-result",
        .type = "function",
        .function = {
            .name = "grep_search",
            .arguments = "{}",
        },
    });
    agent->load_history(
        {
            core::llm::Message{.role = "user", .content = "keep me"},
            std::move(tool_use),
            core::llm::Message{.role = "assistant", .content = "interleaved"},
        },
        {},
        "BUILD");

    const auto restored = agent->get_history();
    REQUIRE(restored.size() == 1);
    CHECK(restored.front().role == "user");
    CHECK(restored.front().content == "keep me");

    send_and_wait(agent, "continue safely");
    REQUIRE(provider->requests_snapshot().size() == 1);
}

TEST_CASE("Loading a complete transcript scrubs malformed tool arguments",
          "[agent][history][load][tool-call][regression]") {
    auto provider = std::make_shared<CapturingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    core::llm::Message assistant{
        .role = "assistant",
        .tool_calls = {core::llm::ToolCall{
            .id = "call_replayed",
            .type = "function",
            .function = {
                .name = "edit",
                .arguments = "{\"path\":",
            },
        }},
    };
    core::llm::Message result{
        .role = "tool",
        .content = R"({"error":"Invalid tool arguments"})",
        .name = "edit",
        .tool_call_id = "call_replayed",
    };
    agent->load_history(
        {std::move(assistant), std::move(result)},
        {},
        "BUILD");

    const auto restored = agent->get_history();
    REQUIRE(restored.size() == 2);
    REQUIRE(restored.front().tool_calls.size() == 1);
    CHECK(restored.front().tool_calls.front().function.arguments == "{}");
}

TEST_CASE("Loading a legacy transcript removes nameless calls without truncating later history",
          "[agent][history][load][tool-call][replay][regression]") {
    auto provider = std::make_shared<CapturingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    core::llm::Message assistant{
        .role = "assistant",
        .content = "I will inspect the project.",
        .tool_calls = {
            {
                .id = "call_valid",
                .type = "function",
                .function = {.name = "read", .arguments = R"({"path":"README.md"})"},
            },
            {
                .id = "call_nameless",
                .type = "function",
                .function = {.name = "", .arguments = "{}"},
            },
        },
    };
    agent->load_history(
        {
            core::llm::Message{.role = "user", .content = "Inspect the project."},
            std::move(assistant),
            core::llm::Message{
                .role = "tool",
                .content = R"({"output":"README"})",
                .name = "read",
                .tool_call_id = "call_valid",
            },
            core::llm::Message{
                .role = "tool",
                .content = R"({"error":"Invalid tool arguments: tool not found"})",
                .tool_call_id = "call_nameless",
            },
            core::llm::Message{.role = "assistant", .content = "The README is present."},
        },
        {},
        "BUILD");

    const auto restored = agent->get_history();
    REQUIRE(restored.size() == 4);
    REQUIRE(restored[1].tool_calls.size() == 1);
    CHECK(restored[1].tool_calls.front().id == "call_valid");
    CHECK(restored[1].tool_calls.front().function.name == "read");
    CHECK(std::ranges::none_of(restored, [](const core::llm::Message& message) {
        return message.role == "tool" && message.tool_call_id == "call_nameless";
    }));
    CHECK(restored.back().content == "The README is present.");

    core::llm::ChatRequest claude_request;
    claude_request.model = "claude-opus-5";
    claude_request.messages = restored;
    const auto payload = core::llm::protocols::ClaudeSerializer::serialize(claude_request);
    CHECK_THAT(payload, !Catch::Matchers::ContainsSubstring(R"("name":"")"));
}

TEST_CASE("Appending assistant tool history scrubs malformed arguments",
          "[agent][history][append][tool-call][regression]") {
    auto provider = std::make_shared<CapturingProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    agent->append_history_message(core::llm::Message{
        .role = "assistant",
        .tool_calls = {core::llm::ToolCall{
            .id = "call_appended",
            .function = {
                .name = "edit",
                .arguments = "{\"path\":",
            },
        }},
    });
    agent->append_history_message(core::llm::Message{
        .role = "tool",
        .content = "{}",
        .name = "edit",
        .tool_call_id = "call_appended",
    });

    const auto history = agent->get_history();
    REQUIRE(history.size() == 2);
    REQUIRE(history.front().tool_calls.size() == 1);
    CHECK(history.front().tool_calls.front().function.arguments == "{}");
}

TEST_CASE("Agent exposes one delegation API even when the MCP registry is shared",
          "[agent][orchestration][tools]") {
    auto provider = std::make_shared<CapturingProvider>();
    auto& tools = core::tools::ToolManager::get_instance();
    tools.register_tool(std::make_shared<core::tools::TaskTool>());
    auto agent = std::make_shared<core::agent::Agent>(
        provider, tools, test_support::make_workspace_session_context());
    send_and_wait(agent, "Inspect the available tools");
    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 1);
    CHECK(std::ranges::any_of(requests.front().tools, [](const auto& tool) { return tool.function.name == "task"; }));
    CHECK(std::ranges::none_of(requests.front().tools, [](const auto& tool) { return tool.function.name == "delegate_task"; }));
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

TEST_CASE("Agent retracts in-progress output when the transport resets an attempt",
          "[agent][loop][retry]") {
    auto provider = std::make_shared<ResetThenRecoverProvider>();
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    std::mutex capture_mutex;
    std::string streamed_text;
    std::vector<std::string> status_logs;
    int reset_count = 0;
    std::mutex done_mutex;
    std::condition_variable done_cv;
    bool done = false;

    agent->send_message(
        "continue the task",
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
            .on_attempt_reset = [&]() {
                std::lock_guard lock(capture_mutex);
                ++reset_count;
                streamed_text.clear();
            },
        });

    {
        std::unique_lock lock(done_mutex);
        REQUIRE(done_cv.wait_for(lock, std::chrono::seconds(5), [&]() { return done; }));
    }

    {
        std::lock_guard lock(capture_mutex);
        CHECK(streamed_text == "recovered after generation failure");
        REQUIRE(status_logs.size() == 1);
        CHECK_THAT(status_logs.front(),
                   Catch::Matchers::ContainsSubstring("retrying after a generation failure"));
        CHECK(reset_count == 1);
    }

    const auto history = agent->get_history();
    REQUIRE_FALSE(history.empty());
    CHECK(history.back().role == "assistant");
    CHECK(history.back().content == "recovered after generation failure");
    CHECK(history.back().tool_calls.empty());
    CHECK(history.back().continuation_items.empty());
    CHECK_FALSE(agent->last_turn_failed());
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
                   Catch::Matchers::ContainsSubstring("claude ended this turn"));
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
        core::llm::protocols::ClaudeSerializer::serialize(requests[0]);
    const auto second_payload =
        core::llm::protocols::ClaudeSerializer::serialize(requests[1]);
    const auto third_payload =
        core::llm::protocols::ClaudeSerializer::serialize(requests[2]);
    const auto first_messages = first_payload.find(R"(,"messages":[)");
    const auto second_messages = second_payload.find(R"(,"messages":[)");
    const auto third_messages = third_payload.find(R"(,"messages":[)");
    REQUIRE(first_messages != std::string::npos);
    REQUIRE(second_messages != std::string::npos);
    REQUIRE(third_messages != std::string::npos);
    // `cch` signs the whole body, so it differs on every request by design;
    // everything else before the messages must stay byte-identical.
    const auto unsigned_prefix = [](const std::string& payload, std::size_t end) {
        auto prefix = payload.substr(0, end);
        if (const auto token = prefix.find("cch="); token != std::string::npos) prefix.replace(token + 4, 5, "00000");
        return prefix;
    };
    CHECK(unsigned_prefix(first_payload, first_messages)
          == unsigned_prefix(second_payload, second_messages));
    CHECK(unsigned_prefix(second_payload, second_messages)
          == unsigned_prefix(third_payload, third_messages));
}

TEST_CASE("Agent keeps live task state out of the cached prompt prefix",
          "[agent][prompt][cache]") {
    TempDir tmp{
        std::filesystem::temp_directory_path()
        / std::format("filo_prompt_task_state_{}",
                      std::chrono::steady_clock::now().time_since_epoch().count())
    };
    ScopedCurrentPath scoped_cwd(tmp.path());

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
        "prompt-task-state-session");
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        session_context);

    std::string graph_state = "GRAPH_V1";
    agent->set_goal_graph_context_fn([&graph_state] { return graph_state; });

    send_and_wait(agent, "QUESTION_ALPHA");
    graph_state = "GRAPH_V2";
    send_and_wait(agent, "QUESTION_BETA");

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 2);
    const auto& first_plan = requests[0].prompt_plan;
    const auto& second_plan = requests[1].prompt_plan;

    // The live task state is the plan's Dynamic tail ...
    CHECK_THAT(first_plan.render_from(core::context::PromptStability::Dynamic),
               Catch::Matchers::ContainsSubstring("GRAPH_V1"));
    CHECK_THAT(second_plan.render_from(core::context::PromptStability::Dynamic),
               Catch::Matchers::ContainsSubstring("GRAPH_V2"));
    // ... and the prefix above it is byte-identical while that tail churns.
    CHECK(first_plan.render_through(core::context::PromptStability::Session)
          == second_plan.render_through(core::context::PromptStability::Session));
    CHECK(first_plan.render_through(core::context::PromptStability::Session)
              .find("GRAPH_") == std::string::npos);

    // The system message keeps the full render for legacy consumers; the wire
    // layout is the serializer's job.
    CHECK(requests[0].messages[0].content == first_plan.render());

    // On the wire the task state appears only after the conversation, as the
    // trailing system message — outside the prefix an implicit cache keys on.
    core::llm::Serializer::Options options;
    options.system_prompt_layout =
        core::llm::Serializer::SystemPromptLayout::DynamicTail;
    const auto first_payload = core::llm::Serializer::serialize(requests[0], options);
    const auto graph_at = first_payload.find("GRAPH_V1");
    const auto question_at = first_payload.find("QUESTION_ALPHA");
    REQUIRE(graph_at != std::string::npos);
    REQUIRE(question_at != std::string::npos);
    CHECK(graph_at > question_at);
    CHECK(first_payload.find("GRAPH_V1", graph_at + 1) == std::string::npos);
}

TEST_CASE("Agent keeps attached files out of the stable prompt prefix",
          "[agent][prompt][attachments]") {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    TempDir tmp{std::filesystem::temp_directory_path()
                / std::format("filo_prompt_attach_project_{}", stamp)};
    TempDir outside{std::filesystem::temp_directory_path()
                    / std::format("filo_prompt_attach_outside_{}", stamp)};
    ScopedCurrentPath scoped_cwd(tmp.path());
    const auto screenshot = outside.path() / "Screenshot.png";
    {
        std::ofstream out(screenshot, std::ios::binary);
        out << "png-bytes";
    }
    std::filesystem::create_directories(outside.path() / "library");

    auto provider = std::make_shared<CapturingProvider>();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        core::tools::ToolManager::get_instance(),
        test_support::make_session_context(
            core::workspace::WorkspaceSnapshot{
                .primary = tmp.path(),
                .enforce = true,
                .version = 1,
            },
            core::context::SessionTransport::cli,
            "prompt-attach-session"));

    send_and_wait(agent, "first");
    REQUIRE(agent->grant_workspace_paths({screenshot}) == 1);
    REQUIRE(agent->workspace_snapshot().allows_read(screenshot));
    send_and_wait(agent, "second");
    REQUIRE(agent->grant_workspace_paths({outside.path() / "library"}) == 1);
    send_and_wait(agent, "third");

    const auto requests = provider->requests_snapshot();
    REQUIRE(requests.size() == 3);
    const auto first = requests[0].prompt_plan.render();
    const auto second = requests[1].prompt_plan.render();
    const auto third = requests[2].prompt_plan.render();

    // Attaching a file leaves the cached prefix byte-identical ...
    CHECK(first == second);
    CHECK(second.find("Screenshot.png") == std::string::npos);
    CHECK(second.find("Additional directories:") == std::string::npos);
    // ... while granting a real directory still reaches the prompt.
    CHECK(third != second);
    CHECK_THAT(third, Catch::Matchers::ContainsSubstring("Additional directories:"));
    CHECK_THAT(third, Catch::Matchers::ContainsSubstring("library"));
    CHECK(third.find("Screenshot.png") == std::string::npos);
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
    core::tools::ToolManager tool_manager;
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

    bool retained_user_request = false;
    for (const auto& message : requests.back().messages) {
        if (message.role == "user"
            && message.content.find(std::string(100, 'x')) != std::string::npos) {
            retained_user_request = true;
            break;
        }
    }

    CHECK(retained_user_request);
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

TEST_CASE("AUTO enforces fresh verification after a successful mutation",
          "[agent][auto][loop][quality]") {
  const auto temp_path =
      std::filesystem::temp_directory_path() /
      std::format("filo_auto_quality_{}",
                  std::chrono::steady_clock::now().time_since_epoch().count());
  TempDir temp(temp_path);
  ScopedCurrentPath cwd(temp.path());
  std::filesystem::create_directories(temp.path() / ".filo");
  {
    std::ofstream config(temp.path() / ".filo" / "verification.json");
    config << R"({"version":1,"recipes":[{"id":"quality","name":"Project quality","kind":"test","executable":"test","arguments":["-f","auto-quality.txt"],"required":true}]})";
  }

  auto &tool_manager = core::tools::ToolManager::get_instance();
  tool_manager.register_tool(std::make_shared<core::tools::WriteFileTool>());
  tool_manager.register_tool(std::make_shared<core::tools::ShellTool>(tool_manager.shell_sessions()));
  tool_manager.register_tool(std::make_shared<core::tools::VerificationTool>());

  auto provider = std::make_shared<AutoQualityProvider>();
  auto agent = std::make_shared<core::agent::Agent>(
      provider, tool_manager,
      test_support::make_session_context(core::workspace::WorkspaceSnapshot{
          .primary = temp.path(),
          .enforce = true,
          .version = 1,
      }));
  agent->set_mode("AUTO");
  agent->set_permission_fn(
      [](std::string_view, std::string_view) { return true; });

  std::string statuses;
  send_and_wait(agent, "Implement the file change and verify it.",
                core::agent::Agent::TurnCallbacks{
                    .on_status_log =
                        [&](const std::string &status) { statuses += status; },
                });

  CHECK(provider->call_count() == 2);
  CHECK(std::filesystem::exists(temp.path() / "auto-quality.txt"));
  CHECK_FALSE(agent->last_turn_failed());
  CHECK_THAT(statuses,
             Catch::Matchers::ContainsSubstring("AUTO · orchestrated path"));
  CHECK_THAT(statuses, Catch::Matchers::ContainsSubstring(
                           "AUTO · completion quality gate passed"));

  const auto requests = provider->requests_snapshot();
  REQUIRE(requests.size() == 3);
  CHECK_THAT(requests[1].prompt_plan.render(),
             Catch::Matchers::ContainsSubstring("AUTO execution contract"));
  CHECK_THAT(
      requests[1].prompt_plan.render(),
      Catch::Matchers::ContainsSubstring("Repository verification recipes"));
  CHECK_FALSE(std::ranges::any_of(
      requests.back().messages, [](const core::llm::Message &message) {
        return message.synthetic && message.content.contains("AUTO quality gate");
      }));
}

TEST_CASE("AUTO explore subagents complete under a parent writer transaction",
          "[agent][auto][loop][lease]") {
  const auto temp_path =
      std::filesystem::temp_directory_path() /
      std::format("filo_auto_explore_{}",
                  std::chrono::steady_clock::now().time_since_epoch().count());
  TempDir temp(temp_path);
  ScopedCurrentPath cwd(temp.path());

  auto &tool_manager = core::tools::ToolManager::get_instance();
  auto provider = std::make_shared<AutoExploreProvider>();
  auto leases = std::make_shared<core::scm::WorkspaceLeaseRegistry>();
  auto agent = std::make_shared<core::agent::Agent>(
      provider, tool_manager,
      test_support::make_session_context(core::workspace::WorkspaceSnapshot{
          .primary = temp.path(),
          .enforce = true,
          .version = 1,
      }),
      core::agent::ToolResultStore::default_root(),
      std::shared_ptr<core::power::SleepInhibitor>{},
      std::shared_ptr<core::session::SessionStatsRegistry>{},
      nullptr,
      std::shared_ptr<core::memory::MemorySystem>{},
      leases);
  agent->set_mode("AUTO");
  agent->set_permission_fn(
      [](std::string_view, std::string_view) { return true; });

  send_and_wait(
      agent,
      "Debug the deadlock in the task scheduler, fix it, and add regression tests.");

  CHECK_FALSE(agent->last_turn_failed());
  CHECK(provider->explore_count() == 2);
  const auto requests = provider->requests_snapshot();
  CHECK(std::ranges::any_of(requests, [](const core::llm::ChatRequest &request) {
    return std::ranges::any_of(request.messages, [](const auto &message) {
      return message.role == "system" &&
             message.content.contains("running in RESEARCH mode");
    });
  }));
  CHECK_FALSE(std::ranges::any_of(
      requests, [](const core::llm::ChatRequest &request) {
        const bool delegated = std::ranges::any_of(
            request.messages, [](const auto &message) {
              return message.content.contains("[Delegated worker]");
            });
        const bool auto_system = std::ranges::any_of(
            request.messages, [](const auto &message) {
              return message.role == "system" &&
                     message.content.contains("running in AUTO mode");
            });
        return delegated && auto_system;
      }));
}

TEST_CASE("A turn publishes and records the net changes of its file tools",
          "[agent][changes]") {
    const auto temp_path =
        std::filesystem::temp_directory_path() /
        std::format("filo_turn_file_changes_{}",
                    std::chrono::steady_clock::now().time_since_epoch().count());
    TempDir temp(temp_path);
    {
        std::ofstream(temp.path() / "notes.txt") << "first\n";
    }

    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<core::tools::WriteFileTool>());
    auto provider = std::make_shared<FileEditingProvider>(std::vector<std::string>{
        R"({"file_path":"notes.txt","content":"first\nsecond\n"})",
        R"({"file_path":"notes.txt","content":"first\nsecond\nthird\n"})",
    });
    auto agent = std::make_shared<core::agent::Agent>(
        provider, tool_manager,
        test_support::make_session_context(core::workspace::WorkspaceSnapshot{
            .primary = temp.path(),
            .enforce = true,
            .version = 1,
        }));
    agent->set_permission_fn([](std::string_view, std::string_view) { return true; });

    std::mutex events_mutex;
    std::vector<std::string> events;
    std::vector<core::changes::TurnChanges> published;
    send_and_wait(agent, "Extend the notes.",
                  core::agent::Agent::TurnCallbacks{
                      .on_step_begin = [&] {
                          std::lock_guard lock(events_mutex);
                          events.emplace_back("step");
                      },
                      .on_tool_finish =
                          [&](const core::llm::ToolCall&, const core::llm::Message&) {
                              std::lock_guard lock(events_mutex);
                              events.emplace_back("tool");
                          },
                      .on_turn_changes =
                          [&](const core::changes::TurnChanges& changes) {
                              std::lock_guard lock(events_mutex);
                              events.emplace_back("changes");
                              published.push_back(changes);
                          },
                  });

    // Published once, after the last step of the turn, never while it runs.
    REQUIRE(!events.empty());
    CHECK(events.back() == "changes");
    CHECK(std::ranges::count(events, std::string("changes")) == 1);
    CHECK(std::ranges::count(events, std::string("tool")) == 2);
    REQUIRE(published.size() == 1);
    REQUIRE(published[0].files.size() == 1);
    const auto& change = published[0].files[0];
    CHECK(change.path == "notes.txt");
    CHECK(change.kind == core::changes::FileChangeKind::Modified);
    CHECK(change.added == 2);
    CHECK(change.deleted == 0);

    const auto history = agent->get_history();
    const auto last_assistant = std::ranges::find_if(
        history | std::views::reverse,
        [](const core::llm::Message& message) { return message.role == "assistant"; });
    REQUIRE(last_assistant != (history | std::views::reverse).end());
    CHECK(last_assistant->content == "edited");
    CHECK(last_assistant->turn_changes == published[0]);
    CHECK(std::ranges::count_if(history, [](const core::llm::Message& message) {
        return !message.turn_changes.empty();
    }) == 1);
}

TEST_CASE("Marking a turn reverted stamps the newest summary, and only once",
          "[agent][changes]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto provider = std::make_shared<FileEditingProvider>(std::vector<std::string>{});
    core::agent::Agent agent(
        provider, tool_manager,
        test_support::make_session_context(core::workspace::WorkspaceSnapshot{
            .primary = std::filesystem::temp_directory_path(),
            .enforce = true,
            .version = 1,
        }));

    // Nothing has been recorded yet, so there is nothing to mark.
    CHECK_FALSE(agent.mark_latest_turn_changes_reverted());

    core::llm::Message answer{.role = "assistant", .content = "done"};
    answer.turn_changes.files = {core::changes::FileChange{
        .kind = core::changes::FileChangeKind::Modified,
        .path = "notes.txt",
        .diff = "--- a/notes.txt\n+++ b/notes.txt\n@@ -1 +1 @@\n-first\n+second\n",
        .added = 1,
        .deleted = 1,
    }};
    agent.append_history_message(core::llm::Message{.role = "user", .content = "edit"});
    agent.append_history_message(answer);

    CHECK(agent.mark_latest_turn_changes_reverted());
    const auto history = agent.get_history();
    const auto marked = std::ranges::find_if(
        history, [](const core::llm::Message& message) {
            return !message.turn_changes.empty();
        });
    REQUIRE(marked != history.end());
    // The summary is still what the turn did; only its standing changed, and it
    // changed in the history a resumed session is built from.
    CHECK(marked->turn_changes.reverted);
    CHECK(marked->turn_changes.files == answer.turn_changes.files);

    // A second revert of the same turn is the caller's to refuse, and says so.
    CHECK_FALSE(agent.mark_latest_turn_changes_reverted());
}

TEST_CASE("Review comments are kept with the turn they belong to",
          "[agent][changes]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto provider = std::make_shared<FileEditingProvider>(std::vector<std::string>{});
    core::agent::Agent agent(
        provider, tool_manager,
        test_support::make_session_context(core::workspace::WorkspaceSnapshot{
            .primary = std::filesystem::temp_directory_path(),
            .enforce = true,
            .version = 1,
        }));

    core::llm::Message answer{.role = "assistant", .content = "done"};
    answer.turn_changes.id = "turn-1";
    answer.turn_changes.files = {core::changes::FileChange{.path = "notes.txt", .diff = "@@ -1 +1 @@\n-a\n+b\n"}};
    agent.append_history_message(core::llm::Message{.role = "user", .content = "edit"});
    agent.append_history_message(answer);

    const std::vector<core::review::Annotation> comments{{
        .id = "15319332-3c51-4bd8-a391-30de79a90551",
        .path = "notes.txt",
        .side = "new",
        .start_line = 1,
        .end_line = 1,
        .selected_text = "b",
        .comment = "Keep a.",
        .revision = "revision-1",
    }};
    CHECK(agent.set_turn_annotations("turn-1", comments));
    const auto history = agent.get_history();
    CHECK(history.back().turn_changes.annotations == comments);
    CHECK(history.back().turn_changes.files == answer.turn_changes.files);

    // A turn the conversation no longer has, or one saved before turns had an id.
    CHECK_FALSE(agent.set_turn_annotations("turn-2", comments));
    CHECK_FALSE(agent.set_turn_annotations("", comments));
}

TEST_CASE("A delegated worker records into the tracker it inherited and publishes nothing",
          "[agent][changes]") {
    const auto temp_path =
        std::filesystem::temp_directory_path() /
        std::format("filo_inherited_changes_{}",
                    std::chrono::steady_clock::now().time_since_epoch().count());
    TempDir temp(temp_path);
    {
        std::ofstream(temp.path() / "notes.txt") << "first\n";
    }

    auto& tool_manager = core::tools::ToolManager::get_instance();
    tool_manager.register_tool(std::make_shared<core::tools::WriteFileTool>());
    auto provider = std::make_shared<FileEditingProvider>(std::vector<std::string>{
        R"({"file_path":"notes.txt","content":"first\nsecond\n"})",
    });
    // The turn that spawned this worker owns the summary; the worker only
    // contributes to it.
    auto owner = std::make_shared<core::changes::TurnChangeTracker>(temp.path());
    auto agent = std::make_shared<core::agent::Agent>(
        provider, tool_manager,
        test_support::make_session_context(core::workspace::WorkspaceSnapshot{
            .primary = temp.path(),
            .enforce = true,
            .version = 1,
        }),
        core::agent::ToolResultStore::default_root(),
        std::shared_ptr<core::power::SleepInhibitor>{},
        std::shared_ptr<core::session::SessionStatsRegistry>{},
        nullptr,
        std::shared_ptr<core::memory::MemorySystem>{},
        std::shared_ptr<core::scm::WorkspaceLeaseRegistry>{},
        owner);
    agent->set_permission_fn([](std::string_view, std::string_view) { return true; });

    std::size_t published = 0;
    send_and_wait(agent, "Extend the notes.",
                  core::agent::Agent::TurnCallbacks{
                      .on_turn_changes =
                          [&](const core::changes::TurnChanges&) { ++published; },
                  });

    // The owner publishes once, at its own turn end; a worker never pre-empts it.
    CHECK(published == 0);
    const auto summary = owner->changes();
    REQUIRE(summary.files.size() == 1);
    CHECK(summary.files[0].path == "notes.txt");
    CHECK(summary.files[0].added == 1);

    // The worker's own history carries no summary to be resurrected later.
    CHECK(std::ranges::count_if(agent->get_history(), [](const core::llm::Message& message) {
        return !message.turn_changes.empty();
    }) == 0);
}

TEST_CASE("Recalled memories are ranked against the prompt that opened the conversation",
          "[agent][memory][relevance]") {
    const auto temp_path =
        std::filesystem::temp_directory_path() /
        std::format("filo_memory_rank_{}",
                    std::chrono::steady_clock::now().time_since_epoch().count());
    TempDir temp(temp_path);
    ScopedCurrentPath cwd(temp.path());

    const auto context = test_support::make_session_context(
        core::workspace::WorkspaceSnapshot{
            .primary = temp.path(),
            .enforce = true,
            .version = 1,
        });
    core::memory::MemoryStore store{temp.path() / "memory.json"};
    auto scoped = store.for_context(context);
    // Saved in the order recency would recall them, so the relevant one is last:
    // a block that still leads with it proves the query decided the order.
    REQUIRE(scoped.remember("TurnRevert replays the hunks of a turn backwards.").ok);
    REQUIRE(scoped.remember("The diff comparer opens patches in Lampo.").ok);
    REQUIRE(scoped.remember("PromptEditor launches the external editor.").ok);

    auto provider = std::make_shared<CapturingProvider>();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        core::tools::ToolManager::get_instance(),
        context,
        core::agent::ToolResultStore::default_root(),
        std::shared_ptr<core::power::SleepInhibitor>{},
        std::shared_ptr<core::session::SessionStatsRegistry>{},
        nullptr,
        std::make_shared<core::memory::MemorySystem>(
            store, std::make_shared<core::memory::NullToolRecoveryMemory>()));

    const auto system_prompt = [](const core::llm::ChatRequest& request) {
        const auto found = std::ranges::find_if(
            request.messages, [](const core::llm::Message& message) {
                return message.role == "system";
            });
        return found == request.messages.end() ? std::string{} : found->content;
    };

    send_and_wait(agent, "PromptEditor never launches the external editor");
    auto requests = provider->requests_snapshot();
    REQUIRE_FALSE(requests.empty());
    const auto first = system_prompt(requests.front());
    const auto editor = first.find("PromptEditor launches the external editor.");
    const auto revert = first.find("TurnRevert replays the hunks");
    REQUIRE(editor != std::string::npos);
    REQUIRE(revert != std::string::npos);
    CHECK(editor < revert);

    // The block lives in the cached prompt prefix, so a later turn on another
    // topic must not re-rank it: that would rewrite the prefix every turn and
    // cost the provider's prompt cache.
    send_and_wait(agent, "Now check what TurnRevert restores");
    requests = provider->requests_snapshot();
    const auto later = system_prompt(requests.back());
    const auto later_editor = later.find("PromptEditor launches the external editor.");
    const auto later_revert = later.find("TurnRevert replays the hunks");
    REQUIRE(later_editor != std::string::npos);
    REQUIRE(later_revert != std::string::npos);
    CHECK(later_editor < later_revert);
}

TEST_CASE("Agent chains a truncated-turn continuation only after the provider stream unwinds",
          "[agent][loop][provider][regression][deadlock]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto provider = std::make_shared<LockAcrossCallbackProvider>();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    // send_and_wait() bounds the turn at 5s, so a self-deadlock on the
    // provider's own lock fails here instead of hanging the suite.
    send_and_wait(agent, "Continue past the output limit.");

    // Two streams: the truncated turn and its recovery. While the recovery was
    // started from inside the terminal chunk callback, it re-entered
    // stream_response() while the provider still held inference_mutex_ and
    // blocked forever on it.
    CHECK(provider->call_count() == 2);
    const auto history = agent->get_history();
    CHECK(std::ranges::any_of(history, [](const core::llm::Message& message) {
        return message.role == "assistant" && message.content.find("finished") != std::string::npos;
    }));
}

TEST_CASE("Agent runs a truncated-turn continuation from a provider that finishes asynchronously",
          "[agent][loop][provider][regression]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto provider = std::make_shared<AsyncTruncatingProvider>();
    auto agent = std::make_shared<core::agent::Agent>(
        provider,
        tool_manager,
        test_support::make_workspace_session_context());

    // A continuation parked "until stream_response() returns" would wait for a
    // return that has already happened, and the turn would never finish.
    send_and_wait(agent, "Continue past the output limit.");
    provider->join_all();

    CHECK(provider->call_count() == 2);
    const auto history = agent->get_history();
    CHECK(std::ranges::any_of(history, [](const core::llm::Message& message) {
        return message.role == "assistant" && message.content.find("finished") != std::string::npos;
    }));
}
