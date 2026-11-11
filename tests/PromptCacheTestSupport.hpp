#pragma once

// Shared fixtures for prompt-cache tests: a provider whose cache profile is
// chosen by the test, and an agent wired to it.

#include "TestSessionContext.hpp"
#include "core/agent/Agent.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/PromptCacheProfile.hpp"
#include "core/tools/ToolManager.hpp"

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace test_support {

/// A provider with the cache profile the test chooses. Every request reports
/// a mostly cached 2k-token prompt; forks share the request log so a
/// keep-warm replay is observable.
class CachingProvider final : public core::llm::LLMProvider {
public:
    struct Log {
        std::mutex mutex;
        std::vector<core::llm::ChatRequest> requests;
        /// Answers with a tool call instead of text, from the next request on.
        bool reply_with_tool_call = false;
    };

    explicit CachingProvider(core::llm::PromptCacheProfile profile,
                             std::shared_ptr<Log> log = std::make_shared<Log>())
        : profile_(profile), log_(std::move(log)) {}

    void stream_response(const core::llm::ChatRequest& request,
                         std::function<void(const core::llm::StreamChunk&)> callback) override {
        bool tool_call = false;
        {
            std::lock_guard lock(log_->mutex);
            log_->requests.push_back(request);
            tool_call = log_->reply_with_tool_call;
        }
        set_last_usage(core::llm::TokenUsage{
            .prompt_tokens = 2'000,
            .completion_tokens = 1,
            .total_tokens = 2'001,
            .cached_prompt_tokens = 1'900,
        });
        if (tool_call) {
            core::llm::ToolCall call;
            call.id = "call_1";
            call.function.name = "read";
            call.function.arguments = "{}";
            callback(core::llm::StreamChunk::make_tools({std::move(call)}));
        } else {
            callback(core::llm::StreamChunk::make_content("ok"));
        }
        callback(core::llm::StreamChunk::make_final());
    }

    void reply_with_tool_call() {
        std::lock_guard lock(log_->mutex);
        log_->reply_with_tool_call = true;
    }

    [[nodiscard]] core::llm::PromptCacheProfile prompt_cache_profile(
        std::string_view) const noexcept override {
        return profile_;
    }

    [[nodiscard]] std::shared_ptr<core::llm::LLMProvider>
    fork_for_parallel_request() const override {
        return std::make_shared<CachingProvider>(profile_, log_);
    }

    [[nodiscard]] std::vector<core::llm::ChatRequest> requests() const {
        std::lock_guard lock(log_->mutex);
        return log_->requests;
    }

private:
    core::llm::PromptCacheProfile profile_;
    std::shared_ptr<Log> log_;
};

inline std::shared_ptr<core::agent::Agent> make_prompt_cache_agent(
    std::shared_ptr<core::llm::LLMProvider> provider) {
    auto agent = std::make_shared<core::agent::Agent>(
        std::move(provider),
        core::tools::ToolManager::get_instance(),
        test_support::make_workspace_session_context(
            core::context::SessionTransport::cli, "prompt-cache-session"));
    agent->set_active_provider_name("caching");
    agent->set_active_model("cache-model");
    return agent;
}

inline void send_prompt_and_wait(const std::shared_ptr<core::agent::Agent>& agent,
                                 const std::string& prompt) {
    std::promise<void> done;
    auto finished = done.get_future();
    agent->send_message(
        prompt,
        [](const std::string&) {},
        [](const std::string&, const std::string&) {},
        [&done]() { done.set_value(); });
    REQUIRE(finished.wait_for(std::chrono::seconds{5}) == std::future_status::ready);
}

} // namespace test_support
