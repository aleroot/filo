#include <catch2/catch_test_macros.hpp>

#include "TestSessionContext.hpp"
#include "core/agent/Agent.hpp"
#include "core/context/SessionContext.hpp"
#include "core/llm/Advisor.hpp"
#include "core/llm/AdvisorSupport.hpp"
#include "core/llm/LLMProvider.hpp"
#include "core/llm/Models.hpp"
#include "core/tools/ToolManager.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

using namespace core::llm;

namespace {

/// An advisor capability that accepts one advisor and refuses the rest.
///
/// @p absorbs_changes: whether the wire can absorb a change to the request's
/// tool list, which is what lets a conversation change its advisor.
class FakeAdvisorSupport final : public IAdvisorSupport {
public:
    explicit FakeAdvisorSupport(std::vector<std::string> accepted,
                                bool absorbs_changes = true)
        : accepted_(std::move(accepted)), absorbs_changes_(absorbs_changes) {}

    [[nodiscard]] bool supports(std::string_view, const AdvisorSpec& advisor) const override {
        return std::ranges::find(accepted_, advisor.model) != accepted_.end();
    }

    [[nodiscard]] std::string unsupported_reason(std::string_view,
                                                 const AdvisorSpec& advisor) const override {
        return std::format("{} is not an accepted advisor.", advisor.model);
    }

    [[nodiscard]] std::string tool_definition_json(const AdvisorSpec&) const override {
        return R"({"type":"advisor_20260301","name":"advisor"})";
    }

    /// Mirrors the real rule: a wire that drops the invalidated reasoning
    /// absorbs any change, and one that does not absorbs only a change made
    /// before the conversation produced any bound reasoning.
    [[nodiscard]] bool absorbs_advisor_changes(
        std::string_view,
        std::span<const core::llm::Message> conversation) const noexcept override {
        if (absorbs_changes_) return true;
        for (const auto& message : conversation) {
            for (const auto& item : message.continuation_items) {
                if (item.kind == "thinking" || item.kind == "redacted_thinking") {
                    return false;
                }
            }
        }
        return true;
    }

private:
    std::vector<std::string> accepted_;
    bool absorbs_changes_ = true;
};

/// A provider that records the requests it serves and answers them.
class AdvisorCapturingProvider final : public LLMProvider {
public:
    explicit AdvisorCapturingProvider(std::shared_ptr<const IAdvisorSupport> support)
        : support_(std::move(support)) {}

    [[nodiscard]] std::shared_ptr<const IAdvisorSupport> advisor_support() const override {
        return support_;
    }

    void stream_response(const ChatRequest& request,
                         std::function<void(const StreamChunk&)> callback) override {
        {
            std::lock_guard lock(mutex_);
            requests_.push_back(request);
        }
        // The turn leaves bound reasoning in the history, the way a model
        // with preserved thinking does.
        StreamChunk chunk = StreamChunk::make_content("ok");
        chunk.continuation_items.push_back(ContinuationItem{
            .provider = "anthropic",
            .kind = "thinking",
            .payload = R"({"type":"thinking","thinking":"","signature":"sig"})",
        });
        callback(chunk);
        callback(StreamChunk::make_final());
    }

    /// The advisors the served requests declared.
    [[nodiscard]] std::vector<std::string> declared_advisors() const {
        std::lock_guard lock(mutex_);
        std::vector<std::string> advisors;
        for (const auto& request : requests_) {
            advisors.push_back(request.advisor ? request.advisor->model : std::string{});
        }
        return advisors;
    }

private:
    std::shared_ptr<const IAdvisorSupport> support_;
    mutable std::mutex mutex_;
    std::vector<ChatRequest> requests_;
};

/// A provider that reports a refusal for the advisor, the way a provider
/// that will not run one does.
class RefusingAdvisorProvider final : public LLMProvider {
public:
    explicit RefusingAdvisorProvider(std::shared_ptr<const IAdvisorSupport> support)
        : support_(std::move(support)) {}

    [[nodiscard]] std::shared_ptr<const IAdvisorSupport> advisor_support() const override {
        return support_;
    }

    void stream_response(const ChatRequest& request,
                         std::function<void(const StreamChunk&)> callback) override {
        {
            std::lock_guard lock(mutex_);
            requests_.push_back(request);
        }
        if (request.advisor) {
            AdvisorEvent event{.state = AdvisorCallState::Refused};
            event.advisor_model = request.advisor->model;
            callback(StreamChunk::make_advisor_event(std::move(event)));
        }
        callback(StreamChunk::make_content("ok"));
        callback(StreamChunk::make_final());
    }

    [[nodiscard]] std::vector<std::string> declared_advisors() const {
        std::lock_guard lock(mutex_);
        std::vector<std::string> advisors;
        for (const auto& request : requests_) {
            advisors.push_back(request.advisor ? request.advisor->model : std::string{});
        }
        return advisors;
    }

private:
    std::shared_ptr<const IAdvisorSupport> support_;
    mutable std::mutex mutex_;
    std::vector<ChatRequest> requests_;
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

} // namespace

TEST_CASE("A turn declares the advisor the session asked for",
          "[agent][advisor]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto provider = std::make_shared<AdvisorCapturingProvider>(
        std::make_shared<FakeAdvisorSupport>(std::vector<std::string>{"fable"}));
    core::agent::Agent agent(
        provider, tool_manager,
        test_support::make_session_context(core::workspace::WorkspaceSnapshot{
            .primary = std::filesystem::temp_directory_path(),
            .enforce = true,
            .version = 1,
        }));

    core::agent::Agent::TurnCallbacks callbacks;
    callbacks.advisor_override = "fable";
    send_and_wait(std::shared_ptr<core::agent::Agent>(&agent, [](auto*) {}),
                  "hello", callbacks);

    const auto advisors = provider->declared_advisors();
    REQUIRE(!advisors.empty());
    CHECK(advisors.front() == "fable");
}

TEST_CASE("A conversation changes its advisor when the wire can absorb it",
          "[agent][advisor][switch]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto provider = std::make_shared<AdvisorCapturingProvider>(
        std::make_shared<FakeAdvisorSupport>(std::vector<std::string>{"fable", "opus"},
                                                       /*absorbs_changes=*/true));
    core::agent::Agent agent(
        provider, tool_manager,
        test_support::make_session_context(core::workspace::WorkspaceSnapshot{
            .primary = std::filesystem::temp_directory_path(),
            .enforce = true,
            .version = 1,
        }));

    // The conversation has not declared an advisor yet.
    CHECK_FALSE(agent.advisor_in_use().has_value());

    core::agent::Agent::TurnCallbacks first;
    first.advisor_override = "fable";
    send_and_wait(std::shared_ptr<core::agent::Agent>(&agent, [](auto*) {}),
                  "hello", first);

    REQUIRE(agent.advisor_in_use().has_value());
    CHECK(*agent.advisor_in_use() == "fable");

    // Another advisor in the same conversation takes effect on the next turn:
    // the advisor is a tool declaration, and this wire can absorb the change.
    core::agent::Agent::TurnCallbacks second;
    second.advisor_override = "opus";
    send_and_wait(std::shared_ptr<core::agent::Agent>(&agent, [](auto*) {}),
                  "again", second);

    const auto advisors = provider->declared_advisors();
    REQUIRE(advisors.size() >= 2);
    CHECK(advisors.front() == "fable");
    CHECK(advisors.back() == "opus");
    CHECK(*agent.advisor_in_use() == "opus");
}

TEST_CASE("A conversation keeps its advisor when the wire cannot absorb a change",
          "[agent][advisor][switch]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto provider = std::make_shared<AdvisorCapturingProvider>(
        std::make_shared<FakeAdvisorSupport>(std::vector<std::string>{"fable", "opus"},
                                                       /*absorbs_changes=*/false));
    core::agent::Agent agent(
        provider, tool_manager,
        test_support::make_session_context(core::workspace::WorkspaceSnapshot{
            .primary = std::filesystem::temp_directory_path(),
            .enforce = true,
            .version = 1,
        }));

    core::agent::Agent::TurnCallbacks first;
    first.advisor_override = "fable";
    send_and_wait(std::shared_ptr<core::agent::Agent>(&agent, [](auto*) {}),
                  "hello", first);
    REQUIRE(agent.advisor_in_use().has_value());
    CHECK(*agent.advisor_in_use() == "fable");

    // The wire cannot absorb the change, so the conversation keeps the
    // advisor it declared.
    core::agent::Agent::TurnCallbacks second;
    second.advisor_override = "opus";
    send_and_wait(std::shared_ptr<core::agent::Agent>(&agent, [](auto*) {}),
                  "again", second);

    const auto advisors = provider->declared_advisors();
    REQUIRE(advisors.size() >= 2);
    CHECK(advisors.front() == "fable");
    CHECK(advisors.back() == "fable");
    CHECK(*agent.advisor_in_use() == "fable");
}

TEST_CASE("A turn without an advisor declares none", "[agent][advisor][regression]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto provider = std::make_shared<AdvisorCapturingProvider>(
        std::make_shared<FakeAdvisorSupport>(std::vector<std::string>{"fable"}));
    core::agent::Agent agent(
        provider, tool_manager,
        test_support::make_session_context(core::workspace::WorkspaceSnapshot{
            .primary = std::filesystem::temp_directory_path(),
            .enforce = true,
            .version = 1,
        }));

    send_and_wait(std::shared_ptr<core::agent::Agent>(&agent, [](auto*) {}),
                  "hello", core::agent::Agent::TurnCallbacks{});

    const auto advisors = provider->declared_advisors();
    REQUIRE(!advisors.empty());
    CHECK(advisors.front().empty());
    CHECK_FALSE(agent.advisor_in_use().has_value());
}

TEST_CASE("An advisor the provider refuses is not declared", "[agent][advisor]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto provider = std::make_shared<AdvisorCapturingProvider>(
        std::make_shared<FakeAdvisorSupport>(std::vector<std::string>{"fable"}));
    core::agent::Agent agent(
        provider, tool_manager,
        test_support::make_session_context(core::workspace::WorkspaceSnapshot{
            .primary = std::filesystem::temp_directory_path(),
            .enforce = true,
            .version = 1,
        }));

    core::agent::Agent::TurnCallbacks callbacks;
    callbacks.advisor_override = "opus";
    send_and_wait(std::shared_ptr<core::agent::Agent>(&agent, [](auto*) {}),
                  "hello", callbacks);

    const auto advisors = provider->declared_advisors();
    REQUIRE(!advisors.empty());
    CHECK(advisors.front().empty());
    CHECK_FALSE(agent.advisor_in_use().has_value());
}

TEST_CASE("A refused advisor is not asked again", "[agent][advisor][refusal]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto provider = std::make_shared<RefusingAdvisorProvider>(
        std::make_shared<FakeAdvisorSupport>(std::vector<std::string>{"fable"}));
    core::agent::Agent agent(
        provider, tool_manager,
        test_support::make_session_context(core::workspace::WorkspaceSnapshot{
            .primary = std::filesystem::temp_directory_path(),
            .enforce = true,
            .version = 1,
        }));

    core::agent::Agent::TurnCallbacks callbacks;
    callbacks.advisor_override = "fable";
    send_and_wait(std::shared_ptr<core::agent::Agent>(&agent, [](auto*) {}),
                  "hello", callbacks);
    send_and_wait(std::shared_ptr<core::agent::Agent>(&agent, [](auto*) {}),
                  "again", callbacks);

    // The provider refused it once; the conversation goes on without one
    // instead of asking again and being refused again.
    const auto advisors = provider->declared_advisors();
    REQUIRE(advisors.size() >= 2);
    CHECK(advisors.front() == "fable");
    CHECK(advisors.back().empty());
}

TEST_CASE("A compacted conversation asks for the advisor again", "[agent][advisor]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto provider = std::make_shared<RefusingAdvisorProvider>(
        std::make_shared<FakeAdvisorSupport>(std::vector<std::string>{"fable"}));
    core::agent::Agent agent(
        provider, tool_manager,
        test_support::make_session_context(core::workspace::WorkspaceSnapshot{
            .primary = std::filesystem::temp_directory_path(),
            .enforce = true,
            .version = 1,
        }));

    core::agent::Agent::TurnCallbacks callbacks;
    callbacks.advisor_override = "fable";
    send_and_wait(std::shared_ptr<core::agent::Agent>(&agent, [](auto*) {}),
                  "hello", callbacks);

    // A refused advisor stays off until the conversation is cleared or
    // compacted.
    agent.compact_history("summary");
    send_and_wait(std::shared_ptr<core::agent::Agent>(&agent, [](auto*) {}),
                  "again", callbacks);

    const auto advisors = provider->declared_advisors();
    REQUIRE(advisors.size() >= 2);
    CHECK(advisors.back() == "fable");
}

TEST_CASE("A cleared conversation asks for the advisor again", "[agent][advisor]") {
    auto& tool_manager = core::tools::ToolManager::get_instance();
    auto provider = std::make_shared<RefusingAdvisorProvider>(
        std::make_shared<FakeAdvisorSupport>(std::vector<std::string>{"fable"}));
    core::agent::Agent agent(
        provider, tool_manager,
        test_support::make_session_context(core::workspace::WorkspaceSnapshot{
            .primary = std::filesystem::temp_directory_path(),
            .enforce = true,
            .version = 1,
        }));

    core::agent::Agent::TurnCallbacks callbacks;
    callbacks.advisor_override = "fable";
    send_and_wait(std::shared_ptr<core::agent::Agent>(&agent, [](auto*) {}),
                  "hello", callbacks);

    agent.clear_history();
    send_and_wait(std::shared_ptr<core::agent::Agent>(&agent, [](auto*) {}),
                  "again", callbacks);

    const auto advisors = provider->declared_advisors();
    REQUIRE(advisors.size() >= 2);
    CHECK(advisors.back() == "fable");
}
