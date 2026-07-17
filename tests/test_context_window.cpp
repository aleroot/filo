#include <catch2/catch_test_macros.hpp>

#include "core/context/ContextWindowTracker.hpp"

#include <string>
#include <vector>

namespace {

class FixedContextProvider final : public core::llm::LLMProvider {
public:
    int max_context_tokens = 0;
    std::string last_model;

    void stream_response(const core::llm::ChatRequest&,
                         std::function<void(const core::llm::StreamChunk&)>) override {}

    [[nodiscard]] int max_context_size() const noexcept override {
        return max_context_tokens;
    }

    [[nodiscard]] std::string get_last_model() const override {
        return last_model;
    }
};

core::llm::Message message_with_chars(std::size_t chars) {
    return core::llm::Message{
        .role = "assistant",
        .content = std::string(chars, 'x'),
    };
}

} // namespace

TEST_CASE("ContextWindowTracker estimates live history snapshots", "[context][window]") {
    std::vector<core::llm::Message> history{
        message_with_chars(400),
    };

    const auto snapshot = core::context::ContextWindowTracker::snapshot(
        history,
        nullptr,
        "kimi-for-coding");

    CHECK(snapshot.estimated_context_tokens == 101);
    CHECK(snapshot.max_context_tokens == 256'000);
    CHECK(snapshot.remaining_pct == 100);
}

TEST_CASE("ContextWindowTracker uses provider-reported context before model registry",
          "[context][window]") {
    auto provider = std::make_shared<FixedContextProvider>();
    provider->max_context_tokens = 64'000;
    provider->last_model = "claude-sonnet-4-6[1m]";

    const auto snapshot = core::context::ContextWindowTracker::snapshot(
        {message_with_chars(4'000)},
        provider,
        "kimi-for-coding");

    CHECK(snapshot.max_context_tokens == 64'000);
    CHECK(snapshot.estimated_context_tokens == 1'001);
    CHECK(snapshot.remaining_pct == 99);
}

TEST_CASE("ContextWindowTracker ceils tiny Kimi 2.7 startup usage to full remaining display",
          "[context][window]") {
    const auto snapshot = core::context::ContextWindowTracker::snapshot(
        {message_with_chars(4'000)},
        nullptr,
        "kimi-k2.7-code");

    CHECK(snapshot.max_context_tokens == 256'000);
    CHECK(snapshot.estimated_context_tokens == 1'001);
    CHECK(snapshot.remaining_pct == 100);
}

TEST_CASE("ContextWindowTracker does not show 99 percent left before one percent is used",
          "[context][window]") {
    const auto just_under_one_percent = core::context::ContextWindowTracker::snapshot(
        {message_with_chars(10'232)},
        nullptr,
        "kimi-k2.7-code");
    const auto just_over_one_percent = core::context::ContextWindowTracker::snapshot(
        {message_with_chars(10'240)},
        nullptr,
        "kimi-k2.7-code");

    CHECK(just_under_one_percent.estimated_context_tokens == 2'559);
    CHECK(just_under_one_percent.remaining_pct == 100);
    CHECK(just_over_one_percent.estimated_context_tokens == 2'561);
    CHECK(just_over_one_percent.remaining_pct == 99);
}

TEST_CASE("ContextWindowTracker excludes stable bootstrap from displayed context pressure",
          "[context][window]") {
    const std::vector<core::llm::Message> bootstrap_only{
        message_with_chars(8'000),
    };
    const std::size_t bootstrap_tokens =
        core::context::ContextWindowTracker::estimate_tokens(bootstrap_only);

    const auto fresh_snapshot = core::context::ContextWindowTracker::snapshot(
        bootstrap_only,
        nullptr,
        "claude-opus-4-8",
        bootstrap_tokens);

    CHECK(fresh_snapshot.estimated_context_tokens == 2'001);
    CHECK(fresh_snapshot.metered_context_tokens == 0);
    CHECK(fresh_snapshot.remaining_pct == 100);

    std::vector<core::llm::Message> with_user_turn = bootstrap_only;
    with_user_turn.push_back(message_with_chars(8'000));
    const auto active_snapshot = core::context::ContextWindowTracker::snapshot(
        with_user_turn,
        nullptr,
        "claude-opus-4-8",
        bootstrap_tokens);

    CHECK(active_snapshot.estimated_context_tokens == 4'001);
    CHECK(active_snapshot.metered_context_tokens == 2'000);
    CHECK(active_snapshot.remaining_pct == 99);
}

TEST_CASE("ContextWindowTracker falls back to provider last model for router-like providers",
          "[context][window]") {
    auto provider = std::make_shared<FixedContextProvider>();
    provider->last_model = "qwen3-coder-plus";

    const auto snapshot = core::context::ContextWindowTracker::snapshot(
        {message_with_chars(4'000)},
        provider,
        "auto");

    CHECK(snapshot.max_context_tokens == 1'000'000);
    CHECK(snapshot.remaining_pct == 100);
}

TEST_CASE("ContextWindowTracker resolves representative provider model windows",
          "[context][window]") {
    struct ModelCase {
        std::string_view model;
        int32_t expected_context;
    };

    const std::vector<ModelCase> cases{
        {"kimi-k3", 1'048'576},
        {"k3", 1'048'576},
        {"kimi-for-coding", 256'000},
        {"kimi-k2.7-code", 256'000},
        {"claude-sonnet-4-6[1m]", 1'000'000},
        {"gpt-5.6-sol", 1'050'000},
        {"gpt-5.6-terra", 1'050'000},
        {"gpt-5.6-luna", 1'050'000},
        {"gpt-4o", 128'000},
        {"mistral-vibe-cli-latest", 256'000},
        {"mistral-medium-3.5", 256'000},
        {"mistral-medium-latest", 256'000},
        {"mistral-small-latest", 256'000},
        {"mistral-large-latest", 256'000},
        {"qwen3-coder-plus", 1'000'000},
        {"gemini-2.5-pro", 1'048'576},
        {"glm-5.2", 1'000'000},
    };

    for (const auto& item : cases) {
        CHECK(core::context::ContextWindowTracker::resolve_max_context_tokens(
                  nullptr,
                  item.model) == item.expected_context);
    }
}

TEST_CASE("ContextWindowTracker compaction policy uses model-aware default threshold",
          "[context][window]") {
    std::vector<core::llm::Message> history{
        message_with_chars(120'000),
    };

    const auto decision = core::context::ContextWindowTracker::compaction_decision(
        history,
        nullptr,
        "test-model",
        core::context::CompactionTriggerPolicy{
            .configured_token_threshold =
                core::context::CompactionTriggerPolicy::kDefaultFixedTokenThreshold,
        });

    CHECK(decision.max_context_tokens == 0);
    CHECK(decision.effective_token_threshold == 25'000);
    CHECK(decision.should_compact);
}

TEST_CASE("ContextWindowTracker keeps default compaction threshold proportional when model window is known",
          "[context][window]") {
    std::vector<core::llm::Message> history{
        message_with_chars(120'000),
    };

    const auto decision = core::context::ContextWindowTracker::compaction_decision(
        history,
        nullptr,
        "kimi-for-coding",
        core::context::CompactionTriggerPolicy{
            .configured_token_threshold =
                core::context::CompactionTriggerPolicy::kDefaultFixedTokenThreshold,
        });

    CHECK(decision.max_context_tokens == 256'000);
    CHECK(decision.effective_token_threshold == 192'000);
    CHECK_FALSE(decision.should_compact);
}

TEST_CASE("ContextWindowTracker preserves explicit threshold equal to built-in default",
          "[context][window]") {
    std::vector<core::llm::Message> history{
        message_with_chars(120'000),
    };

    const auto decision = core::context::ContextWindowTracker::compaction_decision(
        history,
        nullptr,
        "kimi-for-coding",
        core::context::CompactionTriggerPolicy{
            .configured_token_threshold = 25'000,
            .use_model_aware_default_threshold = false,
        });

    CHECK(decision.max_context_tokens == 256'000);
    CHECK(decision.effective_token_threshold == 25'000);
    CHECK(decision.should_compact);
}

TEST_CASE("ContextWindowTracker respects custom compaction thresholds but caps them by window ratio",
          "[context][window]") {
    std::vector<core::llm::Message> history{
        message_with_chars(120'000),
    };

    const auto decision = core::context::ContextWindowTracker::compaction_decision(
        history,
        nullptr,
        "kimi-for-coding",
        core::context::CompactionTriggerPolicy{
            .configured_token_threshold = 20'000,
        });

    CHECK(decision.effective_token_threshold == 20'000);
    CHECK(decision.should_compact);
}
