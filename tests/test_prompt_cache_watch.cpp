#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "PromptCacheTestSupport.hpp"
#include "tui/PromptCacheWatch.hpp"
#include "tui/ThreadModalHost.hpp"

#include <chrono>
#include <condition_variable>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

using namespace std::chrono_literals;
using Catch::Matchers::ContainsSubstring;
using core::llm::PromptCacheProfile;
namespace prompt_cache = tui::prompt_cache;

namespace {

using Clock = std::chrono::system_clock;

/// Stands in for MainApp: records what the watch shows and answers with a
/// scripted label, or leaves the question open when there is none.
class FakeTui {
public:
    explicit FakeTui(std::optional<std::string> answer = std::nullopt)
        : answer_(std::move(answer)) {}

    tui::PromptCacheWatch::Dependencies dependencies(
        core::agent::PromptCacheGuardSettings settings = {.min_context_tokens = 1'000}) {
        return {
            .post_question = [this](std::string session_id,
                                    core::tools::QuestionItem question,
                                    tui::QuestionDialogPromise promise) {
                std::lock_guard lock(mutex_);
                questions_.push_back(question);
                sessions_.push_back(std::move(session_id));
                if (answer_.has_value()) {
                    promise->set_value(tui::QuestionDialogAnswers{{question.question, *answer_}});
                } else {
                    open_ = std::move(promise);
                }
                cv_.notify_all();
            },
            .withdraw_question = [this](std::string_view, const tui::QuestionDialogPromise& promise) {
                std::lock_guard lock(mutex_);
                if (!open_ || open_ != promise) return false;
                open_->set_value(std::nullopt);
                open_.reset();
                ++withdrawn_;
                cv_.notify_all();
                return true;
            },
            .notify = [this](const tui::ThreadRuntime::Ptr&, std::string text) {
                std::lock_guard lock(mutex_);
                notes_.push_back(std::move(text));
                cv_.notify_all();
            },
            .settings = [settings]() { return settings; },
        };
    }

    template <typename Predicate>
    bool wait_until(Predicate predicate, std::chrono::milliseconds timeout = 5s) {
        std::unique_lock lock(mutex_);
        return cv_.wait_for(lock, timeout, [&] { return predicate(*this); });
    }

    std::vector<core::tools::QuestionItem> questions_;
    std::vector<std::string> sessions_;
    std::vector<std::string> notes_;
    int withdrawn_ = 0;

private:
    std::optional<std::string> answer_;
    tui::QuestionDialogPromise open_;
    std::mutex mutex_;
    std::condition_variable cv_;
};

tui::ThreadRuntime::Ptr make_runtime(std::shared_ptr<core::agent::Agent> agent) {
    return std::make_shared<tui::ThreadRuntime>(
        tui::ThreadRuntimeMetadata{
            .session_id = "cache0001",
            .created_at = "2026-10-20T12:00:00Z",
        },
        std::move(agent),
        std::make_shared<tui::ThreadRuntime::Messages>(),
        nullptr);
}

} // namespace

TEST_CASE("The send question carries the session and maps every answer",
          "[prompt-cache][watch]") {
    // A resumed session knows only its estimated history size; a one-token
    // threshold makes any of it heavy.
    auto provider = std::make_shared<test_support::CachingProvider>(
        PromptCacheProfile::documented(5min, true));
    auto agent = test_support::make_prompt_cache_agent(provider);
    test_support::send_prompt_and_wait(agent, "hello");
    agent->restore_prompt_cache_activity(Clock::now() - 2h, "caching", "cache-model");
    auto runtime = make_runtime(agent);

    for (const auto& [label, expected] : std::vector<std::pair<std::string_view, prompt_cache::SendChoice>>{
             {prompt_cache::kSendAnywayLabel, prompt_cache::SendChoice::Send},
             {prompt_cache::kCompactThenSendLabel, prompt_cache::SendChoice::CompactThenSend},
             {prompt_cache::kCancelLabel, prompt_cache::SendChoice::Cancel}}) {
        FakeTui fake{std::string(label)};
        tui::PromptCacheWatch watch(fake.dependencies({.min_context_tokens = 1}));
        CHECK(watch.confirm_send(runtime) == expected);
        REQUIRE(fake.questions_.size() == 1);
        CHECK(fake.sessions_.front() == "cache0001");
        CHECK_THAT(fake.questions_.front().question, ContainsSubstring("has expired"));
    }
}

TEST_CASE("A warm or light cache sends without asking", "[prompt-cache][watch]") {
    auto agent = test_support::make_prompt_cache_agent(
        std::make_shared<test_support::CachingProvider>(PromptCacheProfile::documented(5min, true)));
    test_support::send_prompt_and_wait(agent, "hello");
    auto runtime = make_runtime(agent);

    FakeTui fake{std::string(prompt_cache::kCancelLabel)};
    tui::PromptCacheWatch warm_watch(fake.dependencies());
    CHECK(warm_watch.confirm_send(runtime) == prompt_cache::SendChoice::Send);

    agent->restore_prompt_cache_activity(Clock::now() - 2h, "caching", "cache-model");
    tui::PromptCacheWatch light_watch(fake.dependencies({.min_context_tokens = 10'000'000}));
    CHECK(light_watch.confirm_send(runtime) == prompt_cache::SendChoice::Send);
    CHECK(fake.questions_.empty());
}

TEST_CASE("An unanswered expiry notice is withdrawn when the cache expires",
          "[prompt-cache][watch]") {
    // A three-second documented cache with a two-second lead: the notice
    // fires about a second after the exchange, expiry two seconds later.
    auto agent = test_support::make_prompt_cache_agent(
        std::make_shared<test_support::CachingProvider>(PromptCacheProfile::documented(3s, true)));
    test_support::send_prompt_and_wait(agent, "hello");
    auto runtime = make_runtime(agent);

    FakeTui fake;  // never answers
    tui::PromptCacheWatch watch(fake.dependencies({.min_context_tokens = 1'000, .notice_lead = 2s}));
    watch.track(runtime);

    REQUIRE(fake.wait_until([](const FakeTui& f) { return !f.questions_.empty(); }));
    const auto& notice = fake.questions_.front();
    CHECK(notice.header == "Cache expiring");
    CHECK(notice.options.front().label == prompt_cache::kKeepWarmLabel);

    REQUIRE(fake.wait_until([](const FakeTui& f) { return !f.notes_.empty(); }));
    CHECK(fake.withdrawn_ == 1);
    CHECK_THAT(fake.notes_.front(), ContainsSubstring("has expired"));
}

TEST_CASE("Keep warm from the expiry notice renews the cache and re-arms",
          "[prompt-cache][watch]") {
    auto provider = std::make_shared<test_support::CachingProvider>(
        PromptCacheProfile::documented(3s, true));
    auto agent = test_support::make_prompt_cache_agent(provider);
    test_support::send_prompt_and_wait(agent, "hello");
    auto runtime = make_runtime(agent);

    FakeTui fake{std::string(prompt_cache::kKeepWarmLabel)};
    tui::PromptCacheWatch watch(fake.dependencies({.min_context_tokens = 1'000, .notice_lead = 2s}));
    watch.track(runtime);

    REQUIRE(fake.wait_until([](const FakeTui& f) { return !f.notes_.empty(); }));
    CHECK_THAT(fake.notes_.front(), ContainsSubstring("kept warm"));
    CHECK(provider->requests().size() >= 2);
    CHECK(fake.withdrawn_ == 0);
}

TEST_CASE("Automatic keep-warm renews without asking until its budget is spent",
          "[prompt-cache][watch]") {
    // Read 50%, write 100%: one renewal is worth it, the second would not be.
    auto provider = std::make_shared<test_support::CachingProvider>(
        PromptCacheProfile::documented(3s, true).with_costs(50, 100));
    auto agent = test_support::make_prompt_cache_agent(provider);
    test_support::send_prompt_and_wait(agent, "hello");
    auto runtime = make_runtime(agent);

    FakeTui fake;  // never answers
    tui::PromptCacheWatch watch(fake.dependencies({
        .min_context_tokens = 1'000,
        .notice_lead = 2s,
        .keep_warm = core::agent::PromptCacheKeepWarm::Auto,
    }));
    watch.track(runtime);

    REQUIRE(fake.wait_until([](const FakeTui& f) { return !f.notes_.empty(); }));
    CHECK_THAT(fake.notes_.front(), ContainsSubstring("automatic, 1 of 1"));
    CHECK(provider->requests().size() >= 2);

    // The renewed cache comes due again; with the budget spent, the user is asked.
    REQUIRE(fake.wait_until([](const FakeTui& f) { return !f.questions_.empty(); }));
    CHECK(fake.questions_.front().header == "Cache expiring");
    watch.shutdown();
}

TEST_CASE("ThreadModalHost withdraws only the dialog it is asked about",
          "[prompt-cache][watch][thread-modals]") {
    tui::ThreadModalHost host;
    auto mine = std::make_shared<std::promise<std::optional<tui::QuestionDialogAnswers>>>();
    auto answer = mine->get_future();
    host.post_question(core::tools::QuestionRequest{
        .id = "q1",
        .questions = {prompt_cache::expiry_question({.context_tokens = 1}, false)},
        .session_id = "s1",
        .promise = mine,
    });
    REQUIRE(host.question_visible("s1"));

    const auto someone_else = std::make_shared<std::promise<std::optional<tui::QuestionDialogAnswers>>>();
    CHECK_FALSE(host.withdraw_question("s1", someone_else).has_resolution());
    CHECK(host.question_visible("s1"));

    auto withdrawn = host.withdraw_question("s1", mine);
    REQUIRE(withdrawn.has_resolution());
    CHECK_FALSE(withdrawn.stop_agent);
    withdrawn.resolve();
    CHECK_FALSE(host.question_visible("s1"));
    CHECK_FALSE(answer.get().has_value());
}
