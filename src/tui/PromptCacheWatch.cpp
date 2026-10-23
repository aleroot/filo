#include "PromptCacheWatch.hpp"

#include "core/logging/Logger.hpp"

#include <chrono>
#include <condition_variable>
#include <exception>
#include <future>
#include <optional>
#include <thread>
#include <utility>

namespace tui {

namespace {

using Clock = std::chrono::system_clock;

[[nodiscard]] std::string notice_key(std::string_view session_id) {
    return std::string(session_id) + "#cache-notice";
}

[[nodiscard]] std::string expiry_key(std::string_view session_id) {
    return std::string(session_id) + "#cache-expiry";
}

[[nodiscard]] QuestionDialogPromise make_promise() {
    return std::make_shared<QuestionDialogPromise::element_type>();
}

/// How long a notice waits for the user to stop typing before trying again.
constexpr std::chrono::seconds kComposerRetry{5};

bool run_detached(std::function<void()> task) {
    try {
        std::thread(std::move(task)).detach();
        return true;
    } catch (const std::exception& error) {
        core::logging::warn("Prompt cache watch could not start a worker: {}", error.what());
        return false;
    }
}

} // namespace

struct PromptCacheWatch::Shared final : std::enable_shared_from_this<Shared> {
    explicit Shared(Dependencies deps_) : deps(std::move(deps_)) {}

    const Dependencies deps;
    DeadlineScheduler scheduler;

    std::mutex mutex;
    std::condition_variable idle;
    bool alive = true;
    std::size_t calls_in_flight = 0;
    // The expiry notice currently shown per session, so expiry can withdraw
    // exactly that dialog and a second notice is never stacked on it.
    std::unordered_map<std::string, QuestionDialogPromise> open_notices;
    // Automatic renewals per session since its user last sent a prompt;
    // PromptCacheGuard::should_keep_warm caps them.
    std::unordered_map<std::string, int> auto_renewals;

    /// Runs @p call against the dependencies unless shut down. Calls run
    /// concurrently; shutdown waits for every one in flight.
    template <typename Call>
    bool with_dependencies(Call&& call) {
        {
            std::lock_guard lock(mutex);
            if (!alive) return false;
            ++calls_in_flight;
        }
        try {
            std::forward<Call>(call)(deps);
        } catch (const std::exception& error) {
            core::logging::warn("Prompt cache watch callback failed: {}", error.what());
        } catch (...) {
            core::logging::warn("Prompt cache watch callback failed");
        }
        {
            std::lock_guard lock(mutex);
            --calls_in_flight;
        }
        idle.notify_all();
        return true;
    }

    /// The current policy; after shutdown, one that never asks.
    [[nodiscard]] core::agent::PromptCacheGuard guard() {
        core::agent::PromptCacheGuardSettings settings{.min_context_tokens = 0};
        with_dependencies([&](const Dependencies& d) {
            settings = d.settings ? d.settings() : core::agent::PromptCacheGuardSettings{};
        });
        return core::agent::PromptCacheGuard{settings};
    }

    void notify(const ThreadRuntime::Ptr& runtime, std::string text) {
        with_dependencies([&](const Dependencies& d) {
            if (d.notify) d.notify(runtime, std::move(text));
        });
    }

    /// Asks and waits. nullopt when dismissed, withdrawn or shut down.
    [[nodiscard]] std::optional<QuestionDialogAnswers> ask(
        const std::string& session_id,
        core::tools::QuestionItem question,
        const QuestionDialogPromise& promise) {
        auto answer = promise->get_future();
        bool posted = false;
        with_dependencies([&](const Dependencies& d) {
            if (d.post_question) {
                d.post_question(session_id, std::move(question), promise);
                posted = true;
            }
        });
        if (!posted) return std::nullopt;
        try {
            return answer.get();
        } catch (const std::future_error&) {
            return std::nullopt;  // the dialog went away unanswered
        }
    }

    [[nodiscard]] QuestionDialogPromise take_open_notice(std::string_view session_id) {
        std::lock_guard lock(mutex);
        auto node = open_notices.extract(std::string(session_id));
        return node ? std::move(node.mapped()) : QuestionDialogPromise{};
    }

    /// True when a notice dialog was open and is now closed.
    bool withdraw_open_notice(std::string_view session_id) {
        const auto promise = take_open_notice(session_id);
        if (!promise) return false;
        bool withdrawn = false;
        with_dependencies([&](const Dependencies& d) {
            if (d.withdraw_question) withdrawn = d.withdraw_question(session_id, promise);
        });
        return withdrawn;
    }

    void track(const ThreadRuntime::Ptr& runtime) {
        const auto session_id = runtime->session_id();
        const auto deadlines = guard().deadlines(runtime->agent()->prompt_cache_state());
        if (!deadlines.has_value()) {
            disarm(session_id);
            return;
        }
        scheduler.schedule(notice_key(session_id), deadlines->notice,
                           bind(runtime, &Shared::on_notice));
        scheduler.schedule(expiry_key(session_id), deadlines->expiry,
                           bind(runtime, &Shared::on_expiry));
    }

    /// Wraps @p action so it runs only while both this watch and @p runtime
    /// are still alive; callbacks parked in timers and agents never keep
    /// either alive. The action receives them, then the callback's arguments.
    template <typename Action>
    [[nodiscard]] auto guarded(const ThreadRuntime::Ptr& runtime, Action action) {
        return [weak_self = weak_from_this(),
                weak_runtime = std::weak_ptr<ThreadRuntime>(runtime),
                action = std::move(action)](auto&&... args) {
            const auto self = weak_self.lock();
            const auto target = weak_runtime.lock();
            if (self && target) action(*self, target, std::forward<decltype(args)>(args)...);
        };
    }

    [[nodiscard]] std::function<void()> bind(
        const ThreadRuntime::Ptr& runtime,
        void (Shared::*handler)(const ThreadRuntime::Ptr&)) {
        return guarded(runtime, [handler](Shared& self, const ThreadRuntime::Ptr& target) {
            (self.*handler)(target);
        });
    }

    void disarm(std::string_view session_id) {
        scheduler.cancel(notice_key(session_id));
        scheduler.cancel(expiry_key(session_id));
    }

    void forget(std::string_view session_id) {
        disarm(session_id);
        withdraw_open_notice(session_id);
        std::lock_guard lock(mutex);
        auto_renewals.erase(std::string(session_id));
    }

    /// Counts and returns the next automatic renewal of @p session_id's cache,
    /// or nullopt when @p policy wants the user asked instead.
    [[nodiscard]] std::optional<prompt_cache::AutoRenewal> claim_auto_renewal(
        const std::string& session_id,
        const core::agent::PromptCacheGuard& policy,
        const core::llm::PromptCacheProfile& profile) {
        std::lock_guard lock(mutex);
        const auto it = auto_renewals.find(session_id);
        const int done = it == auto_renewals.end() ? 0 : it->second;
        if (!alive || !policy.should_keep_warm(profile, done)) return std::nullopt;
        auto_renewals[session_id] = done + 1;
        return prompt_cache::AutoRenewal{
            .number = done + 1,
            .budget = core::agent::PromptCacheGuard::keep_warm_budget(profile),
        };
    }

    /// A failed automatic renewal hands the next decision back to the user
    /// rather than retrying into the same error.
    void exhaust_auto_renewals(const std::string& session_id, int budget) {
        std::lock_guard lock(mutex);
        auto_renewals[session_id] = budget;
    }

    // Scheduler thread. Re-reads the state: anything may have happened since
    // the notice was armed, and only a still-idle, still-warm heavy cache
    // deserves the question.
    void on_notice(const ThreadRuntime::Ptr& runtime) {
        if (runtime->turn_active()) return;  // the turn's end re-arms
        const auto state = runtime->agent()->prompt_cache_state();
        const auto policy = guard();
        const auto deadlines = policy.deadlines(state);
        if (!deadlines.has_value()) return;
        const auto now = Clock::now();
        if (now + std::chrono::seconds{1} < deadlines->notice) {
            track(runtime);  // the cache was used since: follow its new lifetime
            return;
        }
        const auto assessment = core::agent::PromptCacheGuard::assess(state, now);
        if (assessment.warmth != core::agent::PromptCacheWarmth::Warm) return;

        const auto session_id = runtime->session_id();
        // A renewal touches neither the keyboard nor the conversation, so it
        // need not wait for the composer.
        if (const auto renewal = claim_auto_renewal(session_id, policy, assessment.profile)) {
            keep_warm(runtime, assessment.profile, renewal);
            return;
        }
        bool typing = false;
        with_dependencies([&](const Dependencies& d) {
            typing = d.composer_in_use && d.composer_in_use(session_id);
        });
        if (typing) {
            if (now + kComposerRetry < deadlines->expiry) {
                scheduler.schedule(notice_key(session_id), now + kComposerRetry,
                                   bind(runtime, &Shared::on_notice));
            }
            return;
        }
        auto promise = make_promise();
        {
            std::lock_guard lock(mutex);
            if (!alive || open_notices.contains(session_id)) return;
            open_notices.emplace(session_id, promise);
        }
        // The answer may take hours; wait for it off the scheduler thread.
        const bool started = run_detached([self = shared_from_this(), runtime, assessment,
                                           session_id, promise]() {
            const bool can_keep_warm =
                core::agent::PromptCacheGuard::can_keep_warm(assessment.profile);
            const auto answers = self->ask(
                session_id, prompt_cache::expiry_question(assessment, can_keep_warm), promise);
            {
                std::lock_guard lock(self->mutex);
                const auto it = self->open_notices.find(session_id);
                if (it != self->open_notices.end() && it->second == promise) {
                    self->open_notices.erase(it);
                }
            }
            if (!answers.has_value()) return;  // dismissed or withdrawn
            self->apply(runtime, prompt_cache::expiry_choice(answers), assessment.profile);
        });
        if (!started) {
            static_cast<void>(take_open_notice(session_id));
        }
    }

    // Scheduler thread. An unanswered notice no longer applies once the cache
    // is gone: take it back and leave a note where the user will look.
    void on_expiry(const ThreadRuntime::Ptr& runtime) {
        // Answered meanwhile: the answer stands and needs no note.
        if (!withdraw_open_notice(runtime->session_id())) return;
        const auto assessment = core::agent::PromptCacheGuard::assess(
            runtime->agent()->prompt_cache_state(), Clock::now());
        if (assessment.expired()) {
            notify(runtime, prompt_cache::expired_notice(assessment));
        }
    }

    /// Replays the last request, reports it, and follows the renewed cache.
    void keep_warm(const ThreadRuntime::Ptr& runtime,
                   const core::llm::PromptCacheProfile& profile,
                   std::optional<prompt_cache::AutoRenewal> renewal) {
        runtime->agent()->refresh_prompt_cache_async(guarded(
            runtime,
            [profile, renewal](Shared& self, const ThreadRuntime::Ptr& target,
                               const core::agent::Agent::PromptCacheRefresh& refresh) {
                if (renewal && !refresh.refreshed) {
                    self.exhaust_auto_renewals(target->session_id(), renewal->budget);
                }
                self.notify(target, prompt_cache::keep_warm_notice(refresh, profile, renewal));
                self.track(target);
            }));
    }

    void apply(const ThreadRuntime::Ptr& runtime,
               prompt_cache::ExpiryChoice choice,
               const core::llm::PromptCacheProfile& profile) {
        switch (choice) {
        case prompt_cache::ExpiryChoice::KeepWarm:
            keep_warm(runtime, profile, std::nullopt);
            break;
        case prompt_cache::ExpiryChoice::CompactNow:
            runtime->agent()->compact_history_then(
                guarded(runtime,
                        [](Shared& self, const ThreadRuntime::Ptr& target,
                           const std::string& status) { self.notify(target, status); }),
                // Compaction starts a new prefix; track() then disarms the
                // old cache's timers.
                guarded(runtime,
                        [](Shared& self, const ThreadRuntime::Ptr& target, bool) {
                            self.track(target);
                        }));
            break;
        case prompt_cache::ExpiryChoice::LetExpire:
            break;
        }
    }

    void shutdown() {
        scheduler.shutdown();
        std::unique_lock lock(mutex);
        alive = false;
        open_notices.clear();
        auto_renewals.clear();
        idle.wait(lock, [this] { return calls_in_flight == 0; });
    }
};

PromptCacheWatch::PromptCacheWatch(Dependencies dependencies)
    : shared_(std::make_shared<Shared>(std::move(dependencies))) {}

PromptCacheWatch::~PromptCacheWatch() {
    shutdown();
}

void PromptCacheWatch::track(const ThreadRuntime::Ptr& runtime) {
    if (runtime) shared_->track(runtime);
}

void PromptCacheWatch::forget(std::string_view session_id) {
    shared_->forget(session_id);
}

prompt_cache::SendChoice PromptCacheWatch::confirm_send(const ThreadRuntime::Ptr& runtime) {
    if (!runtime) return prompt_cache::SendChoice::Send;
    const auto session_id = runtime->session_id();
    // The prompt supersedes any reminder about the idle cache it is about to
    // use; the turn's end arms a fresh one.
    shared_->forget(session_id);

    const auto assessment = core::agent::PromptCacheGuard::assess(
        runtime->agent()->prompt_cache_state(), Clock::now());
    if (!shared_->guard().should_confirm_send(assessment)) {
        return prompt_cache::SendChoice::Send;
    }
    const auto promise = make_promise();
    return prompt_cache::send_choice(
        shared_->ask(session_id, prompt_cache::send_question(assessment), promise));
}

void PromptCacheWatch::shutdown() {
    shared_->shutdown();
}

} // namespace tui
