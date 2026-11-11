#include "ThreadLedger.hpp"

#include <algorithm>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <utility>

namespace core::llm::continuation {

std::string_view to_string(CreateReason reason) noexcept {
    switch (reason) {
        case CreateReason::first_request: return "first_request";
        case CreateReason::expired: return "expired";
        case CreateReason::fixed_fields_changed: return "fixed_fields_changed";
        case CreateReason::create_only_changed: return "create_only_changed";
        case CreateReason::transcript_drift: return "transcript_drift";
        case CreateReason::anchor_rewritten: return "anchor_rewritten";
        case CreateReason::nothing_new: return "nothing_new";
        case CreateReason::replay_after_rejection: return "replay_after_rejection";
    }
    return "unknown";
}

std::string_view to_string(UnthreadedReason reason) noexcept {
    switch (reason) {
        case UnthreadedReason::not_planned: return "not_planned";
        case UnthreadedReason::no_scope: return "no_scope";
        case UnthreadedReason::scope_busy: return "scope_busy";
        case UnthreadedReason::rejected: return "rejected";
        case UnthreadedReason::dropped: return "dropped";
    }
    return "unknown";
}

struct ThreadLedger::State {
    struct Stored {
        std::vector<WireMessage> messages;
        std::uint64_t create_only_digest = 0;
        std::uint64_t fixed_digest = 0;
        std::string response_id;
        std::uint64_t reply_digest = 0;
        Clock::time_point stored_at;
    };
    struct Entry {
        std::optional<Stored> stored;
        bool busy = false;
    };

    explicit State(ThreadPolicy p) : policy(std::move(p)) {}

    ThreadPolicy policy;
    mutable std::mutex mutex;
    std::unordered_map<std::string, Entry> entries;

    [[nodiscard]] Clock::time_point now() const { return policy.now ? policy.now() : Clock::now(); }

    /// Caller holds the mutex. Idle entries always hold a stored thread (an
    /// emptied one is erased on release), so the oldest idle one goes.
    void evict_for_insert() {
        if (entries.size() < std::max<std::size_t>(policy.max_scopes, 1)) return;
        auto victim = entries.end();
        for (auto it = entries.begin(); it != entries.end(); ++it) {
            if (it->second.busy || !it->second.stored) continue;
            if (victim == entries.end() || it->second.stored->stored_at < victim->second.stored->stored_at) {
                victim = it;
            }
        }
        if (victim != entries.end()) entries.erase(victim);
    }

    /// Caller holds the mutex.
    [[nodiscard]] Plan plan(const Entry& entry, const WireTranscript& transcript) const {
        if (!entry.stored) return Create{CreateReason::first_request};
        const auto& stored = *entry.stored;
        const auto live = policy.lifetime - policy.safety_margin;
        if (now() - stored.stored_at >= live) return Create{CreateReason::expired};
        if (stored.fixed_digest != transcript.fixed_digest) return Create{CreateReason::fixed_fields_changed};
        if (stored.create_only_digest != transcript.create_only_digest) {
            return Create{CreateReason::create_only_changed};
        }
        const auto sent = stored.messages.size();
        if (transcript.messages.size() <= sent
            || !std::equal(stored.messages.begin(), stored.messages.end(), transcript.messages.begin())) {
            return Create{CreateReason::transcript_drift};
        }
        if (stored.reply_digest == 0 || transcript.messages[sent].reply_digest != stored.reply_digest) {
            return Create{CreateReason::anchor_rewritten};
        }
        if (transcript.messages.size() == sent + 1) return Create{CreateReason::nothing_new};
        return Continue{.previous_message_id = stored.response_id, .first_new_message = sent + 1};
    }
};

ThreadLedger::ThreadLedger(ThreadPolicy policy)
    : state_(std::make_shared<State>(std::move(policy))) {}

Attempt ThreadLedger::open(std::string_view scope, WireTranscript transcript) const {
    Attempt attempt;
    attempt.transcript_ = std::move(transcript);
    if (scope.empty()) {
        attempt.plan_ = Unthreaded{UnthreadedReason::no_scope};
        return attempt;
    }
    std::lock_guard lock(state_->mutex);
    auto it = state_->entries.find(std::string(scope));
    if (it == state_->entries.end()) {
        state_->evict_for_insert();
        it = state_->entries.emplace(std::string(scope), State::Entry{}).first;
    }
    if (it->second.busy) {
        attempt.plan_ = Unthreaded{UnthreadedReason::scope_busy};
        return attempt;
    }
    attempt.plan_ = state_->plan(it->second, attempt.transcript_);
    it->second.busy = true;
    attempt.ledger_ = state_;
    attempt.scope_ = std::string(scope);
    attempt.owns_scope_ = true;
    return attempt;
}

void ThreadLedger::forget(std::string_view scope) const {
    std::lock_guard lock(state_->mutex);
    if (const auto it = state_->entries.find(std::string(scope)); it != state_->entries.end()) {
        it->second.stored.reset();
        if (!it->second.busy) state_->entries.erase(it);
    }
}

std::size_t ThreadLedger::size() const {
    std::lock_guard lock(state_->mutex);
    return static_cast<std::size_t>(std::ranges::count_if(
        state_->entries, [](const auto& entry) { return entry.second.stored.has_value(); }));
}

Attempt::Attempt(Attempt&& other) noexcept
    : ledger_(std::move(other.ledger_)),
      scope_(std::move(other.scope_)),
      transcript_(std::move(other.transcript_)),
      plan_(std::move(other.plan_)),
      response_id_(std::move(other.response_id_)),
      owns_scope_(std::exchange(other.owns_scope_, false)) {}

Attempt& Attempt::operator=(Attempt&& other) noexcept {
    if (this != &other) {
        release();
        ledger_ = std::move(other.ledger_);
        scope_ = std::move(other.scope_);
        transcript_ = std::move(other.transcript_);
        plan_ = std::move(other.plan_);
        response_id_ = std::move(other.response_id_);
        owns_scope_ = std::exchange(other.owns_scope_, false);
    }
    return *this;
}

Attempt::~Attempt() { release(); }

void Attempt::replan(Plan plan) {
    response_id_.clear();
    if (!is_threaded(plan) || !owns_scope_) {
        plan_ = is_threaded(plan) ? Plan{Unthreaded{UnthreadedReason::dropped}} : std::move(plan);
        release();
        return;
    }
    plan_ = std::move(plan);
}

void Attempt::observe_response_id(std::string_view id) { response_id_ = std::string(id); }

void Attempt::commit(std::uint64_t reply_digest) {
    if (!owns_scope_) return;
    if (!is_threaded(plan_) || response_id_.empty() || reply_digest == 0) {
        release();
        return;
    }
    {
        std::lock_guard lock(ledger_->mutex);
        auto& entry = ledger_->entries[scope_];
        entry.stored = ThreadLedger::State::Stored{
            .messages = transcript_.messages,
            .create_only_digest = transcript_.create_only_digest,
            .fixed_digest = transcript_.fixed_digest,
            .response_id = response_id_,
            .reply_digest = reply_digest,
            .stored_at = ledger_->now(),
        };
        entry.busy = false;
    }
    owns_scope_ = false;
}

void Attempt::abandon() { release(); }

void Attempt::release() noexcept {
    if (!std::exchange(owns_scope_, false) || !ledger_) return;
    try {
        std::lock_guard lock(ledger_->mutex);
        // Nothing stored is idle and empty: the entry goes with it.
        ledger_->entries.erase(scope_);
    } catch (...) {
        // Releasing a scope is bookkeeping; a failure only costs a create later.
    }
}

} // namespace core::llm::continuation
