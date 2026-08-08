#include "ThreadRuntime.hpp"

#include <utility>

namespace tui {

ThreadRuntime::ThreadRuntime(
    ThreadRuntimeMetadata metadata,
    std::shared_ptr<core::agent::Agent> agent,
    std::shared_ptr<Messages> messages,
    core::session::ActiveSessionLease::Ptr lease)
    : metadata_(std::move(metadata))
    , agent_(std::move(agent))
    , messages_(std::move(messages))
    , lease_(std::move(lease)) {}

std::string ThreadRuntime::session_id() const {
    std::lock_guard lock(metadata_mutex_);
    return metadata_.session_id;
}

std::shared_ptr<core::agent::Agent> ThreadRuntime::agent() const noexcept {
    return agent_;
}

std::shared_ptr<ThreadRuntime::Messages> ThreadRuntime::messages() const noexcept {
    return messages_;
}

core::session::ActiveSessionLease::Ptr ThreadRuntime::lease() const {
    std::lock_guard lock(metadata_mutex_);
    return lease_;
}

void ThreadRuntime::set_lease(core::session::ActiveSessionLease::Ptr lease) {
    std::lock_guard lock(metadata_mutex_);
    lease_ = std::move(lease);
}

ThreadRuntimeMetadata ThreadRuntime::metadata() const {
    std::lock_guard lock(metadata_mutex_);
    return metadata_;
}

void ThreadRuntime::update_metadata(ThreadRuntimeMetadata metadata) {
    std::lock_guard lock(metadata_mutex_);
    metadata_ = std::move(metadata);
}

void ThreadRuntime::mutate_metadata(
    const std::function<void(ThreadRuntimeMetadata&)>& mutation) {
    std::lock_guard lock(metadata_mutex_);
    mutation(metadata_);
}

void ThreadRuntime::touch() noexcept {
    last_activity_.store(std::chrono::system_clock::now(), std::memory_order_release);
}

std::chrono::system_clock::time_point ThreadRuntime::last_activity() const noexcept {
    return last_activity_.load(std::memory_order_acquire);
}

bool ThreadRuntime::begin_turn() {
    // Admission, queueing, and completion share one lock. An atomic-only CAS
    // here can otherwise race finish_turn() and lose a newly admitted retry
    // when the finishing worker stores false immediately afterwards.
    std::lock_guard lock(queue_mutex_);
    if (turn_active_.load(std::memory_order_acquire)) {
        return false;
    }
    turn_active_.store(true, std::memory_order_release);
    completion_status_.store(TurnCompletionStatus::None, std::memory_order_release);
    touch();
    return true;
}

bool ThreadRuntime::begin_or_queue(PendingAgentTurn& turn) {
    std::lock_guard lock(queue_mutex_);
    if (turn_active_.load(std::memory_order_acquire)) {
        queued_turns_.push_back(std::move(turn));
        touch();
        return false;
    }
    turn_active_.store(true, std::memory_order_release);
    completion_status_.store(TurnCompletionStatus::None, std::memory_order_release);
    touch();
    return true;
}

void ThreadRuntime::queue_turn(PendingAgentTurn turn) {
    std::lock_guard lock(queue_mutex_);
    queued_turns_.push_back(std::move(turn));
    touch();
}

std::optional<PendingAgentTurn> ThreadRuntime::finish_turn(bool succeeded) {
    std::unique_lock lock(queue_mutex_);
    touch();
    if (!queued_turns_.empty()) {
        auto next = std::move(queued_turns_.front());
        queued_turns_.pop_front();
        return next;
    }
    completion_status_.store(
        succeeded ? TurnCompletionStatus::Succeeded : TurnCompletionStatus::Failed,
        std::memory_order_release);
    turn_active_.store(false, std::memory_order_release);
    lock.unlock();
    idle_cv_.notify_all();
    return std::nullopt;
}

std::size_t ThreadRuntime::discard_queued_turns() {
    std::lock_guard lock(queue_mutex_);
    const std::size_t discarded = queued_turns_.size();
    queued_turns_.clear();
    return discarded;
}

void ThreadRuntime::begin_worker() {
    std::lock_guard lock(queue_mutex_);
    ++workers_in_flight_;
}

void ThreadRuntime::finish_worker() noexcept {
    {
        std::lock_guard lock(queue_mutex_);
        if (workers_in_flight_ > 0) {
            --workers_in_flight_;
        }
    }
    idle_cv_.notify_all();
}

std::size_t ThreadRuntime::workers_in_flight() const {
    std::lock_guard lock(queue_mutex_);
    return workers_in_flight_;
}

bool ThreadRuntime::turn_active() const noexcept {
    return turn_active_.load(std::memory_order_acquire);
}

TurnCompletionStatus ThreadRuntime::completion_status() const noexcept {
    return completion_status_.load(std::memory_order_acquire);
}

std::size_t ThreadRuntime::queued_turn_count() const {
    std::lock_guard lock(queue_mutex_);
    return queued_turns_.size();
}

void ThreadRuntime::reset_turn_state() {
    {
        std::lock_guard lock(queue_mutex_);
        queued_turns_.clear();
        completion_status_.store(TurnCompletionStatus::None, std::memory_order_release);
        turn_active_.store(false, std::memory_order_release);
    }
    activity_timers_.clear();
    idle_cv_.notify_all();
}

void ThreadRuntime::request_stop() {
    if (agent_ && turn_active()) {
        agent_->request_stop();
    }
}

void ThreadRuntime::wait_until_idle() {
    std::unique_lock lock(queue_mutex_);
    idle_cv_.wait(lock, [&]() {
        return !turn_active_.load(std::memory_order_acquire)
            && workers_in_flight_ == 0;
    });
}

std::uint64_t ThreadRuntime::request_save() noexcept {
    return latest_save_requested_.fetch_add(1, std::memory_order_acq_rel) + 1;
}

bool ThreadRuntime::is_latest_save(std::uint64_t generation) const noexcept {
    return generation == latest_save_requested_.load(std::memory_order_acquire);
}

std::mutex& ThreadRuntime::save_mutex() noexcept {
    return save_mutex_;
}

void ThreadRuntime::begin_save() noexcept {
    std::lock_guard lock(save_wait_mutex_);
    ++saves_in_flight_;
}

void ThreadRuntime::finish_save() noexcept {
    {
        std::lock_guard lock(save_wait_mutex_);
        if (saves_in_flight_ > 0) {
            --saves_in_flight_;
        }
    }
    save_wait_cv_.notify_all();
}

void ThreadRuntime::wait_until_saved() {
    std::unique_lock lock(save_wait_mutex_);
    save_wait_cv_.wait(lock, [&]() { return saves_in_flight_ == 0; });
}

ActivityTimerRegistry& ThreadRuntime::activity_timers() noexcept {
    return activity_timers_;
}

bool ThreadRuntimeRegistry::insert(ThreadRuntime::Ptr runtime) {
    if (!runtime || runtime->session_id().empty()) {
        return false;
    }
    const std::string session_id = runtime->session_id();
    std::lock_guard lock(mutex_);
    const auto [_, inserted] = runtimes_.emplace(session_id, std::move(runtime));
    return inserted;
}

ThreadRuntime::Ptr ThreadRuntimeRegistry::find(std::string_view session_id) const {
    std::lock_guard lock(mutex_);
    const auto it = runtimes_.find(std::string{session_id});
    return it == runtimes_.end() ? nullptr : it->second;
}

ThreadRuntime::Ptr ThreadRuntimeRegistry::select(std::string_view session_id) {
    std::lock_guard lock(mutex_);
    const auto it = runtimes_.find(std::string{session_id});
    if (it == runtimes_.end()) {
        return nullptr;
    }
    current_session_id_ = it->first;
    return it->second;
}

bool ThreadRuntimeRegistry::rekey(std::string_view old_session_id,
                                  std::string_view new_session_id) {
    if (new_session_id.empty()) {
        return false;
    }
    std::lock_guard lock(mutex_);
    const auto old_it = runtimes_.find(std::string{old_session_id});
    if (old_it == runtimes_.end() || runtimes_.contains(std::string{new_session_id})) {
        return false;
    }
    auto runtime = std::move(old_it->second);
    runtimes_.erase(old_it);
    runtimes_.emplace(std::string{new_session_id}, std::move(runtime));
    if (current_session_id_ == old_session_id) {
        current_session_id_ = std::string{new_session_id};
    }
    return true;
}

ThreadRuntime::Ptr ThreadRuntimeRegistry::current() const {
    std::lock_guard lock(mutex_);
    const auto it = runtimes_.find(current_session_id_);
    return it == runtimes_.end() ? nullptr : it->second;
}

std::vector<ThreadRuntime::Ptr> ThreadRuntimeRegistry::snapshot() const {
    std::lock_guard lock(mutex_);
    std::vector<ThreadRuntime::Ptr> result;
    result.reserve(runtimes_.size());
    for (const auto& [_, runtime] : runtimes_) {
        result.push_back(runtime);
    }
    return result;
}

std::unordered_set<std::string> ThreadRuntimeRegistry::running_session_ids() const {
    std::lock_guard lock(mutex_);
    std::unordered_set<std::string> result;
    for (const auto& [session_id, runtime] : runtimes_) {
        if (runtime->turn_active()) {
            result.insert(session_id);
        }
    }
    return result;
}

bool ThreadRuntimeRegistry::erase(std::string_view session_id) {
    std::lock_guard lock(mutex_);
    if (session_id == current_session_id_) {
        return false;
    }
    const auto it = runtimes_.find(std::string{session_id});
    if (it == runtimes_.end() || it->second->turn_active()
        || it->second->workers_in_flight() > 0) {
        return false;
    }
    runtimes_.erase(it);
    return true;
}

void ThreadRuntimeRegistry::request_stop_all() {
    for (const auto& runtime : snapshot()) {
        // Drop queued steering turns first: finish_turn() otherwise re-starts
        // them right after the in-flight turn winds down, which would make a
        // shutdown run whole extra LLM turns before the idle barrier passes.
        static_cast<void>(runtime->discard_queued_turns());
        runtime->request_stop();
    }
}

void ThreadRuntimeRegistry::wait_until_all_idle() {
    for (const auto& runtime : snapshot()) {
        runtime->wait_until_idle();
    }
}

void ThreadRuntimeRegistry::wait_until_all_saved() {
    for (const auto& runtime : snapshot()) {
        runtime->wait_until_saved();
    }
}

} // namespace tui
