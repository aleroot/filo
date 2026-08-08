#pragma once

#include "ActivityTimer.hpp"
#include "Conversation.hpp"
#include "core/agent/Agent.hpp"
#include "core/session/ActiveSessionLease.hpp"
#include "core/session/SessionData.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tui {

enum class ModelSelectionMode {
    Manual,
    Router,
    Auto,
};

struct ModelSelectionSnapshot {
    ModelSelectionMode mode = ModelSelectionMode::Manual;
    std::string manual_provider_name;
    std::string manual_model_name;
    std::string router_policy;

    bool operator==(const ModelSelectionSnapshot&) const = default;
};

enum class TurnCompletionStatus {
    None,
    Succeeded,
    Failed,
};

struct PendingAgentTurn {
    std::string text;
    core::agent::Agent::TurnCallbacks callbacks;
};

struct ThreadRuntimeMetadata {
    std::string session_id;
    /// User-facing label for this live runtime. This is intentionally not
    /// persisted as the session name: threads and sessions have independent
    /// identities and lifecycles.
    std::string thread_name;
    /// Name of the persisted conversation currently owned by this thread.
    std::string session_name;
    std::string created_at;
    std::string file_path;
    std::string provider;
    std::string model;
    ModelSelectionSnapshot model_selection;
    std::optional<ModelSelectionSnapshot> previous_model_selection;
    bool yolo_enabled = false;
    std::unordered_set<std::string> permission_rules;
    std::optional<core::session::SessionGoal> goal;
    std::optional<core::session::SessionGoalGraph> goal_graph;
};

// Owns every mutable resource whose lifetime must follow a conversation rather
// than the currently visible TUI page. The TUI may switch away while this
// object continues receiving agent callbacks.
class ThreadRuntime final {
public:
    using Ptr = std::shared_ptr<ThreadRuntime>;
    using Messages = std::vector<UiMessage>;

    ThreadRuntime(ThreadRuntimeMetadata metadata,
                  std::shared_ptr<core::agent::Agent> agent,
                  std::shared_ptr<Messages> messages,
                  core::session::ActiveSessionLease::Ptr lease);

    [[nodiscard]] std::string session_id() const;
    [[nodiscard]] std::shared_ptr<core::agent::Agent> agent() const noexcept;
    [[nodiscard]] std::shared_ptr<Messages> messages() const noexcept;
    [[nodiscard]] core::session::ActiveSessionLease::Ptr lease() const;
    void set_lease(core::session::ActiveSessionLease::Ptr lease);

    [[nodiscard]] ThreadRuntimeMetadata metadata() const;
    void update_metadata(ThreadRuntimeMetadata metadata);
    void mutate_metadata(
        const std::function<void(ThreadRuntimeMetadata&)>& mutation);

    [[nodiscard]] bool begin_turn();
    [[nodiscard]] bool begin_or_queue(PendingAgentTurn& turn);
    void queue_turn(PendingAgentTurn turn);
    [[nodiscard]] std::optional<PendingAgentTurn> finish_turn(bool succeeded);
    [[nodiscard]] bool turn_active() const noexcept;
    [[nodiscard]] TurnCompletionStatus completion_status() const noexcept;
    [[nodiscard]] std::size_t queued_turn_count() const;
    void reset_turn_state();

    /// Drop every queued turn without touching the in-flight one. Used at
    /// shutdown so a runtime never starts fresh work while the app winds
    /// down; steering deliberately keeps its queue (it stops + re-queues).
    [[nodiscard]] std::size_t discard_queued_turns();

    /// Tracks detached turn-entry functions independently from logical turn
    /// state. A turn can become idle just before its worker releases the last
    /// references captured from MainApp.
    void begin_worker();
    void finish_worker() noexcept;
    [[nodiscard]] std::size_t workers_in_flight() const;

    void request_stop();
    void wait_until_idle();

    /// Last observable activity (turn start/end, queueing). Powers the
    /// thread browser's recency grouping for live, unsaved threads.
    [[nodiscard]] std::chrono::system_clock::time_point last_activity() const noexcept;

    [[nodiscard]] std::uint64_t request_save() noexcept;
    [[nodiscard]] bool is_latest_save(std::uint64_t generation) const noexcept;
    [[nodiscard]] std::mutex& save_mutex() noexcept;
    void begin_save() noexcept;
    void finish_save() noexcept;
    void wait_until_saved();

    ActivityTimerRegistry& activity_timers() noexcept;

private:
    mutable std::mutex metadata_mutex_;
    ThreadRuntimeMetadata metadata_;
    std::shared_ptr<core::agent::Agent> agent_;
    std::shared_ptr<Messages> messages_;
    core::session::ActiveSessionLease::Ptr lease_;

    std::atomic_bool turn_active_{false};
    std::atomic<TurnCompletionStatus> completion_status_{TurnCompletionStatus::None};
    mutable std::mutex queue_mutex_;
    std::deque<PendingAgentTurn> queued_turns_;
    std::size_t workers_in_flight_ = 0;
    std::condition_variable idle_cv_;
    ActivityTimerRegistry activity_timers_;
    std::atomic<std::chrono::system_clock::time_point> last_activity_{
        std::chrono::system_clock::now()};

    void touch() noexcept;

    std::mutex save_mutex_;
    std::atomic<std::uint64_t> latest_save_requested_{0};
    std::mutex save_wait_mutex_;
    std::condition_variable save_wait_cv_;
    std::size_t saves_in_flight_ = 0;
};

// Registry and selection are deliberately separate: selecting a thread never
// stops, clears, or replaces another runtime.
class ThreadRuntimeRegistry final {
public:
    [[nodiscard]] bool insert(ThreadRuntime::Ptr runtime);
    [[nodiscard]] ThreadRuntime::Ptr find(std::string_view session_id) const;
    [[nodiscard]] ThreadRuntime::Ptr select(std::string_view session_id);
    [[nodiscard]] bool rekey(std::string_view old_session_id,
                             std::string_view new_session_id);
    [[nodiscard]] ThreadRuntime::Ptr current() const;
    [[nodiscard]] std::vector<ThreadRuntime::Ptr> snapshot() const;
    [[nodiscard]] std::unordered_set<std::string> running_session_ids() const;
    [[nodiscard]] bool erase(std::string_view session_id);

    void request_stop_all();
    void wait_until_all_idle();
    void wait_until_all_saved();

private:
    mutable std::mutex mutex_;
    std::unordered_map<std::string, ThreadRuntime::Ptr> runtimes_;
    std::string current_session_id_;
};

} // namespace tui
