#pragma once

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace core::mcp {

enum class RemoteServerState {
    disabled,
    starting,
    listening,
    failed,
    stopped,
};

enum class RemoteToolStatus {
    running,
    succeeded,
    failed,
    cancelled,
};

struct RemoteClientActivity {
    std::string session_id;
    std::string name{"Client"};
    std::string version;
    bool ready = false;
    bool closed = false;
    std::chrono::steady_clock::time_point last_seen{};
};

struct RemoteToolActivity {
    std::uint64_t id = 0;
    std::string session_id;
    std::string tool_name;
    std::string arguments;
    std::string result;
    RemoteToolStatus status{RemoteToolStatus::running};
    std::chrono::steady_clock::time_point started_at{};
    std::chrono::steady_clock::time_point finished_at{};
    /// True once this failure has been presented to the user. The snapshot's
    /// unacknowledged_errors is *derived* from this flag rather than tracked in
    /// a parallel counter, so evicting a failed entry can never strand a stale
    /// "N issues" badge.
    bool error_acknowledged = false;
};

struct RemoteActivitySnapshot {
    RemoteServerState server_state{RemoteServerState::disabled};
    std::string server_detail;
    std::vector<RemoteClientActivity> clients;
    // Newest activity first.
    std::vector<RemoteToolActivity> activities;
    std::size_t unacknowledged_errors = 0;
};

/// Sanitizes an MCP clientInfo.name for terminal display.
/// Empty or unusable names become "Client".
[[nodiscard]] std::string sanitize_remote_client_name(std::string_view name);

/**
 * Process-wide, UI-independent activity model for inbound MCP HTTP clients.
 *
 * Producers run on daemon/worker threads. The TUI reads immutable snapshots
 * and receives a lightweight wake-up callback; remote activity never enters
 * the local conversation or session store.
 */
class RemoteActivityHub {
public:
    static RemoteActivityHub& get_instance();

    void server_starting();
    void server_listening(std::string detail);
    void server_failed(std::string detail);
    void server_stopped();

    void client_initialized(std::string session_id,
                            std::string_view client_name,
                            std::string_view client_version);
    /// Stateless (MCP 2026-07-28) identity refresh: every request carries
    /// clientInfo in _meta and no initialized notification ever follows.
    void client_identified(std::string session_id,
                           std::string_view client_name,
                           std::string_view client_version);
    void client_ready(std::string_view session_id);
    void client_seen(std::string_view session_id);
    void client_closed(std::string_view session_id);

    [[nodiscard]] std::uint64_t tool_started(std::string_view session_id,
                                             std::string_view tool_name,
                                             std::string_view arguments);
    void tool_finished(std::uint64_t activity_id,
                       std::string_view result,
                       bool failed);
    void tool_cancelled(std::uint64_t activity_id,
                        std::string_view result = {});

    void acknowledge_errors();
    void clear_completed();

    /// Number of retained clients, including sessions already marked closed.
    /// Exposed for tests that assert the retention bound.
    [[nodiscard]] std::size_t client_count() const;

    /// Restores the hub to its default-constructed state and detaches the
    /// notify callback. Test-only seam: production lifecycle resets go through
    /// server_starting().
    void reset_for_testing();

    /// When include_payloads is false, arguments/results are omitted so the
    /// footer can refresh cheaply even after many large tool calls.
    [[nodiscard]] RemoteActivitySnapshot snapshot(bool include_payloads = true) const;

    /**
     * Installs the single TUI wake-up callback; passing {} detaches it.
     *
     * Detaching is a *barrier*: it blocks until no notification is executing
     * and guarantees none can start afterwards. Callbacks are invoked from
     * daemon threads and typically capture TUI-owned state by reference, so
     * this is what makes it safe to destroy that state once detach returns.
     *
     * Precondition: never call this from inside the notify callback itself
     * (it would wait on its own completion).
     */
    void set_notify_callback(std::function<void()> callback);

private:
    RemoteActivityHub() = default;

    void notify();
    /// Returns a reference to the (possibly newly inserted) client record.
    /// The reference stays valid for the remainder of the critical section:
    /// retention pruning never evicts `session_id`.
    RemoteClientActivity& touch_client_locked(std::string_view session_id);
    void prune_activities_locked();
    void prune_clients_locked(std::string_view protected_session_id);
    void detach_notify_callback_locked(std::unique_lock<std::mutex>& lock);
    [[nodiscard]] std::size_t unacknowledged_errors_locked() const;
    [[nodiscard]] RemoteToolActivity* find_activity_locked(std::uint64_t activity_id);

    /// Hard bound on retained activities. Finished entries are evicted first;
    /// if every entry is still running (e.g. an execution that never reported
    /// completion), the oldest running entry is evicted instead so a stuck
    /// producer can never grow the history without limit.
    static constexpr std::size_t kMaxActivities = 100;
    static constexpr std::size_t kMaxStoredArguments = 16 * 1024;
    static constexpr std::size_t kMaxStoredResult = 32 * 1024;

    /// Hard bound on retained client records. An MCP session only disappears
    /// on an explicit HTTP DELETE, which clients are free never to send, so
    /// without this bound the map would grow for the lifetime of the daemon.
    /// Closed sessions are evicted first, then the least recently seen.
    static constexpr std::size_t kMaxClients = 64;

    mutable std::mutex mutex_;
    RemoteServerState server_state_{RemoteServerState::disabled};
    std::string server_detail_;
    std::unordered_map<std::string, RemoteClientActivity> clients_;
    std::vector<RemoteToolActivity> activities_;
    std::uint64_t next_activity_id_ = 1;
    std::function<void()> notify_callback_;
    std::condition_variable notify_idle_cv_;
    std::size_t notify_in_flight_ = 0;
};

} // namespace core::mcp
