#include "RemoteActivity.hpp"

#include <algorithm>
#include <cctype>
#include <utility>

namespace core::mcp {

namespace {

/// A client that keeps issuing requests must not wake the UI once per HTTP
/// request. Pure liveness touches are coalesced to at most one notification
/// per client per interval; genuine state changes always notify immediately.
constexpr std::chrono::seconds kLivenessNotifyInterval{1};

/// Returns the longest prefix of `value` that is at most `max_bytes` long and
/// does not split a UTF-8 code point. Marker-free: callers pick the marker
/// that suits their context.
[[nodiscard]] std::string_view truncate_utf8_prefix(std::string_view value,
                                                    std::size_t max_bytes) {
    if (value.size() <= max_bytes) return value;

    std::size_t end = max_bytes;
    while (end > 0
           && (static_cast<unsigned char>(value[end]) & 0xc0U) == 0x80U) {
        --end;
    }
    return value.substr(0, end);
}

[[nodiscard]] std::string clamp_payload(std::string_view value, std::size_t max_bytes) {
    if (value.size() <= max_bytes) return std::string(value);
    std::string out(truncate_utf8_prefix(value, max_bytes));
    out += "\n[truncated]";
    return out;
}

[[nodiscard]] std::string sanitize_terminal_text(std::string_view value,
                                                 std::size_t max_bytes) {
    std::string sanitized;
    sanitized.reserve(std::min(value.size(), max_bytes));
    bool pending_space = false;

    for (const unsigned char ch : value) {
        const bool ascii_space = ch < 0x80U && std::isspace(ch) != 0;
        const bool ascii_control = ch < 0x20U || ch == 0x7fU;
        if (ascii_space || ascii_control) {
            pending_space = !sanitized.empty();
            continue;
        }
        if (pending_space) {
            sanitized.push_back(' ');
            pending_space = false;
        }
        sanitized.push_back(static_cast<char>(ch));
    }

    if (sanitized.size() <= max_bytes) return sanitized;
    std::string out(truncate_utf8_prefix(sanitized, max_bytes));
    out += "…";
    return out;
}

} // namespace

std::string sanitize_remote_client_name(std::string_view name) {
    std::string sanitized = sanitize_terminal_text(name, 48);
    if (sanitized.empty()) return "Client";
    return sanitized;
}

RemoteActivityHub& RemoteActivityHub::get_instance() {
    static RemoteActivityHub instance;
    return instance;
}

void RemoteActivityHub::notify() {
    std::function<void()> callback;
    {
        std::lock_guard lock(mutex_);
        if (!notify_callback_) return;
        callback = notify_callback_;
        ++notify_in_flight_;
    }

    // Deliberately invoked without mutex_ held: the callback re-enters the hub
    // (snapshot()) and blocks on the UI event queue, so holding the lock here
    // would deadlock and stall every producer thread. The in-flight count is
    // what lets set_notify_callback({}) wait us out instead.
    struct InFlightGuard {
        RemoteActivityHub& hub;
        ~InFlightGuard() {
            {
                std::lock_guard lock(hub.mutex_);
                --hub.notify_in_flight_;
            }
            hub.notify_idle_cv_.notify_all();
        }
    } guard{*this};

    callback();
}

void RemoteActivityHub::detach_notify_callback_locked(
    std::unique_lock<std::mutex>& lock) {
    notify_callback_ = {};
    notify_idle_cv_.wait(lock, [this] { return notify_in_flight_ == 0; });
}

void RemoteActivityHub::server_starting() {
    {
        std::lock_guard lock(mutex_);
        server_state_ = RemoteServerState::starting;
        server_detail_.clear();
        clients_.clear();
        activities_.clear();
    }
    notify();
}

void RemoteActivityHub::server_listening(std::string detail) {
    {
        std::lock_guard lock(mutex_);
        server_state_ = RemoteServerState::listening;
        server_detail_ = std::move(detail);
    }
    notify();
}

void RemoteActivityHub::server_failed(std::string detail) {
    {
        std::lock_guard lock(mutex_);
        server_state_ = RemoteServerState::failed;
        server_detail_ = std::move(detail);
    }
    notify();
}

void RemoteActivityHub::server_stopped() {
    {
        std::lock_guard lock(mutex_);
        server_state_ = RemoteServerState::stopped;
        for (auto& [_, client] : clients_) {
            client.closed = true;
        }
    }
    notify();
}

void RemoteActivityHub::client_initialized(std::string session_id,
                                           std::string_view client_name,
                                           std::string_view client_version) {
    {
        std::lock_guard lock(mutex_);
        auto& client = touch_client_locked(session_id);
        client.name = sanitize_remote_client_name(client_name);
        client.version = sanitize_terminal_text(client_version, 32);
        client.ready = false;
    }
    notify();
}

RemoteClientActivity& RemoteActivityHub::touch_client_locked(
    std::string_view session_id) {
    auto [it, inserted] = clients_.try_emplace(std::string(session_id));
    auto& client = it->second;
    if (inserted) {
        client.session_id = std::string(session_id);
        client.name = "Client";
    }
    client.closed = false;
    client.last_seen = std::chrono::steady_clock::now();
    if (inserted) prune_clients_locked(client.session_id);
    return client;
}

void RemoteActivityHub::prune_clients_locked(
    std::string_view protected_session_id) {
    while (clients_.size() > kMaxClients) {
        auto victim = clients_.end();
        for (auto it = clients_.begin(); it != clients_.end(); ++it) {
            // Never evict the entry the caller is about to hand out: doing so
            // would dangle the reference returned by touch_client_locked().
            if (it->first == protected_session_id) continue;
            if (victim == clients_.end()) {
                victim = it;
                continue;
            }
            if (it->second.closed != victim->second.closed) {
                if (it->second.closed) victim = it;
                continue;
            }
            if (it->second.last_seen < victim->second.last_seen) victim = it;
        }
        if (victim == clients_.end()) return;
        clients_.erase(victim);
    }
}

void RemoteActivityHub::prune_activities_locked() {
    while (activities_.size() > kMaxActivities) {
        const auto evict = std::ranges::find_if(
            activities_,
            [](const RemoteToolActivity& activity) {
                return activity.status != RemoteToolStatus::running;
            });
        // When every entry is still running (e.g. an execution that never
        // reported completion), evict the oldest entry so the history stays
        // bounded. A late completion for an evicted id is dropped silently by
        // find_activity_locked().
        activities_.erase(evict != activities_.end() ? evict : activities_.begin());
    }
}

void RemoteActivityHub::client_ready(std::string_view session_id) {
    {
        std::lock_guard lock(mutex_);
        touch_client_locked(session_id).ready = true;
    }
    notify();
}

void RemoteActivityHub::client_seen(std::string_view session_id) {
    bool notify_ui = false;
    {
        std::lock_guard lock(mutex_);
        const auto it = clients_.find(std::string(session_id));
        // A first sighting or a reopened session is an observable state change
        // and must refresh immediately; a steady request stream from a session
        // that is already open only advances its "last seen" clock, so it is
        // coalesced to avoid one UI wake-up per HTTP request.
        notify_ui = it == clients_.end()
            || it->second.closed
            || std::chrono::steady_clock::now() - it->second.last_seen
                   >= kLivenessNotifyInterval;
        touch_client_locked(session_id);
    }
    if (notify_ui) notify();
}

void RemoteActivityHub::client_closed(std::string_view session_id) {
    {
        std::lock_guard lock(mutex_);
        auto it = clients_.find(std::string(session_id));
        if (it == clients_.end()) return;
        it->second.closed = true;
        it->second.last_seen = std::chrono::steady_clock::now();
    }
    notify();
}

std::uint64_t RemoteActivityHub::tool_started(std::string_view session_id,
                                              std::string_view tool_name,
                                              std::string_view arguments) {
    std::uint64_t activity_id = 0;
    {
        std::lock_guard lock(mutex_);
        touch_client_locked(session_id);
        activity_id = next_activity_id_++;
        activities_.push_back(RemoteToolActivity{
            .id = activity_id,
            .session_id = std::string(session_id),
            .tool_name = std::string(tool_name),
            .arguments = clamp_payload(arguments, kMaxStoredArguments),
            .status = RemoteToolStatus::running,
            .started_at = std::chrono::steady_clock::now(),
        });
        prune_activities_locked();
    }
    notify();
    return activity_id;
}

RemoteToolActivity* RemoteActivityHub::find_activity_locked(std::uint64_t activity_id) {
    const auto it = std::ranges::find(activities_, activity_id, &RemoteToolActivity::id);
    return it == activities_.end() ? nullptr : &*it;
}

void RemoteActivityHub::tool_finished(std::uint64_t activity_id,
                                      std::string_view result,
                                      bool failed) {
    {
        std::lock_guard lock(mutex_);
        auto* activity = find_activity_locked(activity_id);
        if (activity == nullptr || activity->status != RemoteToolStatus::running) return;
        activity->result = clamp_payload(result, kMaxStoredResult);
        activity->status =
            failed ? RemoteToolStatus::failed : RemoteToolStatus::succeeded;
        activity->finished_at = std::chrono::steady_clock::now();
        prune_activities_locked();
    }
    notify();
}

void RemoteActivityHub::tool_cancelled(std::uint64_t activity_id,
                                       std::string_view result) {
    {
        std::lock_guard lock(mutex_);
        auto* activity = find_activity_locked(activity_id);
        if (activity == nullptr || activity->status != RemoteToolStatus::running) return;
        activity->result = clamp_payload(result, kMaxStoredResult);
        activity->status = RemoteToolStatus::cancelled;
        activity->finished_at = std::chrono::steady_clock::now();
        prune_activities_locked();
    }
    notify();
}

void RemoteActivityHub::reset_for_testing() {
    std::unique_lock lock(mutex_);
    detach_notify_callback_locked(lock);
    server_state_ = RemoteServerState::disabled;
    server_detail_.clear();
    clients_.clear();
    activities_.clear();
    next_activity_id_ = 1;
}

std::size_t RemoteActivityHub::unacknowledged_errors_locked() const {
    return static_cast<std::size_t>(std::ranges::count_if(
        activities_,
        [](const RemoteToolActivity& activity) {
            return activity.status == RemoteToolStatus::failed
                && !activity.error_acknowledged;
        }));
}

std::size_t RemoteActivityHub::client_count() const {
    std::lock_guard lock(mutex_);
    return clients_.size();
}

void RemoteActivityHub::acknowledge_errors() {
    {
        std::lock_guard lock(mutex_);
        for (auto& activity : activities_) {
            activity.error_acknowledged = true;
        }
    }
    notify();
}

void RemoteActivityHub::clear_completed() {
    {
        std::lock_guard lock(mutex_);
        std::erase_if(activities_, [](const RemoteToolActivity& activity) {
            return activity.status != RemoteToolStatus::running;
        });
    }
    notify();
}

RemoteActivitySnapshot RemoteActivityHub::snapshot(bool include_payloads) const {
    std::lock_guard lock(mutex_);
    RemoteActivitySnapshot out{
        .server_state = server_state_,
        .server_detail = server_detail_,
        .unacknowledged_errors = unacknowledged_errors_locked(),
    };
    out.clients.reserve(clients_.size());
    for (const auto& [_, client] : clients_) {
        out.clients.push_back(client);
    }
    std::ranges::sort(out.clients, [](const auto& lhs, const auto& rhs) {
        if (lhs.closed != rhs.closed) return !lhs.closed;
        return lhs.last_seen > rhs.last_seen;
    });

    out.activities.reserve(activities_.size());
    for (auto it = activities_.rbegin(); it != activities_.rend(); ++it) {
        if (include_payloads) {
            out.activities.push_back(*it);
        } else {
            out.activities.push_back(RemoteToolActivity{
                .id = it->id,
                .session_id = it->session_id,
                .tool_name = it->tool_name,
                .status = it->status,
                .started_at = it->started_at,
                .finished_at = it->finished_at,
                .error_acknowledged = it->error_acknowledged,
            });
        }
    }
    return out;
}

void RemoteActivityHub::set_notify_callback(std::function<void()> callback) {
    std::unique_lock lock(mutex_);
    if (!callback) {
        detach_notify_callback_locked(lock);
        return;
    }
    notify_callback_ = std::move(callback);
}

} // namespace core::mcp
