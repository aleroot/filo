#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace exec::daemon::detail {

struct McpDispatchResult {
    int status = 200;
    std::string body;
};

// Keeps recently completed request responses for a short time so that retries
// (same session + id + payload) are replayed without re-executing tools.
class RequestReplayCache {
public:
    enum class DecisionKind {
        execute,
        replay_completed,
        wait_inflight,
    };

    struct InflightEntry {
        std::mutex mutex;
        std::condition_variable cv;
        bool ready = false;
        uint64_t generation = 0;
        McpDispatchResult result;
    };

    struct Decision {
        DecisionKind kind{DecisionKind::execute};
        McpDispatchResult replay_result;
        std::shared_ptr<InflightEntry> inflight;
    };

    explicit RequestReplayCache(
        std::chrono::steady_clock::duration replay_ttl = std::chrono::minutes{2},
        std::size_t max_completed_entries = 1024)
        : replay_ttl_(replay_ttl),
          max_completed_entries_(max_completed_entries) {}

    // Stable, delimiter-safe representation of a session id for replay keys.
    // Hex encoding guarantees no '|' characters so key parsing remains unambiguous.
    [[nodiscard]] static std::string session_token(std::string_view session_id) {
        static constexpr char kHex[] = "0123456789abcdef";
        std::string out;
        out.resize(session_id.size() * 2);
        for (std::size_t i = 0; i < session_id.size(); ++i) {
            const unsigned char ch = static_cast<unsigned char>(session_id[i]);
            out[i * 2] = kHex[ch >> 4];
            out[i * 2 + 1] = kHex[ch & 0x0f];
        }
        return out;
    }

    [[nodiscard]] Decision begin(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        prune_expired_locked();
        const std::string session_id = session_from_key(key);
        const uint64_t generation = session_generation_locked(session_id);

        if (auto it = completed_.find(key); it != completed_.end()) {
            return Decision{
                .kind = DecisionKind::replay_completed,
                .replay_result = it->second.result,
                .inflight = nullptr,
            };
        }

        if (auto it = inflight_.find(key); it != inflight_.end()) {
            return Decision{
                .kind = DecisionKind::wait_inflight,
                .replay_result = {},
                .inflight = it->second,
            };
        }

        auto entry = std::make_shared<InflightEntry>();
        entry->generation = generation;
        inflight_[key] = entry;
        return Decision{
            .kind = DecisionKind::execute,
            .replay_result = {},
            .inflight = std::move(entry),
        };
    }

    [[nodiscard]] McpDispatchResult wait_for(const std::shared_ptr<InflightEntry>& entry) const {
        std::unique_lock<std::mutex> lock(entry->mutex);
        entry->cv.wait(lock, [&entry]() { return entry->ready; });
        return entry->result;
    }

    void finish(const std::string& key,
                const std::shared_ptr<InflightEntry>& entry,
                const McpDispatchResult& result,
                bool cache_result) {
        {
            std::lock_guard<std::mutex> lock(entry->mutex);
            entry->result = result;
            entry->ready = true;
        }
        entry->cv.notify_all();

        std::lock_guard<std::mutex> lock(mutex_);
        if (auto inflight_it = inflight_.find(key);
            inflight_it != inflight_.end() && inflight_it->second == entry) {
            inflight_.erase(inflight_it);
        }
        if (!cache_result || max_completed_entries_ == 0) return;
        const std::string session_id = session_from_key(key);
        if (entry->generation != session_generation_locked(session_id)) return;

        completed_[key] = CompletedEntry{
            .result = result,
            .stored_at = std::chrono::steady_clock::now(),
        };
        if (completed_.size() > max_completed_entries_) {
            evict_oldest_locked();
        }
    }

    void clear_session(std::string_view session_id) {
        if (session_id.empty()) return;
        const std::string canonical_session = session_token(session_id);
        const std::string canonical_prefix = canonical_session + "|";

        std::vector<std::shared_ptr<InflightEntry>> to_cancel;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            // Invalidate stale in-flight completions for this logical session.
            ++session_generations_[canonical_session];

            std::erase_if(completed_, [&](const auto& pair) {
                return pair.first.starts_with(canonical_prefix);
            });

            for (auto it = inflight_.begin(); it != inflight_.end();) {
                if (it->first.starts_with(canonical_prefix)) {
                    to_cancel.push_back(it->second);
                    it = inflight_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        for (const auto& entry : to_cancel) {
            {
                std::lock_guard<std::mutex> lock(entry->mutex);
                if (entry->ready) continue;
                entry->result.status = 499;
                entry->result.body = R"({"error":"MCP session was closed"})";
                entry->ready = true;
            }
            entry->cv.notify_all();
        }
    }

private:
    struct CompletedEntry {
        McpDispatchResult result;
        std::chrono::steady_clock::time_point stored_at;
    };

    void prune_expired_locked() {
        const auto now = std::chrono::steady_clock::now();
        std::erase_if(
            completed_,
            [&](const auto& pair) {
                return (now - pair.second.stored_at) > replay_ttl_;
            });
    }

    void evict_oldest_locked() {
        if (completed_.empty()) return;

        auto oldest_it = completed_.begin();
        for (auto it = std::next(completed_.begin()); it != completed_.end(); ++it) {
            if (it->second.stored_at < oldest_it->second.stored_at) {
                oldest_it = it;
            }
        }
        completed_.erase(oldest_it);
    }

    [[nodiscard]] static std::string session_from_key(std::string_view key) {
        const std::size_t sep = key.find('|');
        if (sep == std::string_view::npos) return std::string(key);
        return std::string(key.substr(0, sep));
    }

    [[nodiscard]] uint64_t session_generation_locked(std::string_view session_id) const {
        if (auto it = session_generations_.find(std::string(session_id));
            it != session_generations_.end()) {
            return it->second;
        }
        return 0;
    }

    std::chrono::steady_clock::duration replay_ttl_;
    std::size_t max_completed_entries_;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, CompletedEntry> completed_;
    std::unordered_map<std::string, std::shared_ptr<InflightEntry>> inflight_;
    std::unordered_map<std::string, uint64_t> session_generations_;
};

} // namespace exec::daemon::detail
