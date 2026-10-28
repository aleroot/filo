#pragma once

#include <algorithm>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

namespace core::llm {

/// Per-account cache for a provider's out-of-band usage endpoint.
///
/// Subscription quotas span hours, so a reading a minute old is invisible in
/// the status bar, while the read itself sits on the response path. Each
/// account (identity) has its own lock: concurrent streams of one account
/// share a single read, and network I/O never holds the cache-wide lock. A
/// failed read is cached as "unknown" for `failure_ttl`, so an outage is not
/// retried on every response.
template <class Snapshot>
class UsageSnapshotCache {
public:
    explicit UsageSnapshotCache(std::chrono::steady_clock::duration ttl)
        : UsageSnapshotCache(ttl, ttl) {}
    /// For a snapshot kept long enough that a transient failure should not
    /// hide it for the whole `ttl`.
    UsageSnapshotCache(std::chrono::steady_clock::duration ttl,
                       std::chrono::steady_clock::duration failure_ttl)
        : ttl_(ttl), failure_ttl_(failure_ttl) {}

    /// The snapshot for `identity`, calling `fetch()` once the cached one has
    /// expired. A throwing fetch counts as an unknown reading.
    template <class Fetch>
    std::optional<Snapshot> get(const std::string& identity, Fetch&& fetch) {
        const auto entry = entry_for(identity);
        if (!entry) return std::nullopt;
        std::lock_guard lock(entry->mutex);
        if (entry->expires > std::chrono::steady_clock::now()) return entry->value;
        entry->value.reset();
        try {
            entry->value = fetch();
        } catch (...) { /* Usage is best-effort and must not fail generation. */ }
        entry->expires = std::chrono::steady_clock::now() + (entry->value ? ttl_ : failure_ttl_);
        return entry->value;
    }

private:
    struct Entry {
        std::mutex mutex;
        std::optional<Snapshot> value;
        std::chrono::steady_clock::time_point expires{};
    };

    std::shared_ptr<Entry> entry_for(const std::string& identity) {
        std::lock_guard lock(mutex_);
        if (const auto it = cache_.find(identity); it != cache_.end()) return it->second;
        if (cache_.size() >= kMaxAccounts) {
            // Never evict an in-flight entry: its waiters must share its read.
            const auto idle = std::find_if(cache_.begin(), cache_.end(), [](const auto& item) {
                return item.second.use_count() == 1;
            });
            if (idle == cache_.end()) return {};
            cache_.erase(idle);
        }
        auto entry = std::make_shared<Entry>();
        cache_.emplace(identity, entry);
        return entry;
    }

    static constexpr std::size_t kMaxAccounts = 64;
    std::chrono::steady_clock::duration ttl_;
    std::chrono::steady_clock::duration failure_ttl_;
    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<Entry>> cache_;
};

} // namespace core::llm
