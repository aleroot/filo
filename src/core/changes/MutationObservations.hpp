#pragma once

#include "../utils/PathUtils.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace core::changes {

// Attribution only: observations never exclude or wait for another writer.
class MutationObservations final {
public:
    class Owner final {
        friend class MutationObservations;
    public:
        explicit Owner(std::function<void()> persist = {}) : persist_(std::move(persist)) {}
        [[nodiscard]] bool ambiguous() const noexcept { return ambiguous_.load(); }
    private:
        bool mark_pending() {
            std::lock_guard lock(mutex_);
            ambiguous_.store(true);
            if (pending_ || persisted_) return false;
            pending_ = true;
            return true;
        }
        void persist() {
            const auto complete = [this](std::exception_ptr failure = {}) {
                {
                    std::lock_guard lock(mutex_);
                    failure_ = failure;
                    persisted_ = !failure;
                    pending_ = false;
                }
                finished_.notify_all();
            };
            try { if (persist_) persist_(); }
            catch (...) { complete(std::current_exception()); throw; }
            complete();
        }
        void wait() {
            std::unique_lock lock(mutex_);
            finished_.wait(lock, [this] { return !pending_; });
            if (failure_) std::rethrow_exception(failure_);
        }
        std::atomic<bool> ambiguous_{false};
        std::function<void()> persist_;
        std::mutex mutex_;
        std::condition_variable finished_;
        bool pending_ = false;
        bool persisted_ = false;
        std::exception_ptr failure_;
    };

    class Observation final {
        friend class MutationObservations;
        std::shared_ptr<Owner> owner_;
        std::vector<std::filesystem::path> paths_;
        bool unscoped_ = false;
    };

    [[nodiscard]] std::shared_ptr<Observation> observe(
        std::shared_ptr<Owner> owner,
        std::span<const std::filesystem::path> paths, bool unscoped = false) {
        if (paths.empty()) return {};
        auto observation = std::make_shared<Observation>();
        observation->owner_ = std::move(owner);
        observation->unscoped_ = unscoped;
        for (const auto& path : paths) {
            if (!path.empty()) observation->paths_.push_back(core::utils::path::normalized(path));
        }
        struct Notification { std::shared_ptr<Owner> owner; bool persist; };
        std::vector<Notification> notifications;
        {
            std::lock_guard lock(mutex_);
            std::erase_if(active_, [](const auto& item) { return item.expired(); });
            // Allocate before scheduling notifications so allocation failure
            // cannot strand an owner waiting for an unqueued callback.
            active_.reserve(active_.size() + 1);
            notifications.reserve(active_.size() + 1);
            for (const auto& weak : active_) {
                const auto other = weak.lock();
                if (!other || (observation->owner_ && other->owner_ == observation->owner_)) continue;
                const bool overlaps = std::ranges::any_of(observation->paths_, [&](const auto& a) {
                    return std::ranges::any_of(other->paths_, [&](const auto& b) {
                        return core::utils::path::normalized_within(a, b)
                            || core::utils::path::normalized_within(b, a);
                    });
                });
                if (!overlaps || (!other->unscoped_ && !observation->unscoped_)) continue;
                for (const auto& item : {other, observation}) {
                    if (!item->owner_) continue;
                    if (std::ranges::any_of(notifications, [&](const auto& notification) {
                            return notification.owner == item->owner_;
                        })) continue;
                    notifications.push_back({item->owner_, item->owner_->mark_pending()});
                }
            }
            active_.push_back(observation);
        }
        // Filesystem persistence must never hold the shared registry mutex.
        std::exception_ptr failure;
        // Complete callbacks owned by this invocation before awaiting any
        // other invocation. Otherwise two overlaps could await each other.
        for (const auto& notification : notifications) {
            if (!notification.persist) continue;
            try {
                notification.owner->persist();
            }
            catch (...) { if (!failure) failure = std::current_exception(); }
        }
        for (const auto& notification : notifications) {
            if (notification.persist) continue;
            try { notification.owner->wait(); }
            catch (...) { if (!failure) failure = std::current_exception(); }
        }
        if (failure) std::rethrow_exception(failure);
        return observation;
    }

    void close(const std::shared_ptr<Owner>& owner) {
        {
            std::lock_guard lock(mutex_);
            std::erase_if(active_, [&](const auto& weak) {
                const auto item = weak.lock();
                return !item || item->owner_ == owner;
            });
        }
        // Only this owner's already-scheduled durable notification is awaited.
        owner->wait();
    }

private:
    std::mutex mutex_;
    std::vector<std::weak_ptr<Observation>> active_;
};

} // namespace core::changes
