#include "DetachedWorkTracker.hpp"

namespace core::utils {

DetachedWorkTracker::Guard::Guard(DetachedWorkTracker& tracker) noexcept
    : tracker_(tracker) {
    tracker_.enter();
}

DetachedWorkTracker::Guard::~Guard() {
    tracker_.leave();
}

bool DetachedWorkTracker::wait_for_idle(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return cv_.wait_for(lock, timeout, [&] { return active_ == 0; });
}

std::size_t DetachedWorkTracker::active() const noexcept {
    std::lock_guard lock(mutex_);
    return active_;
}

DetachedWorkTracker& DetachedWorkTracker::instance() noexcept {
    static DetachedWorkTracker* const tracker = new DetachedWorkTracker();
    return *tracker;
}

void DetachedWorkTracker::enter() noexcept {
    std::lock_guard lock(mutex_);
    ++active_;
}

void DetachedWorkTracker::leave() noexcept {
    {
        std::lock_guard lock(mutex_);
        --active_;
    }
    cv_.notify_all();
}

} // namespace core::utils
