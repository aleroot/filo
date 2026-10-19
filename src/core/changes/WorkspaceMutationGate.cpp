#include "WorkspaceMutationGate.hpp"
#include "../utils/PathUtils.hpp"

#include <algorithm>
#include <condition_variable>
#include <mutex>

namespace core::changes {
struct WorkspaceMutationGate::State {
    std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::weak_ptr<Lease>> leases;
};

WorkspaceMutationGate::WorkspaceMutationGate() : state_(std::make_shared<State>()) {}

WorkspaceMutationGate::Lease::Lease(std::shared_ptr<State> state, Access access)
    : state_(std::move(state)), access_(access) {}

WorkspaceMutationGate::Lease::~Lease() {
    state_->changed.notify_all();
}

std::expected<std::shared_ptr<WorkspaceMutationGate::Lease>, std::string>
WorkspaceMutationGate::acquire(std::span<const std::filesystem::path> roots,
                               Access access, std::chrono::milliseconds timeout,
                               const std::function<bool()>& cancelled) {
    const auto& registry = state_;
    auto lease = std::shared_ptr<Lease>(new Lease(registry, access));
    for (const auto& root : roots) if (!root.empty()) lease->roots_.push_back(core::utils::path::directory_for(root));
    std::ranges::sort(lease->roots_);
    lease->roots_.erase(std::ranges::unique(lease->roots_).begin(), lease->roots_.end());
    const auto overlaps = [&](const auto& other) {
        for (const auto& a : lease->roots_) for (const auto& b : other.roots_)
            if (core::utils::path::overlaps(a, b)) return true;
        return false;
    };
    std::unique_lock lock(registry->mutex);
    std::erase_if(registry->leases, [](const auto& item) { return item.expired(); });
    registry->leases.push_back(lease);
    const auto ready = [&] {
        bool earlier = true;
        for (const auto& weak : registry->leases) {
            const auto other = weak.lock();
            if (!other) continue;
            if (other.get() == lease.get()) { earlier = false; continue; }
            if (!overlaps(*other)) continue;
            if ((other->registered_ || earlier) &&
                (access == Access::Exclusive || other->access_ == Access::Exclusive)) return false;
        }
        return true;
    };
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!ready() && !(cancelled && cancelled()) && std::chrono::steady_clock::now() < deadline)
        registry->changed.wait_until(lock, std::min(deadline,
            std::chrono::steady_clock::now() + std::chrono::milliseconds(20)));
    if ((cancelled && cancelled()) || !ready())
        return std::unexpected("Workspace has an active writer; retry after it stops.");
    lease->registered_ = true;
    return lease;
}
} // namespace core::changes
