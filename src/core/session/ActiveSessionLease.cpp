#include "ActiveSessionLease.hpp"

#include "SessionData.hpp"
#include "SessionStore.hpp"

#include <filesystem>
#include <format>
#include <utility>

namespace core::session {

std::filesystem::path active_session_lease_path(
    const std::filesystem::path& session_path) {
    return std::filesystem::path(session_path.string() + ".active.lock");
}

ActiveSessionLease::ActiveSessionLease(
    std::string session_id,
    core::utils::InterprocessFileLock lock) noexcept
    : session_id_(std::move(session_id)), lock_(std::move(lock)) {}

std::expected<ActiveSessionLease::Ptr, std::string> ActiveSessionLease::acquire(
    const SessionStore& store,
    const SessionData& data) {
    const auto lock_path = active_session_lease_path(store.compute_path(data));
    std::string error;
    auto lock = core::utils::InterprocessFileLock::try_acquire(lock_path, &error);
    if (!lock) {
        if (error.empty()) {
            error = std::format(
                "Session '{}' is already open in another Filo process.",
                data.session_id);
        }
        return std::unexpected(std::move(error));
    }
    return Ptr(new ActiveSessionLease(data.session_id, std::move(*lock)));
}

ActiveSessionLeaseManager::Reservation::Reservation(
    std::shared_ptr<State> state,
    ActiveSessionLease::Ptr lease) noexcept
    : state_(std::move(state)), lease_(std::move(lease)) {}

ActiveSessionLeaseManager::Reservation::Reservation(Reservation&& other) noexcept
    : state_(std::move(other.state_)),
      lease_(std::move(other.lease_)) {}

ActiveSessionLeaseManager::Reservation&
ActiveSessionLeaseManager::Reservation::operator=(Reservation&& other) noexcept {
    if (this == &other) return *this;
    state_ = std::move(other.state_);
    lease_ = std::move(other.lease_);
    return *this;
}

void ActiveSessionLeaseManager::Reservation::commit() {
    if (!state_) return;
    auto state = std::move(state_);
    std::lock_guard lock(state->mutex);
    state->selected_session_id = lease_->session_id();
    state->leases.insert_or_assign(state->selected_session_id, std::move(lease_));
}

ActiveSessionLeaseManager::ActiveSessionLeaseManager(const SessionStore& store)
    : store_(store), state_(std::make_shared<State>()) {}

std::expected<ActiveSessionLeaseManager::Reservation, std::string>
ActiveSessionLeaseManager::reserve(const SessionData& data) {
    if (auto active = find_active(data.session_id)) {
        return Reservation(state_, std::move(active));
    }

    auto lease = ActiveSessionLease::acquire(store_, data);
    if (lease) return Reservation(state_, std::move(*lease));

    // A concurrent local commit may have won between the first check and the
    // lock attempt. Treat that as an idempotent reservation.
    if (auto active = find_active(data.session_id)) {
        return Reservation(state_, std::move(active));
    }
    return std::unexpected(std::move(lease.error()));
}

ActiveSessionLease::Ptr ActiveSessionLeaseManager::retain() const {
    std::lock_guard lock(state_->mutex);
    const auto it = state_->leases.find(state_->selected_session_id);
    return it == state_->leases.end() ? nullptr : it->second;
}

ActiveSessionLease::Ptr ActiveSessionLeaseManager::retain(
    std::string_view session_id) const {
    std::lock_guard lock(state_->mutex);
    const auto it = state_->leases.find(std::string{session_id});
    return it == state_->leases.end() ? nullptr : it->second;
}

void ActiveSessionLeaseManager::release(std::string_view session_id) {
    std::lock_guard lock(state_->mutex);
    state_->leases.erase(std::string{session_id});
    if (state_->selected_session_id == session_id) {
        state_->selected_session_id.clear();
    }
}

ActiveSessionLease::Ptr ActiveSessionLeaseManager::find_active(
    std::string_view session_id) const {
    std::lock_guard lock(state_->mutex);
    const auto it = state_->leases.find(std::string{session_id});
    return it == state_->leases.end() ? nullptr : it->second;
}

} // namespace core::session
