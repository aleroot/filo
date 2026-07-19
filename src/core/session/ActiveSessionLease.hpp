#pragma once

#include "core/utils/InterprocessFile.hpp"

#include <expected>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

namespace core::session {

struct SessionData;
class SessionStore;

[[nodiscard]] std::filesystem::path active_session_lease_path(
    const std::filesystem::path& session_path);

// Move-only RAII lease preventing two processes from actively mutating the
// same resumable conversation. The lock file may remain; ownership is the
// kernel-held advisory lock, which is released on destruction or process exit.
class ActiveSessionLease {
public:
    ActiveSessionLease(ActiveSessionLease&&) noexcept = default;
    ActiveSessionLease& operator=(ActiveSessionLease&&) noexcept = default;

    ActiveSessionLease(const ActiveSessionLease&) = delete;
    ActiveSessionLease& operator=(const ActiveSessionLease&) = delete;

    using Ptr = std::shared_ptr<ActiveSessionLease>;

    [[nodiscard]] static std::expected<Ptr, std::string> acquire(
        const SessionStore& store,
        const SessionData& data);

    [[nodiscard]] const std::string& session_id() const noexcept {
        return session_id_;
    }

private:
    ActiveSessionLease(std::string session_id,
                       core::utils::InterprocessFileLock lock) noexcept;

    std::string session_id_;
    core::utils::InterprocessFileLock lock_;
};

// Coordinates the current lease through an explicit two-phase transition:
// reserve the destination first, mutate session state, then commit. Dropping an
// uncommitted reservation releases only the destination lease.
class ActiveSessionLeaseManager {
    struct State;

public:
    class Reservation {
    public:
        Reservation(Reservation&& other) noexcept;
        Reservation& operator=(Reservation&& other) noexcept;

        Reservation(const Reservation&) = delete;
        Reservation& operator=(const Reservation&) = delete;

        void commit();

    private:
        friend class ActiveSessionLeaseManager;
        Reservation(std::shared_ptr<State> state,
                    ActiveSessionLease::Ptr lease) noexcept;

        std::shared_ptr<State> state_;
        ActiveSessionLease::Ptr lease_;
    };

    explicit ActiveSessionLeaseManager(const SessionStore& store);

    ActiveSessionLeaseManager(const ActiveSessionLeaseManager&) = delete;
    ActiveSessionLeaseManager& operator=(const ActiveSessionLeaseManager&) = delete;
    ActiveSessionLeaseManager(ActiveSessionLeaseManager&&) = delete;
    ActiveSessionLeaseManager& operator=(ActiveSessionLeaseManager&&) = delete;

    [[nodiscard]] std::expected<Reservation, std::string> reserve(
        const SessionData& data);

    // Retains the active lease for asynchronous work on its session.
    [[nodiscard]] ActiveSessionLease::Ptr retain() const;

private:
    [[nodiscard]] ActiveSessionLease::Ptr find_active(
        std::string_view session_id) const;

    struct State {
        mutable std::mutex mutex;
        ActiveSessionLease::Ptr active;
    };

    const SessionStore& store_;
    std::shared_ptr<State> state_;
};

} // namespace core::session
