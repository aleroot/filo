#pragma once

#include <chrono>
#include <expected>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace core::changes {

// Runtime-owned reservations for declared mutation and restoration targets.
// Permission grants and unscoped execution never enter this coordinator.
class WorkspaceMutationGate final {
    struct State;
public:
    enum class Failure { Busy, Cancelled };

    [[nodiscard]] static constexpr const char* failure_message(Failure failure) noexcept {
        return failure == Failure::Cancelled
            ? "Workspace mutation cancelled."
            : "A target path has an active writer; retry after it stops.";
    }

    // A reservation can move between threads and outlive its coordinator.
    class Lease final {
    public:
        ~Lease();
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
    private:
        friend class WorkspaceMutationGate;
        explicit Lease(std::shared_ptr<State> state);
        std::shared_ptr<State> state_;
        std::vector<std::filesystem::path> roots_;
    };

    WorkspaceMutationGate();
    WorkspaceMutationGate(const WorkspaceMutationGate&) = delete;
    WorkspaceMutationGate& operator=(const WorkspaceMutationGate&) = delete;

    [[nodiscard]] std::expected<std::shared_ptr<Lease>, Failure>
    acquire(std::span<const std::filesystem::path> roots,
            std::chrono::milliseconds timeout = std::chrono::seconds(30),
            const std::function<bool()>& cancelled = {});

private:
    std::shared_ptr<State> state_;
};

} // namespace core::changes
