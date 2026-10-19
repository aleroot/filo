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

// Owned by the execution runtime and shared explicitly with its tools, hooks
// and recovery operations. Independent runtimes have independent coordinators.
class WorkspaceMutationGate final {
    struct State;
public:
    enum class Access { ScopedWrite, Exclusive };

    // A reservation can move between threads and outlive its coordinator.
    class Lease final {
    public:
        ~Lease();
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
    private:
        friend class WorkspaceMutationGate;
        explicit Lease(std::shared_ptr<State> state, Access access);
        std::shared_ptr<State> state_;
        Access access_;
        std::vector<std::filesystem::path> roots_;
        bool registered_ = false;
    };

    WorkspaceMutationGate();
    WorkspaceMutationGate(const WorkspaceMutationGate&) = delete;
    WorkspaceMutationGate& operator=(const WorkspaceMutationGate&) = delete;

    [[nodiscard]] std::expected<std::shared_ptr<Lease>, std::string>
    acquire(std::span<const std::filesystem::path> roots, Access access,
            std::chrono::milliseconds timeout = std::chrono::seconds(30),
            const std::function<bool()>& cancelled = {});

private:
    std::shared_ptr<State> state_;
};

} // namespace core::changes
