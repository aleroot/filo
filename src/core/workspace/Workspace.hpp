#pragma once
#include "FileAccessScope.hpp"

#include <filesystem>
#include <cstdint>
#include <vector>

namespace core::workspace {

struct WorkspaceSnapshot {
    std::filesystem::path primary;
    std::vector<std::filesystem::path> additional;
    bool enforce{false};
    std::uint64_t version{0};

    /// Directories outside the project roots that tools may touch. Built in
    /// the composition root from the active sandbox posture and carried here
    /// so it flows to every session without a process-wide instance. Empty by
    /// default, which fails closed.
    FileAccessScope scratch;
};

class Workspace {
public:
    static Workspace& get_instance() {
        static Workspace instance;
        return instance;
    }

    // Initialize with primary working directory and any additional directories
    void initialize(std::filesystem::path primary,
                    std::vector<std::filesystem::path> additional,
                    bool enforce = true,
                    FileAccessScope scratch = {});

    [[nodiscard]] const std::filesystem::path& get_primary() const noexcept { return primary_; }
    [[nodiscard]] const std::vector<std::filesystem::path>& get_additional() const noexcept { return additional_; }
    [[nodiscard]] bool is_enforced() const noexcept { return enforce_; }
    [[nodiscard]] WorkspaceSnapshot snapshot() const;
    [[nodiscard]] std::filesystem::path resolve_path(const std::filesystem::path& target_path) const;

    // True when the path is inside the project roots or the readable/writable
    // half of the configured scratch scope.
    [[nodiscard]] bool allows_read(const std::filesystem::path& target_path) const;
    [[nodiscard]] bool allows_write(const std::filesystem::path& target_path) const;

private:
    Workspace() = default;

    std::filesystem::path primary_;
    std::vector<std::filesystem::path> additional_;
    bool enforce_{false};
    FileAccessScope scratch_;
};

} // namespace core::workspace
