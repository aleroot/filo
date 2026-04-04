#pragma once
#include <filesystem>
#include <vector>

namespace core::workspace {

class Workspace {
public:
    static Workspace& get_instance() {
        static Workspace instance;
        return instance;
    }

    // Initialize with primary working directory and any additional directories
    void initialize(std::filesystem::path primary, std::vector<std::filesystem::path> additional, bool enforce = true);

    [[nodiscard]] const std::filesystem::path& get_primary() const noexcept { return primary_; }
    [[nodiscard]] const std::vector<std::filesystem::path>& get_additional() const noexcept { return additional_; }
    [[nodiscard]] bool is_enforced() const noexcept { return enforce_; }

    // Check if the target path is strictly within the primary or additional directories
    [[nodiscard]] bool is_path_allowed(const std::filesystem::path& target_path) const;

private:
    Workspace() = default;

    std::filesystem::path primary_;
    std::vector<std::filesystem::path> additional_;
    bool enforce_{false};
};

} // namespace core::workspace
