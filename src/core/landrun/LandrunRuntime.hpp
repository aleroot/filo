#pragma once

#include <filesystem>

namespace core::landrun {

/** Owns the process-private scratch/home tree used by sandboxed children. */
class LandrunRuntime {
public:
    LandrunRuntime();
    ~LandrunRuntime();

    LandrunRuntime(const LandrunRuntime&) = delete;
    LandrunRuntime& operator=(const LandrunRuntime&) = delete;

    [[nodiscard]] const std::filesystem::path& root() const noexcept { return root_; }

private:
    std::filesystem::path root_;
};

} // namespace core::landrun
