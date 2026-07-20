#pragma once

#include "LandrunPolicy.hpp"

#include <atomic>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace core::landrun {

class LandrunRuntime;

enum class LandrunCapability {
    in_process_untrusted_code,
    unsandboxed_child_process,
};

struct LandrunStartupConfiguration {
    LandrunMode mode{LandrunMode::off};
    std::vector<std::filesystem::path> excluded_paths;
};

/** Process-wide product setting. Tests and library users remain opt-in. */
class LandrunSettings {
public:
    static LandrunSettings& instance() noexcept {
        static LandrunSettings settings;
        return settings;
    }

    [[nodiscard]] LandrunMode mode() const noexcept {
        return mode_.load(std::memory_order_acquire);
    }

    [[nodiscard]] bool enabled() const noexcept {
        return mode() != LandrunMode::off;
    }

    [[nodiscard]] bool permits(LandrunCapability capability) const noexcept {
        (void)capability;
        return !enabled();
    }

    /** Atomically replaces startup-only options until main freezes them. */
    void configure_startup(LandrunStartupConfiguration configuration) {
        std::vector<std::filesystem::path> normalized;
        normalized.reserve(configuration.excluded_paths.size());
        for (const auto& path : configuration.excluded_paths) {
            auto value = normalize_landrun_path(path);
            if (value.empty()
                || std::ranges::find(normalized, value) != normalized.end()) {
                continue;
            }
            normalized.push_back(std::move(value));
        }
        std::lock_guard lock(mutex_);
        if (startup_configuration_frozen_) {
            throw std::logic_error(
                "landrun startup configuration is already frozen");
        }
        excluded_paths_ = std::move(normalized);
        mode_.store(configuration.mode, std::memory_order_release);
    }

    void freeze_startup_configuration() noexcept {
        std::lock_guard lock(mutex_);
        startup_configuration_frozen_ = true;
    }

    [[nodiscard]] LandrunStartupConfiguration startup_configuration() const {
        std::lock_guard lock(mutex_);
        return {
            .mode = mode_.load(std::memory_order_acquire),
            .excluded_paths = excluded_paths_,
        };
    }

    [[nodiscard]] std::vector<std::filesystem::path> excluded_paths() const {
        std::lock_guard lock(mutex_);
        return excluded_paths_;
    }

    [[nodiscard]] bool is_excluded(const std::filesystem::path& path) const {
        const auto normalized = normalize_landrun_path(path);
        std::lock_guard lock(mutex_);
        return std::ranges::any_of(excluded_paths_, [&](const auto& excluded) {
            return is_landrun_path_within(excluded, normalized);
        });
    }

    [[nodiscard]] const std::filesystem::path& host_tmpdir() const noexcept {
        return host_tmpdir_;
    }

    [[nodiscard]] std::filesystem::path runtime_root() const {
        std::lock_guard lock(mutex_);
        return runtime_root_;
    }

    [[nodiscard]] std::filesystem::path runtime_home() const {
        const auto root = runtime_root();
        return root.empty() ? root : root / "home";
    }

    [[nodiscard]] std::filesystem::path runtime_tmp() const {
        const auto root = runtime_root();
        return root.empty() ? root : root / "tmp";
    }

    [[nodiscard]] std::filesystem::path effective_tmpdir() const {
        if (!host_tmpdir_.empty() && !is_excluded(host_tmpdir_)) {
            return host_tmpdir_;
        }
        return runtime_tmp();
    }

private:
    friend class LandrunRuntime;

    LandrunSettings() {
        if (const char* value = std::getenv("TMPDIR"); value && *value) {
            host_tmpdir_ = normalize_landrun_path(value);
        }
    }

    void set_runtime_root(std::filesystem::path root) {
        std::lock_guard lock(mutex_);
        runtime_root_ = std::move(root);
    }

    std::atomic<LandrunMode> mode_{LandrunMode::off};
    std::filesystem::path host_tmpdir_;
    mutable std::mutex mutex_;
    bool startup_configuration_frozen_{false};
    std::filesystem::path runtime_root_;
    std::vector<std::filesystem::path> excluded_paths_;
};

} // namespace core::landrun
