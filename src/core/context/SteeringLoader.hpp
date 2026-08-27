#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace core::context {

enum class SteeringMode {
    Default,    // Project root discovery (standard behavior)
    None,       // All steering disabled
    CustomFile, // Explicit file path (e.g. /tmp/AGENTS.md)
    CustomDir,  // Custom directory root (e.g. /home/user/Data/)
};

struct SteeringPolicy {
    SteeringMode mode = SteeringMode::Default;
    std::filesystem::path custom_path{};
    std::vector<std::string> disabled_sources{};

    [[nodiscard]] bool is_disabled(const std::filesystem::path& file_path, std::string_view label) const;
    void disable_source(std::string_view label_or_path);
    void enable_source(std::string_view label_or_path);
    void clear_disabled() noexcept { disabled_sources.clear(); }
    [[nodiscard]] std::string format() const;

    bool operator==(const SteeringPolicy&) const = default;
};

[[nodiscard]] SteeringPolicy parse_steering_policy(std::string_view spec);
[[nodiscard]] inline std::string format_steering_policy(const SteeringPolicy& policy) {
    return policy.format();
}

struct SteeringFile {
    std::filesystem::path path;
    std::string label;
    std::string content;
    bool enabled = true;
};

struct SteeringLoadResult {
    std::string block;
    std::vector<std::string> source_labels;
    std::vector<SteeringFile> files;
    SteeringMode mode = SteeringMode::Default;
};

[[nodiscard]] SteeringLoadResult load_project_steering_context(
    const std::filesystem::path& project_root,
    const SteeringPolicy& policy = {});

[[nodiscard]] std::string load_project_steering_block(
    const std::filesystem::path& project_root,
    const SteeringPolicy& policy = {});

} // namespace core::context
