#pragma once

#include <array>
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
    // Chain of responsibility across the workspace roots: try the primary, and
    // if it carries no steering at all, fall through to each additional root in
    // order and use the first one that does. Appended last so the existing
    // enumerator values stay put.
    Fallback,
};

/**
 * A steering mode a user can select and persist.
 *
 * CustomFile and CustomDir are deliberately absent: they are parameterized by a
 * path, so they are expressible only as `--steering <path>` / `/steering <path>`
 * and can never be stored as a bare mode token.
 */
struct SteeringModeOption {
    SteeringMode mode;
    std::string_view token;        ///< Round-trips through parse_steering_policy().
    std::string_view label;        ///< Short menu name.
    std::string_view description;  ///< One-line explanation for menus and --help.
};

/**
 * Every persistable steering mode, in the order menus should present them.
 *
 * Single source of truth shared by the CLI parser, the persisted
 * `steering_mode` setting, and the settings panel, so a token can never mean
 * one thing on the command line and another in settings.json.
 */
[[nodiscard]] constexpr std::array<SteeringModeOption, 3> steering_mode_options() noexcept {
    return {{
        {.mode = SteeringMode::Default,
         .token = "default",
         .label = "Default",
         .description = "Load steering from the primary workspace only."},
        {.mode = SteeringMode::Fallback,
         .token = "fallback",
         .label = "Fallback",
         .description = "Use the first workspace root that has steering files."},
        {.mode = SteeringMode::None,
         .token = "none",
         .label = "Disabled",
         .description = "Load no project steering at all."},
    }};
}

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
    /// Workspace root this file was discovered under. Lets the UI attribute a
    /// file to its project instead of implying everything came from the primary.
    std::filesystem::path root;
};

struct SteeringLoadResult {
    std::string block;
    std::vector<std::string> source_labels;
    std::vector<SteeringFile> files;
    SteeringMode mode = SteeringMode::Default;
    /// The root whose files were used; empty when no steering was found.
    std::filesystem::path root;
    /// Roots actually consulted, in chain order. Populated by Fallback so the
    /// UI can explain "looked in X, found nothing, used Y".
    std::vector<std::filesystem::path> searched_roots;
};

/**
 * Ordered steering roots for a workspace: the primary first, then each
 * additional root, empties dropped and CLI order preserved.
 *
 * Fallback mode walks this list; Default mode only ever consults the first
 * entry, so passing the full list is always safe.
 */
[[nodiscard]] std::vector<std::filesystem::path> collect_steering_roots(
    const std::filesystem::path& primary,
    const std::vector<std::filesystem::path>& additional = {});

/**
 * Load steering for a whole workspace.
 *
 * Default mode reads @p roots[0] only, exactly as the single-root overload
 * always has. Fallback mode walks the roots in order and returns the first
 * root that has any steering file at all — a root whose only file the user
 * explicitly unloaded still counts as "has steering", so the chain stops there
 * rather than quietly substituting another project's instructions.
 * CustomFile and CustomDir ignore @p roots entirely.
 */
[[nodiscard]] SteeringLoadResult load_workspace_steering_context(
    const std::vector<std::filesystem::path>& roots,
    const SteeringPolicy& policy = {});

[[nodiscard]] std::string load_workspace_steering_block(
    const std::vector<std::filesystem::path>& roots,
    const SteeringPolicy& policy = {});

[[nodiscard]] SteeringLoadResult load_project_steering_context(
    const std::filesystem::path& project_root,
    const SteeringPolicy& policy = {});

[[nodiscard]] std::string load_project_steering_block(
    const std::filesystem::path& project_root,
    const SteeringPolicy& policy = {});

} // namespace core::context
