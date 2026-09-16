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

/**
 * Every filename Filo reads as project steering, split by how far it is searched.
 *
 * Single source of truth for the names: discovery, the I/O-free candidate
 * filter, the read-side SteeringGuard, and every "is this an instruction file?"
 * classifier in the read tool and the tool-output summarizer all consult these
 * tables, so a file can never be loadable into the prompt yet freely readable
 * through a tool (or the reverse). Adding a name here is what makes it steering
 * everywhere — and removing one stops it being steering everywhere.
 *
 * Anything that needs to ask the question must call is_steering_filename() or
 * is_steering_candidate() rather than spelling names out: a second list is a
 * list that drifts.
 */
/// Searched in a steering root *and* in every directory above it up to the
/// project root. Order is priority: the loader reads only the first name present
/// in a directory and treats the rest as shadowed, but every one of them still
/// exists, so enumerate_steering_files() reports them all.
[[nodiscard]] constexpr std::array<std::string_view, 2> hierarchical_steering_names() noexcept {
    return {{"AGENTS.override.md", "AGENTS.md"}};
}

/// Filo's own steering file: the one `/init` writes and the one read first.
inline constexpr std::string_view kPrimarySteeringFileName = "FILO.md";

/// Searched in a steering root itself, in this order.
[[nodiscard]] constexpr std::array<std::string_view, 6> root_steering_names() noexcept {
    return {{kPrimarySteeringFileName, "GEMINI.md", "CLAUDE.md", "SYSTEM.md",
             "CURSOR.md", "COPILOT.md"}};
}

/// Directory under a steering root whose `.md` files are all steering.
inline constexpr std::string_view kSteeringDirectoryName = ".filo/steering";

/// True when @p filename (a bare name, matched case-insensitively because
/// discovery does) is one of the steering filenames above.
[[nodiscard]] bool is_steering_filename(std::string_view filename) noexcept;

/// True when @p dir is a `.filo/steering` directory.
[[nodiscard]] bool is_steering_directory(const std::filesystem::path& dir) noexcept;

/**
 * I/O-free necessary condition for "a steering *file* could live at this path".
 *
 * The directory rule is deliberately absent: `.filo/steering` is a location that
 * contains steering, not instruction text itself. Consumers that classify the
 * content behind a path (the reader's verbatim instruction path, output
 * summarization) must use this, or they claim the directory and break listing it.
 */
[[nodiscard]] bool is_steering_file_candidate(const std::filesystem::path& path) noexcept;

/**
 * I/O-free necessary condition for "a steering policy could block this path".
 *
 * Lets the per-path authorization gate skip steering enforcement — and skip it
 * without a single stat() — for the overwhelming majority of paths, including
 * every entry of a large directory listing.
 *
 * Wider than is_steering_file_candidate() by exactly the steering directory:
 * a fully blocked `.filo/steering` is itself denied, so that listing it cannot
 * reveal the names inside.
 */
[[nodiscard]] bool is_steering_candidate(const std::filesystem::path& path) noexcept;

/**
 * On-disk identity of a steering file: fully symlink-resolved, so that two
 * discovered entries pointing at the same file (the common
 * `CLAUDE.md -> AGENTS.md` symlink) compare equal.
 */
[[nodiscard]] std::filesystem::path steering_file_identity(const std::filesystem::path& path);

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

/**
 * The persistable token for @p mode, or empty for the path-parameterized modes.
 *
 * Lets the CLI default, the `--no-steering` alias, and anything else that has
 * to *name* a mode read the spelling from the table instead of restating it —
 * a token typed twice is a token that can stop parsing after a rename.
 */
[[nodiscard]] constexpr std::string_view steering_mode_token(SteeringMode mode) noexcept {
    for (const auto& option : steering_mode_options()) {
        if (option.mode == mode) return option.token;
    }
    return {};
}

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
 * Which files a policy selects, without reading any of them.
 *
 * Same shape as the load result minus the rendered text: the selection is what
 * SteeringGuard subtracts from discovery, and keeping it separate means the
 * guard never has to read 48 KiB of instructions it is only trying to classify.
 * `files[].content` is always empty here; `files[].enabled` already reflects
 * `SteeringPolicy::disabled_sources`.
 */
struct SteeringSelection {
    std::vector<SteeringFile> files;
    SteeringMode mode = SteeringMode::Default;
    std::filesystem::path root;
    std::vector<std::filesystem::path> searched_roots;

    [[nodiscard]] bool empty() const noexcept { return files.empty(); }
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
 * Every steering file the loader would read under @p roots, in chain order.
 *
 * Deliberately policy-free and content-free. Note that "would read" is narrower
 * than "exists": AGENTS.override.md shadows a sibling AGENTS.md, so the shadowed
 * file is absent here. Anything deciding what to *load* wants this; anything
 * deciding what to *deny* wants enumerate_steering_files().
 */
[[nodiscard]] std::vector<SteeringFile> discover_steering_files(
    const std::vector<std::filesystem::path>& roots);

/**
 * Every steering file physically present under @p roots, in chain order and
 * de-duplicated by on-disk identity.
 *
 * For an explicitly restrictive policy, the difference between this set and
 * select_steering_files() is what SteeringGuard blocks. Automatic Default and
 * Fallback selection alone does not restrict tool access. This set includes
 * files the loader skips — a shadowed AGENTS.md is still a project
 * instruction file on disk, and `--no-steering` has to withhold it too. Using
 * discover_steering_files() here instead is how a shadowed file escapes every
 * policy at once.
 */
[[nodiscard]] std::vector<SteeringFile> enumerate_steering_files(
    const std::vector<std::filesystem::path>& roots);

/**
 * The steering files @p policy selects under @p roots, labelled but unread.
 *
 * `load_workspace_steering_context()` is this plus rendering; splitting them is
 * what lets the read-side guard and the prompt agree on the selected set by
 * construction instead of by duplicated mode logic.
 */
[[nodiscard]] SteeringSelection select_steering_files(
    const std::vector<std::filesystem::path>& roots,
    const SteeringPolicy& policy);

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
