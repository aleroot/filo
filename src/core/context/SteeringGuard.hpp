#pragma once

#include "SteeringLoader.hpp"

#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace core::context {

struct SessionContext;

/**
 * Tool-side enforcement of a SteeringPolicy.
 *
 * A policy decides which instruction files reach the prompt; the guard decides
 * which ones the agent may fetch for itself. Without the second half,
 * `--no-steering` is advisory only: the model simply calls `read` on AGENTS.md
 * (or `cat` through the shell) and the instructions the user turned off come
 * back in through a tool result.
 *
 * Default and Fallback select which files are added to the prompt; that alone
 * does not restrict tool access, regardless of the number of workspace roots.
 * Enforcement requires an explicit restriction: None, a custom file/directory,
 * or individually disabled sources. For those policies the blocked set is
 * derived from every steering file present minus the files the policy loads.
 *
 * This is a policy barrier, not a security sandbox: it stops an agent from
 * reading steering it was told to ignore, not a determined attacker with
 * shell access (who is already the user).
 */
class SteeringGuard {
public:
    SteeringGuard() = default;

    /**
     * I/O-free necessary condition: does @p policy explicitly restrict steering?
     *
     * When false, the guard is provably empty and callers skip discovery
     * entirely — which keeps the default configuration free of any extra
     * filesystem work on the hot path.
     */
    [[nodiscard]] static bool may_block(const SteeringPolicy& policy) noexcept;

    [[nodiscard]] static SteeringGuard for_roots(
        const std::vector<std::filesystem::path>& roots,
        const SteeringPolicy& policy);

    [[nodiscard]] static SteeringGuard for_context(const SessionContext& context);

    [[nodiscard]] bool empty() const noexcept { return blocked_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return blocked_.size(); }

    /**
     * Why @p resolved_path is off-limits, or nullopt when it is not blocked.
     *
     * @p resolved_path must already be workspace-resolved (as produced by
     * SessionContext::resolve_path); symlinks are resolved again here so an
     * alias such as `CLAUDE.md -> AGENTS.md` is caught by identity rather than
     * by name.
     */
    [[nodiscard]] std::optional<std::string> blocked_reason(
        const std::filesystem::path& resolved_path) const;

    /**
     * Why a free-form command or snippet is off-limits: it references a blocked
     * steering file.
     *
     * Used for the tools whose argument is *code* rather than a path — the
     * shell, the embedded Python interpreter, and anything that builds a
     * command line — where there is no single path to validate. Relative
     * references are resolved against @p base_dir, the directory the command
     * will actually run in. Glob tokens (such as `*.md` or a wildcard inside
     * `.filo/steering`) are matched as patterns, so a wildcard cannot be used to
     * reach a blocked file.
     */
    [[nodiscard]] std::optional<std::string> blocked_text_reason(
        std::string_view text,
        const std::filesystem::path& base_dir) const;

    /// Display labels of every blocked file, in discovery order.
    [[nodiscard]] std::vector<std::string> blocked_labels() const;

    /**
     * The blocked set as filesystem hierarchies, in discovery order: every
     * withheld file plus any steering directory hidden as a whole.
     *
     * This is the only representation that survives a hand-off to something
     * that cannot ask Filo questions — an OS sandbox profile denies a path, it
     * does not resolve tokens. Consumers must not re-derive the set from
     * labels, which are workspace-relative and would silently miss a
     * secondary root's file.
     */
    [[nodiscard]] std::vector<std::filesystem::path> blocked_paths() const;

    /**
     * Prompt-visible statement of the restriction, empty when nothing is
     * blocked. Telling the model the files are unreadable is what keeps it from
     * spending turns rediscovering that fact through rejected tool calls.
     */
    [[nodiscard]] std::string prompt_notice() const;

private:
    struct Entry {
        std::filesystem::path identity;  ///< Symlink-resolved, for comparison.
        std::filesystem::path parent;    ///< Identity's directory, for the directory rule.
        std::string label;               ///< Workspace-relative, for messages.
        bool directory = false;          ///< A steering dir whose whole content is blocked.
    };

    [[nodiscard]] const Entry* match_identity(const std::filesystem::path& identity) const;
    [[nodiscard]] const Entry* match_filename(std::string_view name) const;
    [[nodiscard]] const Entry* match_glob(std::string_view token,
                                          const std::filesystem::path& base_dir) const;
    [[nodiscard]] const Entry* match_reference(std::string_view token,
                                               const std::filesystem::path& base_dir) const;
    [[nodiscard]] std::string describe(const Entry& entry, bool referenced_by_command) const;

    std::vector<Entry> blocked_;
    std::string policy_description_;
};

} // namespace core::context
